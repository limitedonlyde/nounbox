import uuid
from collections import defaultdict
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app import storage
from app.db import get_session
from app.models import Annotation, AnnotationStatus, Document, Image, Project, ProjectClass
from app.services import export as export_service

router = APIRouter(tags=["export"])


async def _project_classes(session: AsyncSession, project_id: uuid.UUID) -> list[str]:
    """Project class names in sort_order: the order defines class_idx in YOLO and
    category_id in COCO, so it has to be the same as in the UI."""
    result = await session.execute(
        select(ProjectClass.name)
        .where(ProjectClass.project_id == project_id)
        .order_by(ProjectClass.sort_order, ProjectClass.created_at)
    )
    return list(result.scalars().all())


async def _reviewed_images(
    session: AsyncSession, project_id: uuid.UUID
) -> list[Image]:
    """Project frames a human has already looked at.

    A frame without a single box is a legitimate negative example, but only
    if it has been checked: it has an annotation in accepted/edited/rejected,
    or Image.reviewed is set. Unreviewed frames stay out of the dataset,
    otherwise an unlabeled object goes into training as background.
    """
    # the frame carries finished labels, so it goes into the dataset with them
    has_exportable = (
        select(Annotation.id)
        .where(
            Annotation.image_id == Image.id,
            Annotation.status.in_(export_service.EXPORTABLE_STATUSES),
        )
        .exists()
    )
    # candidate background frame. "A human decided something here" is not enough: if
    # an unreviewed box is still left on that frame, it is NOT empty but merely
    # unfinished, and an empty label file would teach the model to treat the
    # object as background.
    has_pending = (
        select(Annotation.id)
        .where(
            Annotation.image_id == Image.id,
            Annotation.status == AnnotationStatus.PENDING,
        )
        .exists()
    )
    rejected_only = (
        select(Annotation.id)
        .where(
            Annotation.image_id == Image.id,
            Annotation.status == AnnotationStatus.REJECTED,
        )
        .exists()
    )
    empty_and_checked = and_(
        ~has_pending,
        or_(Image.reviewed.is_(True), rejected_only),
    )
    result = await session.execute(
        select(Image)
        .join(Document, Image.document_id == Document.id)
        .where(
            Document.project_id == project_id,
            or_(has_exportable, empty_and_checked),
        )
        # id is the tiebreaker: created_at is the same for a batch of frames from
        # one document, and file names and image_id in COCO depend on the order
        .order_by(Image.created_at, Image.id)
    )
    return list(result.scalars().all())


async def _exportable_annotations(
    session: AsyncSession, project_id: uuid.UUID
) -> dict[uuid.UUID, list[Annotation]]:
    """Reviewed annotations of the project, grouped by image."""
    result = await session.execute(
        select(Annotation)
        .join(Image, Annotation.image_id == Image.id)
        .join(Document, Image.document_id == Document.id)
        .where(
            Document.project_id == project_id,
            Annotation.status.in_(export_service.EXPORTABLE_STATUSES),
        )
        .order_by(Annotation.created_at)
    )
    by_image: dict[uuid.UUID, list[Annotation]] = defaultdict(list)
    for annotation in result.scalars().all():
        by_image[annotation.image_id].append(annotation)
    return by_image


@router.get("/projects/{project_id}/export/formats")
async def export_formats(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    """Formats available to the project (they depend on task_type)."""
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return {
        "task_type": project.task_type,
        "formats": list(export_service.formats_for(project.task_type)),
    }


@router.get("/projects/{project_id}/export")
async def export_project(
    project_id: uuid.UUID,
    format: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Download a ZIP with the dataset. Reviewed annotations only (accepted/edited)."""
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")

    task_type = project.task_type
    available = export_service.formats_for(task_type)
    format = format or available[0]
    if format not in available:
        raise HTTPException(
            400,
            f"Format {format} is not available for task_type={task_type}. "
            f"Available: {', '.join(available)}",
        )
    classes = await _project_classes(session, project_id)

    images = await _reviewed_images(session, project_id)
    by_image = await _exportable_annotations(session, project_id)

    items = []
    for image in images:
        data = await run_in_threadpool(storage.get_bytes, image.s3_key)
        # annotations may be absent entirely: a reviewed empty frame is a
        # background example
        items.append(
            export_service.ExportItem(image, data, by_image.get(image.id, []))
        )

    try:
        zip_bytes = await run_in_threadpool(
            export_service.build_zip, format, items, str(project_id), task_type, classes
        )
    except export_service.ExportError as exc:
        raise HTTPException(400, str(exc))

    filename = f"dataset_{format}_{date.today().isoformat()}.zip"
    return Response(
        zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
