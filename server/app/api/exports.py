import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app import storage
from app.db import get_session
from app.models import Annotation, Document, Image, Project, ProjectClass
from app.services import export as export_service

router = APIRouter(tags=["export"])


async def _project_classes(session: AsyncSession, project_id: uuid.UUID) -> list[str]:
    """Имена классов проекта в порядке sort_order: порядок задаёт class_idx в YOLO
    и category_id в COCO, поэтому он должен быть таким же, как в UI."""
    result = await session.execute(
        select(ProjectClass.name)
        .where(ProjectClass.project_id == project_id)
        .order_by(ProjectClass.sort_order, ProjectClass.created_at)
    )
    return list(result.scalars().all())


@router.get("/projects/{project_id}/export/formats")
async def export_formats(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    """Форматы, доступные проекту (зависят от task_type)."""
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
    """Скачать ZIP с датасетом. Только проверенные аннотации (accepted/edited)."""
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

    rows = (
        (
            await session.execute(
                select(Image, Annotation)
                .join(Document, Image.document_id == Document.id)
                .join(Annotation, Annotation.image_id == Image.id)
                .where(
                    Document.project_id == project_id,
                    Annotation.status.in_(export_service.EXPORTABLE_STATUSES),
                )
                .order_by(Image.created_at, Annotation.created_at)
            )
        )
        .all()
    )

    # группируем аннотации по изображениям и подтягиваем байты из S3
    by_image: dict[uuid.UUID, dict] = {}
    for image, annotation in rows:
        entry = by_image.setdefault(image.id, {"image": image, "annotations": []})
        entry["annotations"].append(annotation)

    items = []
    for entry in by_image.values():
        data = await run_in_threadpool(storage.get_bytes, entry["image"].s3_key)
        items.append(
            export_service.ExportItem(entry["image"], data, entry["annotations"])
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
