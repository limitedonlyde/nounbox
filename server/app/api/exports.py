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
    """Имена классов проекта в порядке sort_order: порядок задаёт class_idx в YOLO
    и category_id в COCO, поэтому он должен быть таким же, как в UI."""
    result = await session.execute(
        select(ProjectClass.name)
        .where(ProjectClass.project_id == project_id)
        .order_by(ProjectClass.sort_order, ProjectClass.created_at)
    )
    return list(result.scalars().all())


async def _reviewed_images(
    session: AsyncSession, project_id: uuid.UUID
) -> list[Image]:
    """Кадры проекта, которые человек уже смотрел.

    Кадр без единой рамки — легальный негативный пример, но только если он
    проверен: есть аннотация в accepted/edited/rejected либо стоит
    Image.reviewed. Непросмотренные кадры в датасет не идут, иначе
    неразмеченный объект уедет в обучение как фон.
    """
    # кадр несёт готовую разметку — идёт в датасет вместе с ней
    has_exportable = (
        select(Annotation.id)
        .where(
            Annotation.image_id == Image.id,
            Annotation.status.in_(export_service.EXPORTABLE_STATUSES),
        )
        .exists()
    )
    # кадр-кандидат в фон. Одного «человек что-то тут решал» мало: если рядом
    # осталась непросмотренная рамка, кадр НЕ пустой — он недоделанный, и
    # пустой файл меток научил бы модель считать объект фоном.
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
        # id — тайбрейкер: created_at у пачки кадров одного документа совпадает,
        # а от порядка зависят имена файлов и image_id в COCO
        .order_by(Image.created_at, Image.id)
    )
    return list(result.scalars().all())


async def _exportable_annotations(
    session: AsyncSession, project_id: uuid.UUID
) -> dict[uuid.UUID, list[Annotation]]:
    """Проверенные аннотации проекта, сгруппированные по изображению."""
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

    images = await _reviewed_images(session, project_id)
    by_image = await _exportable_annotations(session, project_id)

    items = []
    for image in images:
        data = await run_in_threadpool(storage.get_bytes, image.s3_key)
        # аннотаций может не быть вовсе: проверенный пустой кадр — фоновый пример
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
