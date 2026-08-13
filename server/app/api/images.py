import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app import storage
from app.db import get_session
from app.models import Annotation, AnnotationStatus, Document, Image
from app.schemas import ImageOut, ImageUpdate, ImageWithStats

router = APIRouter(tags=["images"])


@router.get("/projects/{project_id}/images", response_model=list[ImageWithStats])
async def list_project_images(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    """Все изображения проекта со статистикой аннотаций (для списка/прогресса)."""
    total = func.count(Annotation.id)
    pending = func.count(Annotation.id).filter(
        Annotation.status == AnnotationStatus.PENDING
    )
    result = await session.execute(
        select(Image, total.label("total"), pending.label("pending"))
        .join(Document, Image.document_id == Document.id)
        .outerjoin(Annotation, Annotation.image_id == Image.id)
        .where(Document.project_id == project_id)
        .group_by(Image.id)
        .order_by(Image.created_at)
    )
    fields = ImageOut.model_fields.keys()
    return [
        ImageWithStats(
            **{f: getattr(image, f) for f in fields},
            total_annotations=total_count,
            pending_annotations=pending_count,
        )
        for image, total_count, pending_count in result.all()
    ]


@router.get("/documents/{document_id}/images", response_model=list[ImageOut])
async def list_document_images(
    document_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    result = await session.execute(
        select(Image)
        .where(Image.document_id == document_id)
        .order_by(Image.page_index)
    )
    return result.scalars().all()


@router.get("/images/{image_id}", response_model=ImageOut)
async def get_image(image_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "Image not found")
    return image


@router.get("/images/{image_id}/url")
async def get_image_url(image_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """Presigned URL для отображения в review UI."""
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "Image not found")
    url = await run_in_threadpool(storage.presigned_url, image.s3_key)
    return {"url": url}


@router.patch("/images/{image_id}", response_model=ImageOut)
async def update_image(
    image_id: uuid.UUID,
    body: ImageUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Пометить кадр просмотренным.

    Единственный способ сказать «я смотрел, объектов тут нет»: у такого кадра
    нет ни одной аннотации, поэтому иначе он неотличим от неразмеченного и в
    датасет как фоновый пример не попадёт.
    """
    image = await session.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "Image not found")
    image.reviewed = body.reviewed
    await session.commit()
    await session.refresh(image)
    return image
