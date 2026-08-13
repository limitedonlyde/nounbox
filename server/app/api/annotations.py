import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Annotation, AnnotationStatus, Document, Image, Project
from app.schemas import (
    AnnotationCreate,
    AnnotationOut,
    AnnotationUpdate,
    BulkAcceptRequest,
)

router = APIRouter(tags=["annotations"])

HUMAN_SOURCE = {"type": "human", "name": "manual", "version": None}


@router.post("/projects/{project_id}/annotations/bulk-accept")
async def bulk_accept(
    project_id: uuid.UUID,
    body: BulkAcceptRequest,
    session: AsyncSession = Depends(get_session),
):
    """Accept all pending annotations in the project with confidence >= threshold."""
    if await session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")
    project_images = (
        select(Image.id)
        .join(Document, Image.document_id == Document.id)
        .where(Document.project_id == project_id)
    )
    conditions = [
        Annotation.image_id.in_(project_images),
        Annotation.status == AnnotationStatus.PENDING,
        Annotation.confidence >= body.min_confidence,
    ]
    if body.labeler:
        conditions.append(Annotation.source["name"].astext == body.labeler)

    result = await session.execute(
        update(Annotation).where(*conditions).values(status=AnnotationStatus.ACCEPTED)
    )
    await session.commit()
    return {"accepted": result.rowcount}


@router.get("/images/{image_id}/annotations", response_model=list[AnnotationOut])
async def list_annotations(
    image_id: uuid.UUID,
    status: AnnotationStatus | None = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(Annotation).where(Annotation.image_id == image_id)
    if status is not None:
        query = query.where(Annotation.status == status)
    result = await session.execute(query.order_by(Annotation.created_at))
    return result.scalars().all()


@router.post("/images/{image_id}/annotations", response_model=AnnotationOut, status_code=201)
async def create_annotation(
    image_id: uuid.UUID,
    body: AnnotationCreate,
    session: AsyncSession = Depends(get_session),
):
    if await session.get(Image, image_id) is None:
        raise HTTPException(404, "Image not found")
    annotation = Annotation(
        image_id=image_id,
        geometry=body.geometry.model_dump(),
        label=body.label,
        text=body.text,
        attrs=body.attrs,
        confidence=body.confidence,
        parent_id=body.parent_id,
        source=HUMAN_SOURCE,
        status=AnnotationStatus.ACCEPTED,  # manual labeling is accepted right away
    )
    session.add(annotation)
    await session.commit()
    await session.refresh(annotation)
    return annotation


@router.patch("/annotations/{annotation_id}", response_model=AnnotationOut)
async def update_annotation(
    annotation_id: uuid.UUID,
    body: AnnotationUpdate,
    session: AsyncSession = Depends(get_session),
):
    annotation = await session.get(Annotation, annotation_id)
    if annotation is None:
        raise HTTPException(404, "Annotation not found")

    updates = body.model_dump(exclude_none=True)
    for field, value in updates.items():
        if field == "geometry" and value is not None:
            value = body.geometry.model_dump()
        setattr(annotation, field, value)

    # any edit by a human -> edited (except an explicit status-only change)
    if updates and set(updates) != {"status"}:
        annotation.status = AnnotationStatus.EDITED
    annotation.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

    await session.commit()
    await session.refresh(annotation)
    return annotation


@router.delete("/annotations/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    annotation = await session.get(Annotation, annotation_id)
    if annotation is None:
        raise HTTPException(404, "Annotation not found")
    await session.delete(annotation)
    await session.commit()
