"""Documents: source file upload. Full ingest (normalization into images) is in the worker."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app import storage
from app.db import get_session
from app.models import Document, Job, JobType, Project
from app.schemas import DocumentOut

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=201)
async def upload_document(
    project_id: uuid.UUID,
    file: UploadFile,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    if await session.get(Project, project_id) is None:
        raise HTTPException(404, "Project not found")

    data = await file.read()
    doc = Document(
        project_id=project_id,
        filename=file.filename or "unnamed",
        content_type=file.content_type or "",
        size_bytes=len(data),
        s3_key=f"raw/{project_id}/{uuid.uuid4()}/{file.filename}",
    )
    await run_in_threadpool(storage.put_bytes, doc.s3_key, data, doc.content_type)
    session.add(doc)
    await session.flush()  # get doc.id before creating the job

    # ingest: normalization -> images (in the background via the worker)
    job = Job(
        project_id=project_id,
        type=JobType.INGEST,
        payload={"document_id": str(doc.id)},
    )
    session.add(job)
    await session.commit()
    await session.refresh(doc)

    await request.app.state.arq.enqueue_job("run_ingest", str(job.id))
    return doc


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    result = await session.execute(
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()
