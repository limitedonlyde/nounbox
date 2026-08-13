import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app import storage
from app.db import get_session
from app.models import Annotation, Document, Image, Job, Project, ProjectClass
from app.schemas import ProjectCreate, ProjectOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate, session: AsyncSession = Depends(get_session)
):
    project = Project(
        name=body.name, description=body.description, task_type=body.task_type
    )
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut])
async def list_projects(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Project).order_by(Project.created_at.desc()))
    return result.scalars().all()


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    """Удалить проект вместе со всем содержимым.

    Каскадов у внешних ключей нет, поэтому строки сносим сами и строго от
    листьев к корню: иначе postgres отвечает IntegrityError и запрос падает
    с 500 на любом непустом проекте.
    """
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")

    documents = (
        await session.execute(
            select(Document.id, Document.s3_key).where(
                Document.project_id == project_id
            )
        )
    ).all()
    document_ids = [row.id for row in documents]

    images: list = []
    if document_ids:
        images = (
            await session.execute(
                select(Image.id, Image.s3_key).where(
                    Image.document_id.in_(document_ids)
                )
            )
        ).all()
    image_ids = [row.id for row in images]

    if image_ids:
        # parent_id аннотаций разрешается внутри одного statement: FK-проверки
        # postgres выполняет по завершении запроса
        await session.execute(
            delete(Annotation).where(Annotation.image_id.in_(image_ids))
        )
        await session.execute(delete(Image).where(Image.id.in_(image_ids)))
    if document_ids:
        await session.execute(delete(Document).where(Document.id.in_(document_ids)))
    await session.execute(
        delete(ProjectClass).where(ProjectClass.project_id == project_id)
    )
    await session.execute(delete(Job).where(Job.project_id == project_id))
    await session.delete(project)
    await session.commit()

    # файлы — после коммита: хранилище не участвует в транзакции, и сбой
    # удаления объекта не повод оставлять пользователю проект-призрак
    keys = [row.s3_key for row in documents] + [row.s3_key for row in images]
    if keys:
        try:
            await run_in_threadpool(storage.delete_objects, keys)
        except Exception:
            logger.warning(
                "Project %s deleted, but S3 cleanup failed for %d object(s)",
                project_id,
                len(keys),
                exc_info=True,
            )
