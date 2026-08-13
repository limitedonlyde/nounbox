"""Проекты: удаление непустого проекта уносит всё содержимое и файлы в S3."""

import uuid

import pytest
from sqlalchemy import func, select

from app import storage
from app.models import (
    Annotation,
    AnnotationStatus,
    Document,
    Image,
    Job,
    JobType,
    ProjectClass,
    Project,
)

BASE = "/api/v1"


@pytest.fixture
def deleted_keys(monkeypatch) -> list[str]:
    """Ключи, которые ручка попросила удалить из S3."""
    recorded: list[str] = []
    monkeypatch.setattr(storage, "delete_objects", lambda keys: recorded.extend(keys))
    return recorded


async def make_project(client) -> str:
    response = await client.post(f"{BASE}/projects", json={"name": "flats"})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def fill_project(session_factory, project_id: str) -> dict:
    """Документ + кадр + аннотация + класс + задача. Возвращает s3-ключи."""
    async with session_factory() as session:
        document = Document(
            project_id=uuid.UUID(project_id),
            filename="a.png",
            s3_key=f"raw/{project_id}/a.png",
        )
        session.add(document)
        await session.flush()
        image = Image(
            document_id=document.id,
            s3_key=f"images/{project_id}/a.png",
            width=640,
            height=480,
        )
        session.add(image)
        await session.flush()
        session.add(
            Annotation(
                image_id=image.id,
                geometry={"type": "bbox", "x": 1, "y": 2, "width": 3, "height": 4},
                label="carpet",
                attrs={},
                confidence=0.9,
                source={"type": "engine", "name": "owlv2"},
                status=AnnotationStatus.ACCEPTED,
            )
        )
        session.add(ProjectClass(project_id=uuid.UUID(project_id), name="carpet"))
        session.add(
            Job(
                project_id=uuid.UUID(project_id),
                type=JobType.INGEST,
                payload={"document_id": str(document.id)},
            )
        )
        await session.commit()
        return {"document": document.s3_key, "image": image.s3_key}


async def count(session_factory, model) -> int:
    async with session_factory() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar()


async def test_delete_project_with_content_removes_everything(
    client, session_factory, deleted_keys
):
    project_id = await make_project(client)
    await fill_project(session_factory, project_id)

    response = await client.delete(f"{BASE}/projects/{project_id}")
    assert response.status_code == 204, response.text

    for model in (Project, Document, Image, Annotation, ProjectClass, Job):
        assert await count(session_factory, model) == 0, model.__name__


async def test_delete_project_cleans_up_s3_objects(
    client, session_factory, deleted_keys
):
    project_id = await make_project(client)
    keys = await fill_project(session_factory, project_id)

    assert (await client.delete(f"{BASE}/projects/{project_id}")).status_code == 204
    assert sorted(deleted_keys) == sorted(keys.values())


async def test_storage_failure_does_not_break_deletion(
    client, session_factory, monkeypatch
):
    """S3 — не транзакция: недоступное хранилище не должно ронять удаление."""

    def boom(keys):
        raise RuntimeError("minio is down")

    monkeypatch.setattr(storage, "delete_objects", boom)
    project_id = await make_project(client)
    await fill_project(session_factory, project_id)

    assert (await client.delete(f"{BASE}/projects/{project_id}")).status_code == 204
    assert await count(session_factory, Document) == 0


async def test_delete_project_keeps_other_projects_intact(
    client, session_factory, deleted_keys
):
    doomed = await make_project(client)
    kept = await make_project(client)
    await fill_project(session_factory, doomed)
    await fill_project(session_factory, kept)

    assert (await client.delete(f"{BASE}/projects/{doomed}")).status_code == 204
    for model in (Document, Image, Annotation, ProjectClass, Job):
        assert await count(session_factory, model) == 1, model.__name__
    assert (await client.get(f"{BASE}/projects/{kept}")).status_code == 200


async def test_delete_unknown_project_is_404(client, deleted_keys):
    response = await client.delete(f"{BASE}/projects/{uuid.uuid4()}")
    assert response.status_code == 404
    assert deleted_keys == []


async def test_empty_project_deletion_does_not_touch_storage(
    client, session_factory, deleted_keys
):
    project_id = await make_project(client)
    assert (await client.delete(f"{BASE}/projects/{project_id}")).status_code == 204
    assert deleted_keys == []
