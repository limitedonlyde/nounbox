"""Воркер: резолв конфига движка и задача развёртывания GPU."""

import uuid

import nounbox_sdk
import pytest
from nounbox_sdk import Annotation as SdkAnnotation
from sqlalchemy import select

from app import storage
from app.crypto import encrypt_secret
from app.models import (
    Annotation,
    Document,
    GpuStatus,
    Image,
    InstanceSettings,
    Job,
    JobStatus,
    JobType,
    Project,
    TaskType,
)
from app.services import modal_deploy
from app.workers import tasks

ENDPOINT = "https://ws--nounbox-gpu-fastapi-app.modal.run"
PREDICT = f"{ENDPOINT}/predict"


class RecordingLabeler:
    version = "0.1.0"

    def __init__(self, name: str) -> None:
        self.name = name
        self.configs: list[dict] = []

    def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
        self.configs.append(config)
        return [
            SdkAnnotation(
                geometry=[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
                label="text_line",
                text="АКТ",
                confidence=0.98,
            )
        ]


def install(monkeypatch, *labelers: RecordingLabeler) -> None:
    monkeypatch.setattr(
        nounbox_sdk, "load_labelers", lambda: {la.name: la for la in labelers}
    )


async def make_project_with_image(
    session_factory, task_type: TaskType = TaskType.OCR
) -> uuid.UUID:
    """Движки в этом файле — OCR-овые, поэтому проект по умолчанию ocr:
    detection-проект требует классов (см. test_autolabel_classes.py)."""
    async with session_factory() as session:
        project = Project(name="p", description="", task_type=task_type)
        session.add(project)
        await session.flush()
        document = Document(project_id=project.id, filename="a.png", s3_key="doc/a.png")
        session.add(document)
        await session.flush()
        session.add(Image(document_id=document.id, s3_key="img/a.png"))
        await session.commit()
        return project.id


async def enqueue_autolabel(session_factory, project_id, labeler, config=None) -> str:
    async with session_factory() as session:
        job = Job(
            project_id=project_id,
            type=JobType.AUTOLABEL,
            payload={"labeler": labeler, "config": config or {}},
        )
        session.add(job)
        await session.commit()
        return str(job.id)


async def set_gpu(session_factory, status: GpuStatus, endpoint: str | None = None) -> None:
    async with session_factory() as session:
        session.add(
            InstanceSettings(
                modal_token_id="ak-1234567890abcdef",
                modal_token_secret_encrypted=encrypt_secret("as-secret"),
                gpu_status=status,
                gpu_endpoint_url=endpoint,
            )
        )
        await session.commit()


async def get_job(session_factory, job_id: str) -> Job:
    async with session_factory() as session:
        return await session.get(Job, uuid.UUID(job_id))


async def test_modal_gpu_requested_but_not_ready_fails_job(
    session_factory, key_path, monkeypatch
):
    install(monkeypatch, RecordingLabeler("modal_gpu"))
    project_id = await make_project_with_image(session_factory)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)  # воркер не должен упасть

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.FAILED
    assert "modal_gpu" in job.result["error"]
    assert "A Modal token is required" in job.result["error"]


async def test_all_labelers_run_skips_unready_gpu(session_factory, key_path, monkeypatch):
    install(monkeypatch, RecordingLabeler("modal_gpu"))
    project_id = await make_project_with_image(session_factory)
    await set_gpu(session_factory, GpuStatus.DEPLOYING)
    job_id = await enqueue_autolabel(session_factory, project_id, None)

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.FAILED
    assert "deploying" in job.result["error"]


async def test_modal_gpu_config_gets_endpoint_from_settings(
    session_factory, key_path, monkeypatch
):
    labeler = RecordingLabeler("modal_gpu")
    install(monkeypatch, labeler)
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    project_id = await make_project_with_image(session_factory)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu", {"lang": "ru"})

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    assert job.result["annotations_created"] == 1
    assert labeler.configs == [{"lang": "ru", "endpoint": PREDICT}]

    async with session_factory() as session:
        annotation = (await session.execute(select(Annotation))).scalar_one()
    assert annotation.source["name"] == "modal_gpu"
    assert annotation.geometry["type"] == "polygon"


async def test_cpu_labeler_config_untouched(session_factory, key_path, monkeypatch):
    labeler = RecordingLabeler("rapidocr")
    install(monkeypatch, labeler)
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    project_id = await make_project_with_image(session_factory)
    job_id = await enqueue_autolabel(session_factory, project_id, "rapidocr", {"lang": "ru"})

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    assert labeler.configs == [{"lang": "ru"}]


async def test_deploy_gpu_saves_endpoint_and_takes_token_from_settings(
    session_factory, key_path, monkeypatch
):
    await set_gpu(session_factory, GpuStatus.NOT_CONFIGURED)
    calls = []

    async def fake_deploy(token_id, token_secret, app_name=None, path=None, **kwargs):
        calls.append((token_id, token_secret, app_name))
        return modal_deploy.DeployedApp(
            app_id="ap-1",
            app_page_url="https://modal.com/apps/ap-1",
            endpoint_url=ENDPOINT,
            warnings=[],
        )

    monkeypatch.setattr(modal_deploy, "deploy_gpu_app", fake_deploy)
    async with session_factory() as session:
        job = Job(type=JobType.DEPLOY_GPU, payload={"app_name": "nounbox-gpu"})
        session.add(job)
        await session.commit()
        job_id = str(job.id)

    await tasks.run_deploy_gpu({}, job_id)

    assert calls == [("ak-1234567890abcdef", "as-secret", "nounbox-gpu")]
    async with session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert job.status == JobStatus.DONE
    assert job.result["endpoint_url"] == ENDPOINT
    assert "as-secret" not in str(job.result) + str(job.payload)
    assert row.gpu_status == GpuStatus.READY
    assert row.gpu_endpoint_url == ENDPOINT
    assert row.gpu_error is None


async def test_deploy_gpu_failure_lands_in_gpu_error(session_factory, key_path, monkeypatch):
    await set_gpu(session_factory, GpuStatus.NOT_CONFIGURED)

    async def failing_deploy(token_id, token_secret, app_name=None, path=None, **kwargs):
        raise modal_deploy.ModalDeployError(modal_deploy.PROXY_TOKEN_HINT)

    monkeypatch.setattr(modal_deploy, "deploy_gpu_app", failing_deploy)
    async with session_factory() as session:
        job = Job(type=JobType.DEPLOY_GPU, payload={})
        session.add(job)
        await session.commit()
        job_id = str(job.id)

    await tasks.run_deploy_gpu({}, job_id)

    async with session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert job.status == JobStatus.FAILED
    assert row.gpu_status == GpuStatus.FAILED
    assert "proxy" in row.gpu_error.lower()
    assert "as-secret" not in row.gpu_error
    assert row.gpu_endpoint_url is None


async def test_deploy_gpu_scrubs_secret_from_unexpected_error(
    session_factory, key_path, monkeypatch
):
    await set_gpu(session_factory, GpuStatus.NOT_CONFIGURED)

    async def leaking_deploy(token_id, token_secret, app_name=None, path=None, **kwargs):
        raise RuntimeError(f"gRPC failed with credentials {token_secret}")

    monkeypatch.setattr(modal_deploy, "deploy_gpu_app", leaking_deploy)
    async with session_factory() as session:
        job = Job(type=JobType.DEPLOY_GPU, payload={})
        session.add(job)
        await session.commit()
        job_id = str(job.id)

    await tasks.run_deploy_gpu({}, job_id)

    async with session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert "as-secret" not in row.gpu_error
    assert "as-secret" not in str(job.result)
    assert "***" in row.gpu_error


async def test_deploy_gpu_without_token_fails_gracefully(
    session_factory, key_path, monkeypatch
):
    async with session_factory() as session:
        job = Job(type=JobType.DEPLOY_GPU, payload={})
        session.add(job)
        await session.commit()
        job_id = str(job.id)

    await tasks.run_deploy_gpu({}, job_id)

    async with session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert job.status == JobStatus.FAILED
    assert "No Modal token saved" in job.result["error"]
    assert row.gpu_status == GpuStatus.FAILED


@pytest.mark.parametrize(
    ("token_id", "token_secret", "expect_problem"),
    [
        ("ak-good", "as-good", False),
        ("wk-proxy", "ws-proxy", True),
        ("garbage", "garbage", True),
        ("ak-good", "garbage", True),
        ("", "", True),
    ],
)
def test_validate_token_pair(token_id, token_secret, expect_problem):
    problem = modal_deploy.validate_token_pair(token_id, token_secret)

    assert (problem is not None) == expect_problem


def test_resolved_recipe_exists():
    assert modal_deploy.recipe_path().is_file()


def test_recipe_is_self_contained():
    # Modal монтирует ровно один .py — локальных импортов в рецепте быть не должно
    source = modal_deploy.recipe_path().read_text()

    assert "modal.App(" in source
    assert "from app" not in source
    assert "from nounbox" not in source


def test_recipe_path_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom.py"
    custom.write_text("app = None\n")
    monkeypatch.setattr(modal_deploy.settings, "modal_gpu_recipe_path", str(custom))

    assert modal_deploy.recipe_path() == custom
