"""Worker: engine config resolution and the GPU deployment job."""

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
    GpuDeployment,
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
from app.services.gpu_recipes import GPU_RECIPES, MODAL_GPU, MODAL_GPU_DETECT
from app.workers import tasks

ENDPOINT = "https://ws--nounbox-gpu-fastapi-app.modal.run"
PREDICT = f"{ENDPOINT}/predict"
DETECT_ENDPOINT = "https://ws--nounbox-gpu-detect-fastapi-app.modal.run"

# captured before the autouse fixture below swaps it out for a recorder
REAL_WARM_UP = tasks._warm_up


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
    """The engines in this file are OCR ones, so the project defaults to ocr:
    a detection project requires classes (see test_autolabel_classes.py)."""
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
    """The pre-upgrade shape: GPU state in the legacy settings columns. Still
    exercised on purpose — this is what an existing installation looks like."""
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


async def set_deployment(
    session_factory,
    engine: str,
    status: GpuStatus,
    endpoint: str | None = None,
    *,
    with_token: bool = True,
) -> None:
    async with session_factory() as session:
        if with_token and (await session.execute(select(InstanceSettings))).first() is None:
            session.add(
                InstanceSettings(
                    modal_token_id="ak-1234567890abcdef",
                    modal_token_secret_encrypted=encrypt_secret("as-secret"),
                )
            )
        session.add(
            GpuDeployment(
                engine=engine,
                app_name=engine,
                status=status,
                endpoint_url=endpoint,
            )
        )
        await session.commit()


@pytest.fixture(autouse=True)
def warmups(monkeypatch):
    """The deploy job warms the fresh endpoint with a 1x1 PNG. Tests must not
    make that request; they check that it was asked for instead."""
    calls: list[tuple] = []

    async def fake_warm_up(endpoint_url, path, gpu_token, config):
        calls.append((endpoint_url, path, gpu_token))
        return True

    monkeypatch.setattr(tasks, "_warm_up", fake_warm_up)
    return calls


async def get_job(session_factory, job_id: str) -> Job:
    async with session_factory() as session:
        return await session.get(Job, uuid.UUID(job_id))


async def test_modal_gpu_requested_but_not_ready_fails_job(
    session_factory, key_path, monkeypatch
):
    install(monkeypatch, RecordingLabeler("modal_gpu"))
    project_id = await make_project_with_image(session_factory)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)  # the worker must not crash

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


async def test_modal_gpu_detect_config_gets_its_own_endpoint(
    session_factory, key_path, monkeypatch
):
    labeler = RecordingLabeler("modal_gpu_detect")
    install(monkeypatch, labeler)
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    project_id = await make_project_with_image(session_factory, TaskType.DETECTION)
    async with session_factory() as session:
        from app.models import ProjectClass

        session.add(ProjectClass(project_id=project_id, name="sofa"))
        await session.commit()
    await set_deployment(
        session_factory, MODAL_GPU, GpuStatus.READY, ENDPOINT
    )
    await set_deployment(
        session_factory, MODAL_GPU_DETECT, GpuStatus.READY, DETECT_ENDPOINT
    )
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu_detect")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    # the classes come from the project, the endpoint from ITS OWN deployment
    assert labeler.configs == [
        {"classes": ["sofa"], "endpoint": f"{DETECT_ENDPOINT}/predict"}
    ]


async def test_ocr_engine_requested_in_a_detection_project_fails_the_job(
    session_factory, key_path, monkeypatch
):
    """The defect being fixed: modal_gpu serves OCR, so pointing a detection
    project at it produced text_line boxes that export threw away."""
    install(monkeypatch, RecordingLabeler("modal_gpu"))
    project_id = await make_project_with_image(session_factory, TaskType.DETECTION)
    await set_deployment(session_factory, MODAL_GPU, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.FAILED
    assert "handles ocr projects" in job.result["error"]
    assert "this project is detection" in job.result["error"]


async def test_fan_out_skips_engines_of_the_wrong_task(
    session_factory, key_path, monkeypatch
):
    """With no engine named the worker runs every installed one — the OCR
    engines must sit that one out instead of polluting a detection project."""
    detector = RecordingLabeler("owlv2")
    install(monkeypatch, detector, RecordingLabeler("rapidocr"))
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    project_id = await make_project_with_image(session_factory, TaskType.DETECTION)
    async with session_factory() as session:
        from app.models import ProjectClass

        session.add(ProjectClass(project_id=project_id, name="sofa"))
        await session.commit()
    job_id = await enqueue_autolabel(session_factory, project_id, None)

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    assert job.result["labelers"] == ["owlv2"]
    assert "rapidocr" in job.result["skipped_labelers"]
    assert len(detector.configs) == 1


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
    # the deployment row is the new source of truth...
    async with session_factory() as session:
        deployment = (await session.execute(select(GpuDeployment))).scalar_one()
    assert deployment.engine == MODAL_GPU
    assert deployment.status == GpuStatus.READY
    assert deployment.endpoint_url == ENDPOINT
    # ...and the legacy columns keep being written for one release, so that
    # rolling back to the previous image still finds a working OCR endpoint
    assert row.gpu_status == GpuStatus.READY
    assert row.gpu_endpoint_url == ENDPOINT
    assert row.gpu_error is None


async def test_deploy_detection_gpu_uses_its_own_recipe_and_row(
    session_factory, key_path, monkeypatch, warmups
):
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    calls = []

    async def fake_deploy(token_id, token_secret, **kwargs):
        calls.append(kwargs)
        return modal_deploy.DeployedApp(
            app_id="ap-2",
            app_page_url="https://modal.com/apps/ap-2",
            endpoint_url=DETECT_ENDPOINT,
            warnings=[],
        )

    monkeypatch.setattr(modal_deploy, "deploy_gpu_app", fake_deploy)
    async with session_factory() as session:
        job = Job(
            type=JobType.DEPLOY_GPU,
            payload={"engine": MODAL_GPU_DETECT, "app_name": "nounbox-gpu-detect"},
        )
        session.add(job)
        await session.commit()
        job_id = str(job.id)

    await tasks.run_deploy_gpu({}, job_id)

    assert calls[0]["engine"] == MODAL_GPU_DETECT
    assert calls[0]["app_name"] == "nounbox-gpu-detect"
    async with session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        rows = {
            r.engine: r
            for r in (await session.execute(select(GpuDeployment))).scalars().all()
        }
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert job.status == JobStatus.DONE, job.result
    assert rows[MODAL_GPU_DETECT].endpoint_url == DETECT_ENDPOINT
    # the OCR GPU is untouched — no redeploy, same endpoint, same legacy mirror
    assert rows[MODAL_GPU].endpoint_url == ENDPOINT
    assert row.gpu_endpoint_url == ENDPOINT
    assert row.gpu_status == GpuStatus.READY
    # the fresh endpoint is warmed so the first real photo is not a cold start
    assert warmups == [(DETECT_ENDPOINT, "predict", warmups[0][2])]
    assert job.result["warmed"] is True


async def test_warm_up_never_raises(monkeypatch):
    """The warm-up runs inside the deploy job's try block: if it could raise,
    an unreachable fresh endpoint would turn a successful deploy into a failed
    one. Losing the warm-up costs a slow first image and nothing else."""
    import httpx

    class Unreachable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(httpx, "AsyncClient", Unreachable)

    # the real one — the autouse fixture above replaces tasks._warm_up
    assert await REAL_WARM_UP(ENDPOINT, "predict", "token", {}) is False


async def test_deploy_of_an_unknown_engine_fails_without_touching_anything(
    session_factory, key_path
):
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    async with session_factory() as session:
        job = Job(type=JobType.DEPLOY_GPU, payload={"engine": "modal_gpu_magic"})
        session.add(job)
        await session.commit()
        job_id = str(job.id)

    await tasks.run_deploy_gpu({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.FAILED
    assert "modal_gpu_magic" in job.result["error"]
    async with session_factory() as session:
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert row.gpu_status == GpuStatus.READY  # the working OCR GPU is untouched


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


@pytest.mark.parametrize("engine", sorted(GPU_RECIPES))
def test_resolved_recipe_exists(engine):
    assert modal_deploy.recipe_path(engine).is_file()


@pytest.mark.parametrize("engine", sorted(GPU_RECIPES))
def test_recipe_is_self_contained(engine):
    # Modal mounts exactly one .py — the recipe must contain no local imports
    source = modal_deploy.recipe_path(engine).read_text()

    assert "modal.App(" in source
    assert "from app" not in source
    assert "from nounbox" not in source


def test_recipe_path_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom.py"
    custom.write_text("app = None\n")
    monkeypatch.setattr(modal_deploy.settings, "modal_gpu_recipe_path", str(custom))

    # the default argument is still the OCR engine, for pre-existing callers
    assert modal_deploy.recipe_path() == custom
    assert modal_deploy.recipe_path(MODAL_GPU) == custom
    # one override must not redirect the other engine — that would deploy the
    # OCR recipe as the detection engine, silently
    assert modal_deploy.recipe_path(MODAL_GPU_DETECT).name == "owlv2_modal.py"


def test_detection_recipe_path_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom_detect.py"
    custom.write_text("app = None\n")
    monkeypatch.setattr(
        modal_deploy.settings, "modal_gpu_detect_recipe_path", str(custom)
    )

    assert modal_deploy.recipe_path(MODAL_GPU_DETECT) == custom
    assert modal_deploy.recipe_path(MODAL_GPU).name == "paddleocr_modal.py"


def test_each_recipe_is_imported_under_its_own_module_name():
    # Modal serializes a function by the name of its module: two recipes under
    # one sys.modules key would make the second deploy overwrite the first
    names = {modal_deploy._module_name(engine) for engine in GPU_RECIPES}

    assert len(names) == len(GPU_RECIPES)
