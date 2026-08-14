"""Worker: engine config resolution and the GPU deployment job."""

import threading
import time
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


# --- concurrent dispatch to remote engines ---
class ConcurrencyProbe:
    """Records how many predict() calls were ever in flight at the same time.

    predict runs in a threadpool, so the counter is guarded by a lock rather
    than relying on the event loop for mutual exclusion.
    """

    version = "0.1.0"

    def __init__(self, name: str, delay: float = 0.05) -> None:
        self.name = name
        self.delay = delay
        self.peak = 0
        self.calls = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
        with self._lock:
            self._in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self._in_flight)
        try:
            time.sleep(self.delay)
            return []
        finally:
            with self._lock:
                self._in_flight -= 1


async def make_project_with_images(session_factory, count: int) -> uuid.UUID:
    async with session_factory() as session:
        project = Project(name="p", description="", task_type=TaskType.OCR)
        session.add(project)
        await session.flush()
        document = Document(project_id=project.id, filename="a.png", s3_key="doc/a.png")
        session.add(document)
        await session.flush()
        for index in range(count):
            session.add(Image(document_id=document.id, s3_key=f"img/{index}.png"))
        await session.commit()
        return project.id


async def test_remote_engine_gets_several_images_at_once(
    session_factory, key_path, monkeypatch
):
    probe = ConcurrencyProbe("modal_gpu")
    install(monkeypatch, probe)
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 8)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    assert probe.calls == 8
    assert probe.peak > 1, "a remote engine must receive several images at once"
    assert probe.peak <= 4, "and never more than the configured concurrency"


async def test_local_engine_still_gets_one_image_at_a_time(
    session_factory, key_path, monkeypatch
):
    """A CPU engine already occupies the worker's cores — overlapping calls
    would make it slower, not faster, so the run must stay sequential."""
    probe = ConcurrencyProbe("rapidocr")
    install(monkeypatch, probe)
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 6)
    job_id = await enqueue_autolabel(session_factory, project_id, "rapidocr")

    await tasks.run_autolabel({}, job_id)

    assert probe.calls == 6
    assert probe.peak == 1


async def test_mixed_local_and_remote_run_stays_sequential(
    session_factory, key_path, monkeypatch
):
    """The local engine would be the bottleneck anyway; running it several
    times over would only slow the run down."""
    remote = ConcurrencyProbe("modal_gpu")
    local = ConcurrencyProbe("rapidocr")
    install(monkeypatch, remote, local)
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 6)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, None)

    await tasks.run_autolabel({}, job_id)

    assert remote.peak == 1
    assert local.peak == 1


class FailingOnOne:
    """Fails on one image, works on the rest."""

    version = "0.1.0"
    name = "modal_gpu"

    def __init__(self, fail_on: bytes, error: Exception) -> None:
        self.fail_on = fail_on
        self.error = error

    def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
        if image == self.fail_on:
            raise self.error
        return [
            SdkAnnotation(
                geometry=[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
                label="text_line",
                confidence=0.9,
            )
        ]


async def test_one_failing_image_does_not_take_down_the_concurrent_run(
    session_factory, key_path, monkeypatch
):
    install(monkeypatch, FailingOnOne(b"bad", RuntimeError("boom")))

    def get_bytes(key):
        return b"bad" if key.endswith("2.png") else b"png-bytes"

    monkeypatch.setattr(storage, "get_bytes", get_bytes)
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 6)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    assert job.result["failed_images"] == 1
    assert job.result["annotations_created"] == 5


async def test_config_error_still_fails_the_whole_concurrent_run(
    session_factory, key_path, monkeypatch
):
    """A ValueError means the same thing will happen to every remaining frame.
    Concurrency must not turn that into 'one image failed' repeated N times."""
    install(monkeypatch, FailingOnOne(b"png-bytes", ValueError("no classes")))
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 8)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.FAILED
    assert "no classes" in job.result["error"]
# --- appended to server/tests/test_worker_tasks.py ---


class EchoingLabeler:
    """Returns an annotation that names the bytes it was handed.

    Every existing concurrency test hands each image the SAME bytes and gets
    back the SAME annotation, so no assertion can tell which image a row came
    from. This one makes every image's result unique, which is what makes the
    image <-> result pairing checkable at all.
    """

    version = "0.1.0"
    name = "modal_gpu"

    def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
        return [
            SdkAnnotation(
                geometry=[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
                label="text_line",
                text=image.decode(),
                confidence=0.9,
            )
        ]


async def annotations_by_image(session_factory) -> dict[str, list[str]]:
    """{image s3_key: [text of each annotation stored against it]}"""
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Image.s3_key, Annotation.text).join(
                    Annotation, Annotation.image_id == Image.id
                )
            )
        ).all()
    out: dict[str, list[str]] = {}
    for key, text in rows:
        out.setdefault(key, []).append(text)
    return out


async def test_each_annotation_lands_on_the_image_it_was_predicted_from(
    session_factory, key_path, monkeypatch
):
    """Concurrency must not cross the wires between images.

    A window predicts N images at once and then applies the results to the
    session. If the results are ever re-paired with the window by anything
    other than position — completion order, a zip against a reordered list, a
    shared `data` buffer — every count in job.result stays exactly right while
    the boxes land on the wrong pages. The counters cannot see it; only the
    image_id can.
    """
    install(monkeypatch, EchoingLabeler())
    # each image gets bytes that identify it, so the annotation it produces
    # can be traced back to the image it came from
    monkeypatch.setattr(storage, "get_bytes", lambda key: key.encode())
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 10)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    assert job.result["annotations_created"] == 10

    stored = await annotations_by_image(session_factory)
    assert stored == {f"img/{i}.png": [f"img/{i}.png"] for i in range(10)}


class FailingOnKey:
    """Raises for one image's bytes; succeeds on the rest. Counts its calls."""

    version = "0.1.0"
    name = "modal_gpu"

    def __init__(self, fail_on: bytes, error: Exception) -> None:
        self.fail_on = fail_on
        self.error = error
        self.calls = 0

    def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
        self.calls += 1
        if image == self.fail_on:
            raise self.error
        return [
            SdkAnnotation(
                geometry=[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
                label="text_line",
                confidence=0.9,
            )
        ]


async def test_config_error_fails_the_job_even_when_the_window_had_successes(
    session_factory, key_path, monkeypatch
):
    """The existing ValueError test fails EVERY image, so it cannot tell a
    "config errors fail the job" rule from a "windows where nothing worked fail
    the job" one. Here images 0-2 of the window succeed and image 3 raises: the
    job must still fail, and must not keep spending frames afterwards.
    """
    labeler = FailingOnKey(b"img/3.png", ValueError("no classes"))
    install(monkeypatch, labeler)
    monkeypatch.setattr(storage, "get_bytes", lambda key: key.encode())
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 12)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.FAILED, job.result
    assert "no classes" in job.result["error"]
    # a config error repeats on every frame: nothing past the window that hit
    # it may be sent, or the run burns the whole project's quota on it
    assert labeler.calls <= 4, f"kept labeling after a config error ({labeler.calls})"


@pytest.mark.parametrize("knob", [2, 3])
async def test_window_size_follows_the_configured_concurrency(
    session_factory, key_path, monkeypatch, knob
):
    """Pinned to values that are NOT the default (4): setting the knob to its
    own default proves only that some window exists, not that the setting is
    read at all.
    """
    probe = ConcurrencyProbe("modal_gpu")
    install(monkeypatch, probe)
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", knob)
    project_id = await make_project_with_images(session_factory, 12)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    assert probe.calls == 12
    assert probe.peak == knob


class OrderRecorder:
    """Records every (engine, image key) predict call, in order."""

    version = "0.1.0"

    def __init__(self, name: str, log: list, error: Exception | None = None) -> None:
        self.name = name
        self.log = log
        self.error = error

    def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
        self.log.append((self.name, image.decode()))
        if self.error is not None:
            raise self.error
        return []


async def test_config_error_stops_the_other_engines_on_that_image(
    session_factory, key_path, monkeypatch
):
    """A ValueError fails the whole job, so nothing else on that image is worth
    computing. The sequential loop raised immediately and never reached the
    second engine; running it now would burn a billable GPU inference whose
    result is thrown away."""
    log: list = []
    first = OrderRecorder("modal_gpu", log, ValueError("no classes"))
    second = OrderRecorder("modal_gpu_detect", log)
    install(monkeypatch, first, second)
    monkeypatch.setattr(storage, "get_bytes", lambda key: key.encode())
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 1)
    project_id = await make_project_with_images(session_factory, 3)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    await set_deployment(
        session_factory, MODAL_GPU_DETECT, GpuStatus.READY, DETECT_ENDPOINT
    )
    job_id = await enqueue_autolabel(session_factory, project_id, None)

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.FAILED
    assert log == [("modal_gpu", "img/0.png")], log


async def test_failures_of_different_kinds_across_windows_are_all_counted(
    session_factory, key_path, monkeypatch
):
    """The case windowing newly creates: several images failing differently, and
    not all inside the first window."""
    bad_fetch = "img/1.png"
    bad_predict = {"img/2.png", "img/5.png"}  # one per window at concurrency 4

    def get_bytes(key):
        if key == bad_fetch:
            raise RuntimeError("gone from S3")
        return key.encode()

    class Engine:
        version = "0.1.0"
        name = "modal_gpu"

        def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
            if image.decode() in bad_predict:
                raise RuntimeError("engine blew up")
            return [
                SdkAnnotation(
                    geometry=[(1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)],
                    label="text_line",
                    confidence=0.9,
                )
            ]

    install(monkeypatch, Engine())
    monkeypatch.setattr(storage, "get_bytes", get_bytes)
    monkeypatch.setattr(tasks.settings, "remote_labeler_concurrency", 4)
    project_id = await make_project_with_images(session_factory, 8)
    await set_gpu(session_factory, GpuStatus.READY, ENDPOINT)
    job_id = await enqueue_autolabel(session_factory, project_id, "modal_gpu")

    await tasks.run_autolabel({}, job_id)

    job = await get_job(session_factory, job_id)
    assert job.status == JobStatus.DONE, job.result
    assert job.result["failed_images"] == 3
    assert job.result["annotations_created"] == 5
