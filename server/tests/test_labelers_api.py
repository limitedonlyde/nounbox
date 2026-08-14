"""GET /labelers: installed plugins + GPU availability taken from the settings."""

from sqlalchemy import select

from app.api import labelers as labelers_api
from app.models import GpuDeployment, GpuStatus, InstanceSettings


class FakeLabeler:
    version = "0.1.0"

    def __init__(self, name: str) -> None:
        self.name = name

    def predict(self, image: bytes, config: dict) -> list:
        return []


def install(monkeypatch, *names: str) -> None:
    monkeypatch.setattr(
        labelers_api, "load_labelers", lambda: {n: FakeLabeler(n) for n in names}
    )


async def with_token(session_factory, token: str = "ak-1234567890abcdef") -> None:
    async with session_factory() as session:
        session.add(
            InstanceSettings(
                modal_token_id=token, modal_token_secret_encrypted="encrypted"
            )
        )
        await session.commit()


async def ready_gpu(
    session_factory,
    engine: str = "modal_gpu",
    token: str = "ak-1234567890abcdef",
) -> None:
    await with_token(session_factory, token)
    async with session_factory() as session:
        session.add(
            GpuDeployment(
                engine=engine,
                app_name=engine,
                status=GpuStatus.READY,
                endpoint_url=f"https://ws--{engine}-fastapi-app.modal.run",
            )
        )
        await session.commit()


async def test_rapidocr_is_default_and_always_available(client, monkeypatch):
    install(monkeypatch, "rapidocr", "modal_gpu")

    items = (await client.get("/api/v1/labelers")).json()

    assert items[0]["name"] == "rapidocr"
    assert items[0]["requires"] == "cpu"
    assert items[0]["available"] is True
    assert items[0]["reason"] is None
    assert items[0]["title"]


async def test_modal_gpu_unavailable_without_token(client, monkeypatch):
    install(monkeypatch, "rapidocr", "modal_gpu")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    gpu = items["modal_gpu"]
    assert gpu["requires"] == "modal"
    assert gpu["available"] is False
    assert gpu["reason"] == "A Modal token is required"


async def test_modal_gpu_available_when_gpu_ready(client, session_factory, monkeypatch):
    install(monkeypatch, "rapidocr", "modal_gpu")
    await ready_gpu(session_factory)

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["modal_gpu"]["available"] is True
    assert items["modal_gpu"]["reason"] is None


async def test_deploying_gpu_reports_progress_reason(client, session_factory, monkeypatch):
    install(monkeypatch, "rapidocr", "modal_gpu")
    await with_token(session_factory)
    async with session_factory() as session:
        session.add(
            GpuDeployment(
                engine="modal_gpu",
                app_name="nounbox-gpu",
                status=GpuStatus.DEPLOYING,
            )
        )
        await session.commit()

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["modal_gpu"]["available"] is False
    assert "deploying" in items["modal_gpu"]["reason"]


async def test_gpu_engines_are_split_by_task(client, monkeypatch):
    install(monkeypatch, "rapidocr", "modal_gpu", "modal_gpu_detect")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    # this is the fix: the OCR GPU no longer offers itself to a detection
    # project, and there is a separate engine that actually draws boxes
    assert items["modal_gpu"]["tasks"] == ["ocr"]
    assert items["modal_gpu"]["requires"] == "modal"
    assert items["modal_gpu_detect"]["tasks"] == ["detection"]
    assert items["modal_gpu_detect"]["requires"] == "modal"


async def test_gpu_availability_is_per_engine(client, session_factory, monkeypatch):
    """Deploying the OCR app must not make the detection engine look ready."""
    install(monkeypatch, "modal_gpu", "modal_gpu_detect")
    await ready_gpu(session_factory, "modal_gpu")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["modal_gpu"]["available"] is True
    assert items["modal_gpu_detect"]["available"] is False
    assert "detection GPU" in items["modal_gpu_detect"]["reason"]


async def test_detection_gpu_available_once_deployed(client, session_factory, monkeypatch):
    install(monkeypatch, "modal_gpu", "modal_gpu_detect")
    await ready_gpu(session_factory, "modal_gpu_detect")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["modal_gpu_detect"]["available"] is True
    assert items["modal_gpu_detect"]["reason"] is None
    assert items["modal_gpu"]["available"] is False


async def test_core_labelers_listed_even_if_plugin_missing(client, monkeypatch):
    install(monkeypatch, "http")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["rapidocr"]["available"] is False
    assert "not installed" in items["rapidocr"]["reason"]
    assert items["http"]["available"] is True
    assert items["http"]["requires"] == "config"


async def test_detection_engines_are_marked_by_task(client, monkeypatch):
    install(monkeypatch, "rapidocr", "owlv2", "llmdet")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["owlv2"]["tasks"] == ["detection"]
    assert items["owlv2"]["requires"] == "cpu"
    assert items["owlv2"]["available"] is True
    assert items["llmdet"]["tasks"] == ["detection"]
    assert items["rapidocr"]["tasks"] == ["ocr"]


async def test_owlv2_listed_even_if_plugin_missing(client, monkeypatch):
    """The default detection engine has to stay visible with a reason, otherwise
    the engine list of a detection project is empty with no explanation."""
    install(monkeypatch, "rapidocr")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["owlv2"]["available"] is False
    assert "not installed" in items["owlv2"]["reason"]


async def test_owlv2_is_first_detection_engine(client, monkeypatch):
    install(monkeypatch, "rapidocr", "owlv2", "llmdet", "http")

    items = (await client.get("/api/v1/labelers")).json()

    detection = [i["name"] for i in items if "detection" in i["tasks"]]
    assert detection[0] == "owlv2"


async def test_unknown_plugin_is_offered_for_both_tasks(client, monkeypatch):
    install(monkeypatch, "my_custom_engine")

    items = {i["name"]: i for i in (await client.get("/api/v1/labelers")).json()}

    assert items["my_custom_engine"]["tasks"] == ["detection", "ocr"]


async def test_labelers_endpoint_does_not_create_settings_row(
    client, session_factory, monkeypatch
):
    install(monkeypatch, "rapidocr")

    await client.get("/api/v1/labelers")

    async with session_factory() as session:
        assert (await session.execute(select(InstanceSettings))).first() is None
