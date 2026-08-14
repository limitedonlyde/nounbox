"""Resolving the engine config out of the settings and the GPU deployments."""

import pytest
from sqlalchemy import select

from app.models import GpuDeployment, GpuStatus, InstanceSettings
from app.services import settings_store
from app.services.gpu_recipes import MODAL_GPU, MODAL_GPU_DETECT

ENDPOINT = "https://ws--nounbox-gpu-fastapi-app.modal.run"
# the recipe listens on POST /predict — the resolver appends the path itself
PREDICT = f"{ENDPOINT}/predict"
DETECT_ENDPOINT = "https://ws--nounbox-gpu-detect-fastapi-app.modal.run"
DETECT_PREDICT = f"{DETECT_ENDPOINT}/predict"


def row(token: bool = True):
    """Settings row. Since the GPU state moved into gpu_deployments, all this
    row decides is whether a Modal token exists at all."""
    return InstanceSettings(
        modal_token_id="ak-1234567890abcdef" if token else None,
        modal_token_secret_encrypted="encrypted" if token else None,
    )


def deployment(
    status: GpuStatus,
    endpoint: str | None = ENDPOINT,
    engine: str = MODAL_GPU,
    token: str | None = None,
):
    return GpuDeployment(
        engine=engine,
        app_name="nounbox-gpu",
        status=status,
        endpoint_url=endpoint,
        access_token_encrypted=token,
    )


def deployments(*rows: GpuDeployment) -> dict[str, GpuDeployment]:
    return {row.engine: row for row in rows}


def test_cpu_labeler_config_passes_through():
    config = {"lang": "ru"}

    resolved = settings_store.resolve_labeler_config("rapidocr", config, None, {})

    assert resolved == {"lang": "ru"}
    assert resolved is not config


def test_modal_gpu_gets_endpoint_from_deployment():
    resolved = settings_store.resolve_labeler_config(
        MODAL_GPU, {"lang": "ru"}, row(), deployments(deployment(GpuStatus.READY))
    )

    assert resolved == {"lang": "ru", "endpoint": PREDICT}


def test_modal_gpu_detect_gets_its_own_endpoint():
    resolved = settings_store.resolve_labeler_config(
        MODAL_GPU_DETECT,
        {"classes": ["sofa"]},
        row(),
        deployments(
            deployment(GpuStatus.READY),
            deployment(GpuStatus.READY, DETECT_ENDPOINT, MODAL_GPU_DETECT),
        ),
    )

    assert resolved == {"classes": ["sofa"], "endpoint": DETECT_PREDICT}


def test_detection_never_falls_back_to_the_ocr_endpoint():
    """The bug this whole change exists to remove: a detection project must
    fail loudly rather than send photos to the OCR app."""
    with pytest.raises(settings_store.LabelerNotReadyError) as exc:
        settings_store.resolve_labeler_config(
            MODAL_GPU_DETECT,
            {"classes": ["sofa"]},
            row(),
            deployments(deployment(GpuStatus.READY)),  # OCR only
        )

    assert MODAL_GPU_DETECT in str(exc.value)
    assert ENDPOINT not in str(exc.value)


def test_gpu_token_is_injected_as_api_key(monkeypatch):
    monkeypatch.setattr(settings_store.app_config, "nounbox_gpu_token", "s3cret")

    resolved = settings_store.resolve_labeler_config(
        MODAL_GPU, {}, row(), deployments(deployment(GpuStatus.READY))
    )

    assert resolved == {"endpoint": PREDICT, "api_key": "s3cret"}


def test_per_deployment_token_is_decrypted(key_path):
    from app.crypto import encrypt_secret

    resolved = settings_store.resolve_labeler_config(
        MODAL_GPU,
        {},
        row(),
        deployments(
            deployment(GpuStatus.READY, token=encrypt_secret("endpoint-secret"))
        ),
    )

    assert resolved["api_key"] == "endpoint-secret"


def test_no_api_key_without_gpu_token():
    resolved = settings_store.resolve_labeler_config(
        MODAL_GPU, {}, row(), deployments(deployment(GpuStatus.READY))
    )

    assert "api_key" not in resolved


def test_explicit_endpoint_wins():
    resolved = settings_store.resolve_labeler_config(
        MODAL_GPU,
        {"endpoint": "http://localhost:9999"},
        row(),
        deployments(deployment(GpuStatus.READY)),
    )

    assert resolved["endpoint"] == "http://localhost:9999"


@pytest.mark.parametrize(
    ("settings_row", "gpu"),
    [
        (None, None),
        (row(token=False), None),
        (row(), None),
        (row(), deployment(GpuStatus.DEPLOYING, endpoint=None)),
        (row(), deployment(GpuStatus.FAILED, endpoint=None)),
        (row(), deployment(GpuStatus.READY, endpoint=None)),
    ],
)
def test_modal_gpu_not_ready_raises(settings_row, gpu):
    with pytest.raises(settings_store.LabelerNotReadyError) as exc:
        settings_store.resolve_labeler_config(
            MODAL_GPU, {}, settings_row, deployments(gpu) if gpu else {}
        )

    assert MODAL_GPU in str(exc.value)


def test_labeler_supports_task_is_the_contract_the_ui_filters_on():
    assert settings_store.labeler_supports_task(MODAL_GPU, "ocr")
    assert not settings_store.labeler_supports_task(MODAL_GPU, "detection")
    assert settings_store.labeler_supports_task(MODAL_GPU_DETECT, "detection")
    assert not settings_store.labeler_supports_task(MODAL_GPU_DETECT, "ocr")
    # an unknown third-party plugin is not hidden from anything
    assert settings_store.labeler_supports_task("my_engine", "detection")
    assert settings_store.labeler_supports_task("my_engine", "ocr")


def test_to_out_never_exposes_secret():
    out = settings_store.to_out(
        row(),
        deployments(
            deployment(GpuStatus.READY, token="encrypted-endpoint-token")
        ),
    )

    assert set(out) == {
        "modal_configured",
        "modal_token_id_masked",
        "gpu_status",
        "gpu_endpoint_url",
        "gpu_error",
        "gpus",
        "access_protected",
    }
    assert "encrypted" not in str(out)
    assert "ak-1234567890abcdef" not in str(out)
    assert out["modal_token_id_masked"] == "ak-1234...cdef"
    assert out["modal_configured"] is True


def test_to_out_flat_fields_mirror_the_ocr_gpu():
    out = settings_store.to_out(
        row(),
        deployments(
            deployment(GpuStatus.READY),
            deployment(GpuStatus.FAILED, None, MODAL_GPU_DETECT),
        ),
    )

    assert out["gpu_status"] == GpuStatus.READY
    assert out["gpu_endpoint_url"] == ENDPOINT
    assert [g["engine"] for g in out["gpus"]] == [MODAL_GPU, MODAL_GPU_DETECT]
    assert out["gpus"][1]["status"] == GpuStatus.FAILED
    assert out["gpus"][1]["task"] == "detection"


def test_to_out_of_missing_row():
    out = settings_store.to_out(None)

    assert out["modal_configured"] is False
    assert out["modal_token_id_masked"] is None
    assert out["gpu_status"] == GpuStatus.NOT_CONFIGURED
    assert out["gpu_endpoint_url"] is None
    assert out["gpu_error"] is None
    assert out["access_protected"] is False
    assert [g["status"] for g in out["gpus"]] == [
        GpuStatus.NOT_CONFIGURED,
        GpuStatus.NOT_CONFIGURED,
    ]


async def test_legacy_settings_row_is_seeded_as_the_ocr_deployment(session_factory):
    """An installation that deployed the GPU before this table existed keeps
    its endpoint working without a redeploy — and only as the OCR engine."""
    async with session_factory() as session:
        session.add(
            InstanceSettings(
                modal_token_id="ak-1234567890abcdef",
                modal_token_secret_encrypted="encrypted",
                gpu_status=GpuStatus.READY,
                gpu_endpoint_url=ENDPOINT,
                gpu_access_token_encrypted="old-token",
            )
        )
        await session.commit()

    # a read path gets the seeded deployment without writing anything: this is
    # called from GET /settings, GET /labelers and run_autolabel, and none of
    # them commit, so a row added here would be rolled back every time
    async with session_factory() as session:
        loaded = await settings_store.load_deployments(session)
        await session.commit()

    assert set(loaded) == {MODAL_GPU}
    assert loaded[MODAL_GPU].endpoint_url == ENDPOINT
    assert loaded[MODAL_GPU].status == GpuStatus.READY
    assert loaded[MODAL_GPU].access_token_encrypted == "old-token"

    async with session_factory() as session:
        rows = (await session.execute(select(GpuDeployment))).scalars().all()
    assert rows == [], "a read path must not write the seeded row"

    # the callers that own a commit ask for it explicitly, and then it sticks
    async with session_factory() as session:
        await settings_store.load_deployments(session, persist=True)
        await session.commit()

    async with session_factory() as session:
        rows = (await session.execute(select(GpuDeployment))).scalars().all()
    assert len(rows) == 1
    assert rows[0].endpoint_url == ENDPOINT

    # idempotent: once the row exists, loading again neither duplicates nor re-seeds
    async with session_factory() as session:
        again = await settings_store.load_deployments(session, persist=True)
        await session.commit()
        rows = (await session.execute(select(GpuDeployment))).scalars().all()

    assert len(rows) == 1
    assert set(again) == {MODAL_GPU}


async def test_no_settings_row_seeds_nothing(session_factory):
    async with session_factory() as session:
        assert await settings_store.load_deployments(session) == {}
        # reading the deployments must not conjure a settings row either
        assert (await session.execute(select(InstanceSettings))).first() is None
