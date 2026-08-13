"""Резолв конфига движка из настроек."""

import pytest

from app.models import GpuStatus, InstanceSettings
from app.services import settings_store

ENDPOINT = "https://ws--autolabelui-gpu-fastapi-app.modal.run"
# рецепт слушает POST /predict — резолвер дописывает путь сам
PREDICT = f"{ENDPOINT}/predict"


def row(status: GpuStatus, endpoint: str | None = ENDPOINT, token: bool = True):
    return InstanceSettings(
        modal_token_id="ak-1234567890abcdef" if token else None,
        modal_token_secret_encrypted="encrypted" if token else None,
        gpu_status=status,
        gpu_endpoint_url=endpoint,
    )


def test_cpu_labeler_config_passes_through():
    config = {"lang": "ru"}

    resolved = settings_store.resolve_labeler_config("rapidocr", config, None)

    assert resolved == {"lang": "ru"}
    assert resolved is not config


def test_modal_gpu_gets_endpoint_from_settings():
    resolved = settings_store.resolve_labeler_config(
        "modal_gpu", {"lang": "ru"}, row(GpuStatus.READY)
    )

    assert resolved == {"lang": "ru", "endpoint": PREDICT}


def test_gpu_token_is_injected_as_api_key(monkeypatch):
    monkeypatch.setattr(settings_store.app_config, "autolabelui_gpu_token", "s3cret")

    resolved = settings_store.resolve_labeler_config("modal_gpu", {}, row(GpuStatus.READY))

    assert resolved == {"endpoint": PREDICT, "api_key": "s3cret"}


def test_no_api_key_without_gpu_token():
    resolved = settings_store.resolve_labeler_config("modal_gpu", {}, row(GpuStatus.READY))

    assert "api_key" not in resolved


def test_explicit_endpoint_wins():
    resolved = settings_store.resolve_labeler_config(
        "modal_gpu", {"endpoint": "http://localhost:9999"}, row(GpuStatus.READY)
    )

    assert resolved["endpoint"] == "http://localhost:9999"


@pytest.mark.parametrize(
    "settings_row",
    [
        None,
        row(GpuStatus.NOT_CONFIGURED, endpoint=None, token=False),
        row(GpuStatus.DEPLOYING, endpoint=None),
        row(GpuStatus.FAILED, endpoint=None),
        row(GpuStatus.READY, endpoint=None),
    ],
)
def test_modal_gpu_not_ready_raises(settings_row):
    with pytest.raises(settings_store.LabelerNotReadyError) as exc:
        settings_store.resolve_labeler_config("modal_gpu", {}, settings_row)

    assert "modal_gpu" in str(exc.value)


def test_to_out_never_exposes_secret():
    out = settings_store.to_out(row(GpuStatus.READY))

    assert set(out) == {
        "modal_configured",
        "modal_token_id_masked",
        "gpu_status",
        "gpu_endpoint_url",
        "gpu_error",
    }
    assert "encrypted" not in str(out)
    assert "ak-1234567890abcdef" not in str(out)
    assert out["modal_token_id_masked"] == "ak-1234...cdef"
    assert out["modal_configured"] is True


def test_to_out_of_missing_row():
    assert settings_store.to_out(None) == {
        "modal_configured": False,
        "modal_token_id_masked": None,
        "gpu_status": GpuStatus.NOT_CONFIGURED,
        "gpu_endpoint_url": None,
        "gpu_error": None,
    }
