"""Instance settings: reading/creating the row, engine catalog, config resolution.

Two labeling modes: "easy" — rapidocr on CPU (always available),
"hard" — modal_gpu, available only once the GPU is deployed (gpu_status=ready).
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_config
from app.crypto import mask_token_id
from app.security import access_token_configured
from app.models import GpuStatus, InstanceSettings

RAPIDOCR = "rapidocr"
MODAL_GPU = "modal_gpu"

# the path the GPU recipe listens on (deploy/modal/paddleocr_modal.py)
GPU_PREDICT_PATH = "predict"


def predict_url(endpoint_url: str | None) -> str | None:
    """Root of the deployed app -> URL of the prediction endpoint."""
    if not endpoint_url:
        return endpoint_url
    base = endpoint_url.rstrip("/")
    if base.endswith(f"/{GPU_PREDICT_PATH}"):
        return base
    return f"{base}/{GPU_PREDICT_PATH}"

OWLV2 = "owlv2"
LLMDET = "llmdet"

# Engines the UI must always know about — even when the plugin is not installed
# (otherwise the user has nowhere to see that GPU mode exists at all, and in a
# detection project the engine list would come out empty with no explanation).
CORE_LABELERS = (RAPIDOCR, MODAL_GPU, OWLV2)

DETECTION = ("detection",)
OCR = ("ocr",)
BOTH = ("detection", "ocr")

# tasks — the project task types the engine is good for: the UI shows only
# the fitting ones. An unknown third-party plugin is not hidden (BOTH).
CATALOG: dict[str, dict] = {
    RAPIDOCR: {
        "title": "RapidOCR — CPU, works out of the box",
        "requires": "cpu",
        "tasks": OCR,
    },
    MODAL_GPU: {
        "title": "GPU in your own Modal account",
        "requires": "modal",
        "tasks": OCR,
    },
    OWLV2: {
        "title": "OWLv2 — boxes for your classes, CPU",
        "requires": "cpu",
        "tasks": DETECTION,
    },
    LLMDET: {
        "title": "LLMDet — boxes for your classes, CPU",
        "requires": "cpu",
        "tasks": DETECTION,
    },
    "consensus": {
        "title": "Consensus of several engines",
        "requires": "config",
        "tasks": BOTH,
    },
    "vlm": {
        "title": "VLM — OpenAI-compatible endpoint",
        "requires": "config",
        "tasks": OCR,
    },
    "http": {"title": "External HTTP endpoint", "requires": "config", "tasks": BOTH},
    "paddleocr": {"title": "PaddleOCR — local CPU", "requires": "cpu", "tasks": OCR},
}

ORDER = (
    RAPIDOCR,
    MODAL_GPU,
    OWLV2,
    LLMDET,
    "consensus",
    "vlm",
    "http",
    "paddleocr",
)


class LabelerNotReadyError(RuntimeError):
    """The engine is installed but not ready (required settings missing)."""


async def get_row(session: AsyncSession) -> InstanceSettings | None:
    result = await session.execute(
        select(InstanceSettings)
        .where(InstanceSettings.user_id.is_(None))
        .order_by(InstanceSettings.id)
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_or_create(session: AsyncSession) -> InstanceSettings:
    row = await get_row(session)
    if row is None:
        row = InstanceSettings()
        session.add(row)
        await session.flush()
    return row


def to_out(row: InstanceSettings | None) -> dict:
    """Public view of the settings. The secret is never handed out."""
    if row is None:
        return {
            "modal_configured": False,
            "modal_token_id_masked": None,
            "gpu_status": GpuStatus.NOT_CONFIGURED,
            "gpu_endpoint_url": None,
            "gpu_error": None,
            "access_protected": access_token_configured(),
        }
    return {
        "modal_configured": bool(
            row.modal_token_id and row.modal_token_secret_encrypted
        ),
        "modal_token_id_masked": mask_token_id(row.modal_token_id),
        "access_protected": access_token_configured(),
        "gpu_status": row.gpu_status,
        "gpu_endpoint_url": row.gpu_endpoint_url,
        "gpu_error": row.gpu_error,
    }


def gpu_ready(row: InstanceSettings | None) -> bool:
    return bool(
        row is not None
        and row.gpu_status == GpuStatus.READY
        and row.gpu_endpoint_url
    )


def gpu_blocker(row: InstanceSettings | None) -> str | None:
    """Why modal_gpu is unavailable; None — it is available."""
    if gpu_ready(row):
        return None
    if row is None or not row.modal_token_secret_encrypted:
        return "A Modal token is required"
    if row.gpu_status == GpuStatus.DEPLOYING:
        return "The GPU is deploying — please wait"
    if row.gpu_status == GpuStatus.FAILED:
        return "The GPU deploy failed — see the settings page"
    return "Click “Connect GPU” on the settings page"


def resolve_labeler_config(
    name: str, config: dict, row: InstanceSettings | None
) -> dict:
    """Fill the engine config from the settings (nobody writes that JSON by hand)."""
    resolved = dict(config)
    if name != MODAL_GPU:
        return resolved
    blocker = gpu_blocker(row)
    if blocker is not None:
        raise LabelerNotReadyError(f"Engine {MODAL_GPU} is not ready: {blocker}")
    # settings hold the root of the Modal app; the recipe listens on POST /predict
    resolved.setdefault("endpoint", predict_url(row.gpu_endpoint_url))
    gpu_token = app_config.nounbox_gpu_token
    if not gpu_token and row.gpu_access_token_encrypted:
        from app.crypto import decrypt_secret

        gpu_token = decrypt_secret(row.gpu_access_token_encrypted)
    if gpu_token:
        resolved.setdefault("api_key", gpu_token)
    return resolved


def build_labelers(
    installed: Iterable[str], row: InstanceSettings | None
) -> list[dict]:
    """Installed plugins + availability from the settings, in UI order."""
    installed = set(installed)
    names = [n for n in ORDER if n in installed or n in CORE_LABELERS]
    names += sorted(installed - set(names))

    items = []
    for name in names:
        meta = CATALOG.get(name, {})
        available, reason = True, None
        if name not in installed:
            available, reason = False, "Engine is not installed in this image"
        elif name == MODAL_GPU:
            reason = gpu_blocker(row)
            available = reason is None
        items.append(
            {
                "name": name,
                "title": meta.get("title", name),
                "requires": meta.get("requires", "config"),
                "tasks": list(meta.get("tasks", BOTH)),
                "available": available,
                "reason": reason,
            }
        )
    return items
