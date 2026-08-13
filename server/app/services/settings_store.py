"""Настройки инсталляции: чтение/создание строки, каталог движков, резолв конфига.

Два режима разметки: «просто» — rapidocr на CPU (доступен всегда),
«сложно» — modal_gpu, доступен только когда GPU развёрнут (gpu_status=ready).
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

# путь, который слушает GPU-рецепт (deploy/modal/paddleocr_modal.py)
GPU_PREDICT_PATH = "predict"


def predict_url(endpoint_url: str | None) -> str | None:
    """Корень развёрнутого приложения -> URL ручки предсказания."""
    if not endpoint_url:
        return endpoint_url
    base = endpoint_url.rstrip("/")
    if base.endswith(f"/{GPU_PREDICT_PATH}"):
        return base
    return f"{base}/{GPU_PREDICT_PATH}"

OWLV2 = "owlv2"
LLMDET = "llmdet"

# Движки, о которых UI должен знать всегда — даже если плагин не установлен
# (иначе пользователю негде увидеть, что GPU-режим существует, а в проекте
# детекции список движков оказался бы пустым без объяснения).
CORE_LABELERS = (RAPIDOCR, MODAL_GPU, OWLV2)

DETECTION = ("detection",)
OCR = ("ocr",)
BOTH = ("detection", "ocr")

# tasks — типы задач проекта, для которых движок пригоден: UI показывает
# только подходящие. Неизвестный сторонний плагин не прячем (BOTH).
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
    """Движок установлен, но не готов к работе (нет обязательных настроек)."""


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
    """Публичное представление настроек. Секрет не отдаётся никогда."""
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
    """Почему modal_gpu недоступен; None — доступен."""
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
    """Дополнить конфиг движка данными из настроек (руками JSON никто не пишет)."""
    resolved = dict(config)
    if name != MODAL_GPU:
        return resolved
    blocker = gpu_blocker(row)
    if blocker is not None:
        raise LabelerNotReadyError(f"Engine {MODAL_GPU} is not ready: {blocker}")
    # в настройках лежит корень приложения Modal, рецепт слушает POST /predict
    resolved.setdefault("endpoint", predict_url(row.gpu_endpoint_url))
    gpu_token = app_config.autolabelui_gpu_token
    if not gpu_token and row.gpu_access_token_encrypted:
        from app.crypto import decrypt_secret

        gpu_token = decrypt_secret(row.gpu_access_token_encrypted)
    if gpu_token:
        resolved.setdefault("api_key", gpu_token)
    return resolved


def build_labelers(
    installed: Iterable[str], row: InstanceSettings | None
) -> list[dict]:
    """Установленные плагины + доступность из настроек, в порядке для UI."""
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
