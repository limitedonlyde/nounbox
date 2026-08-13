"""Развёртывание GPU-рецепта в аккаунт Modal пользователя.

Креды передаются явно (modal.Client.from_credentials — официальный путь для
«managing Modal on behalf of third-party users»): ни ~/.modal.toml, ни
MODAL_TOKEN_*, ни интерактива. URL web-эндпоинта deploy_app не возвращает —
его берём отдельно через Function.from_name + hydrate(client=...).
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

# случайно примонтированный ~/.modal.toml (два активных профиля) сломал бы деплой
os.environ.setdefault("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")

RECIPE_MODULE_NAME = "autolabelui_gpu_recipe"
APP_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")
# имя переменной, из которой рецепт забирает Bearer-токен эндпоинта
GPU_TOKEN_ENV = "AUTOLABELUI_GPU_TOKEN"

TOKEN_ID_PREFIX = "ak-"
TOKEN_SECRET_PREFIX = "as-"
PROXY_PREFIXES = ("wk-", "ws-")

PROXY_TOKEN_HINT = (
    "Похоже, введён proxy-токен (wk-…/ws-…) — он годится только для вызова "
    "эндпоинта. Нужен API-токен из `modal token new`: token_id начинается "
    "с ak-, секрет — с as-."
)


class ModalDeployError(RuntimeError):
    """Развернуть GPU не удалось; текст пригоден для показа пользователю."""


@dataclass(slots=True)
class DeployedApp:
    app_id: str
    app_page_url: str
    endpoint_url: str
    warnings: list[str] = field(default_factory=list)


def validate_token_pair(token_id: str, token_secret: str) -> str | None:
    """Локальная проверка формата пары токенов; None — формат в порядке."""
    if not token_id or not token_secret:
        return "Укажите и token_id, и token_secret"
    if token_id.startswith(PROXY_PREFIXES) or token_secret.startswith(PROXY_PREFIXES):
        return PROXY_TOKEN_HINT
    if not token_id.startswith(TOKEN_ID_PREFIX):
        return "token_id должен начинаться с ak- (создаётся командой `modal token new`)"
    if not token_secret.startswith(TOKEN_SECRET_PREFIX):
        return "token_secret должен начинаться с as- (создаётся командой `modal token new`)"
    return None


# Рецепт один — тот, что в монорепо (в образе лежит в /deploy/modal).
# Дубликат внутри пакета не держим: две копии разъезжаются молча.
MONOREPO_RECIPES = (
    Path("/deploy/modal/paddleocr_modal.py"),
    Path(__file__).resolve().parents[3] / "deploy" / "modal" / "paddleocr_modal.py",
)


def recipe_path() -> Path:
    if settings.modal_gpu_recipe_path:
        return Path(settings.modal_gpu_recipe_path)
    for candidate in MONOREPO_RECIPES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "GPU-рецепт не найден: ожидается /deploy/modal/paddleocr_modal.py "
        "(в образе — COPY deploy/modal) либо путь в MODAL_GPU_RECIPE_PATH"
    )


def scrub_secrets(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def _friendly(exc: Exception, *secrets: str) -> str:
    message = scrub_secrets(f"{type(exc).__name__}: {exc}", *secrets)
    if "Token ID is malformed" in message:
        return PROXY_TOKEN_HINT
    if "Token not found" in message:
        return "Modal не знает такой токен — проверьте пару token_id/token_secret."
    return message


async def _call(func, *args, **kwargs):
    """Асинхронный вызов синхронной обёртки modal (в FastAPI нужен .aio)."""
    aio = getattr(func, "aio", None)
    if aio is not None:
        return await aio(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def load_recipe_app(path: Path):
    spec = importlib.util.spec_from_file_location(RECIPE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ModalDeployError(f"GPU-рецепт не найден: {path}")
    module = importlib.util.module_from_spec(spec)
    # функции рецепта сериализуются по имени модуля — регистрируем его заранее
    sys.modules[RECIPE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as exc:
        raise ModalDeployError(f"GPU-рецепт не найден: {path}") from exc
    app = getattr(module, "app", None)
    if app is None:
        raise ModalDeployError(f"В рецепте {path} нет объекта app (modal.App)")
    return app


async def deploy_gpu_app(
    token_id: str,
    token_secret: str,
    app_name: str | None = None,
    path: Path | str | None = None,
    gpu_token: str | None = None,
) -> DeployedApp:
    import modal
    import modal.runner

    problem = validate_token_pair(token_id, token_secret)
    if problem:
        raise ModalDeployError(problem)

    resolved_path = Path(path) if path else recipe_path()
    # рецепт запекает Bearer-токен в Secret на уровне модуля, читая эту
    # переменную, — выставляем её ДО импорта рецепта
    if gpu_token:
        os.environ[GPU_TOKEN_ENV] = gpu_token
    app = load_recipe_app(resolved_path)
    endpoints = list(getattr(app, "registered_web_endpoints", []))
    if not endpoints:
        raise ModalDeployError("В GPU-рецепте нет web-эндпоинта (@modal.asgi_app)")

    # пустой modal_gpu_app_name — деплой под именем из самого рецепта
    app_name = app_name or settings.modal_gpu_app_name or app.name
    if not APP_NAME_RE.match(app_name):
        raise ModalDeployError(
            f"Недопустимое имя приложения Modal {app_name!r}: "
            "разрешены [a-zA-Z0-9._-], не длиннее 64 символов"
        )

    logger.info("Deploying Modal app %s (recipe %s)", app_name, resolved_path)
    try:
        client = await _call(modal.Client.from_credentials, token_id, token_secret)
        result = await _call(modal.runner.deploy_app, app, name=app_name, client=client)
        function = modal.Function.from_name(app_name, endpoints[0], client=client)
        # без явной гидрации ленивый get_web_url уйдёт в Client.from_env()
        await _call(function.hydrate, client=client)
        endpoint_url = function.get_web_url()
    except ModalDeployError:
        raise
    except Exception as exc:
        raise ModalDeployError(_friendly(exc, token_secret, token_id)) from None

    if not endpoint_url:
        raise ModalDeployError(
            f"Modal развернул {app_name}, но не вернул URL web-эндпоинта"
        )
    logger.info("Modal app %s deployed: %s", app_name, endpoint_url)
    return DeployedApp(
        app_id=result.app_id,
        app_page_url=result.app_page_url,
        endpoint_url=endpoint_url,
        warnings=list(getattr(result, "warnings", []) or []),
    )
