"""Deploying the GPU recipe into the user's own Modal account.

Credentials are passed explicitly (modal.Client.from_credentials — the official
path for "managing Modal on behalf of third-party users"): no ~/.modal.toml, no
MODAL_TOKEN_*, no interactive prompt. deploy_app does not return the web
endpoint URL — we take it separately via Function.from_name + hydrate(client=...).
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

# a stray mounted ~/.modal.toml (two active profiles) would break the deploy
os.environ.setdefault("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")

RECIPE_MODULE_NAME = "nounbox_gpu_recipe"
APP_NAME_RE = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")
# name of the variable the recipe takes the endpoint's Bearer token from
GPU_TOKEN_ENV = "NOUNBOX_GPU_TOKEN"

TOKEN_ID_PREFIX = "ak-"
TOKEN_SECRET_PREFIX = "as-"
PROXY_PREFIXES = ("wk-", "ws-")

PROXY_TOKEN_HINT = (
    "This looks like a proxy token (wk-…/ws-…) — those only work for calling an "
    "endpoint. A deploy needs an API token from `modal token new`: the token_id "
    "starts with ak- and the secret with as-."
)


class ModalDeployError(RuntimeError):
    """The GPU deploy failed; the message is safe to show to the user."""


@dataclass(slots=True)
class DeployedApp:
    app_id: str
    app_page_url: str
    endpoint_url: str
    warnings: list[str] = field(default_factory=list)


def validate_token_pair(token_id: str, token_secret: str) -> str | None:
    """Local format check of the token pair; None — the format is fine."""
    if not token_id or not token_secret:
        return "Provide both token_id and token_secret"
    if token_id.startswith(PROXY_PREFIXES) or token_secret.startswith(PROXY_PREFIXES):
        return PROXY_TOKEN_HINT
    if not token_id.startswith(TOKEN_ID_PREFIX):
        return "token_id must start with ak- (run `modal token new` to get one)"
    if not token_secret.startswith(TOKEN_SECRET_PREFIX):
        return "token_secret must start with as- (run `modal token new` to get one)"
    return None


# There is one recipe only — the one in the monorepo (in the image it sits at
# /deploy/modal). No copy inside the package: two copies drift apart silently.
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
        "GPU recipe not found: expected /deploy/modal/paddleocr_modal.py "
        "(the image gets it from COPY deploy/modal), or a path in MODAL_GPU_RECIPE_PATH"
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
        return "Modal does not know this token — check the token_id/token_secret pair."
    return message


async def _call(func, *args, **kwargs):
    """Call modal's sync wrapper asynchronously (under FastAPI we need .aio)."""
    aio = getattr(func, "aio", None)
    if aio is not None:
        return await aio(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def load_recipe_app(path: Path):
    spec = importlib.util.spec_from_file_location(RECIPE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ModalDeployError(f"GPU recipe not found: {path}")
    module = importlib.util.module_from_spec(spec)
    # recipe functions are serialized by module name — register it up front
    sys.modules[RECIPE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as exc:
        raise ModalDeployError(f"GPU recipe not found: {path}") from exc
    app = getattr(module, "app", None)
    if app is None:
        raise ModalDeployError(f"Recipe {path} has no app object (modal.App)")
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
    # the recipe bakes the Bearer token into a Secret at module level, reading
    # this variable — so set it BEFORE importing the recipe
    if gpu_token:
        os.environ[GPU_TOKEN_ENV] = gpu_token
    app = load_recipe_app(resolved_path)
    endpoints = list(getattr(app, "registered_web_endpoints", []))
    if not endpoints:
        raise ModalDeployError("The GPU recipe has no web endpoint (@modal.asgi_app)")

    # empty modal_gpu_app_name — deploy under the name from the recipe itself
    app_name = app_name or settings.modal_gpu_app_name or app.name
    if not APP_NAME_RE.match(app_name):
        raise ModalDeployError(
            f"Invalid Modal app name {app_name!r}: "
            "allowed characters are [a-zA-Z0-9._-], at most 64 of them"
        )

    logger.info("Deploying Modal app %s (recipe %s)", app_name, resolved_path)
    try:
        client = await _call(modal.Client.from_credentials, token_id, token_secret)
        result = await _call(modal.runner.deploy_app, app, name=app_name, client=client)
        function = modal.Function.from_name(app_name, endpoints[0], client=client)
        # without explicit hydration, the lazy get_web_url goes to Client.from_env()
        await _call(function.hydrate, client=client)
        endpoint_url = function.get_web_url()
    except ModalDeployError:
        raise
    except Exception as exc:
        raise ModalDeployError(_friendly(exc, token_secret, token_id)) from None

    if not endpoint_url:
        raise ModalDeployError(
            f"Modal deployed {app_name} but returned no web endpoint URL"
        )
    logger.info("Modal app %s deployed: %s", app_name, endpoint_url)
    return DeployedApp(
        app_id=result.app_id,
        app_page_url=result.app_page_url,
        endpoint_url=endpoint_url,
        warnings=list(getattr(result, "warnings", []) or []),
    )
