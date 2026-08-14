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

# re-exported on purpose: the app name / recipe path now come from
# gpu_recipes (per engine), but `modal_deploy.settings` stays a valid handle
# on the same settings singleton for callers and tests that patch it
from app.config import settings  # noqa: F401
from app.services import gpu_recipes
from app.services.gpu_recipes import GPU_RECIPES, MODAL_GPU

logger = logging.getLogger(__name__)

# a stray mounted ~/.modal.toml (two active profiles) would break the deploy
os.environ.setdefault("MODAL_CONFIG_PATH", "/nonexistent/modal.toml")

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


# Recipes live in the monorepo only (in the image at /deploy/modal, via
# COPY deploy/modal). No copy inside the package: two copies drift apart
# silently, and the recipe is the thing that must match the CPU engine exactly.
MONOREPO_DIRS = (
    Path("/deploy/modal"),
    Path(__file__).resolve().parents[3] / "deploy" / "modal",
)

# environment variable that overrides each engine's recipe path (for the error
# message only — the value itself comes from gpu_recipes)
_PATH_ENV = {
    "modal_gpu": "MODAL_GPU_RECIPE_PATH",
    "modal_gpu_detect": "MODAL_GPU_DETECT_RECIPE_PATH",
}


def recipe_path(engine: str = MODAL_GPU) -> Path:
    """Path of the recipe file this engine deploys."""
    override = gpu_recipes.configured_recipe_path(engine)
    if override:
        return Path(override)
    filename = GPU_RECIPES[engine].filename
    for directory in MONOREPO_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"GPU recipe not found: expected /deploy/modal/{filename} "
        f"(the image gets it from COPY deploy/modal), or a path in "
        f"{_PATH_ENV.get(engine, 'MODAL_GPU_RECIPE_PATH')}"
    )


def _module_name(engine: str) -> str:
    """sys.modules key the recipe is registered under.

    Per engine, not a constant: Modal serializes a function by the name of the
    module it was defined in, so importing two recipes under one name would
    make the second deploy overwrite the first one's functions.
    """
    return f"nounbox_gpu_recipe_{engine}"


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


def load_recipe_app(path: Path, module_name: str = _module_name(MODAL_GPU)):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModalDeployError(f"GPU recipe not found: {path}")
    module = importlib.util.module_from_spec(spec)
    # recipe functions are serialized by module name — register it up front
    sys.modules[module_name] = module
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
    *,
    engine: str = MODAL_GPU,
    app_name: str | None = None,
    path: Path | str | None = None,
    gpu_token: str | None = None,
) -> DeployedApp:
    import modal
    import modal.runner

    problem = validate_token_pair(token_id, token_secret)
    if problem:
        raise ModalDeployError(problem)

    resolved_path = Path(path) if path else recipe_path(engine)
    # The recipe bakes the Bearer token into a Secret at module level, reading
    # this variable — so it has to be set BEFORE the import, and restored right
    # after. The worker process is long-lived and deploys both recipes: leaving
    # the variable set would hand the OCR app's token to the detection deploy
    # that follows, and leave a live secret sitting in the environment of every
    # later job for no reason.
    previous = os.environ.get(GPU_TOKEN_ENV)
    if gpu_token:
        os.environ[GPU_TOKEN_ENV] = gpu_token
    try:
        app = load_recipe_app(resolved_path, _module_name(engine))
    finally:
        if gpu_token:
            if previous is None:
                os.environ.pop(GPU_TOKEN_ENV, None)
            else:
                os.environ[GPU_TOKEN_ENV] = previous
    endpoints = list(getattr(app, "registered_web_endpoints", []))
    if not endpoints:
        raise ModalDeployError("The GPU recipe has no web endpoint (@modal.asgi_app)")

    # empty app name in the environment — deploy under the recipe's own name
    app_name = app_name or gpu_recipes.configured_app_name(engine) or app.name
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
        # gpu_token too: it is the live bearer of the endpoint being deployed,
        # and modal's errors quote the arguments they were called with
        raise ModalDeployError(
            _friendly(exc, token_secret, token_id, gpu_token or "")
        ) from None

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
