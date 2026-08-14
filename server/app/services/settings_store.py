"""Instance settings: reading/creating the row, engine catalog, config resolution.

Two labeling modes: CPU (owlv2 for detection, rapidocr for OCR — always
available, no account) and GPU in the user's own Modal account. The GPU side is
one engine per task: modal_gpu runs the OCR recipe, modal_gpu_detect runs the
detection one, and each has its own deployment row (see services/gpu_recipes).
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_config
from app.crypto import mask_token_id
from app.security import access_token_configured
from app.models import GpuDeployment, GpuStatus, InstanceSettings
from app.services import gpu_recipes
from app.services.gpu_recipes import GPU_RECIPES, MODAL_GPU, MODAL_GPU_DETECT

RAPIDOCR = "rapidocr"

# the path the GPU recipes listen on (see GpuRecipe.predict_path)
GPU_PREDICT_PATH = "predict"


def predict_url(endpoint_url: str | None, path: str = GPU_PREDICT_PATH) -> str | None:
    """Root of the deployed app -> URL of the prediction endpoint."""
    if not endpoint_url:
        return endpoint_url
    base = endpoint_url.rstrip("/")
    if base.endswith(f"/{path}"):
        return base
    return f"{base}/{path}"

OWLV2 = "owlv2"
LLMDET = "llmdet"

# Engines the UI must always know about — even when the plugin is not installed
# (otherwise the user has nowhere to see that GPU mode exists at all, and in a
# detection project the engine list would come out empty with no explanation).
CORE_LABELERS = (RAPIDOCR, MODAL_GPU, MODAL_GPU_DETECT, OWLV2)

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

# The GPU entries are DERIVED from the recipe registry, so an engine can never
# advertise a task its deployed recipe does not serve. That mismatch is the
# whole bug: modal_gpu offering itself to a detection project sends photos to
# a PaddleOCR endpoint and gets back text_line boxes that export throws away.
CATALOG.update(
    {
        recipe.engine: {
            "title": recipe.title,
            "requires": recipe.requires,
            "tasks": (recipe.task,),
        }
        for recipe in GPU_RECIPES.values()
    }
)

# Engines whose predict() spends its time waiting on another machine. The
# labeling run may send images to these concurrently; a local engine already
# occupies the worker's cores, and firing several at once would only make them
# fight each other.
#
# Not derived from "requires": that says what the user must supply, and it puts
# http and vlm (remote endpoints) in the same "config" bucket as consensus,
# which runs its member engines in this very process — parallelizing consensus
# would multiply whatever those engines are already doing.
# Only the platform's own GPU recipes. vlm and http were here and should not
# have been: their endpoint is whatever the user configured, and vlm's shipped
# default is VLM_BASE_URL = http://host.docker.internal:11434/v1 — Ollama on the
# very machine running the worker. Sending it four images at once would split
# one local GPU's memory four ways, which is the opposite of the intent. For
# those two engines "remote" is a property of a config the platform cannot see.
REMOTE_ENGINES = frozenset(GPU_RECIPES)


def is_remote(name: str) -> bool:
    """Whether the engine's work happens on another machine.

    An unknown engine (a third-party plugin) counts as local: assuming local is
    merely slow, while assuming remote would let one plugin run several copies
    of a model on the worker's own cores.
    """
    return name in REMOTE_ENGINES

# MODAL_GPU_DETECT sits AFTER OWLV2: the first detection engine a user sees
# must stay the CPU one that needs no account.
ORDER = (
    RAPIDOCR,
    MODAL_GPU,
    OWLV2,
    LLMDET,
    MODAL_GPU_DETECT,
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


async def load_deployments(
    session: AsyncSession, *, persist: bool = False
) -> dict[str, GpuDeployment]:
    """Every GPU deployment, by engine name.

    Includes a one-time seed for installations that deployed the GPU before
    this table existed: their state lives in the legacy settings.gpu_* columns,
    and it describes an OCR endpoint, so it becomes the modal_gpu row — a
    ready endpoint keeps working without a redeploy, and a failed one keeps its
    error message. Nothing seeds modal_gpu_detect: that app has never been
    deployed there, and inventing a row for it would point a detection project
    at the OCR endpoint, which is the bug this whole change removes.

    The seed is TRANSIENT unless persist=True. Most callers are read paths —
    GET /settings, GET /labelers, run_autolabel — and get_session does not
    commit, so a row added here would be rolled back at the end of every
    request and seeded again on the next one, forever. A detached object serves
    those callers just as well. Only the two callers that own a commit and go
    on to mutate the row ask for persist=True.
    """
    rows = (await session.execute(select(GpuDeployment))).scalars().all()
    deployments = {row.engine: row for row in rows}
    if MODAL_GPU in deployments:
        return deployments

    legacy = await get_row(session)  # never creates the row
    if legacy is None or (
        not legacy.gpu_endpoint_url
        and legacy.gpu_status == GpuStatus.NOT_CONFIGURED
    ):
        return deployments
    seeded = GpuDeployment(
        engine=MODAL_GPU,
        app_name=gpu_recipes.configured_app_name(MODAL_GPU) or "nounbox-paddleocr",
        status=legacy.gpu_status,
        endpoint_url=legacy.gpu_endpoint_url,
        error=legacy.gpu_error,
        access_token_encrypted=legacy.gpu_access_token_encrypted,
    )
    if persist:
        session.add(seeded)
        await session.flush()
    deployments[MODAL_GPU] = seeded
    return deployments


def to_out(
    row: InstanceSettings | None,
    deployments: dict[str, GpuDeployment] | None = None,
) -> dict:
    """Public view of the settings. The secret is never handed out."""
    deployments = deployments or {}
    gpu = deployments.get(MODAL_GPU)
    # the three flat gpu_* fields stay in the response and keep describing the
    # OCR GPU: an external reader written against the old shape must not break
    if gpu is not None:
        flat = {
            "gpu_status": gpu.status,
            "gpu_endpoint_url": gpu.endpoint_url,
            "gpu_error": gpu.error,
        }
    elif row is not None:
        flat = {
            "gpu_status": row.gpu_status,
            "gpu_endpoint_url": row.gpu_endpoint_url,
            "gpu_error": row.gpu_error,
        }
    else:
        flat = {
            "gpu_status": GpuStatus.NOT_CONFIGURED,
            "gpu_endpoint_url": None,
            "gpu_error": None,
        }

    gpus = []
    for recipe in GPU_RECIPES.values():
        dep = deployments.get(recipe.engine)
        gpus.append(
            {
                "engine": recipe.engine,
                "title": recipe.title,
                "task": recipe.task,
                "status": dep.status if dep else GpuStatus.NOT_CONFIGURED,
                "endpoint_url": dep.endpoint_url if dep else None,
                "error": dep.error if dep else None,
            }
        )

    return {
        "modal_configured": bool(
            row is not None
            and row.modal_token_id
            and row.modal_token_secret_encrypted
        ),
        "modal_token_id_masked": mask_token_id(row.modal_token_id) if row else None,
        "access_protected": access_token_configured(),
        **flat,
        "gpus": gpus,
    }


def gpu_ready(deployment: GpuDeployment | None) -> bool:
    return bool(
        deployment is not None
        and deployment.status == GpuStatus.READY
        and deployment.endpoint_url
    )


def gpu_blocker(
    row: InstanceSettings | None,
    deployment: GpuDeployment | None,
    engine: str = MODAL_GPU,
) -> str | None:
    """Why this GPU engine is unavailable; None — it is available."""
    if gpu_ready(deployment):
        return None
    if row is None or not row.modal_token_secret_encrypted:
        return "A Modal token is required"
    if deployment is not None and deployment.status == GpuStatus.DEPLOYING:
        return "The GPU is deploying — please wait"
    if deployment is not None and deployment.status == GpuStatus.FAILED:
        return "The GPU deploy failed — see the settings page"
    if engine == MODAL_GPU_DETECT:
        return "Deploy the detection GPU on the settings page"
    return "Click “Connect GPU” on the settings page"


def labeler_supports_task(name: str, task: str) -> bool:
    """Whether the engine handles this project's task type.

    Same tuple the UI filters the engine picker on — enforced on the server too,
    because AutolabelRequest.labeler is a free string and a run with no engine
    named fans out over every installed engine.
    """
    return task in CATALOG.get(name, {}).get("tasks", BOTH)


def resolve_labeler_config(
    name: str,
    config: dict,
    row: InstanceSettings | None,
    deployments: dict[str, GpuDeployment] | None = None,
) -> dict:
    """Fill the engine config from the settings (nobody writes that JSON by hand)."""
    resolved = dict(config)
    if name not in GPU_RECIPES:
        return resolved

    deployments = deployments or {}
    deployment = deployments.get(name)
    blocker = gpu_blocker(row, deployment, name)
    if blocker is not None:
        # deliberately no fallback to another engine's endpoint: a detection
        # project with only the OCR app deployed must fail loudly instead of
        # quietly sending photos to PaddleOCR
        raise LabelerNotReadyError(f"Engine {name} is not ready: {blocker}")

    # the deployment holds the root of the Modal app; the recipe listens on /predict
    resolved.setdefault(
        "endpoint", predict_url(deployment.endpoint_url, GPU_RECIPES[name].predict_path)
    )
    gpu_token = app_config.nounbox_gpu_token
    if not gpu_token and deployment.access_token_encrypted:
        from app.crypto import decrypt_secret

        gpu_token = decrypt_secret(deployment.access_token_encrypted)
    if gpu_token:
        resolved.setdefault("api_key", gpu_token)
    return resolved


def build_labelers(
    installed: Iterable[str],
    row: InstanceSettings | None,
    deployments: dict[str, GpuDeployment] | None = None,
) -> list[dict]:
    """Installed plugins + availability from the settings, in UI order."""
    installed = set(installed)
    deployments = deployments or {}
    names = [n for n in ORDER if n in installed or n in CORE_LABELERS]
    names += sorted(installed - set(names))

    items = []
    for name in names:
        meta = CATALOG.get(name, {})
        available, reason = True, None
        if name not in installed:
            available, reason = False, "Engine is not installed in this image"
        elif name in GPU_RECIPES:
            reason = gpu_blocker(row, deployments.get(name), name)
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
