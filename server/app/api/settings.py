"""Instance settings: the Modal token and the "Connect GPU" buttons.

The token secret is never handed out by any endpoint, only the masked token_id.
There is one GPU app per task (OCR / detection), each deployed separately.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.crypto import encrypt_secret
from app.db import get_session
from app.models import GpuDeployment, GpuStatus, Job, JobType
from app.schemas import GpuDeployRequest, JobOut, ModalTokenUpdate, SettingsOut
from app.security import require_access
from app.services import gpu_recipes, modal_deploy, settings_store
from app.services.gpu_recipes import GPU_RECIPES, MODAL_GPU

router = APIRouter(prefix="/settings", tags=["settings"])


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@router.get("", response_model=SettingsOut)
async def get_settings(session: AsyncSession = Depends(get_session)):
    row = await settings_store.get_row(session)
    deployments = await settings_store.load_deployments(session)
    return settings_store.to_out(row, deployments)


@router.put("", response_model=SettingsOut, dependencies=[Depends(require_access)])
async def put_modal_token(
    body: ModalTokenUpdate, session: AsyncSession = Depends(get_session)
):
    token_id = body.modal_token_id.strip()
    token_secret = body.modal_token_secret.strip()
    problem = modal_deploy.validate_token_pair(token_id, token_secret)
    if problem:
        raise HTTPException(400, problem)

    row = await settings_store.get_or_create(session)
    if row.modal_token_id != token_id:
        # a different account — the endpoints deployed into the previous one
        # have nothing to do with it, for either task. No need to seed the
        # legacy row first: the delete below and the three assignments after it
        # clear both the table and the legacy columns anyway.
        await session.execute(delete(GpuDeployment))
        row.gpu_status = GpuStatus.NOT_CONFIGURED
        row.gpu_endpoint_url = None
        row.gpu_error = None
    row.modal_token_id = token_id
    row.modal_token_secret_encrypted = encrypt_secret(token_secret)
    row.updated_at = _now()
    await session.commit()
    deployments = await settings_store.load_deployments(session)
    return settings_store.to_out(row, deployments)


@router.delete("/modal", response_model=SettingsOut, dependencies=[Depends(require_access)])
async def delete_modal_token(session: AsyncSession = Depends(get_session)):
    row = await settings_store.get_row(session)
    if row is None:
        return settings_store.to_out(None, {})
    await session.execute(delete(GpuDeployment))
    row.modal_token_id = None
    row.modal_token_secret_encrypted = None
    row.gpu_status = GpuStatus.NOT_CONFIGURED
    row.gpu_endpoint_url = None
    row.gpu_error = None
    row.updated_at = _now()
    await session.commit()
    return settings_store.to_out(row, {})


@router.post("/gpu/deploy", response_model=JobOut, status_code=202, dependencies=[Depends(require_access)])
async def deploy_gpu(
    request: Request,
    body: GpuDeployRequest | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Deploy one GPU recipe into the user's Modal account. Progress: GET /jobs/{id}."""
    engine = (body.engine if body else MODAL_GPU) or MODAL_GPU
    if engine not in GPU_RECIPES:
        raise HTTPException(
            400, f"Unknown GPU engine {engine!r}; valid: {', '.join(GPU_RECIPES)}"
        )

    row = await settings_store.get_or_create(session)
    if not (row.modal_token_id and row.modal_token_secret_encrypted):
        raise HTTPException(400, "Save a Modal token in the settings first")

    # payload: non-secret data only — the worker fetches the token from the
    # settings on its own
    job = Job(
        type=JobType.DEPLOY_GPU,
        payload={
            "engine": engine,
            "app_name": gpu_recipes.configured_app_name(engine),
        },
    )
    session.add(job)

    # persist: the seeded legacy row is mutated below, so it has to be a real
    # row in the session rather than a detached copy whose changes go nowhere
    deployments = await settings_store.load_deployments(session, persist=True)
    deployment = deployments.get(engine)
    if deployment is None:
        deployment = GpuDeployment(
            engine=engine, app_name=gpu_recipes.configured_app_name(engine)
        )
        session.add(deployment)
    deployment.status = GpuStatus.DEPLOYING
    deployment.error = None
    deployment.updated_at = _now()
    if engine == MODAL_GPU:
        # legacy mirror, so a rollback to the previous image sees the same state
        row.gpu_status = GpuStatus.DEPLOYING
        row.gpu_error = None
    row.updated_at = _now()
    await session.commit()
    await session.refresh(job)

    await request.app.state.arq.enqueue_job("run_deploy_gpu", str(job.id))
    return job
