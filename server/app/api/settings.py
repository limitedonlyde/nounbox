"""Instance settings: the Modal token and the "Connect GPU" button.

The token secret is never handed out by any endpoint, only the masked token_id.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings as app_settings
from app.crypto import encrypt_secret
from app.db import get_session
from app.models import GpuStatus, Job, JobType
from app.schemas import JobOut, ModalTokenUpdate, SettingsOut
from app.security import require_access
from app.services import modal_deploy, settings_store

router = APIRouter(prefix="/settings", tags=["settings"])


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@router.get("", response_model=SettingsOut)
async def get_settings(session: AsyncSession = Depends(get_session)):
    return settings_store.to_out(await settings_store.get_row(session))


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
        # a different account — the previous endpoint has nothing to do with it
        row.gpu_status = GpuStatus.NOT_CONFIGURED
        row.gpu_endpoint_url = None
        row.gpu_error = None
    row.modal_token_id = token_id
    row.modal_token_secret_encrypted = encrypt_secret(token_secret)
    row.updated_at = _now()
    await session.commit()
    return settings_store.to_out(row)


@router.delete("/modal", response_model=SettingsOut, dependencies=[Depends(require_access)])
async def delete_modal_token(session: AsyncSession = Depends(get_session)):
    row = await settings_store.get_row(session)
    if row is None:
        return settings_store.to_out(None)
    row.modal_token_id = None
    row.modal_token_secret_encrypted = None
    row.gpu_status = GpuStatus.NOT_CONFIGURED
    row.gpu_endpoint_url = None
    row.gpu_error = None
    row.updated_at = _now()
    await session.commit()
    return settings_store.to_out(row)


@router.post("/gpu/deploy", response_model=JobOut, status_code=202, dependencies=[Depends(require_access)])
async def deploy_gpu(request: Request, session: AsyncSession = Depends(get_session)):
    """Deploy the GPU recipe into the user's Modal account. Progress: GET /jobs/{id}."""
    row = await settings_store.get_or_create(session)
    if not (row.modal_token_id and row.modal_token_secret_encrypted):
        raise HTTPException(400, "Save a Modal token in the settings first")

    # payload: non-secret data only — the worker fetches the token from the
    # settings on its own
    job = Job(
        type=JobType.DEPLOY_GPU,
        payload={"app_name": app_settings.modal_gpu_app_name},
    )
    session.add(job)
    row.gpu_status = GpuStatus.DEPLOYING
    row.gpu_error = None
    row.updated_at = _now()
    await session.commit()
    await session.refresh(job)

    await request.app.state.arq.enqueue_job("run_deploy_gpu", str(job.id))
    return job
