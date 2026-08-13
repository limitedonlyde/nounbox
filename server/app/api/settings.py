"""Настройки инсталляции: токен Modal и кнопка «Подключить GPU».

Секрет токена наружу не отдаётся ни в одной ручке — только маска token_id.
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
        # другой аккаунт — прежний endpoint к нему не относится
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
    """Развернуть GPU-рецепт в аккаунт Modal пользователя. Прогресс — в GET /jobs/{id}."""
    row = await settings_store.get_or_create(session)
    if not (row.modal_token_id and row.modal_token_secret_encrypted):
        raise HTTPException(400, "Сначала сохраните токен Modal в настройках")

    # payload — только неsecret-данные: токен воркер берёт из настроек сам
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
