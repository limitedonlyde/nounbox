"""Engine list for the UI: installed plugins + availability from the settings."""

from nounbox_sdk import load_labelers
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.db import get_session
from app.schemas import LabelerOut
from app.services import settings_store

router = APIRouter(prefix="/labelers", tags=["labelers"])


@router.get("", response_model=list[LabelerOut])
async def list_labelers(session: AsyncSession = Depends(get_session)):
    installed = await run_in_threadpool(load_labelers)
    row = await settings_store.get_row(session)
    return settings_store.build_labelers(installed, row)
