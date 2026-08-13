from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app import storage
from app.api import api_router
from app.config import settings
from app.db import engine
from app.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Создание таблиц при старте — только для каркаса v0.1.
    # TODO: заменить на Alembic-миграции до публикации.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await run_in_threadpool(storage.ensure_bucket)

    app.state.arq = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    yield
    await app.state.arq.close()
    await engine.dispose()


app = FastAPI(title="AutoLabelUi API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 без эха присланного тела.

    Штатный обработчик FastAPI возвращает поле input с тем, что прислал клиент,
    и на PUT /settings это отправило бы токен Modal обратно в браузер (а оттуда
    в логи прокси и консоль). Оставляем только тип, путь и сообщение.
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": [
                {"type": e.get("type"), "loc": e.get("loc"), "msg": e.get("msg")}
                for e in exc.errors()
            ]
        },
    )


app.include_router(api_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
