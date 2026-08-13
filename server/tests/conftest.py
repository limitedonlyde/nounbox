"""Test environment: sqlite instead of postgres, a local encryption key, a fake arq.

The environment is set up before app.* is imported, because app.config.settings
is read at import time.
"""

import os
import tempfile
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

TMP = Path(tempfile.mkdtemp(prefix="nounbox-tests-"))
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{TMP / 'test.db'}"
os.environ["SETTINGS_KEY_PATH"] = str(TMP / "settings.key")
os.environ.pop("SETTINGS_ENCRYPTION_KEY", None)

from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import event  # noqa: E402
from sqlalchemy.dialects.postgresql import JSONB, UUID  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.ext.compiler import compiles  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app import crypto, db  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from app.models import Base  # noqa: E402
from app.workers import tasks as worker_tasks  # noqa: E402


# postgres-specific schema types map to their sqlite equivalents (the project has
# no migrations, the schema is built by create_all, so this is enough)
@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(type_, compiler, **kw):
    return compiler.visit_JSON(type_, **kw)


@compiles(UUID, "sqlite")
def _compile_uuid_sqlite(type_, compiler, **kw):
    return "CHAR(32)"


class FakeArq:
    """Stand-in for the arq pool: remembers what would have been enqueued."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[str, tuple]] = []

    async def enqueue_job(self, name: str, *args, **kwargs):
        self.enqueued.append((name, args))
        return None


@pytest_asyncio.fixture
async def session_factory(tmp_path, monkeypatch):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'test.db'}", poolclass=NullPool
    )

    # sqlite does not enforce foreign keys by default while postgres does:
    # without this the tests would miss both orphaned rows and IntegrityError
    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", factory)
    monkeypatch.setattr(worker_tasks, "SessionLocal", factory)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield factory
    await engine.dispose()


@pytest.fixture
def key_path(tmp_path, monkeypatch):
    """A fresh encryption key for every test."""
    path = tmp_path / "keys" / "settings.key"
    monkeypatch.setattr(crypto.settings, "settings_key_path", str(path))
    monkeypatch.setattr(crypto.settings, "settings_encryption_key", "")
    crypto.reset_key_cache()
    yield path
    crypto.reset_key_cache()


@pytest.fixture
def arq() -> FakeArq:
    pool = FakeArq()
    fastapi_app.state.arq = pool
    return pool


@pytest_asyncio.fixture
async def client(session_factory, key_path, arq):
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture
def token_pair() -> tuple[str, str]:
    return f"ak-{uuid.uuid4().hex}", f"as-{uuid.uuid4().hex}"
