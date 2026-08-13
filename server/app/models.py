"""Схема БД. user_id и provenance аннотаций заложены с первого дня.

provenance (source / confidence / status) — не служебное поле, а то, из чего
считается качество движка на данных пользователя и подбирается порог.
"""

import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TaskType(StrEnum):
    """Тип задачи проекта. Детекция объектов по своим классам — основной сценарий,
    OCR остаётся вторым типом (движки OCR — те же плагины)."""

    DETECTION = "detection"
    OCR = "ocr"


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class AnnotationStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobType(StrEnum):
    INGEST = "ingest"
    AUTOLABEL = "autolabel"
    DEPLOY_GPU = "deploy_gpu"


class GpuStatus(StrEnum):
    NOT_CONFIGURED = "not_configured"
    DEPLOYING = "deploying"
    READY = "ready"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    task_type: Mapped[TaskType] = mapped_column(
        SAEnum(TaskType, native_enum=False, length=20),
        default=TaskType.DETECTION,
        server_default=TaskType.DETECTION.value,
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ProjectClass(Base):
    """Класс объектов проекта: английское имя-запрос для движка + цвет для UI.

    Annotation.label хранит именно name — связи по id нет намеренно: аннотация
    остаётся читаемой после удаления класса, а экспорт не зависит от таблицы.
    """

    __tablename__ = "project_classes"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_project_classes_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE")
    )
    # длина совпадает с Annotation.label — имя класса уезжает туда как есть
    name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(20), default="#3b82f6")
    sort_order: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Document(Base):
    """Исходный загруженный файл (фото, PDF, архив...)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id"))
    filename: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(200), default="")
    size_bytes: Mapped[int] = mapped_column(default=0)
    s3_key: Mapped[str] = mapped_column(String(1000))
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, native_enum=False, length=20),
        default=DocumentStatus.UPLOADED,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Image(Base):
    """Нормализованная страница/кадр, полученная из Document при ingest."""

    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    page_index: Mapped[int] = mapped_column(default=0)
    s3_key: Mapped[str] = mapped_column(String(1000))
    width: Mapped[int] = mapped_column(default=0)
    height: Mapped[int] = mapped_column(default=0)
    # SHA-256 нормализованных байтов (точный дедуп); perceptual hash — позже отдельной колонкой
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    blur_score: Mapped[float | None] = mapped_column(nullable=True)
    # Кадр просмотрен человеком. Нужен ровно для одного различия: «проверил,
    # объектов нет» против «ещё не смотрели». Первое — валидный фоновый пример
    # для детектора, второе в датасет пускать нельзя. Само по себе наличие
    # проверенных аннотаций (accepted/edited/rejected) тоже считается проверкой,
    # флаг ставится явно для кадров, где рамок не было вовсе.
    reviewed: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Annotation(Base):
    """Универсальная аннотация: detection / recognition / layout / KIE."""

    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("images.id"))
    # {"type": "bbox", "x":.., "y":.., "width":.., "height":..}
    # {"type": "polygon", "points": [[x, y], ...]}
    geometry: Mapped[dict] = mapped_column(JSONB)
    label: Mapped[str] = mapped_column(String(100), default="text_line")
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    attrs: Mapped[dict] = mapped_column(JSONB, default=dict)  # KIE: {"field": "total"}
    confidence: Mapped[float] = mapped_column(default=1.0)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("annotations.id"), nullable=True
    )
    # provenance: {"type": "engine"|"human", "name": "paddleocr", "version": "2.7"}
    source: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[AnnotationStatus] = mapped_column(
        SAEnum(AnnotationStatus, native_enum=False, length=20),
        default=AnnotationStatus.PENDING,
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Job(Base):
    """Фоновая задача (ingest / autolabel / deploy_gpu)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # deploy_gpu — задача уровня инсталляции, без проекта
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True
    )
    type: Mapped[JobType] = mapped_column(
        SAEnum(JobType, native_enum=False, length=20)
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, native_enum=False, length=20), default=JobStatus.QUEUED
    )
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class InstanceSettings(Base):
    """Настройки инсталляции: одна строка (user_id зарезервирован на мультиаренду).

    Секрет токена Modal хранится только в зашифрованном виде (app.crypto).
    """

    __tablename__ = "settings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    modal_token_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    modal_token_secret_encrypted: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    gpu_status: Mapped[GpuStatus] = mapped_column(
        SAEnum(GpuStatus, native_enum=False, length=20),
        default=GpuStatus.NOT_CONFIGURED,
    )
    gpu_endpoint_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    gpu_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Bearer-токен развёрнутого GPU-эндпоинта: генерируется при деплое, иначе
    # эндпоинт открыт любому, кто узнал URL, и тратит деньги владельца аккаунта
    gpu_access_token_encrypted: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
