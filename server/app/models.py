"""Database schema. Annotation user_id and provenance are baked in from day one.

provenance (source / confidence / status) is not bookkeeping: it is what we
measure engine quality on the user's own photos from, and what the accept
threshold is picked from.
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
    """Project task type. Detecting objects by your own classes is the main
    scenario; OCR stays the second type (OCR engines are the same plugins)."""

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
    """A project object class: an English query name for the engine + a UI color.

    Annotation.label stores the name, not an id — deliberately: an annotation
    stays readable after its class is deleted, and export does not depend on
    the classes table at all.
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
    # length matches Annotation.label — the class name travels there as is
    name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(20), default="#3b82f6")
    sort_order: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Document(Base):
    """A source file as uploaded (photo, PDF, archive...)."""

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
    """A normalized page/frame produced from a Document during ingest."""

    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    page_index: Mapped[int] = mapped_column(default=0)
    s3_key: Mapped[str] = mapped_column(String(1000))
    width: Mapped[int] = mapped_column(default=0)
    height: Mapped[int] = mapped_column(default=0)
    # SHA-256 of the normalized bytes (exact dedup); perceptual hash later, in its own column
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    blur_score: Mapped[float | None] = mapped_column(nullable=True)
    # Frame looked at by a human. Needed for exactly one distinction: "checked,
    # there are no objects" versus "not looked at yet". The first is a valid
    # background example for the detector, the second must never reach the
    # dataset. Reviewed annotations (accepted/edited/rejected) already count as
    # a review; the flag is set explicitly for frames that had no boxes at all.
    reviewed: Mapped[bool] = mapped_column(default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Annotation(Base):
    """A universal annotation: detection / recognition / layout / KIE."""

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
    """A background job (ingest / autolabel / deploy_gpu)."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # deploy_gpu is an installation-level job, with no project
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
    """Installation settings: a single row (user_id is reserved for multi-tenancy).

    The Modal token secret is stored encrypted only (app.crypto).
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
    # Bearer token of the deployed GPU endpoint: generated at deploy time,
    # otherwise anyone who learns the URL can use it on the account owner's bill
    gpu_access_token_encrypted: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
