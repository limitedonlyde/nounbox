"""Pydantic schemas for the API."""

import re
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field

from app.models import (
    AnnotationStatus,
    DocumentStatus,
    GpuStatus,
    JobStatus,
    JobType,
    TaskType,
)

# A class name goes to the engine as a text query ("carpet", "coffee table"),
# hence English: Cyrillic or CJK silently produce empty labeling.
CLASS_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 '\-_/().]*$")
CLASS_NAME_HINT = (
    "A class name is written in English words: Latin letters, digits, "
    "spaces and hyphens (for example “coffee table”)"
)


def normalize_class_name(value: str) -> str:
    name = " ".join(value.split())
    if not name:
        raise ValueError("A class name cannot be empty")
    if not CLASS_NAME_RE.match(name):
        raise ValueError(CLASS_NAME_HINT)
    return name


ClassName = Annotated[
    str, Field(min_length=1, max_length=100), AfterValidator(normalize_class_name)
]
ClassColor = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]


# --- Projects ---
class ProjectCreate(BaseModel):
    name: str = Field(max_length=200)
    description: str = ""
    task_type: TaskType = TaskType.DETECTION


class ProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    task_type: TaskType
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Project classes (detection) ---
class ProjectClassCreate(BaseModel):
    name: ClassName
    color: ClassColor | None = None  # None — next color from the palette


class ProjectClassReplace(BaseModel):
    """PUT: the project's class list as a whole (colors assigned automatically)."""

    names: list[ClassName]


class ProjectClassUpdate(BaseModel):
    name: ClassName | None = None
    color: ClassColor | None = None
    sort_order: int | None = None


class ProjectClassOut(BaseModel):
    id: uuid.UUID
    name: str
    color: str
    sort_order: int

    model_config = {"from_attributes": True}


# --- Documents / Images ---
class DocumentOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    page_index: int
    width: int
    height: int
    content_hash: str | None
    blur_score: float | None
    # a human confirmed they looked at the frame: needed so that a frame with
    # NO objects lands in the dataset as a background example instead of being lost
    reviewed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageUpdate(BaseModel):
    reviewed: bool


class ImageWithStats(ImageOut):
    total_annotations: int
    pending_annotations: int


# --- Annotations ---
class BBoxGeometry(BaseModel):
    type: Literal["bbox"]
    x: float
    y: float
    width: float
    height: float


class PolygonGeometry(BaseModel):
    type: Literal["polygon"]
    points: list[tuple[float, float]]


Geometry = BBoxGeometry | PolygonGeometry


class AnnotationCreate(BaseModel):
    geometry: Geometry
    label: str = "text_line"
    text: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    parent_id: uuid.UUID | None = None


class AnnotationUpdate(BaseModel):
    geometry: Geometry | None = None
    label: str | None = None
    text: str | None = None
    attrs: dict[str, Any] | None = None
    status: AnnotationStatus | None = None


class AnnotationOut(BaseModel):
    id: uuid.UUID
    image_id: uuid.UUID
    geometry: dict[str, Any]
    label: str
    text: str | None
    attrs: dict[str, Any]
    confidence: float
    parent_id: uuid.UUID | None
    source: dict[str, Any]
    status: AnnotationStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BulkAcceptRequest(BaseModel):
    min_confidence: float = 0.9
    labeler: str | None = None  # restrict to one specific engine


# --- Jobs ---
class AutolabelRequest(BaseModel):
    labeler: str | None = None  # engine name; None — every installed one
    config: dict[str, Any] = Field(default_factory=dict)
    # Re-run over images that are already labeled. Needed when the class list
    # changed: without it the engine skips every frame and silently returns zero.
    # Pending boxes of this engine are replaced, human-accepted ones are not.
    rerun: bool = False


class JobOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID | None  # deploy_gpu is a job without a project
    type: JobType
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any]
    created_at: datetime
    finished_at: datetime | None

    model_config = {"from_attributes": True}


# --- Settings / labelers ---
class GpuDeploymentOut(BaseModel):
    """One GPU recipe deployed (or not) into the user's Modal account."""

    engine: str
    title: str
    task: TaskType
    status: GpuStatus
    endpoint_url: str | None
    error: str | None


class SettingsOut(BaseModel):
    modal_configured: bool
    modal_token_id_masked: str | None
    # The three flat gpu_* fields describe the OCR GPU (engine modal_gpu) and
    # are kept for compatibility: they were the whole GPU state before there
    # was more than one GPU app. New readers should use `gpus`.
    gpu_status: GpuStatus
    gpu_endpoint_url: str | None
    gpu_error: str | None
    gpus: list[GpuDeploymentOut] = []
    # whether the endpoints that control the Modal token and the deploy are
    # protected; false — the UI shows a warning instead of pretending all is well
    access_protected: bool = False


class GpuDeployRequest(BaseModel):
    """Which GPU recipe to deploy. Optional body: a browser tab opened before
    the upgrade posts nothing, and that still means the OCR GPU it knew."""

    engine: str = "modal_gpu"


class ModalTokenUpdate(BaseModel):
    """No length/format constraints in the schema: pydantic's 422 echoes the
    invalid value back in the response body, and the secret must never leave —
    the format is checked in the endpoint instead."""

    modal_token_id: str
    modal_token_secret: str


class LabelerOut(BaseModel):
    name: str
    title: str
    requires: Literal["cpu", "modal", "config"]
    # task types this engine is suitable for — the UI hides the rest
    tasks: list[TaskType]
    available: bool
    reason: str | None
