"""Pydantic-схемы API."""

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

# Имя класса уходит в движок как текстовый запрос ("carpet", "coffee table"),
# поэтому оно английское: кириллица/иероглифы дадут молча пустую разметку.
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
    color: ClassColor | None = None  # None — цвет из палитры по порядку


class ProjectClassReplace(BaseModel):
    """PUT: список классов проекта целиком (цвета назначаются автоматически)."""

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
    created_at: datetime

    model_config = {"from_attributes": True}


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
    labeler: str | None = None  # ограничить конкретным движком


# --- Jobs ---
class AutolabelRequest(BaseModel):
    labeler: str | None = None  # имя движка; None — все установленные
    config: dict[str, Any] = Field(default_factory=dict)
    # Перезапуск на уже размеченных изображениях. Нужен, когда изменился список
    # классов: без него движок пропустит все кадры и вернёт ноль без объяснений.
    # Непроверенные рамки этого движка заменяются, принятые человеком — нет.
    rerun: bool = False


class JobOut(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID | None  # deploy_gpu — задача без проекта
    type: JobType
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any]
    created_at: datetime
    finished_at: datetime | None

    model_config = {"from_attributes": True}


# --- Settings / labelers ---
class SettingsOut(BaseModel):
    modal_configured: bool
    modal_token_id_masked: str | None
    gpu_status: GpuStatus
    gpu_endpoint_url: str | None
    gpu_error: str | None
    # защищены ли ручки, распоряжающиеся токеном Modal и деплоем; false —
    # UI показывает предупреждение, а не делает вид, что всё в порядке
    access_protected: bool = False


class ModalTokenUpdate(BaseModel):
    """Без ограничений длины/формата в схеме: 422 от pydantic возвращает
    невалидное значение в теле ответа, а секрет наружу выходить не должен —
    формат проверяется в ручке."""

    modal_token_id: str
    modal_token_secret: str


class LabelerOut(BaseModel):
    name: str
    title: str
    requires: Literal["cpu", "modal", "config"]
    # типы задач, для которых движок пригоден — UI прячет лишнее
    tasks: list[TaskType]
    available: bool
    reason: str | None
