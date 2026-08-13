"""Типы данных контракта. Зависимостей нет — только stdlib."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


class Capability(str, Enum):
    """Что умеет labeler."""

    DETECTION = "detection"  # боксы/полигоны текста
    RECOGNITION = "recognition"  # транскрипция внутри геометрии
    LAYOUT = "layout"  # структура документа (заголовки, таблицы...)
    KIE = "kie"  # ключевые поля (attrs)


@dataclass
class BBox:
    """Прямоугольник в абсолютных пикселях (верхний левый угол — начало)."""

    x: float
    y: float
    width: float
    height: float


# Полигон — список точек [x, y] в абсолютных пикселях, по часовой стрелке.
Polygon: TypeAlias = list[tuple[float, float]]


@dataclass
class Annotation:
    """Универсальная аннотация — покрывает detection/recognition/layout/KIE.

    - geometry: BBox | Polygon — где находится объект
    - label: тип ("text_line", "paragraph", "table", "title", ...)
    - text: транскрипция (recognition), если есть
    - attrs: произвольные атрибуты, для KIE — {"field": "total"}
    - confidence: уверенность движка 0..1 (для bulk-accept и очередей ревью)
    """

    geometry: BBox | Polygon
    label: str = "text_line"
    text: str | None = None
    confidence: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)
