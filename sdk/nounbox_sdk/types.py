"""Contract data types. No dependencies — stdlib only."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias


class Capability(str, Enum):
    """What a labeler can do."""

    DETECTION = "detection"  # boxes/polygons around text
    RECOGNITION = "recognition"  # transcription inside the geometry
    LAYOUT = "layout"  # document structure (headings, tables...)
    KIE = "kie"  # key fields (attrs)


@dataclass
class BBox:
    """Rectangle in absolute pixels (origin is the top-left corner)."""

    x: float
    y: float
    width: float
    height: float


# A polygon is a list of [x, y] points in absolute pixels, clockwise.
Polygon: TypeAlias = list[tuple[float, float]]


@dataclass
class Annotation:
    """Universal annotation — covers detection/recognition/layout/KIE.

    - geometry: BBox | Polygon — where the object is
    - label: type ("text_line", "paragraph", "table", "title", ...)
    - text: transcription (recognition), if any
    - attrs: arbitrary attributes, for KIE — {"field": "total"}
    - confidence: engine confidence 0..1 (for bulk-accept and review queues)
    """

    geometry: BBox | Polygon
    label: str = "text_line"
    text: str | None = None
    confidence: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)
