"""Nounbox SDK — контракт labeler-плагинов.

Плагин — это класс, реализующий протокол Labeler. Публикация через entry points:

    [project.entry-points."nounbox.labelers"]
    paddleocr = "my_plugin:PaddleOCRLabeler"
"""

from nounbox_sdk.labeler import Labeler, load_labelers
from nounbox_sdk.types import Annotation, BBox, Capability, Polygon

__all__ = [
    "Annotation",
    "BBox",
    "Capability",
    "Labeler",
    "Polygon",
    "load_labelers",
]
