"""AutoLabelUi SDK — контракт labeler-плагинов.

Плагин — это класс, реализующий протокол Labeler. Публикация через entry points:

    [project.entry-points."autolabelui.labelers"]
    paddleocr = "my_plugin:PaddleOCRLabeler"
"""

from autolabelui_sdk.labeler import Labeler, load_labelers
from autolabelui_sdk.types import Annotation, BBox, Capability, Polygon

__all__ = [
    "Annotation",
    "BBox",
    "Capability",
    "Labeler",
    "Polygon",
    "load_labelers",
]
