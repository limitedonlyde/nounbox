"""Nounbox SDK — the labeler plugin contract.

A plugin is a class implementing the Labeler protocol. Published via entry points:

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
