"""Dataset export into formats for model training.

ONLY human-reviewed annotations (accepted/edited) are exported — pending and
rejected never reach the dataset. Every export ships a manifest.json (a
snapshot version of the dataset: task_type, classes, counts).

Reviewed frames without a single box go into the dataset as well: for a
detector those are negative (background) examples, they cut down false
positives, and YOLO handles them natively (an empty .txt next to the image),
COCO — as an entry in images with no annotations. A frame counts as reviewed
when it has at least one annotation in status accepted/edited/rejected, or
when Image.reviewed is set; unreviewed frames do not get into the dataset at
all — otherwise an unlabeled object would go into training as background.

The set of formats depends on Project.task_type:

detection
- yolo_detect: images/{train,val}/ + labels/{train,val}/*.txt
  (`class_idx cx cy w h`, normalized to 0..1) + data.yaml
- coco: images/ + annotations.json (instances: categories from project classes)

ocr
- paddleocr_det: images/ + label.txt (`path\\t[{"transcription":..., "points":...}]`)
- paddleocr_rec: crops/ + label.txt (`path\\ttext`) — crops of text lines
- coco: images/ + annotations.json (detection, text goes into attributes)
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from PIL import Image as PILImage

from app.models import Annotation, AnnotationStatus, Image

EXPORTABLE_STATUSES = (AnnotationStatus.ACCEPTED, AnnotationStatus.EDITED)
# statuses that prove a human looked at the frame (rejected is a human decision
# too: there was a box and a human removed it)
REVIEWED_STATUSES = (
    AnnotationStatus.ACCEPTED,
    AnnotationStatus.EDITED,
    AnnotationStatus.REJECTED,
)

TaskType = Literal["detection", "ocr"]
ExportFormat = Literal["yolo_detect", "coco", "paddleocr_det", "paddleocr_rec"]

DEFAULT_TASK_TYPE: TaskType = "detection"
FORMATS_BY_TASK: dict[str, tuple[str, ...]] = {
    "detection": ("yolo_detect", "coco"),
    "ocr": ("paddleocr_det", "paddleocr_rec", "coco"),
}
FORMATS: tuple[str, ...] = ("yolo_detect", "coco", "paddleocr_det", "paddleocr_rec")

# val share of the yolo split; the split is deterministic (see _split_of)
VAL_FRACTION = 0.2


class ExportError(ValueError):
    pass


def formats_for(task_type: str | None) -> tuple[str, ...]:
    """Formats available to a project. Unknown/empty task_type acts as detection."""
    return FORMATS_BY_TASK.get(
        task_type or DEFAULT_TASK_TYPE, FORMATS_BY_TASK[DEFAULT_TASK_TYPE]
    )


class ExportItem:
    """An image + its bytes (PNG after ingest) + its reviewed annotations.

    An empty annotation list is legal: a reviewed frame with no objects is a
    background example of the dataset.
    """

    def __init__(self, image: Image, data: bytes, annotations: list[Annotation]):
        self.image = image
        self.data = data
        self.annotations = annotations


@dataclass
class ExportContext:
    """Everything a builder needs to know about the project besides the images."""

    project_id: str
    task_type: str = DEFAULT_TASK_TYPE
    # project class names in sort_order: the index here is the class_idx in YOLO
    classes: tuple[str, ...] = ()
    # annotation labels absent from the project classes (class deleted/renamed)
    skipped_labels: dict[str, int] = field(default_factory=dict)

    def skip(self, label: str) -> None:
        self.skipped_labels[label] = self.skipped_labels.get(label, 0) + 1


def _points(geometry: dict[str, Any]) -> list[list[float]]:
    """A polygon of 4 points (a bbox is expanded into one)."""
    if geometry["type"] == "polygon":
        return [[float(x), float(y)] for x, y in geometry["points"]]
    x, y = geometry["x"], geometry["y"]
    w, h = geometry["width"], geometry["height"]
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    """(x, y, w, h). A polygon is reduced to its bounding box."""
    if geometry["type"] == "bbox":
        return (
            float(geometry["x"]),
            float(geometry["y"]),
            float(geometry["width"]),
            float(geometry["height"]),
        )
    arr = np.asarray(geometry["points"], dtype=np.float64)
    (min_x, min_y), (max_x, max_y) = arr.min(axis=0), arr.max(axis=0)
    return float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)


def _image_name(index: int) -> str:
    return f"images/{index:05d}.png"


def _flatten_ws(text: str) -> str:
    """Tabs/newlines break the tab-separated label.txt — we swap them for a space."""
    return text.replace("\r\n", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _write_common(zf: zipfile.ZipFile, items: list[ExportItem]) -> None:
    for index, item in enumerate(items, start=1):
        zf.writestr(_image_name(index), item.data)


# --- ocr formats ---
def _build_det(
    zf: zipfile.ZipFile, items: list[ExportItem], ctx: ExportContext
) -> dict[str, Any]:
    # background frames help a detector but not OCR: the PaddleOCR loader
    # throws away entries with zero boxes, they are only junk in the dataset
    items = [i for i in items if i.annotations]
    _write_common(zf, items)
    lines = []
    count = 0
    for index, item in enumerate(items, start=1):
        entries = [
            {"transcription": a.text or "", "points": _points(a.geometry)}
            for a in item.annotations
        ]
        count += len(entries)
        lines.append(f"{_image_name(index)}\t{json.dumps(entries, ensure_ascii=False)}")
    zf.writestr("label.txt", "\n".join(lines))
    return {"annotations": count, "images_written": len(items)}


def _build_rec(
    zf: zipfile.ZipFile, items: list[ExportItem], ctx: ExportContext
) -> dict[str, Any]:
    items = [i for i in items if i.annotations]
    _write_common(zf, items)
    lines = []
    count = 0
    for index, item in enumerate(items, start=1):
        pil = PILImage.open(io.BytesIO(item.data))
        for ann_index, a in enumerate(item.annotations):
            if not a.text:
                continue  # the rec format needs text
            x, y, w, h = _bbox(a.geometry)
            crop = pil.crop((int(x), int(y), int(x + w), int(y + h)))
            if crop.width < 2 or crop.height < 2:
                continue
            name = f"crops/{index:05d}_{ann_index:03d}.png"
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            zf.writestr(name, buffer.getvalue())
            lines.append(f"{name}\t{_flatten_ws(a.text)}")
            count += 1
    zf.writestr("label.txt", "\n".join(lines))
    return {"annotations": count, "images_written": len(items)}


# --- yolo (detection) ---
def _yolo_stem(image: Image) -> str:
    """Stable file name: the image id, not its index in the selection.

    The split is derived from the name, so it must survive adding or removing
    neighboring images — otherwise a repeat export would reshuffle train/val.
    """
    return image.id.hex if isinstance(image.id, uuid.UUID) else str(image.id)


def _bucket_of(stem: str) -> int:
    """Bucket 0..99 derived from the file name.

    blake2b and not the built-in hash(): the latter is salted with
    PYTHONHASHSEED and is not reproducible across processes.
    """
    digest = hashlib.blake2b(stem.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def _split_of(stem: str) -> str:
    """Deterministic train/val: one name always lands in the same part."""
    return "val" if _bucket_of(stem) < round(VAL_FRACTION * 100) else "train"


def _assign_splits(stems: list[str]) -> dict[str, tuple[str, ...]]:
    """Distribution of the images over train/val.

    Both parts must be non-empty — ultralytics crashes on a missing folder. On
    a small selection the hash may yield no val at all; then the image with the
    boundary bucket is moved there deterministically, and a lone image goes
    into both train and val.
    """
    if len(stems) == 1:
        return {stems[0]: ("train", "val")}
    splits = {stem: (_split_of(stem),) for stem in stems}
    parts = {split for (split,) in splits.values()}
    if parts == {"train"}:
        splits[min(stems, key=_bucket_of)] = ("val",)
    elif parts == {"val"}:
        splits[max(stems, key=_bucket_of)] = ("train",)
    return splits


def _clipped_bbox(
    geometry: dict[str, Any], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """Box clipped to the frame: (x1, y1, x2, y2). None means degenerate.

    Shared by YOLO and COCO: otherwise one and the same project would give two
    different datasets — the engines can return boxes partly off the edge.
    """
    if width <= 0 or height <= 0:
        return None
    x, y, w, h = _bbox(geometry)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _yolo_line(
    class_index: int, geometry: dict[str, Any], width: int, height: int
) -> str | None:
    """`class_idx cx cy w h`, normalized to 0..1. None if the box is degenerate."""
    clipped = _clipped_bbox(geometry, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    return (
        f"{class_index} {cx:.6f} {cy:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def _data_yaml(classes: Sequence[str]) -> str:
    """data.yaml for ultralytics. json.dumps yields a correctly escaped quoted
    YAML scalar, so pyyaml is not needed among the dependencies."""
    lines = [
        "# Nounbox export",
        # the path key is deliberately omitted: without it ultralytics takes the
        # data.yaml folder as the dataset root, while `path: .` resolves to the
        # current working directory (data/utils.py: path.exists() → path as is)
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
        *(
            f"  {i}: {json.dumps(name, ensure_ascii=False)}"
            for i, name in enumerate(classes)
        ),
    ]
    return "\n".join(lines) + "\n"


def _build_yolo(
    zf: zipfile.ZipFile, items: list[ExportItem], ctx: ExportContext
) -> dict[str, Any]:
    class_index = {name: i for i, name in enumerate(ctx.classes)}
    stems = [_yolo_stem(item.image) for item in items]
    assigned = _assign_splits(stems)
    images = {"train": 0, "val": 0}
    count = 0
    for stem, item in zip(stems, items):
        lines = []
        for a in item.annotations:
            index = class_index.get(a.label)
            if index is None:
                ctx.skip(a.label)
                continue
            line = _yolo_line(index, a.geometry, item.image.width, item.image.height)
            if line is not None:
                lines.append(line)
        for split in assigned[stem]:
            # an empty .txt is a legal negative example, the image goes in anyway
            zf.writestr(f"images/{split}/{stem}.png", item.data)
            zf.writestr(f"labels/{split}/{stem}.txt", "\n".join(lines))
            images[split] += 1
        count += len(lines)
    zf.writestr("data.yaml", _data_yaml(ctx.classes))
    return {
        "annotations": count,
        "images_written": images["train"] + images["val"],
        "train_images": images["train"],
        "val_images": images["val"],
    }


# --- coco ---
def _build_coco(
    zf: zipfile.ZipFile, items: list[ExportItem], ctx: ExportContext
) -> dict[str, Any]:
    _write_common(zf, items)
    detection = ctx.task_type == "detection"
    if detection:
        # categories are project classes in sort_order, ids 1-based (COCO convention)
        names: Sequence[str] = ctx.classes
    else:
        names = sorted({a.label for item in items for a in item.annotations})
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]
    cat_id = {c["name"]: c["id"] for c in categories}

    coco: dict[str, Any] = {
        "info": {"description": "Nounbox export", "version": "0.1.0"},
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    ann_id = 0
    for index, item in enumerate(items, start=1):
        coco["images"].append(
            {
                "id": index,
                "file_name": _image_name(index),
                "width": item.image.width,
                "height": item.image.height,
            }
        )
        for a in item.annotations:
            category_id = cat_id.get(a.label)
            if category_id is None:
                ctx.skip(a.label)
                continue
            if detection:
                # the same clipping as in YOLO: exports of one project must
                # contain an identical set of boxes
                clipped = _clipped_bbox(a.geometry, item.image.width, item.image.height)
                if clipped is None:
                    continue
                x1, y1, x2, y2 = clipped
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
            else:
                x, y, w, h = _bbox(a.geometry)
            ann_id += 1
            entry: dict[str, Any] = {
                "id": ann_id,
                "image_id": index,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
            if detection:
                # boxes only (the owner's decision) — no segmentation in the dataset
                entry["segmentation"] = []
                entry["attributes"] = {
                    "confidence": a.confidence,
                    "source": (a.source or {}).get("name"),
                }
            else:
                entry["segmentation"] = [[v for p in _points(a.geometry) for v in p]]
                entry["attributes"] = {
                    "text": a.text,
                    "confidence": a.confidence,
                    "source": (a.source or {}).get("name"),
                }
            coco["annotations"].append(entry)
    zf.writestr("annotations.json", json.dumps(coco, ensure_ascii=False, indent=1))
    return {"annotations": ann_id, "images_written": len(items)}


_BUILDERS = {
    "yolo_detect": _build_yolo,
    "coco": _build_coco,
    "paddleocr_det": _build_det,
    "paddleocr_rec": _build_rec,
}


def build_zip(
    fmt: str,
    items: list[ExportItem],
    project_id: str,
    task_type: str = DEFAULT_TASK_TYPE,
    classes: Sequence[str] = (),
) -> bytes:
    task_type = task_type or DEFAULT_TASK_TYPE
    available = formats_for(task_type)
    if fmt not in _BUILDERS:
        raise ExportError(f"Unknown format: {fmt}. Available: {', '.join(available)}")
    if fmt not in available:
        raise ExportError(
            f"Format {fmt} is not available for task_type={task_type}. "
            f"Available: {', '.join(available)}"
        )
    if not items:
        raise ExportError(
            "No reviewed images to export: review images before exporting"
        )
    # in detection both class_idx (YOLO) and category_id (COCO) are a position in
    # the project class list, without it there is nothing to export
    if task_type == "detection" and not classes:
        raise ExportError("Project has no classes: add project classes before export")

    ctx = ExportContext(
        project_id=project_id, task_type=task_type, classes=tuple(classes)
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        stats = _BUILDERS[fmt](zf, items, ctx)
        # An empty ZIP must not be served, but "zero boxes" is not the same as
        # "nothing to serve": a dataset of nothing but reviewed background
        # frames is meaningful for a detector. We refuse only when not a single
        # frame made it into the archive.
        if stats.get("images_written", len(items)) == 0:
            raise ExportError(
                f"Nothing to export in format {fmt}: review some images first"
            )
        manifest = {
            "generator": "nounbox",
            "generator_version": "0.1.0",
            "format": fmt,
            "task_type": task_type,
            "project_id": project_id,
            "classes": list(ctx.classes),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "images": len(items),
            # reviewed frames without objects are the negative examples
            "background_images": sum(1 for item in items if not item.annotations),
            "statuses_included": list(EXPORTABLE_STATUSES),
            "skipped_labels": ctx.skipped_labels,
            **stats,
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    return buffer.getvalue()
