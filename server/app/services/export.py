"""Экспорт датасета в форматы для обучения моделей.

Экспортируются ТОЛЬКО проверенные человеком аннотации (accepted/edited) —
pending/rejected в датасет не попадают. Каждый экспорт содержит manifest.json
(snapshot-версия датасета: task_type, классы, счётчики).

В датасет идут и проверенные кадры без единой рамки: для детектора это
негативные (фоновые) примеры, они снижают ложные срабатывания, и YOLO их ест
штатно (пустой .txt рядом с картинкой), COCO — записью в images без
аннотаций. Проверенным считается кадр, у которого есть хотя бы одна
аннотация в статусе accepted/edited/rejected либо выставлен Image.reviewed;
непросмотренные кадры не попадают в датасет вообще — иначе неразмеченный
объект уехал бы в обучение как фон.

Набор форматов зависит от Project.task_type:

detection
- yolo_detect: images/{train,val}/ + labels/{train,val}/*.txt
  (`class_idx cx cy w h`, нормировано 0..1) + data.yaml
- coco: images/ + annotations.json (instances: categories из классов проекта)

ocr
- paddleocr_det: images/ + label.txt (`path\\t[{"transcription":..., "points":...}]`)
- paddleocr_rec: crops/ + label.txt (`path\\ttext`) — кропы текстовых строк
- coco: images/ + annotations.json (детекция, текст — в attributes)
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
# статусы, доказывающие, что кадр смотрел человек (rejected — тоже решение
# человека: рамка была, её убрали)
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

# доля val в yolo-сплите; сплит детерминирован (см. _split_of)
VAL_FRACTION = 0.2


class ExportError(ValueError):
    pass


def formats_for(task_type: str | None) -> tuple[str, ...]:
    """Форматы, доступные проекту. Неизвестный/пустой task_type — как detection."""
    return FORMATS_BY_TASK.get(
        task_type or DEFAULT_TASK_TYPE, FORMATS_BY_TASK[DEFAULT_TASK_TYPE]
    )


class ExportItem:
    """Изображение + его байты (PNG после ingest) + проверенные аннотации.

    Пустой список аннотаций легален: проверенный кадр без объектов — фоновый
    пример датасета.
    """

    def __init__(self, image: Image, data: bytes, annotations: list[Annotation]):
        self.image = image
        self.data = data
        self.annotations = annotations


@dataclass
class ExportContext:
    """Всё, что билдеру нужно знать о проекте помимо самих картинок."""

    project_id: str
    task_type: str = DEFAULT_TASK_TYPE
    # имена классов проекта в порядке sort_order: индекс здесь = class_idx в YOLO
    classes: tuple[str, ...] = ()
    # метки аннотаций, которых нет среди классов проекта (класс удалили/переименовали)
    skipped_labels: dict[str, int] = field(default_factory=dict)

    def skip(self, label: str) -> None:
        self.skipped_labels[label] = self.skipped_labels.get(label, 0) + 1


def _points(geometry: dict[str, Any]) -> list[list[float]]:
    """Полигон из 4 точек (bbox разворачивается)."""
    if geometry["type"] == "polygon":
        return [[float(x), float(y)] for x, y in geometry["points"]]
    x, y = geometry["x"], geometry["y"]
    w, h = geometry["width"], geometry["height"]
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def _bbox(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    """(x, y, w, h). Полигон сводится к ограничивающему прямоугольнику."""
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
    """Табуляции/переводы строк ломают tab-separated label.txt — заменяем пробелом."""
    return text.replace("\r\n", " ").replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _write_common(zf: zipfile.ZipFile, items: list[ExportItem]) -> None:
    for index, item in enumerate(items, start=1):
        zf.writestr(_image_name(index), item.data)


# --- ocr-форматы ---
def _build_det(
    zf: zipfile.ZipFile, items: list[ExportItem], ctx: ExportContext
) -> dict[str, Any]:
    # фоновые кадры полезны детектору, но не OCR: загрузчик PaddleOCR
    # выбрасывает записи с нулём боксов, в датасете от них только мусор
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
                continue  # rec-формату нужен текст
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
    """Стабильное имя файла: id изображения, а не порядковый номер в выборке.

    От имени считается сплит, поэтому имя обязано пережить добавление/удаление
    соседних снимков — иначе повторный экспорт перетасовал бы train/val.
    """
    return image.id.hex if isinstance(image.id, uuid.UUID) else str(image.id)


def _bucket_of(stem: str) -> int:
    """Корзина 0..99 по имени файла.

    blake2b, а не встроенный hash(): тот солится PYTHONHASHSEED и между
    процессами не воспроизводится.
    """
    digest = hashlib.blake2b(stem.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100


def _split_of(stem: str) -> str:
    """Детерминированный train/val: одно имя — всегда одна и та же часть."""
    return "val" if _bucket_of(stem) < round(VAL_FRACTION * 100) else "train"


def _assign_splits(stems: list[str]) -> dict[str, tuple[str, ...]]:
    """Раскладка снимков по train/val.

    Обе части обязаны быть непустыми — ultralytics падает на отсутствующей
    папке. На маленькой выборке хеш может не дать ни одного val, тогда туда
    детерминированно уезжает снимок с граничной корзиной; единственный снимок
    попадает и в train, и в val.
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
    """Бокс, обрезанный по кадру: (x1, y1, x2, y2). None — вырожденный.

    Общая для YOLO и COCO: иначе один и тот же проект давал бы два разных
    датасета — движки умеют возвращать рамки, частично уехавшие за край.
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
    """`class_idx cx cy w h`, нормировано 0..1. None — бокс вырожденный."""
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
    """data.yaml для ultralytics. json.dumps даёт корректно экранированный
    YAML-скаляр в кавычках, так что pyyaml в зависимостях не нужен."""
    lines = [
        "# AutoLabelUi export",
        # ключ path намеренно не пишем: без него ultralytics берёт за корень
        # датасета папку самого data.yaml, а `path: .` он разрешает в текущую
        # рабочую директорию (data/utils.py: path.exists() → путь берётся как есть)
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
            # пустой .txt — легальный негативный пример, картинку всё равно кладём
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
        # категории — классы проекта в порядке sort_order, id 1-based (конвенция COCO)
        names: Sequence[str] = ctx.classes
    else:
        names = sorted({a.label for item in items for a in item.annotations})
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(names)]
    cat_id = {c["name"]: c["id"] for c in categories}

    coco: dict[str, Any] = {
        "info": {"description": "AutoLabelUi export", "version": "0.1.0"},
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
                # тот же клип, что в YOLO: экспорты одного проекта обязаны
                # содержать одинаковый набор рамок
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
                # только боксы (решение владельца) — сегментации в датасете нет
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
    # в detection и class_idx (YOLO), и category_id (COCO) — это позиция в списке
    # классов проекта, без него экспортировать нечего
    if task_type == "detection" and not classes:
        raise ExportError("Project has no classes: add project classes before export")

    ctx = ExportContext(
        project_id=project_id, task_type=task_type, classes=tuple(classes)
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        stats = _BUILDERS[fmt](zf, items, ctx)
        # Пустой ZIP отдавать нельзя, но «ноль рамок» — не то же самое, что
        # «нечего отдавать»: датасет из одних проверенных фоновых кадров
        # осмысленен для детектора. Отказываем, только если в архив вообще
        # не попало ни одного кадра.
        if stats.get("images_written", len(items)) == 0:
            raise ExportError(
                f"Nothing to export in format {fmt}: review some images first"
            )
        manifest = {
            "generator": "autolabelui",
            "generator_version": "0.1.0",
            "format": fmt,
            "task_type": task_type,
            "project_id": project_id,
            "classes": list(ctx.classes),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "images": len(items),
            # проверенные кадры без объектов — негативные примеры
            "background_images": sum(1 for item in items if not item.annotations),
            "statuses_included": list(EXPORTABLE_STATUSES),
            "skipped_labels": ctx.skipped_labels,
            **stats,
        }
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    return buffer.getvalue()
