"""OWLv2: открытословарная детекция боксов по названиям классов.

Дефолтный движок режима «просто»: работает на CPU без ключей и аккаунтов,
классы задаёт пользователь произвольным английским текстом.

Замер на 79 фото / 323 объектах (LVIS val): F1 0.823, точность 0.83,
полнота 0.82, ~2 с на фото на CPU M2. Пороги оттуда же: 0.25 показывать
человеку, 0.40 принимать пачкой (точность 0.94).

ПОЧЕМУ СВОЯ ПОСТОБРАБОТКА, а не post_process_grounded_object_detection:
штатная функция отдаёт на каждую рамку только argmax по запросам, из-за чего
на длинном списке классов теряется часть находок — они переподписываются
соседним классом (kitchen sink -> sink, table lamp -> lamp). Замер: 37 рамок
из 301. Эмиссия по каждому классу-запросу отдельно эту потерю снимает
полностью и делает результат инвариантным к длине списка классов: добавление
класса в проект не сдвигает уже принятые человеком рамки.

Координаты: OWLv2 дополняет изображение до КВАДРАТА и масштабирует в 960x960,
поэтому pred_boxes нормированы относительно стороны квадрата, а не кадра.
Пересчёт идёт через side = max(width, height) с последующим клипом по границам.
"""

from __future__ import annotations

import os
import threading

from autolabelui_sdk import Annotation, BBox, Capability

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_IOU = 0.4
# рамки тоньше этого после клипа по кадру — мусор от краевых патчей
MIN_SIDE_PX = 2.0


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def nms_per_class(detections: list[dict], threshold: float) -> list[dict]:
    """Жадное подавление дублей ОТДЕЛЬНО по каждому классу.

    Разные классы не подавляют друг друга намеренно: ковёр под столом —
    это два объекта в одном месте, и человек должен увидеть оба.
    """
    by_class: dict[str, list[dict]] = {}
    for det in detections:
        by_class.setdefault(det["label"], []).append(det)

    kept: list[dict] = []
    for group in by_class.values():
        selected: list[dict] = []
        for det in sorted(group, key=lambda d: -d["score"]):
            if all(iou(det["bbox"], s["bbox"]) <= threshold for s in selected):
                selected.append(det)
        kept.extend(selected)
    return sorted(kept, key=lambda d: -d["score"])


def square_box_to_image(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """cxcywh, нормированные к дополненному квадрату -> xyxy в пикселях кадра."""
    side = max(width, height)
    cx, cy, bw, bh = box
    x1 = max(0.0, min((cx - bw / 2) * side, float(width)))
    y1 = max(0.0, min((cy - bh / 2) * side, float(height)))
    x2 = max(0.0, min((cx + bw / 2) * side, float(width)))
    y2 = max(0.0, min((cy + bh / 2) * side, float(height)))
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    return (x1, y1, x2, y2)


def to_annotations(detections: list[dict]) -> list[Annotation]:
    return [
        Annotation(
            geometry=BBox(
                x=det["bbox"][0],
                y=det["bbox"][1],
                width=det["bbox"][2] - det["bbox"][0],
                height=det["bbox"][3] - det["bbox"][1],
            ),
            label=det["label"],
            text=None,
            confidence=det["score"],
        )
        for det in detections
    ]


class Owlv2Labeler:
    name = "owlv2"
    version = "0.1.0"
    capabilities = {Capability.DETECTION}

    def __init__(self) -> None:
        self._models: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def _load(self, model_name: str):
        with self._lock:
            if model_name not in self._models:
                import torch
                from transformers import AutoProcessor, Owlv2ForObjectDetection

                torch.set_grad_enabled(False)
                cache = os.environ.get("OVD_MODEL_DIR") or None
                processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache)
                model = Owlv2ForObjectDetection.from_pretrained(
                    model_name, cache_dir=cache, dtype=torch.float32
                ).eval()
                self._models[model_name] = (processor, model)
            return self._models[model_name]

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        classes = [str(c).strip() for c in (config.get("classes") or []) if str(c).strip()]
        if not classes:
            raise ValueError(
                "owlv2: no classes given — add the project classes before labeling"
            )

        score_threshold = float(config.get("score_threshold", DEFAULT_SCORE_THRESHOLD))
        nms_iou = float(config.get("nms_iou", DEFAULT_NMS_IOU))
        model_name = str(config.get("model") or DEFAULT_MODEL)

        import io

        import torch
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(image)) as pil:
            pil = pil.convert("RGB")
            width, height = pil.size
            processor, model = self._load(model_name)
            inputs = processor(text=[classes], images=pil, return_tensors="pt")
            outputs = model(**inputs)

        # logits: [патчи, запросы] — вероятность КАЖДОГО класса в КАЖДОЙ рамке,
        # берём все пары выше порога, а не argmax по запросам
        probs = torch.sigmoid(outputs.logits[0])
        boxes = outputs.pred_boxes[0]

        detections: list[dict] = []
        for patch, query in (probs >= score_threshold).nonzero().tolist():
            box = square_box_to_image(tuple(boxes[patch].tolist()), width, height)
            if box is None:
                continue
            detections.append(
                {
                    "label": classes[query],
                    "bbox": box,
                    "score": float(probs[patch, query]),
                }
            )

        return to_annotations(nms_per_class(detections, nms_iou))
