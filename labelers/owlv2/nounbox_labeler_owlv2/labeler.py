"""OWLv2: open-vocabulary box detection driven by class names.

The default engine of the "easy" mode: runs on CPU with no keys and no
accounts, and the user gives the classes as free-form English text.

Measured on 79 photos / 323 objects (LVIS val): F1 0.823, precision 0.83,
recall 0.82, ~2 s per photo on an M2 CPU. The thresholds come from the same
run: 0.25 to show to a human, 0.40 to accept in bulk (precision 0.94).

WHY OUR OWN POST-PROCESSING rather than post_process_grounded_object_detection:
the stock function hands back only the argmax over the queries for each box, so
on a long class list part of the findings is lost — they get relabeled with a
neighboring class (kitchen sink -> sink, table lamp -> lamp). Measured: 37 of
301 boxes. Emitting each query class separately removes that loss entirely and
makes the result invariant to the length of the class list: adding a class to
the project does not shift boxes a human has already accepted.

Coordinates: OWLv2 pads the image to a SQUARE and rescales it to 960x960, so
pred_boxes are normalized against the square's side, not against the frame.
Converting back goes via side = max(width, height), then clipping to the edges.
"""

from __future__ import annotations

import os
import threading

from nounbox_sdk import Annotation, BBox, Capability

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_IOU = 0.4
# --- begin owlv2 post-processing (kept byte-identical in deploy/modal/owlv2_modal.py) ---
# boxes thinner than this after clipping to the frame are junk from edge patches
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
    """Greedy duplicate suppression, SEPARATELY for each class.

    Different classes deliberately do not suppress each other: a carpet under
    a table is two objects in one place, and the human must see both.
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
    """cxcywh normalized to the padded square -> xyxy in frame pixels."""
    side = max(width, height)
    cx, cy, bw, bh = box
    x1 = max(0.0, min((cx - bw / 2) * side, float(width)))
    y1 = max(0.0, min((cy - bh / 2) * side, float(height)))
    x2 = max(0.0, min((cx + bw / 2) * side, float(width)))
    y2 = max(0.0, min((cy + bh / 2) * side, float(height)))
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    return (x1, y1, x2, y2)
# --- end owlv2 post-processing ---


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
            # per call, not the one-time set_grad_enabled in _load: grad mode
            # is thread-local, and predict runs in a threadpool
            with torch.inference_mode():
                outputs = model(**inputs)

        # logits: [patches, queries] — probability of EVERY class in EVERY box;
        # take every pair above the threshold, not the argmax over the queries
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
