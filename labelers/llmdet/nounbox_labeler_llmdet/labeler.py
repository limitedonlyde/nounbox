"""LLMDet labeler: open-vocabulary box detection from English class names.

Second engine for task_type=detection: the "run it more accurately" mode, measurably
more accurate than the default OWLv2, and a second voice for consensus. The model
is iSEE-Laboratory/llmdet_base
(MM-Grounding-DINO fine-tuned with LLM supervision), Apache-2.0, CPU.

Platform benchmark (79 photos / 323 LVIS objects, manual ground truth, IoU 0.5,
CPU M2):

    engine               F1      P      R    s/photo   RSS
    llmdet_base       0.853  0.881  0.827      4.0    ~4.3 GB
    llmdet_tiny       0.831  0.866  0.799      2.8    ~3.8 GB
    owlv2 (default)   0.823                 2.0-2.4   ~1.5 GB

measured at the default score_threshold=0.35 and nms_iou=0.4 — hence those
defaults. With nms_iou=0.7 (the MM-GDINO default) the same runs give 0.848 /
0.822: duplicates on one object are not suppressed and go to a human reviewer.
The 0.35 threshold is the benchmark's compromise: at 0.30 F1 is slightly higher
(0.857), but precision drops from 0.881 to 0.851, i.e. every seventh box is false.

Config:
    classes: list[str]            — the project's English class names (required)
    score_threshold: float = 0.35 — boxes below the threshold are not returned
    nms_iou: float = 0.4          — suppresses duplicates within a single class
    model: str = "base"           — "base" | "tiny" | a full HF repo id; every
                                    model named here stays in the worker's
                                    memory until the process exits (base ~4.3 GB)
    max_detections: int = 100     — the cap on boxes per image
    max_pixels: int               — image area limit (env LLMDET_MAX_PIXELS)

Weights are cached in OVD_MODEL_DIR (default /data/ovd-models — the same docker
volume owlv2 uses; if the directory cannot be created, the plain huggingface
cache). base is 1.6 GB, tiny is 0.7 GB, and both are downloaded once.

THE 91-CLASS LIMIT. The model's classification head is fixed at max_text_len=256
tokens, and the prompt is every class in one string, "carpet. sofa. chandelier.".
A prompt over 256 tokens crashes the forward inside transformers with an opaque
"RuntimeError: The size of tensor a (256) must match the size of tensor b (319)".
So the limit is caught UP FRONT, before the model loads, from the class count,
and a second time on the actual tokenization (long multi-word names eat it faster).

Why not chunking. Splitting 100 classes into batches of 30 and merging the
results is technically possible, but the benchmark forbids it: with chunking 27%
of the boxes land on an object with a label from another batch — with no
competing classes in the prompt, the model confidently calls a sofa a carpet.
Better to send the user clearly to owlv2, which is invariant to the list length.

Why our own scores instead of post_process_grounded_object_detection. The stock
post-processor decodes a PHRASE per box and glues neighboring classes together
("cutting board toilet paper"), after which the box maps to no project class at
all and is simply lost. Here the class score is computed the way MM-GDINO does
it officially — the mean of the sigmoid of the logits over that class's tokens
(the positive map) — so every box gets its own score for EVERY requested class,
and boxes are emitted per class independently.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

from nounbox_sdk import Annotation, BBox, Capability

logger = logging.getLogger(__name__)

ENV_MODEL_DIR = "OVD_MODEL_DIR"
ENV_MAX_PIXELS = "LLMDET_MAX_PIXELS"

# weight cache shared with owlv2: one volume for every open-vocabulary detector
DEFAULT_MODEL_DIR = "/data/ovd-models"

MODEL_ALIASES = {
    "base": "iSEE-Laboratory/llmdet_base",
    "tiny": "iSEE-Laboratory/llmdet_tiny",
}
DEFAULT_MODEL = MODEL_ALIASES["base"]

# how many classes fit in the prompt: the head's 256 tokens / ~2.8 tokens per class
MAX_CLASSES = 91
# the model's config.max_text_len; read from the loaded model, this is the default
DEFAULT_MAX_TEXT_TOKENS = 256

DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_NMS_IOU = 0.4
DEFAULT_MAX_DETECTIONS = 100

# ingest normalizes pages to 4096 on the long side (~16.8 MP) — a generous limit
DEFAULT_MAX_PIXELS = 40_000_000

# a box thinner than a pixel is useless for training and is not drawn in review
MIN_BOX_SIDE = 1.0


@dataclass(frozen=True)
class Detection:
    """A raw box from the engine: class, score, xyxy in absolute pixels."""

    label: str
    score: float
    box: tuple[float, float, float, float]


# --- configuration ---


def resolve_model(config: dict) -> str:
    """Model name from config: the base/tiny alias or a full huggingface repo id."""
    raw = str(config.get("model") or "").strip() or DEFAULT_MODEL
    alias = MODEL_ALIASES.get(raw.lower())
    if alias:
        return alias
    if "/" not in raw:
        known = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(
            f"llmdet: unknown model {raw!r}; use one of {known} or a full HF repo id"
        )
    return raw


def resolve_classes(config: dict) -> list[str]:
    """Project classes from config: as-is (case preserved), no blanks or dupes.

    The ORIGINAL names are returned — they go into Annotation.label and have to
    match project_classes.name; they reach the prompt lowercased (see
    build_prompt), because that is what the model's tokenizer wants.
    """
    raw = config.get("classes")
    if raw is None:
        raise ValueError(
            "llmdet: config['classes'] is required — add the project classes "
            "before starting a labeling run"
        )
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        raise ValueError(
            "llmdet: config['classes'] must be a list of class names, got "
            f"{type(raw).__name__}"
        )

    names: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = " ".join(str(item).split())
        if not name:
            continue
        if "." in name:
            raise ValueError(
                f"llmdet: class name {name!r} contains '.', which separates classes "
                "in the model prompt — rename the class"
            )
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)

    if not names:
        raise ValueError(
            "llmdet: config['classes'] is empty — add the project classes "
            "before starting a labeling run"
        )
    check_class_limit(names)
    return names


def check_class_limit(classes: Sequence[str]) -> None:
    """Prompt limit, BEFORE the model loads: otherwise an opaque RuntimeError."""
    if len(classes) > MAX_CLASSES:
        raise ValueError(
            f"llmdet: LLMDet takes at most {MAX_CLASSES} classes per request, "
            f"and this project has {len(classes)} — use owlv2 instead"
        )


def check_prompt_tokens(n_tokens: int, max_text_len: int, n_classes: int) -> None:
    """Second gate: long multi-word names hit the limit before 91 classes do."""
    if n_tokens > max_text_len:
        raise ValueError(
            f"llmdet: a prompt with {n_classes} classes takes {n_tokens} tokens, "
            f"and the model limit is {max_text_len} — shorten the class names "
            "or use owlv2 instead"
        )


def option(config: dict, key: str, default: float) -> float:
    """Number from config; a null (from the job's JSON payload) means the default."""
    value = config.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"llmdet: config[{key!r}] must be a number, got {value!r}"
        ) from exc


def model_dir() -> str | None:
    """Weight cache dir: env -> default (docker volume) -> None (stock HF cache)."""
    path = Path(os.environ.get(ENV_MODEL_DIR) or DEFAULT_MODEL_DIR)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "llmdet: model cache %s is unusable (%s), falling back to the default "
            "huggingface cache",
            path,
            exc,
        )
        return None
    return str(path)


def max_pixels(config: dict) -> int:
    """Image area limit: config -> env -> default."""
    raw = (
        config.get("max_pixels") or os.environ.get(ENV_MAX_PIXELS) or DEFAULT_MAX_PIXELS
    )
    value = int(raw)
    if value <= 0:
        raise ValueError(f"llmdet: max_pixels must be positive, got {value}")
    return value


def load_image(image: bytes, limit: int) -> PILImage.Image:
    """Bytes -> an RGB image.

    Corrupt data and decompression bombs turn into a clear error rather than a
    crash somewhere inside the processor.
    """
    if not image:
        raise ValueError("llmdet: empty image payload")
    try:
        pil = PILImage.open(io.BytesIO(image))
        width, height = pil.size
    except PILImage.DecompressionBombError as exc:
        raise ValueError(
            f"llmdet: image rejected as a decompression bomb: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"llmdet: cannot decode image ({len(image)} bytes, "
            f"starts with {image[:8]!r}): {type(exc).__name__}: {exc}"
        ) from exc
    if width * height > limit:
        raise ValueError(
            f"llmdet: image {width}x{height} ({width * height} px) exceeds the "
            f"limit of {limit} px (config.max_pixels / {ENV_MAX_PIXELS})"
        )
    try:
        return pil.convert("RGB")
    except Exception as exc:
        raise ValueError(
            f"llmdet: cannot decode image ({len(image)} bytes): "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# --- the prompt and parsing the model output ---


def build_prompt(classes: Sequence[str]) -> str:
    """Classes -> the prompt string "carpet. sofa. chandelier.".

    Exactly the format the processor builds from a label list itself: lowercase,
    ". " as the separator, a trailing dot. We build it ourselves because that
    same dot is what the class token spans are later recovered from.
    """
    labels = [" ".join(str(name).strip().lower().split()) for name in classes]
    return ". ".join(labels) + "."


def class_spans(
    input_ids: Sequence[int],
    dot_id: int,
    special_ids: Iterable[int],
    n_classes: int,
) -> list[list[int]]:
    """Token indices for each class — the prompt is split on the dots.

    A multi-word class ("cutting board") and a class broken into wordpieces
    ("chan ##del ##ier") give several indices: the class score is their mean.
    """
    skip = set(special_ids)
    spans: list[list[int]] = []
    current: list[int] = []
    for index, token in enumerate(input_ids):
        if token in skip:
            continue
        if token == dot_id:
            if current:
                spans.append(current)
            current = []
        else:
            current.append(index)
    if current:
        spans.append(current)
    return spans[:n_classes]


def scale_box(
    cx: float,
    cy: float,
    w: float,
    h: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """cxcywh as image fractions -> xyxy in pixels, clipped to the frame."""
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def nms_per_class(detections: Sequence[Detection], nms_iou: float) -> list[Detection]:
    """Greedy NMS WITHIN a class: two objects of different classes may overlap.

    A "wine bottle" box on top of a "bottle" box is legitimate labeling and must
    not be suppressed; two identical labels on one object, on the other hand,
    are a duplicate a human is forced to delete by hand.
    """
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.score):
        if all(
            other.label != detection.label or iou(other.box, detection.box) <= nms_iou
            for other in kept
        ):
            kept.append(detection)
    return kept


def to_annotations(
    detections: Sequence[Detection],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[Annotation]:
    """Raw boxes -> SDK annotations: threshold, per-class NMS, cap, BBox."""
    above = [d for d in detections if d.score >= score_threshold]
    degenerate = 0
    survivors: list[Detection] = []
    for detection in nms_per_class(above, nms_iou):
        x1, y1, x2, y2 = detection.box
        if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
            degenerate += 1
            continue
        survivors.append(detection)
    if degenerate:
        logger.debug("llmdet: skipped %d degenerate box(es)", degenerate)

    survivors.sort(key=lambda d: -d.score)
    if len(survivors) > max_detections:
        logger.info(
            "llmdet: %d box(es) above threshold, keeping top %d",
            len(survivors),
            max_detections,
        )
        survivors = survivors[:max_detections]

    return [
        Annotation(
            geometry=BBox(
                x=d.box[0],
                y=d.box[1],
                width=d.box[2] - d.box[0],
                height=d.box[3] - d.box[1],
            ),
            label=d.label,
            text=None,
            confidence=d.score,
        )
        for d in survivors
    ]


# --- the loaded model ---


class Engine:
    """Processor + model for one repo id; lazy, kept until the process exits."""

    def __init__(self, model_id: str) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        cache_dir = model_dir()
        logger.info("Loading LLMDet %s (cache_dir=%s)...", model_id, cache_dir)
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model.eval()

        tokenizer = self.processor.tokenizer
        self.dot_id = tokenizer.convert_tokens_to_ids(".")
        self.special_ids = {
            token_id
            for token_id in (
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )
            if token_id is not None
        }
        self.max_text_len = int(
            getattr(self.model.config, "max_text_len", DEFAULT_MAX_TEXT_TOKENS)
        )

    def detect(
        self,
        image: PILImage.Image,
        classes: Sequence[str],
        score_threshold: float,
    ) -> list[Detection]:
        """One forward -> boxes for EVERY class (positive map, see module docstring)."""
        import torch

        prompt = build_prompt(classes)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()
        check_prompt_tokens(len(input_ids), self.max_text_len, len(classes))

        spans = class_spans(input_ids, self.dot_id, self.special_ids, len(classes))
        if len(spans) != len(classes):
            raise ValueError(
                f"llmdet: prompt tokenized into {len(spans)} class span(s) for "
                f"{len(classes)} class(es) — cannot map boxes to classes"
            )

        with torch.no_grad():
            outputs = self.model(**inputs)
        # logits: (queries, max_text_len) — box score for every prompt token
        scores_per_token = outputs.logits[0].sigmoid()
        boxes = outputs.pred_boxes[0]  # (queries, 4), cxcywh as frame fractions
        width, height = image.size

        found: list[Detection] = []
        for name, span in zip(classes, spans):
            index = torch.tensor(span, dtype=torch.long)
            scores = scores_per_token[:, index].mean(-1)
            keep = scores >= score_threshold
            if not bool(keep.any()):
                continue
            for score, box in zip(scores[keep].tolist(), boxes[keep].tolist()):
                found.append(
                    Detection(
                        label=name,
                        score=float(score),
                        box=scale_box(*box, width, height),
                    )
                )
        return found


class LLMDetLabeler:
    name = "llmdet"
    version = "0.1.0"
    capabilities = {Capability.DETECTION}

    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {}  # model id -> Engine
        self._lock = threading.Lock()

    # --- engine initialization (lazy, once per model) ---

    def _engine(self, model_id: str) -> Engine:
        with self._lock:
            if model_id not in self._engines:
                self._engines[model_id] = self._create_engine(model_id)
            return self._engines[model_id]

    @staticmethod
    def _create_engine(model_id: str) -> Engine:
        return Engine(model_id)

    # --- the SDK contract ---

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        # config is validated first: an error in the classes must not cost
        # a 1.6 GB download of the weights
        classes = resolve_classes(config)
        model_id = resolve_model(config)
        score_threshold = option(config, "score_threshold", DEFAULT_SCORE_THRESHOLD)
        nms_iou = option(config, "nms_iou", DEFAULT_NMS_IOU)
        max_detections = int(option(config, "max_detections", DEFAULT_MAX_DETECTIONS))
        if max_detections <= 0:
            raise ValueError(
                f"llmdet: max_detections must be positive, got {max_detections}"
            )

        pil = load_image(image, max_pixels(config))
        detections = self._engine(model_id).detect(pil, classes, score_threshold)
        annotations = to_annotations(
            detections, score_threshold, nms_iou, max_detections
        )
        logger.debug(
            "llmdet: %dx%d, %d class(es) -> %d box(es)",
            pil.width,
            pil.height,
            len(classes),
            len(annotations),
        )
        return annotations
