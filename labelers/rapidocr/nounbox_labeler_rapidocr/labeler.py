"""RapidOCR labeler: per-line detection + recognition on CPU (onnxruntime).

The platform's default engine — works right after `docker compose up`, with no
GPU, no accounts and no keys. Returns per-line polygons (4 points
TL->TR->BR->BL; on a skewed scan, genuinely slanted quadrilaterals) and a REAL
confidence: the mean softmax probability of the characters from the CTC decoder
(not a constant).

Config:
    lang: str = "ru"            — language code -> PP-OCRv5 rec model (see LANG_ALIASES)
    min_confidence: float = 0.0 — drop lines below the threshold; 0.0 gives the human
                                  everything, including what the engine could not read
    box_thresh: float | None    — DB detection threshold (None = engine default)
    unclip_ratio: float | None  — line contour expansion (None = engine default)
    max_pixels: int             — image area limit (default from RAPIDOCR_MAX_PIXELS)

Weights are cached in RAPIDOCR_MODEL_DIR (default /data/rapidocr-models, on a
docker volume; if that directory cannot be created — ~/.cache/rapidocr-models).
For Russian that is three onnx files, 13.5 MB, downloaded once; after that the
engine runs offline.

The engine parameters were picked by measurement, change them with care:
- Global.text_score=0.0: the default 0.5 SILENTLY throws away lines the engine
  could not read. For a labeling platform that is the worst possible behavior —
  the human never sees what has to be fixed. The platform's own threshold is
  min_confidence, and it is explicit.
- Global.use_cls=False: the default text line orientation classifier flips long
  Cyrillic lines by 180° (4 garbage lines out of 16 on a dense A4 page) and on
  top of that slows the run down.
- Det.limit_type="max" + limit_side_len=1600: the default "min"/736 upscales small
  crops and breaks lines apart into separate words (51 polygons instead of 8).
- Rec.lang_type: the bundled PP-OCRv6 models do not recognize Cyrillic at all, so
  rec/det are taken from PP-OCRv5 explicitly.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image as PILImage

from nounbox_sdk import Annotation, Capability, Polygon

logger = logging.getLogger(__name__)

ENV_MODEL_DIR = "RAPIDOCR_MODEL_DIR"
ENV_MAX_PIXELS = "RAPIDOCR_MAX_PIXELS"

DEFAULT_MODEL_DIR = "/data/rapidocr-models"
FALLBACK_MODEL_DIR = Path.home() / ".cache" / "rapidocr-models"

# ingest normalizes pages to 4096 px on the long side (~16.8 MP) — generous limit
DEFAULT_MAX_PIXELS = 40_000_000

DEFAULT_LANG = "ru"

# LangRec values from rapidocr 3.9.x — the available PP-OCRv5 rec models
REC_LANGS = frozenset(
    {
        "arabic",
        "ch",
        "ch_doc",
        "chinese_cht",
        "cyrillic",
        "devanagari",
        "el",
        "en",
        "eslav",
        "japan",
        "ka",
        "korean",
        "latin",
        "ta",
        "te",
        "th",
    }
)

# UI language codes -> rec model; a value from REC_LANGS passes through as is
LANG_ALIASES = {
    "be": "cyrillic",
    "bg": "cyrillic",
    "kk": "cyrillic",
    "mk": "cyrillic",
    "mn": "cyrillic",
    "ru": "cyrillic",
    "sr": "cyrillic",
    "uk": "cyrillic",
    "cs": "latin",
    "de": "latin",
    "es": "latin",
    "fr": "latin",
    "hu": "latin",
    "it": "latin",
    "pl": "latin",
    "pt": "latin",
    "ro": "latin",
    "tr": "latin",
    "ar": "arabic",
    "hi": "devanagari",
    "ja": "japan",
    "jp": "japan",
    "ko": "korean",
    "kr": "korean",
    "zh": "ch",
    "zh-tw": "chinese_cht",
}


def resolve_rec_lang(lang: str | None) -> str:
    """Language code from config -> RapidOCR rec model name."""
    key = str(lang or DEFAULT_LANG).strip().lower().replace("_", "-")
    rec_lang = LANG_ALIASES.get(key, key)
    if rec_lang not in REC_LANGS:
        supported = ", ".join(sorted(REC_LANGS | set(LANG_ALIASES)))
        raise ValueError(f"rapidocr: unsupported lang {lang!r}; supported: {supported}")
    return rec_lang


def model_dir() -> str:
    """Weight cache directory: env -> default (docker volume) -> ~/.cache."""
    path = Path(os.environ.get(ENV_MODEL_DIR) or DEFAULT_MODEL_DIR)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "rapidocr: model cache %s is unusable (%s), falling back to %s",
            path,
            exc,
            FALLBACK_MODEL_DIR,
        )
        path = FALLBACK_MODEL_DIR
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def max_pixels(config: dict) -> int:
    """Image area limit: config -> env -> default.

    Every source is resolved by PRESENCE, not truthiness: max_pixels=0 in the
    config is set, and has to reach the check below instead of silently falling
    through to the env var and passing as the default. An unset env var and an
    empty one are the same thing, but "0" in it is a value and gets rejected.
    """
    raw = config.get("max_pixels")
    if raw is None:
        env = os.environ.get(ENV_MAX_PIXELS, "").strip()
        raw = env if env else DEFAULT_MAX_PIXELS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"rapidocr: max_pixels must be positive, got {value}")
    return value


def probe_image(image: bytes, limit: int) -> tuple[int, int]:
    """Image size from the header — without decoding the pixels.

    It also turns corrupt bytes into a comprehensible error: rapidocr itself
    dies with UnidentifiedImageError, saying nothing about who passed it what.
    """
    if not image:
        raise ValueError("rapidocr: empty image payload")
    try:
        with PILImage.open(io.BytesIO(image)) as pil:
            width, height = pil.size
    except PILImage.DecompressionBombError as exc:
        raise ValueError(
            f"rapidocr: image rejected as a decompression bomb: {exc}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"rapidocr: cannot decode image ({len(image)} bytes, "
            f"starts with {image[:8]!r}): {type(exc).__name__}: {exc}"
        ) from exc
    if width * height > limit:
        raise ValueError(
            f"rapidocr: image {width}x{height} ({width * height} px) exceeds the "
            f"limit of {limit} px (config.max_pixels / {ENV_MAX_PIXELS})"
        )
    return width, height


def to_polygon(box: Iterable[Sequence[float]]) -> Polygon | None:
    """Points as-is: rapidocr returns TL, TR, BR, BL — already clockwise (SDK contract).

    Coordinates are in pixels of the ORIGINAL image even when the engine
    downscales internally.
    """
    points: Polygon = [(float(x), float(y)) for x, y in box]
    return points if len(points) >= 3 else None


def to_annotations(
    boxes,
    texts,
    scores,
    min_confidence: float = 0.0,
) -> list[Annotation]:
    """RapidOCROutput -> SDK annotations. Empty page (boxes=None) -> empty list."""
    if boxes is None or texts is None or scores is None:
        return []

    annotations: list[Annotation] = []
    degenerate = 0
    filtered = 0
    for box, text, score in zip(boxes, texts, scores):
        polygon = to_polygon(box)
        if polygon is None:
            degenerate += 1
            continue
        confidence = float(score)
        if confidence < min_confidence:
            filtered += 1
            continue
        annotations.append(
            Annotation(
                geometry=polygon,
                label="text_line",
                text=str(text),
                confidence=confidence,
            )
        )
    if degenerate:
        logger.warning("rapidocr: skipped %d box(es) with less than 3 points", degenerate)
    if filtered:
        logger.info(
            "rapidocr: dropped %d line(s) below min_confidence=%s",
            filtered,
            min_confidence,
        )
    return annotations


class RapidOCRLabeler:
    name = "rapidocr"
    version = "0.1.0"
    capabilities = {Capability.DETECTION, Capability.RECOGNITION}

    def __init__(self) -> None:
        self._engines: dict[str, object] = {}  # rec_lang -> RapidOCR instance
        self._lock = threading.Lock()

    # --- engine initialization (lazily, once per rec model) ---

    def _engine(self, config: dict):
        rec_lang = resolve_rec_lang(config.get("lang"))
        with self._lock:
            if rec_lang not in self._engines:
                self._engines[rec_lang] = self._create_engine(rec_lang)
            return self._engines[rec_lang]

    @staticmethod
    def _create_engine(rec_lang: str):
        from rapidocr import LangDet, ModelType, OCRVersion, RapidOCR

        logger.info("Loading RapidOCR (Rec.lang_type=%s)...", rec_lang)
        return RapidOCR(
            params={
                "Global.model_root_dir": model_dir(),
                "Global.text_score": 0.0,
                "Global.use_cls": False,
                "Global.log_level": "warning",
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.lang_type": LangDet.CH,
                "Det.model_type": ModelType.MOBILE,
                "Det.limit_type": "max",
                "Det.limit_side_len": 1600,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.lang_type": rec_lang,
                "Rec.model_type": ModelType.MOBILE,
            }
        )

    # --- SDK contract ---

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        width, height = probe_image(image, max_pixels(config))
        engine = self._engine(config)
        result = engine(
            image,
            box_thresh=config.get("box_thresh"),
            unclip_ratio=config.get("unclip_ratio"),
        )
        annotations = to_annotations(
            result.boxes,
            result.txts,
            result.scores,
            float(config.get("min_confidence", 0.0)),
        )
        logger.debug("rapidocr: %dx%d -> %d line(s)", width, height, len(annotations))
        return annotations
