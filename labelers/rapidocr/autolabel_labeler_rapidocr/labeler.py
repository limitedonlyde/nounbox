"""RapidOCR labeler: построчная детекция + распознавание на CPU (onnxruntime).

Дефолтный движок платформы — работает сразу после `docker compose up`, без GPU,
аккаунтов и ключей. Отдаёт построчные полигоны (4 точки TL->TR->BR->BL, на
перекошенном сканe — настоящие наклонные четырёхугольники) и РЕАЛЬНЫЙ
confidence: средняя softmax-вероятность символов из CTC-декодера (не константа).

Config:
    lang: str = "ru"            — код языка -> rec-модель PP-OCRv5 (см. LANG_ALIASES)
    min_confidence: float = 0.0 — отсечь строки ниже порога; 0.0 отдаёт человеку всё,
                                  включая то, что движок прочитать не смог
    box_thresh: float | None    — порог DB-детекции (None = дефолт движка)
    unclip_ratio: float | None  — раздутие контура строки (None = дефолт движка)
    max_pixels: int             — лимит площади картинки (дефолт из RAPIDOCR_MAX_PIXELS)

Кеш весов — RAPIDOCR_MODEL_DIR (дефолт /data/rapidocr-models, под docker volume;
если каталог не создать — ~/.cache/rapidocr-models). Для русского это три onnx
на 13.5 МБ, качаются один раз, дальше движок работает офлайн.

Параметры движка подобраны замерами, менять с осторожностью:
- Global.text_score=0.0: дефолт 0.5 МОЛЧА выбрасывает строки, которые движок не
  смог прочитать. Для платформы разметки это худшее поведение — человек не увидит
  того, что нужно исправить. Порог платформы — min_confidence, он явный.
- Global.use_cls=False: дефолтный классификатор ориентации строки переворачивает
  длинные кириллические строки на 180° (4 мусорные строки из 16 на плотном A4)
  и при этом замедляет прогон.
- Det.limit_type="max" + limit_side_len=1600: дефолт "min"/736 растягивает мелкие
  кропы и дробит строки на отдельные слова (51 полигон вместо 8).
- Rec.lang_type: бандл-модели PP-OCRv6 кириллицу не распознают вообще, поэтому
  rec/det берутся из PP-OCRv5 явно.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image as PILImage

from autolabelui_sdk import Annotation, Capability, Polygon

logger = logging.getLogger(__name__)

ENV_MODEL_DIR = "RAPIDOCR_MODEL_DIR"
ENV_MAX_PIXELS = "RAPIDOCR_MAX_PIXELS"

DEFAULT_MODEL_DIR = "/data/rapidocr-models"
FALLBACK_MODEL_DIR = Path.home() / ".cache" / "rapidocr-models"

# ingest нормализует страницы до 4096 по длинной стороне (~16.8 Мп) — лимит с запасом
DEFAULT_MAX_PIXELS = 40_000_000

DEFAULT_LANG = "ru"

# значения LangRec из rapidocr 3.9.x — доступные rec-модели PP-OCRv5
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

# коды языков UI -> rec-модель; значение из REC_LANGS проходит как есть
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
    """Код языка из config -> имя rec-модели RapidOCR."""
    key = str(lang or DEFAULT_LANG).strip().lower().replace("_", "-")
    rec_lang = LANG_ALIASES.get(key, key)
    if rec_lang not in REC_LANGS:
        supported = ", ".join(sorted(REC_LANGS | set(LANG_ALIASES)))
        raise ValueError(f"rapidocr: unsupported lang {lang!r}; supported: {supported}")
    return rec_lang


def model_dir() -> str:
    """Каталог кеша весов: env -> дефолт (docker volume) -> ~/.cache."""
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
    """Лимит площади изображения: config -> env -> дефолт."""
    raw = config.get("max_pixels") or os.environ.get(ENV_MAX_PIXELS) or DEFAULT_MAX_PIXELS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"rapidocr: max_pixels must be positive, got {value}")
    return value


def probe_image(image: bytes, limit: int) -> tuple[int, int]:
    """Размер картинки по заголовку — без декодирования пикселей.

    Заодно превращает битые байты во внятную ошибку: сам rapidocr падает
    UnidentifiedImageError без указания на то, кто и что ему передал.
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
    """Точки как есть: rapidocr отдаёт TL, TR, BR, BL — уже по часовой (контракт SDK).

    Координаты — в пикселях ИСХОДНОЙ картинки даже при внутреннем даунскейле.
    """
    points: Polygon = [(float(x), float(y)) for x, y in box]
    return points if len(points) >= 3 else None


def to_annotations(
    boxes,
    texts,
    scores,
    min_confidence: float = 0.0,
) -> list[Annotation]:
    """RapidOCROutput -> аннотации SDK. Пустая страница (boxes=None) -> пустой список."""
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

    # --- инициализация движка (лениво, один раз на rec-модель) ---

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

    # --- SDK-контракт ---

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
