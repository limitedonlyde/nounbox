"""LLMDet labeler: открытословарная детекция боксов по английским именам классов.

Второй движок для task_type=detection: режим «посчитать точнее», чем дефолтный
OWLv2, и второй голос для consensus. Модель — iSEE-Laboratory/llmdet_base
(MM-Grounding-DINO, дообученный с LLM-супервизией), Apache-2.0, CPU.

Замер платформы (79 фото / 323 объекта LVIS, ручной эталон, IoU 0.5, CPU M2):

    движок               F1      P      R     с/фото   RSS
    llmdet_base       0.853  0.881  0.827      4.0    ~4.3 ГБ
    llmdet_tiny       0.831  0.866  0.799      2.8    ~3.8 ГБ
    owlv2 (дефолт)    0.823                 2.0-2.4   ~1.5 ГБ

при дефолтных score_threshold=0.35 и nms_iou=0.4 — отсюда и дефолты. С nms_iou=0.7
(дефолт MM-GDINO) те же прогоны дают 0.848 / 0.822: дубли на одном объекте не
гасятся и уходят человеку в ревью. Порог 0.35 — компромисс замера: на 0.30 F1
чуть выше (0.857), но точность падает с 0.881 до 0.851, то есть каждая седьмая
рамка ложная.

Config:
    classes: list[str]            — английские имена классов проекта (обязательно)
    score_threshold: float = 0.35 — ниже порога рамки не отдаются
    nms_iou: float = 0.4          — подавление дублей внутри одного класса
    model: str = "base"           — "base" | "tiny" | полный repo id с HF; каждая
                                    названная модель остаётся в памяти воркера
                                    до конца процесса (base ~4.3 ГБ)
    max_detections: int = 100     — потолок рамок на изображение
    max_pixels: int               — лимит площади картинки (env LLMDET_MAX_PIXELS)

Веса кешируются в OVD_MODEL_DIR (дефолт /data/ovd-models — тот же docker-volume,
что у owlv2; если каталог не создать — обычный кеш huggingface). base — 1.6 ГБ,
tiny — 0.7 ГБ, качаются один раз.

ЛИМИТ 91 КЛАСС. У модели фиксированная голова классификации на max_text_len=256
токенов, а промпт — это все классы одной строкой «carpet. sofa. chandelier.».
Промпт длиннее 256 токенов роняет forward внутри transformers невнятным
«RuntimeError: The size of tensor a (256) must match the size of tensor b (319)».
Поэтому лимит ловится ЗАРАНЕЕ, до загрузки модели, по числу классов, и второй раз
по факту токенизации (длинные многословные имена съедают лимит быстрее).

Почему не чанкинг. Разбить 100 классов на пачки по 30 и слить результаты
технически можно, но замер это запрещает: при чанкинге 27% рамок садятся на
объект с ярлыком из чужой пачки — модель без конкурирующих классов в промпте
уверенно называет диван ковром. Лучше внятно отправить пользователя на owlv2,
который инвариантен к длине списка.

Почему свои скоры, а не post_process_grounded_object_detection. Стоковый
постпроцессор декодирует ФРАЗУ по каждой рамке и склеивает соседние классы
(«cutting board toilet paper»), после чего рамка не отображается ни на один
класс проекта и просто теряется. Здесь скор класса считается официальным для
MM-GDINO способом — средним сигмоиды логитов по токенам этого класса
(positive map), — то есть каждая рамка получает свой скор по КАЖДОМУ классу
запроса, и рамки эмитируются по классам независимо.
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

from autolabelui_sdk import Annotation, BBox, Capability

logger = logging.getLogger(__name__)

ENV_MODEL_DIR = "OVD_MODEL_DIR"
ENV_MAX_PIXELS = "LLMDET_MAX_PIXELS"

# общий с owlv2 кеш весов: один volume на все открытословарные детекторы
DEFAULT_MODEL_DIR = "/data/ovd-models"

MODEL_ALIASES = {
    "base": "iSEE-Laboratory/llmdet_base",
    "tiny": "iSEE-Laboratory/llmdet_tiny",
}
DEFAULT_MODEL = MODEL_ALIASES["base"]

# сколько классов помещается в промпт: 256 токенов головы / ~2.8 токена на класс
MAX_CLASSES = 91
# config.max_text_len модели; берётся из конфига загруженной модели, это дефолт
DEFAULT_MAX_TEXT_TOKENS = 256

DEFAULT_SCORE_THRESHOLD = 0.35
DEFAULT_NMS_IOU = 0.4
DEFAULT_MAX_DETECTIONS = 100

# ingest нормализует страницы до 4096 по длинной стороне (~16.8 Мп) — лимит с запасом
DEFAULT_MAX_PIXELS = 40_000_000

# рамка тоньше пикселя бесполезна для обучения и не рисуется в ревью
MIN_BOX_SIDE = 1.0


@dataclass(frozen=True)
class Detection:
    """Сырая рамка движка: класс, скор, xyxy в абсолютных пикселях."""

    label: str
    score: float
    box: tuple[float, float, float, float]


# --- конфигурация ---


def resolve_model(config: dict) -> str:
    """Имя модели из config: алиас base/tiny или полный repo id huggingface."""
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
    """Классы проекта из config: как есть (регистр сохраняем), без пустых и дублей.

    Возвращаются ОРИГИНАЛЬНЫЕ имена — они уходят в Annotation.label и должны
    совпасть с project_classes.name; в промпт они попадают в нижнем регистре
    (см. build_prompt), так этого требует токенизатор модели.
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
    """Лимит промпта — ДО загрузки модели: иначе будет невнятный RuntimeError."""
    if len(classes) > MAX_CLASSES:
        raise ValueError(
            f"llmdet: LLMDet takes at most {MAX_CLASSES} classes per request, "
            f"and this project has {len(classes)} — use owlv2 instead"
        )


def check_prompt_tokens(n_tokens: int, max_text_len: int, n_classes: int) -> None:
    """Второй рубеж: длинные многословные имена выбирают лимит раньше 91 класса."""
    if n_tokens > max_text_len:
        raise ValueError(
            f"llmdet: a prompt with {n_classes} classes takes {n_tokens} tokens, "
            f"and the model limit is {max_text_len} — shorten the class names "
            "or use owlv2 instead"
        )


def option(config: dict, key: str, default: float) -> float:
    """Число из config; null (а он приезжает из JSON-payload задачи) — это дефолт."""
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
    """Каталог кеша весов: env -> дефолт (docker volume) -> None (штатный кеш HF)."""
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
    """Лимит площади изображения: config -> env -> дефолт."""
    raw = (
        config.get("max_pixels") or os.environ.get(ENV_MAX_PIXELS) or DEFAULT_MAX_PIXELS
    )
    value = int(raw)
    if value <= 0:
        raise ValueError(f"llmdet: max_pixels must be positive, got {value}")
    return value


def load_image(image: bytes, limit: int) -> PILImage.Image:
    """Байты -> RGB-картинка.

    Битые данные и бомбы превращаются во внятную ошибку, а не в падение
    где-то внутри processor'а.
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


# --- промпт и разбор выхода модели ---


def build_prompt(classes: Sequence[str]) -> str:
    """Классы -> строка промпта «carpet. sofa. chandelier.».

    Формат ровно тот, что собирает сам processor из списка меток: нижний регистр,
    разделитель «. », точка в конце. Собираем сами, потому что по этой же точке
    потом восстанавливаются токенные диапазоны классов.
    """
    labels = [" ".join(str(name).strip().lower().split()) for name in classes]
    return ". ".join(labels) + "."


def class_spans(
    input_ids: Sequence[int],
    dot_id: int,
    special_ids: Iterable[int],
    n_classes: int,
) -> list[list[int]]:
    """Индексы токенов по каждому классу — промпт режется по точкам.

    Многословный класс («cutting board») и класс, разбитый на wordpiece'ы
    («chan ##del ##ier»), дают несколько индексов: скор класса — среднее по ним.
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
    """cxcywh в долях картинки -> xyxy в пикселях, обрезано по границам кадра."""
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
    """IoU двух xyxy-рамок."""
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
    """Жадный NMS ВНУТРИ класса: два объекта разных классов могут перекрываться.

    Рамка «wine bottle» поверх рамки «bottle» — нормальная разметка, глушить её
    нельзя; а два одинаковых ярлыка на одном объекте — дубль, который человек
    вынужден удалять руками.
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
    """Сырые рамки -> аннотации SDK: порог, NMS по классам, потолок, BBox."""
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


# --- загруженная модель ---


class Engine:
    """processor + модель одного repo id. Создаётся лениво, живёт до конца процесса."""

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
        """Один forward -> рамки по КАЖДОМУ классу запроса (positive map, см. модуль)."""
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
        # logits: (queries, max_text_len) — скор рамки по каждому токену промпта
        scores_per_token = outputs.logits[0].sigmoid()
        boxes = outputs.pred_boxes[0]  # (queries, 4), cxcywh в долях кадра
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

    # --- инициализация движка (лениво, один раз на модель) ---

    def _engine(self, model_id: str) -> Engine:
        with self._lock:
            if model_id not in self._engines:
                self._engines[model_id] = self._create_engine(model_id)
            return self._engines[model_id]

    @staticmethod
    def _create_engine(model_id: str) -> Engine:
        return Engine(model_id)

    # --- SDK-контракт ---

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        # конфиг проверяется первым: ошибка в классах не должна стоить
        # загрузки 1.6 ГБ весов
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
