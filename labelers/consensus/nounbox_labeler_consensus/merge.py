"""Чистая логика консенсуса: сведение выводов N движков в один список.

Первый движок — primary: его геометрия считается точнее (обычно OCR),
она и попадает в результат. Аннотации остальных (secondary) сопоставляются
с primary жадным матчингом по убыванию IoU (любая геометрия приводится
к ограничивающему прямоугольнику; порог — iou_threshold).

Сопоставляются ТОЛЬКО аннотации с одинаковым label: в детекции по своим
классам перекрытие рамок разных классов — норма (собака на диване), и
матчинг по одной геометрии склеил бы dog с sofa, отдав в датасет неверный
класс с высокой уверенностью. Регистр и лишние пробелы в label при сравнении
не учитываются, в результате остаётся label primary.

Confidence — из степени согласия. Шкал две, потому что тексты есть не всегда:

  тексты сравнимы (обе стороны непустые) — OCR:
    геометрия совпала, sim >= 0.95       -> 0.95
    геометрия совпала, текст частично    -> 0.55 + 0.35 * sim
  текста нет (детекция) или он есть лишь у одного движка — только геометрия:
    геометрия совпала                    -> 0.60 + 0.30 * q,
                                            q = (iou - threshold) / (1 - threshold)
  primary без пары                       -> 0.45
  secondary без пары                     -> 0.35 (добавляется со своей геометрией)

Потолок геометрической шкалы (0.90) намеренно ниже 0.95: совпадение одних
рамок — более слабое свидетельство, чем рамки плюс совпавший текст. Раньше
шкалы не было, и для детекции (text везде None) difflib на двух пустых
строках давал sim=1.0 — консенсус всегда выдавал максимум 0.95, то есть
уверенность была синтетической.

Каждая аннотация получает attrs["consensus"]:
    {"engines": [...подтвердившие...], "text_similarity": float | None,
     "iou": float,                     # только у подтверждённых
     "alt_text": {"<engine>": "<их вариант>"}}  # только если тексты разошлись
"""

from __future__ import annotations

import difflib
from dataclasses import replace
from typing import Any

from nounbox_sdk import Annotation, BBox

DEFAULT_IOU_THRESHOLD = 0.4

FULL_AGREEMENT_SIM = 0.95
CONF_FULL_AGREEMENT = 0.95
CONF_PARTIAL_BASE = 0.55
CONF_PARTIAL_SPAN = 0.35
# согласие только по геометрии (текста нет): 0.60 при совпадении на пороге,
# 0.90 при полностью совпавших рамках
CONF_GEOMETRY_BASE = 0.60
# потолок 0.89 — строго ниже дефолтного порога bulk-accept (0.9):
# согласие одних рамок не должно приниматься пачкой без человека
CONF_GEOMETRY_SPAN = 0.29
CONF_PRIMARY_ONLY = 0.45
CONF_SECONDARY_ONLY = 0.35

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def to_bbox(geometry: BBox | list[tuple[float, float]]) -> Box:
    """Любая геометрия -> (x1, y1, x2, y2); полигон — ограничивающим прямоугольником."""
    if isinstance(geometry, BBox):
        return (
            geometry.x,
            geometry.y,
            geometry.x + geometry.width,
            geometry.y + geometry.height,
        )
    xs = [p[0] for p in geometry]
    ys = [p[1] for p in geometry]
    return (min(xs), min(ys), max(xs), max(ys))


def iou(a: Box, b: Box) -> float:
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def normalize_text(text: str | None) -> str:
    """Casefold + схлопывание пробелов; None -> ""."""
    return " ".join((text or "").split()).casefold()


def label_key(label: str | None) -> str:
    """Ключ сравнения классов: регистр и лишние пробелы движки пишут по-разному."""
    return normalize_text(label)


def text_similarity(a: str | None, b: str | None) -> float | None:
    """Сходство текстов 0..1; None — сравнивать нечего (хотя бы один пустой).

    Пустой текст не значит «совпало»: детектор без распознавания текста
    не обещал ничего, а difflib на двух пустых строках возвращает 1.0.
    """
    left, right = normalize_text(a), normalize_text(b)
    if not left or not right:
        return None
    return difflib.SequenceMatcher(None, left, right).ratio()


def _geometry_confidence(best_iou: float, iou_threshold: float) -> float:
    """Уверенность по одному лишь совпадению рамок: порог -> BASE, полное -> BASE+SPAN."""
    span = 1.0 - iou_threshold
    quality = 1.0 if span <= 0 else (best_iou - iou_threshold) / span
    return CONF_GEOMETRY_BASE + CONF_GEOMETRY_SPAN * min(1.0, max(0.0, quality))


def _greedy_match(
    primary: list[tuple[Box, str]],
    secondary: list[tuple[Box, str]],
    iou_threshold: float,
) -> dict[int, tuple[int, float]]:
    """Жадный матчинг по убыванию IoU внутри одного класса.

    Вход — (bbox, ключ класса); выход — {primary_idx: (secondary_idx, iou)},
    каждая аннотация участвует не более одного раза.
    """
    scored = [
        (score, i, j)
        for i, (pb, plabel) in enumerate(primary)
        for j, (sb, slabel) in enumerate(secondary)
        if plabel == slabel and (score := iou(pb, sb)) >= iou_threshold
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    matches: dict[int, tuple[int, float]] = {}
    used_secondary: set[int] = set()
    for score, i, j in scored:
        if i in matches or j in used_secondary:
            continue
        matches[i] = (j, score)
        used_secondary.add(j)
    return matches


def _keyed(annotations: list[Annotation]) -> list[tuple[Box, str]]:
    return [(to_bbox(a.geometry), label_key(a.label)) for a in annotations]


def _with_consensus(
    ann: Annotation,
    confidence: float,
    engines: list[str],
    sim: float | None,
    alt_text: dict[str, str],
    best_iou: float | None = None,
) -> Annotation:
    consensus: dict[str, Any] = {"engines": engines, "text_similarity": sim}
    if best_iou is not None:
        consensus["iou"] = best_iou
    if alt_text:
        consensus["alt_text"] = alt_text
    return replace(ann, confidence=confidence, attrs={**ann.attrs, "consensus": consensus})


def merge(
    annotations_by_engine: dict[str, list[Annotation]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[Annotation]:
    """Свести выводы движков (в порядке следования, первый — primary) в один список."""
    if not annotations_by_engine:
        return []
    engine_names = list(annotations_by_engine)
    primary_name = engine_names[0]
    primary = annotations_by_engine[primary_name]
    primary_keyed = _keyed(primary)

    # per primary_idx: [(engine, их аннотация, iou)]; unmatched secondary -> extras
    confirmations: list[list[tuple[str, Annotation, float]]] = [[] for _ in primary]
    extras: list[tuple[str, Annotation]] = []

    for engine_name in engine_names[1:]:
        secondary = annotations_by_engine[engine_name]
        matches = _greedy_match(primary_keyed, _keyed(secondary), iou_threshold)
        for i, (j, score) in matches.items():
            confirmations[i].append((engine_name, secondary[j], score))
        matched = {j for j, _ in matches.values()}
        extras.extend(
            (engine_name, ann) for j, ann in enumerate(secondary) if j not in matched
        )

    merged: list[Annotation] = []
    for ann, confirmed in zip(primary, confirmations):
        if not confirmed:
            merged.append(_with_consensus(ann, CONF_PRIMARY_ONLY, [primary_name], None, {}))
            continue
        sims = [
            sim
            for _, other, _ in confirmed
            if (sim := text_similarity(ann.text, other.text)) is not None
        ]
        best_iou = max(score for _, _, score in confirmed)
        if sims:
            best_sim: float | None = max(sims)
            confidence = (
                CONF_FULL_AGREEMENT
                if best_sim >= FULL_AGREEMENT_SIM
                else CONF_PARTIAL_BASE + CONF_PARTIAL_SPAN * best_sim
            )
        else:
            # текста нет (детекция) или он есть только у одного движка —
            # согласие оцениваем по рамкам
            best_sim = None
            confidence = _geometry_confidence(best_iou, iou_threshold)
        alt_text = {
            name: other.text
            for name, other, _ in confirmed
            if other.text is not None
            and normalize_text(other.text) != normalize_text(ann.text)
        }
        engines = [primary_name, *(name for name, _, _ in confirmed)]
        merged.append(
            _with_consensus(ann, confidence, engines, best_sim, alt_text, best_iou)
        )

    for engine_name, ann in extras:
        merged.append(_with_consensus(ann, CONF_SECONDARY_ONLY, [engine_name], None, {}))
    return merged
