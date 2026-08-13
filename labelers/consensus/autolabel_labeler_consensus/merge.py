"""Чистая логика консенсуса: сведение выводов N движков в один список.

Первый движок — primary: его геометрия считается точнее (обычно OCR),
она и попадает в результат. Аннотации остальных (secondary) сопоставляются
с primary жадным матчингом по убыванию IoU (любая геометрия приводится
к ограничивающему прямоугольнику; порог — iou_threshold).

Confidence — из степени согласия:
    геометрия совпала, текст sim >= 0.95 -> 0.95
    геометрия совпала, текст частично    -> 0.55 + 0.35 * sim
    primary без пары                     -> 0.45
    secondary без пары                   -> 0.35 (добавляется со своей геометрией)

Каждая аннотация получает attrs["consensus"]:
    {"engines": [...подтвердившие...], "text_similarity": float | None,
     "alt_text": {"<engine>": "<их вариант>"}}  # только если тексты разошлись
"""

from __future__ import annotations

import difflib
from dataclasses import replace
from typing import Any

from autolabelui_sdk import Annotation, BBox

DEFAULT_IOU_THRESHOLD = 0.4

FULL_AGREEMENT_SIM = 0.95
CONF_FULL_AGREEMENT = 0.95
CONF_PARTIAL_BASE = 0.55
CONF_PARTIAL_SPAN = 0.35
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


def text_similarity(a: str | None, b: str | None) -> float:
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _greedy_match(
    primary_boxes: list[Box], secondary_boxes: list[Box], iou_threshold: float
) -> dict[int, int]:
    """Жадный матчинг по убыванию IoU: {primary_idx: secondary_idx}, каждый — не более одного раза."""
    scored = [
        (score, i, j)
        for i, pb in enumerate(primary_boxes)
        for j, sb in enumerate(secondary_boxes)
        if (score := iou(pb, sb)) >= iou_threshold
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    matches: dict[int, int] = {}
    used_secondary: set[int] = set()
    for _score, i, j in scored:
        if i in matches or j in used_secondary:
            continue
        matches[i] = j
        used_secondary.add(j)
    return matches


def _with_consensus(
    ann: Annotation,
    confidence: float,
    engines: list[str],
    sim: float | None,
    alt_text: dict[str, str],
) -> Annotation:
    consensus: dict[str, Any] = {"engines": engines, "text_similarity": sim}
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
    primary_boxes = [to_bbox(a.geometry) for a in primary]

    # per primary_idx: [(engine, их аннотация)]; unmatched secondary -> extras
    confirmations: list[list[tuple[str, Annotation]]] = [[] for _ in primary]
    extras: list[tuple[str, Annotation]] = []

    for engine_name in engine_names[1:]:
        secondary = annotations_by_engine[engine_name]
        secondary_boxes = [to_bbox(a.geometry) for a in secondary]
        matches = _greedy_match(primary_boxes, secondary_boxes, iou_threshold)
        for i, j in matches.items():
            confirmations[i].append((engine_name, secondary[j]))
        matched = set(matches.values())
        extras.extend(
            (engine_name, ann) for j, ann in enumerate(secondary) if j not in matched
        )

    merged: list[Annotation] = []
    for ann, confirmed in zip(primary, confirmations):
        if not confirmed:
            merged.append(_with_consensus(ann, CONF_PRIMARY_ONLY, [primary_name], None, {}))
            continue
        best_sim = max(text_similarity(ann.text, other.text) for _, other in confirmed)
        confidence = (
            CONF_FULL_AGREEMENT
            if best_sim >= FULL_AGREEMENT_SIM
            else CONF_PARTIAL_BASE + CONF_PARTIAL_SPAN * best_sim
        )
        alt_text = {
            name: other.text
            for name, other in confirmed
            if other.text is not None
            and normalize_text(other.text) != normalize_text(ann.text)
        }
        engines = [primary_name, *(name for name, _ in confirmed)]
        merged.append(_with_consensus(ann, confidence, engines, best_sim, alt_text))

    for engine_name, ann in extras:
        merged.append(_with_consensus(ann, CONF_SECONDARY_ONLY, [engine_name], None, {}))
    return merged
