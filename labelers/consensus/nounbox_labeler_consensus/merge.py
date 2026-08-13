"""Pure consensus logic: merging the outputs of N engines into one list.

The first engine is the primary: its geometry is taken to be the more precise
one (usually OCR), and that is what ends up in the result. The annotations of
the others (secondary) are matched against the primary greedily by descending
IoU (any geometry is reduced to its bounding box; the cut-off is iou_threshold).

ONLY annotations with the same label are matched: in detection over the
project's own classes, boxes of different classes overlapping is normal (a dog
on a sofa), and matching on geometry alone would glue dog to sofa and hand the
dataset a wrong class with high confidence. Case and extra whitespace in the
label are ignored while comparing; the primary's label is what survives.

Confidence comes from the degree of agreement. Two scales, as text is not always there:

  texts are comparable (both sides non-empty) — OCR:
    geometry matched, sim >= 0.95        -> 0.95
    geometry matched, text partially     -> 0.55 + 0.35 * sim
  no text (detection) or only one engine has it — geometry only:
    geometry matched                     -> 0.60 + 0.29 * q,
                                            q = (iou - threshold) / (1 - threshold)
  primary with no partner                -> 0.45
  secondary with no partner              -> 0.35 (added with its own geometry)

The ceiling of the geometry scale (0.89) is deliberately below 0.95: boxes
agreeing is weaker evidence than boxes plus matching text. There used to be no
such scale, and for detection (text is None everywhere) difflib on two empty
strings gave sim=1.0 — consensus always emitted the maximum 0.95, i.e. the
confidence was synthetic.

Every annotation gets attrs["consensus"]:
    {"engines": [...the confirming ones...], "text_similarity": float | None,
     "iou": float,                     # only on confirmed ones
     "alt_text": {"<engine>": "<their variant>"}}  # only if the texts differ
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
# geometry-only agreement (no text): 0.60 for a match right at the threshold,
# 0.89 for boxes that coincide exactly
CONF_GEOMETRY_BASE = 0.60
# ceiling 0.89 — strictly below the default bulk-accept threshold (0.9):
# agreement on boxes alone must not be accepted in bulk without a human
CONF_GEOMETRY_SPAN = 0.29
CONF_PRIMARY_ONLY = 0.45
CONF_SECONDARY_ONLY = 0.35

Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def to_bbox(geometry: BBox | list[tuple[float, float]]) -> Box:
    """Any geometry -> (x1, y1, x2, y2); a polygon via its bounding box."""
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
    """Casefold + whitespace collapsing; None -> ""."""
    return " ".join((text or "").split()).casefold()


def label_key(label: str | None) -> str:
    """Class comparison key: engines write case and stray spaces differently."""
    return normalize_text(label)


def text_similarity(a: str | None, b: str | None) -> float | None:
    """Text similarity 0..1; None — nothing to compare (at least one side empty).

    Empty text does not mean "matched": a detector without text recognition
    promised nothing, while difflib on two empty strings returns 1.0.
    """
    left, right = normalize_text(a), normalize_text(b)
    if not left or not right:
        return None
    return difflib.SequenceMatcher(None, left, right).ratio()


def _geometry_confidence(best_iou: float, iou_threshold: float) -> float:
    """Confidence from box overlap alone: threshold -> BASE, exact -> BASE+SPAN."""
    span = 1.0 - iou_threshold
    quality = 1.0 if span <= 0 else (best_iou - iou_threshold) / span
    return CONF_GEOMETRY_BASE + CONF_GEOMETRY_SPAN * min(1.0, max(0.0, quality))


def _greedy_match(
    primary: list[tuple[Box, str]],
    secondary: list[tuple[Box, str]],
    iou_threshold: float,
) -> dict[int, tuple[int, float]]:
    """Greedy matching by descending IoU within a single class.

    Input — (bbox, class key); output — {primary_idx: (secondary_idx, iou)},
    with every annotation taking part at most once.
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
    """Merge engine outputs (in order given, the first is primary) into one list."""
    if not annotations_by_engine:
        return []
    engine_names = list(annotations_by_engine)
    primary_name = engine_names[0]
    primary = annotations_by_engine[primary_name]
    primary_keyed = _keyed(primary)

    # per primary_idx: [(engine, its annotation, iou)]; unmatched secondary -> extras
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
            # no text (detection), or only one engine has it — judge the
            # agreement by the boxes
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
