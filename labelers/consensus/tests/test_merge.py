"""Тесты чистой merge-логики консенсуса — без реальных движков."""

import difflib

import pytest

from autolabelui_sdk import Annotation, BBox
from autolabel_labeler_consensus.merge import iou, merge, text_similarity, to_bbox


def ann(x, y, w, h, text=None):
    return Annotation(geometry=BBox(x=x, y=y, width=w, height=h), text=text)


def test_full_agreement_high_confidence():
    merged = merge(
        {
            "paddleocr": [ann(10, 10, 100, 20, "Hello World")],
            "vlm": [ann(12, 11, 98, 19, "hello  world")],
        }
    )
    assert len(merged) == 1
    result = merged[0]
    assert result.confidence == pytest.approx(0.95)
    assert result.text == "Hello World"
    consensus = result.attrs["consensus"]
    assert consensus["engines"] == ["paddleocr", "vlm"]
    assert consensus["text_similarity"] == pytest.approx(1.0)
    assert "alt_text" not in consensus


def test_geometry_matched_text_diverged():
    merged = merge(
        {
            "paddleocr": [ann(0, 0, 100, 20, "Invoice 12345")],
            "vlm": [ann(0, 0, 100, 20, "Invoice 12845")],
        }
    )
    assert len(merged) == 1
    result = merged[0]
    sim = difflib.SequenceMatcher(None, "invoice 12345", "invoice 12845").ratio()
    assert 0 < sim < 0.95
    assert result.confidence == pytest.approx(0.55 + 0.35 * sim)
    assert result.text == "Invoice 12345"
    consensus = result.attrs["consensus"]
    assert consensus["engines"] == ["paddleocr", "vlm"]
    assert consensus["text_similarity"] == pytest.approx(sim)
    assert consensus["alt_text"] == {"vlm": "Invoice 12845"}


def test_primary_only():
    merged = merge({"paddleocr": [ann(0, 0, 100, 20, "alone")], "vlm": []})
    assert len(merged) == 1
    result = merged[0]
    assert result.confidence == pytest.approx(0.45)
    assert result.text == "alone"
    assert result.attrs["consensus"] == {
        "engines": ["paddleocr"],
        "text_similarity": None,
    }


def test_secondary_only_keeps_own_geometry_and_text():
    merged = merge({"paddleocr": [], "vlm": [ann(5, 5, 50, 10, "extra")]})
    assert len(merged) == 1
    result = merged[0]
    assert result.confidence == pytest.approx(0.35)
    assert result.text == "extra"
    assert to_bbox(result.geometry) == (5, 5, 55, 15)
    assert result.attrs["consensus"] == {"engines": ["vlm"], "text_similarity": None}


def test_polygon_matches_bbox_and_primary_geometry_wins():
    polygon = [(10.0, 10.0), (110.0, 10.0), (110.0, 30.0), (10.0, 30.0)]
    merged = merge(
        {
            "paddleocr": [Annotation(geometry=polygon, text="line")],
            "vlm": [ann(10, 10, 100, 20, "line")],
        }
    )
    assert len(merged) == 1
    result = merged[0]
    assert result.confidence == pytest.approx(0.95)
    assert result.geometry == polygon


def test_iou_just_below_threshold_no_match():
    # IoU(A, B) = 5700 / 14300 ≈ 0.399 < 0.4
    box_a, box_b = ann(0, 0, 100, 100, "x"), ann(0, 43, 100, 100, "x")
    assert iou(to_bbox(box_a.geometry), to_bbox(box_b.geometry)) < 0.4
    merged = merge({"a": [box_a], "b": [box_b]})
    assert sorted(m.confidence for m in merged) == pytest.approx([0.35, 0.45])


def test_iou_just_above_threshold_match():
    # IoU(A, B) = 6000 / 14000 ≈ 0.429 >= 0.4
    box_a, box_b = ann(0, 0, 100, 100, "x"), ann(0, 40, 100, 100, "x")
    assert iou(to_bbox(box_a.geometry), to_bbox(box_b.geometry)) >= 0.4
    merged = merge({"a": [box_a], "b": [box_b]})
    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(0.95)


def test_custom_iou_threshold():
    boxes = {"a": [ann(0, 0, 100, 100, "x")], "b": [ann(0, 40, 100, 100, "x")]}
    assert len(merge(boxes, iou_threshold=0.4)) == 1
    assert len(merge(boxes, iou_threshold=0.5)) == 2


def test_empty_outputs():
    assert merge({}) == []
    assert merge({"a": [], "b": []}) == []


def test_three_engines():
    merged = merge(
        {
            "paddleocr": [ann(0, 0, 100, 20, "Total: 42"), ann(0, 50, 100, 20, "Only OCR")],
            "vlm": [ann(1, 1, 99, 19, "Total: 42")],
            "http": [ann(0, 0, 100, 20, "Total 42"), ann(0, 200, 80, 15, "footer")],
        }
    )
    assert len(merged) == 3

    confirmed_by_all = merged[0]
    assert confirmed_by_all.confidence == pytest.approx(0.95)  # лучший sim = 1.0 (vlm)
    consensus = confirmed_by_all.attrs["consensus"]
    assert consensus["engines"] == ["paddleocr", "vlm", "http"]
    assert consensus["alt_text"] == {"http": "Total 42"}

    primary_only = merged[1]
    assert primary_only.text == "Only OCR"
    assert primary_only.confidence == pytest.approx(0.45)

    secondary_only = merged[2]
    assert secondary_only.text == "footer"
    assert secondary_only.confidence == pytest.approx(0.35)
    assert secondary_only.attrs["consensus"]["engines"] == ["http"]


def test_greedy_matching_prefers_best_iou():
    merged = merge(
        {
            "paddleocr": [ann(0, 0, 100, 100, "A"), ann(0, 20, 100, 100, "B")],
            "vlm": [ann(0, 5, 100, 100, "A")],
        }
    )
    by_text = {m.text: m for m in merged}
    assert by_text["A"].confidence == pytest.approx(0.95)
    assert by_text["B"].confidence == pytest.approx(0.45)


def test_inputs_not_mutated():
    primary, secondary = ann(0, 0, 10, 10, "x"), ann(0, 0, 10, 10, "x")
    merge({"a": [primary], "b": [secondary]})
    assert primary.attrs == {} and primary.confidence == 1.0
    assert secondary.attrs == {} and secondary.confidence == 1.0


def test_text_similarity_normalization():
    assert text_similarity("Hello  World", "hello world") == pytest.approx(1.0)
    assert text_similarity(None, None) == pytest.approx(1.0)
    assert text_similarity("abc", None) == pytest.approx(0.0)
