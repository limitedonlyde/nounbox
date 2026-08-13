"""Постобработка OWLv2 — чистые функции, без загрузки модели."""

import pytest

from nounbox_labeler_owlv2.labeler import (
    MIN_SIDE_PX,
    iou,
    nms_per_class,
    square_box_to_image,
    to_annotations,
)


def det(label, bbox, score):
    return {"label": label, "bbox": bbox, "score": score}


def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_touching_edges_is_zero():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap():
    # пересечение 5x10=50, объединение 10*10 + 10*10 - 50 = 150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_nms_suppresses_duplicate_of_same_class():
    kept = nms_per_class(
        [det("sofa", (0, 0, 10, 10), 0.9), det("sofa", (1, 1, 11, 11), 0.5)], 0.4
    )
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9  # остаётся более уверенная


def test_nms_keeps_different_classes_in_same_place():
    # ковёр под столом — два объекта в одной точке, оба нужны человеку
    kept = nms_per_class(
        [det("carpet", (0, 0, 10, 10), 0.9), det("table", (0, 0, 10, 10), 0.8)], 0.4
    )
    assert len(kept) == 2


def test_nms_keeps_distinct_instances_of_same_class():
    kept = nms_per_class(
        [det("chair", (0, 0, 10, 10), 0.9), det("chair", (50, 50, 60, 60), 0.8)], 0.4
    )
    assert len(kept) == 2


def test_nms_output_sorted_by_score():
    kept = nms_per_class(
        [
            det("a", (0, 0, 10, 10), 0.3),
            det("b", (50, 0, 60, 10), 0.9),
            det("c", (0, 50, 10, 60), 0.6),
        ],
        0.4,
    )
    assert [d["score"] for d in kept] == [0.9, 0.6, 0.3]


def test_nms_keeps_pair_at_exactly_threshold():
    # перекрытие ровно на пороге НЕ подавляется (условие подавления — строго больше)
    a, b = (0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)
    overlap = iou(a, b)  # 50/150 = 1/3
    assert nms_per_class([det("x", a, 0.9), det("x", b, 0.5)], overlap) != []
    assert len(nms_per_class([det("x", a, 0.9), det("x", b, 0.5)], overlap)) == 2
    # чуть более строгий порог — дубль подавляется
    assert len(nms_per_class([det("x", a, 0.9), det("x", b, 0.5)], overlap - 0.01)) == 1


def test_square_box_landscape_uses_long_side():
    # кадр 200x100: квадрат 200, бокс в центре половинного размера
    box = square_box_to_image((0.5, 0.25, 0.5, 0.5), 200, 100)
    assert box == pytest.approx((50.0, 0.0, 150.0, 100.0))


def test_square_box_portrait_uses_long_side():
    box = square_box_to_image((0.25, 0.5, 0.5, 0.5), 100, 200)
    assert box == pytest.approx((0.0, 50.0, 100.0, 150.0))


def test_square_box_clipped_to_frame():
    # кадр 200x100, квадрат 200: бокс y 50..130 наполовину уходит в паддинг
    box = square_box_to_image((0.5, 0.45, 0.4, 0.4), 200, 100)
    assert box is not None
    assert box[1] == pytest.approx(50.0)
    assert box[3] == pytest.approx(100.0)  # обрезано по нижней границе кадра


def test_square_box_degenerate_after_clip_is_dropped():
    # целиком в области паддинга снизу — после клипа высота нулевая
    assert square_box_to_image((0.5, 0.99, 0.02, 0.02), 200, 100) is None


def test_square_box_thinner_than_minimum_is_dropped():
    tiny = (MIN_SIDE_PX / 2) / 100
    assert square_box_to_image((0.5, 0.5, tiny, tiny), 100, 100) is None


def test_to_annotations_converts_xyxy_to_bbox_width_height():
    anns = to_annotations([det("sofa", (10.0, 20.0, 50.0, 60.0), 0.77)])
    assert len(anns) == 1
    ann = anns[0]
    assert ann.label == "sofa"
    assert ann.text is None
    assert ann.confidence == pytest.approx(0.77)
    assert ann.geometry.x == pytest.approx(10.0)
    assert ann.geometry.y == pytest.approx(20.0)
    assert ann.geometry.width == pytest.approx(40.0)
    assert ann.geometry.height == pytest.approx(40.0)


def test_empty_detections_give_empty_annotations():
    assert to_annotations(nms_per_class([], 0.4)) == []
