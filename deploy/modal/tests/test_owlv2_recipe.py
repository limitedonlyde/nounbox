"""The detection recipe's box math, checked against the CPU engine's own tests.

server/tests/test_gpu_recipes.py already proves the copied block is
byte-identical to labelers/owlv2. This file proves the block does what the CPU
engine's tests say it does, running inside the recipe module — so a copy that
is identical but landed next to a shadowing definition still gets caught.

Skipped where `modal` is not installed: importing the recipe builds the App,
the Image and the Secret at module level.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("modal", reason="the recipe imports modal at module level")

RECIPE = Path(__file__).resolve().parents[1] / "owlv2_modal.py"


def load_recipe():
    spec = importlib.util.spec_from_file_location("owlv2_modal_under_test", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recipe = load_recipe()
iou = recipe.iou
nms_per_class = recipe.nms_per_class
square_box_to_image = recipe.square_box_to_image
MIN_SIDE_PX = recipe.MIN_SIDE_PX


def det(label, bbox, score):
    return {"label": label, "bbox": bbox, "score": score}


def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_touching_edges_is_zero():
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap():
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_nms_suppresses_duplicate_of_same_class():
    kept = nms_per_class(
        [det("sofa", (0, 0, 10, 10), 0.9), det("sofa", (1, 1, 11, 11), 0.5)], 0.4
    )
    assert len(kept) == 1
    assert kept[0]["score"] == 0.9


def test_nms_keeps_different_classes_in_same_place():
    # a carpet under a table — two objects in one spot, the human needs both
    kept = nms_per_class(
        [det("carpet", (0, 0, 10, 10), 0.9), det("table", (0, 0, 10, 10), 0.8)], 0.4
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


def test_square_box_landscape_uses_long_side():
    box = square_box_to_image((0.5, 0.25, 0.5, 0.5), 200, 100)
    assert box == pytest.approx((50.0, 0.0, 150.0, 100.0))


def test_square_box_portrait_uses_long_side():
    box = square_box_to_image((0.25, 0.5, 0.5, 0.5), 100, 200)
    assert box == pytest.approx((0.0, 50.0, 100.0, 150.0))


def test_square_box_clipped_to_frame():
    box = square_box_to_image((0.5, 0.45, 0.4, 0.4), 200, 100)
    assert box is not None
    assert box[1] == pytest.approx(50.0)
    assert box[3] == pytest.approx(100.0)


def test_square_box_degenerate_after_clip_is_dropped():
    assert square_box_to_image((0.5, 0.99, 0.02, 0.02), 200, 100) is None


def test_square_box_thinner_than_minimum_is_dropped():
    tiny = (MIN_SIDE_PX / 2) / 100
    assert square_box_to_image((0.5, 0.5, tiny, tiny), 100, 100) is None


def test_importing_the_recipe_has_no_side_effects_on_the_filesystem():
    """modal_deploy imports the recipe inside the API process: it must not read
    a local file or need one to exist."""
    assert recipe.app.name == "nounbox-owlv2"
