"""The GPU recipe registry, and the anti-drift check on the copied box math.

The detection recipe may not import the CPU labeler (Modal mounts exactly one
.py), so the post-processing lives in both files. Nothing but a test keeps them
the same, and "the same" has to mean byte-identical: a stray rounding change in
square_box_to_image would move boxes between the two engines, and a project
labeled half on CPU and half on GPU would stop being one dataset.

Deliberately pure text — no imports of either module, no torch, no modal. This
runs everywhere, including the CI job that has none of the engine extras.
"""

from pathlib import Path

import pytest

from app.services import modal_deploy
from app.services.gpu_recipes import GPU_RECIPES, MODAL_GPU, MODAL_GPU_DETECT

BEGIN = (
    "# --- begin owlv2 post-processing "
    "(kept byte-identical in deploy/modal/owlv2_modal.py) ---"
)
END = "# --- end owlv2 post-processing ---"

REPO = Path(__file__).resolve().parents[2]
CPU_LABELER = REPO / "labelers" / "owlv2" / "nounbox_labeler_owlv2" / "labeler.py"
RECIPE = REPO / "deploy" / "modal" / "owlv2_modal.py"


def sentinel_block(path: Path) -> str:
    text = path.read_text()
    start = text.find(BEGIN)
    end = text.find(END)
    assert start != -1, f"{path} has no begin sentinel"
    assert end > start, f"{path} has no end sentinel after the begin one"
    return text[start + len(BEGIN) : end]


def test_copied_post_processing_is_byte_identical():
    cpu = sentinel_block(CPU_LABELER)
    gpu = sentinel_block(RECIPE)

    assert cpu == gpu, (
        "The OWLv2 post-processing has drifted between the CPU labeler and the "
        "Modal recipe. Edit it in the labeler and copy the block over verbatim."
    )


def test_the_block_actually_contains_the_box_math():
    """Guards the guard: two empty blocks are also byte-identical."""
    block = sentinel_block(RECIPE)

    for name in ("MIN_SIDE_PX", "def iou(", "def nms_per_class(", "def square_box_to_image("):
        assert name in block


def test_every_engine_has_a_recipe_file():
    for engine in GPU_RECIPES:
        assert modal_deploy.recipe_path(engine).is_file()


def test_recipes_do_not_share_a_file():
    files = {r.filename for r in GPU_RECIPES.values()}

    assert len(files) == len(GPU_RECIPES)


@pytest.mark.parametrize(
    ("engine", "task", "model"),
    [(MODAL_GPU, "ocr", "PaddleOCR"), (MODAL_GPU_DETECT, "detection", "Owlv2")],
)
def test_engine_task_matches_what_the_recipe_serves(engine, task, model):
    """The whole defect in one assertion: an engine must not advertise a task
    its deployed recipe cannot serve. modal_gpu resolving to a PaddleOCR file
    while claiming detection is exactly what sent photos to an OCR endpoint."""
    assert GPU_RECIPES[engine].task == task
    assert model in modal_deploy.recipe_path(engine).read_text()


def test_detection_recipe_reports_its_task_on_health():
    # the post-deploy check a human can run: curl {endpoint}/health
    assert '"task": "detection"' in modal_deploy.recipe_path(MODAL_GPU_DETECT).read_text()
