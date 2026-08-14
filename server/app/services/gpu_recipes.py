"""Registry of the GPU recipes: one entry per engine name.

One engine name = one recipe file = one Modal app = one row in gpu_deployments
= one value of Annotation.source["name"]. That identity is deliberate: the
engine name is already the de facto key of everything durable (entry points,
the settings catalog, the re-run delete in workers/tasks.py, the
"already labeled by this engine" skip), so splitting a single name across two
models would make those queries mean two different things at once.

Why two apps instead of one recipe that dispatches on the task: the OCR image
(PaddleOCR, Paddle's own CUDA wheels) and the detection image (OWLv2, torch's
CUDA wheels) have nothing in common. Fusing them makes one ~12 GB image, drags
OCR cold starts down for code that did not change, and puts an unproven
dependency mix on the one GPU path that already works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings as app_config


@dataclass(frozen=True)
class GpuRecipe:
    """A recipe the platform can deploy into the user's own Modal account."""

    engine: str
    filename: str  # inside deploy/modal — mounted into the container as is
    task: str  # the project task type this recipe serves ("detection"/"ocr")
    title: str
    requires: str = "modal"
    # path the recipe's FastAPI app listens on; settings_store appends it to
    # the deployed app's root URL
    predict_path: str = "predict"
    # config for the post-deploy warm-up request. NOT decoration: the detection
    # recipe rejects a request without classes before it ever loads the model,
    # so warming it with an empty config would answer 400 in milliseconds and
    # leave the weights undownloaded — the exact hang the warm-up exists to
    # prevent, now hidden behind a job that says it warmed up.
    warmup_config: dict = field(default_factory=dict)


MODAL_GPU = "modal_gpu"
MODAL_GPU_DETECT = "modal_gpu_detect"

# modal_gpu keeps its name and its recipe: an installation that already
# deployed it has an OCR endpoint under that name, and renaming the engine
# would orphan every annotation whose source says "modal_gpu".
GPU_RECIPES: dict[str, GpuRecipe] = {
    MODAL_GPU: GpuRecipe(
        engine=MODAL_GPU,
        filename="paddleocr_modal.py",
        task="ocr",
        title="GPU OCR in your own Modal account (PaddleOCR PP-OCRv5)",
    ),
    MODAL_GPU_DETECT: GpuRecipe(
        engine=MODAL_GPU_DETECT,
        filename="owlv2_modal.py",
        task="detection",
        title="GPU boxes in your own Modal account (OWLv2)",
        # one throwaway class, and a threshold high enough that the 1x1 warm-up
        # PNG cannot produce an annotation worth serializing
        warmup_config={"classes": ["object"], "score_threshold": 0.99},
    ),
}


def get(engine: str) -> GpuRecipe:
    try:
        return GPU_RECIPES[engine]
    except KeyError:
        raise KeyError(
            f"Unknown GPU engine {engine!r}; known: {', '.join(GPU_RECIPES)}"
        ) from None


def configured_app_name(engine: str) -> str:
    """Modal app name from the environment, per engine (may be empty)."""
    get(engine)
    if engine == MODAL_GPU_DETECT:
        return app_config.modal_gpu_detect_app_name
    return app_config.modal_gpu_app_name


def configured_recipe_path(engine: str) -> str:
    """Recipe path override from the environment, per engine (may be empty).

    Deliberately NOT one shared variable: a single override pointing both
    deploys at one file would silently deploy the OCR recipe as the detection
    engine — exactly the failure this registry exists to prevent.
    """
    get(engine)
    if engine == MODAL_GPU_DETECT:
        return app_config.modal_gpu_detect_recipe_path
    return app_config.modal_gpu_recipe_path
