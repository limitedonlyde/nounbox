"""Modal recipe: OWLv2 open-vocabulary box detection on GPU as an HTTP-labeler backend.

Implements the labelers/http convention: POST /predict (image bytes) ->
{"annotations": [{geometry, label, confidence}]}. Config comes in the
X-Labeler-Config header ({"classes": ["cat", "dog"], "score_threshold": 0.25}).

This is the GPU twin of labelers/owlv2 (the CPU default of the "easy" mode):
same model, same thresholds, same post-processing, ~0.1-0.2 s per photo on a T4
instead of ~2 s on an M2 CPU. The boxes must come out IDENTICAL — a project
half-labeled on CPU and half on GPU is one dataset, and a human accepting a box
should not be able to tell which one drew it.

WHY THE POST-PROCESSING IS COPIED IN, not imported: exactly one .py file is
mounted into the Modal container, so the recipe may not import anything from the
monorepo (not nounbox_sdk either). The block between the sentinel comments below
is a byte-for-byte copy of the same block in
labelers/owlv2/nounbox_labeler_owlv2/labeler.py, and
server/tests/test_gpu_recipes.py compares the two as text. Edit it in the CPU
labeler and copy it over — never retype it here.

Two things in that block are load-bearing and easy to get wrong:
  - per-query emission. The stock post_process_grounded_object_detection keeps
    only the argmax class per box, so on a long class list part of the findings
    is relabeled with a neighbouring class (kitchen sink -> sink). Measured: 37
    of 301 boxes. We take every (patch, query) pair above the threshold instead.
  - OWLv2 pads the image to a SQUARE before rescaling, so pred_boxes are
    normalized against the square's side, not against the frame. Converting back
    goes via side = max(width, height) and then clipping to the edges.

The response deliberately carries NO "text" key: the HTTP labeler does
item.get("text"), so an absent key becomes text=None — exactly what the CPU
labeler produces for a detection box. Sending "" would store an empty string
instead. "confidence" and "label" are always present for the same reason: the
parser defaults a missing confidence to 1.0 (which would poison bulk-accept at
0.40) and a missing label to "text_line" (which no detection project has).

Auto-deploy from the platform (Settings -> "Connect GPU"): the `app` object sits
at module level, importing has no side effects and reads no local files, and the
deploy call is modal.runner.deploy_app(app, name=..., client=...). The file is
self-contained (EXACTLY this one .py is mounted into the Modal container), so
there must be no imports from the monorepo here.

Access token (optional): if the deploying process has NOUNBOX_GPU_TOKEN set, it
is baked into the app's Secret and /predict starts requiring
`Authorization: Bearer <token>` (config.api_key in the HTTP labeler). Without
the variable the endpoint stays open behind an unguessable URL, as before.

Deploy:  NOUNBOX_GPU_TOKEN=... modal deploy deploy/modal/owlv2_modal.py
Endpoint: https://<workspace>--nounbox-owlv2-fastapi-app.modal.run/predict
— goes into the HTTP labeler as config.endpoint.

Stop it: modal app stop nounbox-owlv2
"""

import os

import modal

APP_NAME = "nounbox-owlv2"
GPU = "T4"
TOKEN_ENV = "NOUNBOX_GPU_TOKEN"

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"
# the same two numbers the CPU engine and the README are built on: 0.25 to show
# to a human, 0.40 to accept in bulk (F1 0.823 on 79 photos / 323 objects)
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_NMS_IOU = 0.4

app = modal.App(APP_NAME)

# ~620 MB of weights per model: without the volume every cold start re-downloads
# them from HuggingFace, and the first photo of a batch looks like a hang
model_cache = modal.Volume.from_name("nounbox-owlv2-cache", create_if_missing=True)

# the token is read from the deploying process; no variable -> no key in the Secret
auth_secret = modal.Secret.from_dict({TOKEN_ENV: os.environ.get(TOKEN_ENV) or None})

image = (
    modal.Image.debian_slim(python_version="3.12")
    # CUDA wheels, NOT the /whl/cpu index the server image uses: this is the
    # whole point of the GPU path. Kept in its own pip_install because that
    # index carries torch only — transformers & co. are not published there.
    .pip_install("torch>=2.2", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=4.57",
        "Pillow>=10.0",
        "numpy>=1.26",
        # mandatory, not optional: Owlv2ImageProcessor calls
        # requires_backends(self, "scipy") for antialiased resizing. Without it
        # the recipe deploys fine and then fails on the first image.
        "scipy>=1.11",
        "fastapi[standard]>=0.115",
    )
    # HF_HOME points at the mounted volume, so the weights survive scale-to-zero
    .env({"HF_HOME": "/root/.cache/huggingface", "PYTHONUNBUFFERED": "1"})
)


# --- begin owlv2 post-processing (kept byte-identical in deploy/modal/owlv2_modal.py) ---
# boxes thinner than this after clipping to the frame are junk from edge patches
MIN_SIDE_PX = 2.0


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def nms_per_class(detections: list[dict], threshold: float) -> list[dict]:
    """Greedy duplicate suppression, SEPARATELY for each class.

    Different classes deliberately do not suppress each other: a carpet under
    a table is two objects in one place, and the human must see both.
    """
    by_class: dict[str, list[dict]] = {}
    for det in detections:
        by_class.setdefault(det["label"], []).append(det)

    kept: list[dict] = []
    for group in by_class.values():
        selected: list[dict] = []
        for det in sorted(group, key=lambda d: -d["score"]):
            if all(iou(det["bbox"], s["bbox"]) <= threshold for s in selected):
                selected.append(det)
        kept.extend(selected)
    return sorted(kept, key=lambda d: -d["score"])


def square_box_to_image(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """cxcywh normalized to the padded square -> xyxy in frame pixels."""
    side = max(width, height)
    cx, cy, bw, bh = box
    x1 = max(0.0, min((cx - bw / 2) * side, float(width)))
    y1 = max(0.0, min((cy - bh / 2) * side, float(height)))
    x2 = max(0.0, min((cx + bw / 2) * side, float(width)))
    y2 = max(0.0, min((cy + bh / 2) * side, float(height)))
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    return (x1, y1, x2, y2)
# --- end owlv2 post-processing ---


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": model_cache},
    secrets=[auth_secret],
    timeout=1800,
    scaledown_window=120,
    # Deliberately below the 10 concurrent GPUs a Modal Starter workspace
    # allows. Measured ceiling on a T4: ~296 images/min at this value, ~682 at
    # 10 — but the platform's labeling run posts images ONE AT A TIME, so today
    # neither ceiling is reachable through the product and the extra containers
    # would only widen the blast radius of a runaway job. Raise it together with
    # making the worker send requests concurrently, not before.
    max_containers=4,
)
@modal.asgi_app()
def fastapi_app():
    import hmac
    import io
    import json
    import threading

    from anyio.to_thread import run_sync
    from fastapi import FastAPI, HTTPException, Request
    from PIL import Image as PILImage

    token = os.environ.get(TOKEN_ENV, "").strip()

    # with the token on we also drop the auto-generated docs: they spell out
    # the contract of the endpoint we have just locked down
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None) if token else FastAPI()
    models: dict[str, tuple] = {}
    lock = threading.Lock()

    def check_auth(request: Request) -> None:
        if not token:
            return
        scheme, _, value = request.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            value.strip().encode(), token.encode()
        ):
            raise HTTPException(
                401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"}
            )

    def parse_config(request: Request) -> dict:
        raw = request.headers.get("X-Labeler-Config") or "{}"
        try:
            config = json.loads(raw)
        except ValueError:
            raise HTTPException(400, "X-Labeler-Config: invalid JSON")
        if not isinstance(config, dict):
            raise HTTPException(400, "X-Labeler-Config: JSON object expected")
        return config

    def read_float(config: dict, key: str, default: float) -> float:
        """config[key] as a float, absent/null -> the CPU engine's default.

        A garbage value is the caller's mistake, so it comes back as a readable
        400 instead of a 500 from float("0.4a") in the middle of a batch.
        """
        value = config.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            raise HTTPException(
                400, f"X-Labeler-Config: {key} must be a number, got {value!r}"
            )

    def get_model(model_name: str):
        with lock:
            if model_name not in models:
                import torch
                from transformers import AutoProcessor, Owlv2ForObjectDetection

                torch.set_grad_enabled(False)
                processor = AutoProcessor.from_pretrained(model_name)
                # float32, NOT fp16: half precision shifts the sigmoid scores
                # across the 0.25/0.40 thresholds the CPU numbers rest on, and
                # the two engines would stop agreeing on the same photo
                model = (
                    Owlv2ForObjectDetection.from_pretrained(
                        model_name, dtype=torch.float32
                    )
                    .eval()
                    .to("cuda")
                )
                models[model_name] = (processor, model)
                try:
                    # the weights have just landed in the mounted volume; commit
                    # so the next cold start reads them instead of pulling
                    # ~620 MB from HuggingFace again
                    model_cache.commit()
                except Exception:
                    # a failed commit costs a slower cold start, never a request
                    pass
            return models[model_name]

    def run_detect(
        body: bytes,
        classes: list[str],
        score_threshold: float,
        nms_iou: float,
        model_name: str,
    ) -> list[dict]:
        import torch

        try:
            with PILImage.open(io.BytesIO(body)) as raw:
                pil = raw.convert("RGB")
        except Exception:
            # UnidentifiedImageError/OSError/ValueError, and also
            # DecompressionBombError — it inherits straight from Exception.
            # 422, NOT 400: the HTTP labeler treats every 400-level answer as a
            # config error that will repeat on every frame and aborts the whole
            # batch, while the CPU engine merely skips a corrupt image. 422 is
            # excluded from that rule, so one bad file costs one frame.
            raise HTTPException(422, "Body is not a readable image")
        width, height = pil.size

        try:
            processor, model = get_model(model_name)
        except HTTPException:
            raise
        except Exception as exc:
            # Only a NON-default name can be the caller's fault. On the default
            # model a get_model failure is ours — a HuggingFace outage, a full
            # cache volume, CUDA OOM — and answering 400 would tell the labeler
            # the request is hopeless and kill the entire batch. Re-raise so
            # FastAPI returns 500 and the platform retries the frame instead.
            if model_name == DEFAULT_MODEL:
                raise
            raise HTTPException(
                400, f"Bad model {model_name!r}: {type(exc).__name__}: {exc}"
            )

        inputs = processor(text=[classes], images=pil, return_tensors="pt").to("cuda")
        # per call, not the one-time set_grad_enabled in get_model: grad mode
        # is thread-local and that ran once, in whichever thread lost the race
        # to load the model — every other FastAPI worker thread would still be
        # building a graph it never uses
        with torch.inference_mode():
            outputs = model(**inputs)

        # logits: [patches, queries] — probability of EVERY class in EVERY box;
        # take every pair above the threshold, not the argmax over the queries.
        # Both tensors go to the CPU once, up front: the loop below indexes them
        # element by element, and doing that on a CUDA tensor is one device sync
        # per detection. float32 on both sides, and the post-processing below
        # is the same source text, so the two engines agree to fp32 tolerance
        # — not bit for bit: CUDA and CPU kernels reduce in a different order.
        probs = torch.sigmoid(outputs.logits[0]).float().cpu()
        boxes = outputs.pred_boxes[0].float().cpu()

        detections: list[dict] = []
        for patch, query in (probs >= score_threshold).nonzero().tolist():
            box = square_box_to_image(tuple(boxes[patch].tolist()), width, height)
            if box is None:
                continue
            detections.append(
                {
                    "label": classes[query],
                    "bbox": box,
                    "score": float(probs[patch, query]),
                }
            )

        return nms_per_class(detections, nms_iou)

    def to_annotations(detections: list[dict]) -> list[dict]:
        """xyxy -> the HTTP-labeler wire format (x/y/width/height).

        No "text" key at all: the labeler reads item.get("text"), so absent
        becomes None — the same thing the CPU engine stores for a box.
        """
        return [
            {
                "geometry": {
                    "type": "bbox",
                    "x": det["bbox"][0],
                    "y": det["bbox"][1],
                    "width": det["bbox"][2] - det["bbox"][0],
                    "height": det["bbox"][3] - det["bbox"][1],
                },
                # the class string EXACTLY as it arrived: the platform matches
                # it against the project classes, and a lower()/strip() here
                # would send the annotation into export's skipped_labels
                "label": det["label"],
                "confidence": det["score"],
            }
            for det in detections
        ]

    @web.post("/predict")
    async def predict(request: Request):
        check_auth(request)
        config = parse_config(request)

        # "classes" is the key the platform already fills in for every detection
        # job (workers/tasks.py writes the project classes into the engine
        # config), and the same key the CPU engine reads
        classes = [str(c).strip() for c in (config.get("classes") or []) if str(c).strip()]
        if not classes:
            # an empty list is an error with a hint, not "label whatever you
            # find": staying quiet ends the job with zero annotations
            raise HTTPException(
                400, "owlv2: no classes given — add the project classes before labeling"
            )

        score_threshold = read_float(config, "score_threshold", DEFAULT_SCORE_THRESHOLD)
        nms_iou = read_float(config, "nms_iou", DEFAULT_NMS_IOU)
        model_name = str(config.get("model") or DEFAULT_MODEL)
        body = await request.body()

        detections = await run_sync(
            run_detect, body, classes, score_threshold, nms_iou, model_name
        )
        return {
            "annotations": to_annotations(detections),
            "meta": {"model": model_name, "classes": len(classes)},
        }

    @web.get("/health")
    async def health():
        return {
            "status": "ok",
            "auth": "bearer" if token else "open",
            "task": "detection",
            "model": DEFAULT_MODEL,
            "models_loaded": len(models),
        }

    return web
