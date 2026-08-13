"""Modal recipe: PaddleOCR PP-OCRv5 (Cyrillic) on GPU as an HTTP-labeler backend.

Implements the labelers/http convention: POST /predict (image bytes) ->
{"annotations": [{geometry, label, text, confidence}]}. Config is passed through
in the X-Labeler-Config header ({"lang": "ru"}).

Models (paddleocr 3.7, lang + ocr_version="PP-OCRv5"); the detector is always
PP-OCRv5_server_det, rec is chosen by lang:
  ru | uk | be              -> eslav_PP-OCRv5_mobile_rec      (default, lang="ru")
  bg | kk | ky | mk | tt | … -> cyrillic_PP-OCRv5_mobile_rec
  en                        -> en_PP-OCRv5_mobile_rec
  ch | japan | chinese_cht   -> PP-OCRv5_server_rec
An explicit model pair — config {"det_model": ..., "rec_model": ...} (e.g.
rec_model="cyrillic_PP-OCRv5_mobile_rec" for Russian instead of eslav).
ocr_version is set explicitly: without it paddleocr 3.7 routes en/Latin to
PP-OCRv6, which has no Cyrillic at all.
Polygons are per line (4 points per line), confidence is the real recognition
score. The text_rec_score_thresh=0.0 threshold keeps lines the engine could not
read from being dropped silently — they go to a human in the Review UI instead.

Auto-deploy from the platform (Settings -> "Connect GPU"): the `app` object sits
at module level, importing has no side effects and reads no local files, and the
deploy call is modal.runner.deploy_app(app, name=..., client=...). The file is
self-contained (EXACTLY this one .py is mounted into the Modal container), so
there must be no imports from the monorepo here.

Access token (optional): if the deploying process has NOUNBOX_GPU_TOKEN set, it
is baked into the app's Secret and /predict starts requiring
`Authorization: Bearer <token>` (config.api_key in the HTTP labeler). Without
the variable the endpoint stays open behind an unguessable URL, as before.

Deploy:  NOUNBOX_GPU_TOKEN=... modal deploy deploy/modal/paddleocr_modal.py
Endpoint: https://<workspace>--nounbox-paddleocr-fastapi-app.modal.run/predict
— goes into the HTTP labeler as config.endpoint.

Stop it: modal app stop nounbox-paddleocr
"""

import os

import modal

APP_NAME = "nounbox-paddleocr"
GPU = "T4"
TOKEN_ENV = "NOUNBOX_GPU_TOKEN"

DEFAULT_LANG = "ru"
DEFAULT_OCR_VERSION = "PP-OCRv5"
DEFAULT_DET_MODEL = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL = "eslav_PP-OCRv5_mobile_rec"

# pass-through detection/recognition parameters that paddleocr accepts on predict
PREDICT_PARAMS = (
    "text_det_limit_type",
    "text_det_limit_side_len",
    "text_det_thresh",
    "text_det_box_thresh",
    "text_det_unclip_ratio",
    "text_rec_score_thresh",
)

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("nounbox-paddlex-cache", create_if_missing=True)

# the token is read from the deploying process; no variable -> no key in the Secret
auth_secret = modal.Secret.from_dict({TOKEN_ENV: os.environ.get(TOKEN_ENV) or None})

image = (
    modal.Image.from_registry(
        # paddlepaddle-gpu 3.x requires Python >= 3.12
        "nvidia/cuda:12.6.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        # paddle 3.x GPU builds are published in Paddle's own index, not on PyPI
        "paddlepaddle-gpu>=3.0,<4",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .pip_install(
        # version pinned: the lang -> PP-OCRv5 model map was verified on it
        "paddleocr==3.7.0",
        "fastapi[standard]>=0.115",
        "Pillow>=10.0",
        "numpy>=1.26",
    )
    # weights come from HuggingFace into /root/.paddlex — faster than bcebos
    # from Modal's own regions
    .env({"PADDLE_PDX_MODEL_SOURCE": "huggingface", "PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.paddlex": model_cache},
    secrets=[auth_secret],
    timeout=1800,
    scaledown_window=120,
    max_containers=4,
)
@modal.asgi_app()
def fastapi_app():
    import hmac
    import io
    import json
    import threading

    import numpy as np
    from anyio.to_thread import run_sync
    from fastapi import FastAPI, HTTPException, Request
    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

    token = os.environ.get(TOKEN_ENV, "").strip()

    # with the token on we also drop the auto-generated docs: they spell out
    # the contract of the endpoint we have just locked down
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None) if token else FastAPI()
    engines: dict[tuple, object] = {}
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

    def model_kwargs(config: dict) -> dict:
        # explicit model names (e.g. rec_model=cyrillic_PP-OCRv5_mobile_rec)
        # turn off paddleocr's lang auto-pick, so we set the whole pair here
        det_model, rec_model = config.get("det_model"), config.get("rec_model")
        if det_model or rec_model:
            return {
                "text_detection_model_name": str(det_model or DEFAULT_DET_MODEL),
                "text_recognition_model_name": str(rec_model or DEFAULT_REC_MODEL),
            }
        return {
            "lang": str(config.get("lang") or DEFAULT_LANG).lower(),
            "ocr_version": str(config.get("ocr_version") or DEFAULT_OCR_VERSION),
        }

    def get_engine(models: dict):
        key = tuple(sorted(models.items()))
        with lock:
            if key not in engines:
                from paddleocr import PaddleOCR

                engines[key] = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    # the textline orientation classifier flips long Cyrillic
                    # lines by 180 degrees -> turned off
                    use_textline_orientation=False,
                    # 0.0: do not lose lines the engine could not read
                    text_rec_score_thresh=0.0,
                    device="gpu",
                    **models,
                )
            return engines[key]

    def run_ocr(body: bytes, models: dict, params: dict) -> list[dict]:
        try:
            arr = np.asarray(PILImage.open(io.BytesIO(body)).convert("RGB"))
        except Exception:
            # UnidentifiedImageError/OSError/ValueError, and also
            # DecompressionBombError — it inherits straight from Exception
            raise HTTPException(400, "Body is not a readable image")

        try:
            ocr = get_engine(models)
        except HTTPException:
            raise
        except Exception as exc:
            # unknown model name or language — the request's fault, not ours
            raise HTTPException(
                400, f"Bad model config {models}: {type(exc).__name__}: {exc}"
            )
        annotations = []
        for res in ocr.predict(input=arr, **params) or []:
            data = res.json if hasattr(res, "json") else res
            inner = data.get("res", data) if isinstance(data, dict) else data
            texts = inner.get("rec_texts", [])
            scores = inner.get("rec_scores", [])
            # rec_polys lines up with rec_texts/rec_scores; dt_polys holds
            # all detections
            polys = inner.get("rec_polys")
            if polys is None:
                polys = inner.get("dt_polys", [])
            for poly, text, score in zip(polys, texts, scores):
                annotations.append(
                    {
                        "geometry": {
                            "type": "polygon",
                            "points": [[float(x), float(y)] for x, y in np.asarray(poly)],
                        },
                        "label": "text_line",
                        "text": str(text),
                        "confidence": float(score),
                    }
                )
        return annotations

    @web.post("/predict")
    async def predict(request: Request):
        check_auth(request)
        config = parse_config(request)
        models = model_kwargs(config)
        params = {k: config[k] for k in PREDICT_PARAMS if config.get(k) is not None}
        body = await request.body()

        annotations = await run_sync(run_ocr, body, models, params)
        return {"annotations": annotations, "meta": models}

    @web.get("/health")
    async def health():
        return {
            "status": "ok",
            "auth": "bearer" if token else "open",
            "default_lang": DEFAULT_LANG,
            "ocr_version": DEFAULT_OCR_VERSION,
            "engines_loaded": len(engines),
        }

    return web
