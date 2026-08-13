"""Modal-рецепт: PaddleOCR PP-OCRv5 (кириллица) на GPU как HTTP-labeler backend.

Реализует конвенцию labelers/http: POST /predict (image bytes) ->
{"annotations": [{geometry, label, text, confidence}]}. Конфиг пробрасывается
заголовком X-Labeler-Config ({"lang": "ru"}).

Модели (paddleocr 3.7, lang + ocr_version="PP-OCRv5"); детектор всегда
PP-OCRv5_server_det, rec выбирается по lang:
  ru | uk | be              -> eslav_PP-OCRv5_mobile_rec      (дефолт, lang="ru")
  bg | kk | ky | mk | tt | … -> cyrillic_PP-OCRv5_mobile_rec
  en                        -> en_PP-OCRv5_mobile_rec
  ch | japan | chinese_cht   -> PP-OCRv5_server_rec
Явная пара моделей — config {"det_model": ..., "rec_model": ...} (например,
rec_model="cyrillic_PP-OCRv5_mobile_rec" для русского вместо eslav).
ocr_version задаётся явно: без него paddleocr 3.7 уводит en/латиницу в PP-OCRv6,
где кириллицы нет вообще.
Полигоны построчные (4 точки на строку), confidence — реальный скор распознавания.
Порог text_rec_score_thresh=0.0: строки, которые движок не смог прочитать, не
выбрасываются молча, а уходят человеку в Review UI.

Автодеплой платформой («Настройки» -> «Подключить GPU»): объект `app` лежит на
уровне модуля, импорт не имеет побочных эффектов и не читает локальные файлы,
деплой — modal.runner.deploy_app(app, name=..., client=...). Файл самодостаточен
(в контейнер Modal монтируется РОВНО этот один .py), поэтому никаких импортов
из монорепо здесь быть не должно.

Токен доступа (опционально): если у процесса, который деплоит, задана переменная
AUTOLABELUI_GPU_TOKEN, она запекается в Secret приложения и /predict начинает
требовать `Authorization: Bearer <token>` (в HTTP-labeler это config.api_key).
Переменной нет — эндпоинт открыт по неугадываемому URL, как раньше.

Деплой:  AUTOLABELUI_GPU_TOKEN=... modal deploy deploy/modal/paddleocr_modal.py
Endpoint: https://<workspace>--autolabelui-paddleocr-fastapi-app.modal.run/predict
— подставляется в HTTP-labeler как config.endpoint.

Остановить: modal app stop autolabelui-paddleocr
"""

import os

import modal

APP_NAME = "autolabelui-paddleocr"
GPU = "T4"
TOKEN_ENV = "AUTOLABELUI_GPU_TOKEN"

DEFAULT_LANG = "ru"
DEFAULT_OCR_VERSION = "PP-OCRv5"
DEFAULT_DET_MODEL = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL = "eslav_PP-OCRv5_mobile_rec"

# сквозные параметры детекции/распознавания, которые paddleocr принимает на predict
PREDICT_PARAMS = (
    "text_det_limit_type",
    "text_det_limit_side_len",
    "text_det_thresh",
    "text_det_box_thresh",
    "text_det_unclip_ratio",
    "text_rec_score_thresh",
)

app = modal.App(APP_NAME)

model_cache = modal.Volume.from_name("autolabelui-paddlex-cache", create_if_missing=True)

# токен читается у процесса-деплойщика; нет переменной -> ключ в Secret не попадает
auth_secret = modal.Secret.from_dict({TOKEN_ENV: os.environ.get(TOKEN_ENV) or None})

image = (
    modal.Image.from_registry(
        # paddlepaddle-gpu 3.x требует Python >= 3.12
        "nvidia/cuda:12.6.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        # GPU-сборки paddle 3.x публикуются в официальном индексе Paddle, не на PyPI
        "paddlepaddle-gpu>=3.0,<4",
        extra_index_url="https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    )
    .pip_install(
        # версия зафиксирована: на ней проверена карта lang -> PP-OCRv5-модель
        "paddleocr==3.7.0",
        "fastapi[standard]>=0.115",
        "Pillow>=10.0",
        "numpy>=1.26",
    )
    # веса тянутся с HuggingFace (быстрее bcebos из регионов Modal) в /root/.paddlex
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

    # при включённом токене не публикуем и автодокументацию: она раскрывает
    # контракт эндпоинта, который мы только что закрыли
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
        # явные имена моделей (напр. rec_model=cyrillic_PP-OCRv5_mobile_rec)
        # отключают в paddleocr автоподбор по lang, поэтому задаём пару целиком
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
                    # классификатор ориентации строки переворачивает длинные
                    # кириллические строки на 180 градусов -> выключен
                    use_textline_orientation=False,
                    # 0.0: не терять строки, которые движок не смог прочитать
                    text_rec_score_thresh=0.0,
                    device="gpu",
                    **models,
                )
            return engines[key]

    def run_ocr(body: bytes, models: dict, params: dict) -> list[dict]:
        try:
            arr = np.asarray(PILImage.open(io.BytesIO(body)).convert("RGB"))
        except Exception:
            # UnidentifiedImageError/OSError/ValueError, а также
            # DecompressionBombError — она наследуется прямо от Exception
            raise HTTPException(400, "Body is not a readable image")

        try:
            ocr = get_engine(models)
        except HTTPException:
            raise
        except Exception as exc:
            # неизвестное имя модели или язык — вина запроса, не сервера
            raise HTTPException(
                400, f"Bad model config {models}: {type(exc).__name__}: {exc}"
            )
        annotations = []
        for res in ocr.predict(input=arr, **params) or []:
            data = res.json if hasattr(res, "json") else res
            inner = data.get("res", data) if isinstance(data, dict) else data
            texts = inner.get("rec_texts", [])
            scores = inner.get("rec_scores", [])
            # rec_polys выровнены с rec_texts/rec_scores; dt_polys — все детекции
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
