FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Лёгкое ядро по умолчанию: тяжёлый инференс — на Modal/Ollama через vlm/http.
# Локальный PaddleOCR (paddlepaddle+paddleocr, гигабайты) — opt-in: --build-arg WITH_PADDLE=1
ARG WITH_PADDLE=0

# libGL/libglib — рантайм-зависимости opencv, нужны только локальному PaddleOCR
RUN if [ "$WITH_PADDLE" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /code

# SDK и labeler'ы ставятся из монорепо (build context — корень проекта)
COPY sdk /sdk
COPY server/pyproject.toml /code/pyproject.toml
COPY server/app /code/app
COPY labelers /labelers

# SDK и labeler'ы — editable: в dev-режиме ./sdk и ./labelers смонтированы volume'ами.
# Дефолт — rapidocr (режим «просто», CPU) + тонкие клиенты (vlm/http/consensus);
# локальный paddleocr — только при WITH_PADDLE=1
# torch ставим CPU-колесом отдельным индексом: дефолтное тянет CUDA на гигабайты,
# а GPU-инференс у нас уходит на Modal, а не в этот контейнер
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2"

RUN pip install --no-cache-dir -e /sdk -e /labelers/owlv2 -e /labelers/llmdet \
        -e /labelers/rapidocr -e /labelers/vlm \
        -e /labelers/http -e /labelers/consensus /code \
    && if [ "$WITH_PADDLE" = "1" ]; then \
        pip install --no-cache-dir -e /labelers/paddleocr; \
    fi \
    # rapidocr тянет opencv-python, который требует X11-библиотек (libxcb, libGL)
    # и падает в slim-образе на import cv2. headless-сборка даёт тот же модуль
    # cv2 без оконных зависимостей — и весит меньше.
    && pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir "opencv-python-headless>=4.10" \
    && python -c "import cv2, rapidocr; print('cv2', cv2.__version__)"

# GPU-рецепт, который платформа разворачивает в аккаунт Modal пользователя
# (кнопка «Подключить GPU»); переопределяется MODAL_GPU_RECIPE_PATH
COPY deploy/modal /deploy/modal

EXPOSE 8000
