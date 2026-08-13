FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Light core by default: heavy inference goes to Modal/Ollama via vlm/http.
# Local PaddleOCR (paddlepaddle+paddleocr, gigabytes) is opt-in: --build-arg WITH_PADDLE=1
ARG WITH_PADDLE=0

# libGL/libglib are opencv runtime deps, needed only by local PaddleOCR
RUN if [ "$WITH_PADDLE" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /code

# SDK and labelers come from the monorepo (build context is the repo root)
COPY sdk /sdk
COPY server/pyproject.toml /code/pyproject.toml
COPY server/app /code/app
COPY labelers /labelers

# SDK and labelers are installed editable: in dev mode ./sdk and ./labelers
# are mounted as volumes.
# The default set is rapidocr (the "easy" mode, CPU) plus the thin
# clients (vlm/http/consensus); local paddleocr only with WITH_PADDLE=1
# torch comes from a CPU wheel off a separate index: the default one pulls in
# gigabytes of CUDA, and our GPU inference goes to Modal, not this container
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2"

RUN pip install --no-cache-dir -e /sdk -e /labelers/owlv2 -e /labelers/llmdet \
        -e /labelers/rapidocr -e /labelers/vlm \
        -e /labelers/http -e /labelers/consensus /code \
    && if [ "$WITH_PADDLE" = "1" ]; then \
        pip install --no-cache-dir -e /labelers/paddleocr; \
    fi \
    # rapidocr pulls in opencv-python, which needs X11 libraries (libxcb, libGL)
    # and fails on import cv2 in a slim image. The headless build gives the same
    # cv2 module without the windowing deps — and weighs less.
    && pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true \
    && pip install --no-cache-dir "opencv-python-headless>=4.10" \
    && python -c "import cv2, rapidocr; print('cv2', cv2.__version__)"

# The GPU recipe the platform deploys into the user's Modal account
# (the "Connect GPU" button); overridden by MODAL_GPU_RECIPE_PATH
COPY deploy/modal /deploy/modal

EXPOSE 8000
