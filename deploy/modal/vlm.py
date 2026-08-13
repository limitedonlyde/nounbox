"""Modal-рецепт: Qwen2.5-VL как OpenAI-compatible endpoint через vLLM.

Для стандартных моделей проще managed Modal Endpoints (OpenAI-compatible,
без своего кода) — этот рецепт нужен для кастома: своя модель, свои
параметры vLLM, свой прогрев.

Деплой:  modal deploy deploy/modal/vlm.py
После деплоя endpoint вида https://<workspace>--nounbox-vlm-serve.modal.run
подставляется в VLM-labeler как base_url (+ "/v1"), model = MODEL ниже.

Деньги: gpu="L4", scale-to-zero (scaledown_window) — платим только за разметку.
Первый запуск: сборка образа + скачивание модели (~15GB) — минуты; дальше
модель кешируется в Volume. Остановить: modal app stop nounbox-vlm

Производственное замечание: endpoint публичный по неугадываемому URL.
Для защиты добавьте проверку Authorization в обёртку или modal ProxyAuth.
"""

import modal

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
GPU = "L4"  # 24GB — достаточно для 7B bf16 + KV-cache

app = modal.App("nounbox-vlm")

hf_cache = modal.Volume.from_name("nounbox-hf-cache", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "vllm>=0.10",
    "transformers>=4.49",
    "qwen-vl-utils>=0.0.10",
)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=3600,
    scaledown_window=300,  # 5 минут простоя -> scale to zero
)
@modal.web_server(port=8000, startup_timeout=1500)
def serve():
    import subprocess

    subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--max-model-len",
            "8192",
            "--limit-mm-per-prompt",
            '{"image": 1}',
        ]
    )
