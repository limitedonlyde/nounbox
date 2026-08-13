"""Modal recipe: Qwen2.5-VL as an OpenAI-compatible endpoint via vLLM.

For stock models managed Modal Endpoints are simpler (OpenAI-compatible,
no code of your own) — this recipe is for the custom case: your own model,
your own vLLM parameters, your own warmup.

Deploy:  modal deploy deploy/modal/vlm.py
Once deployed, an endpoint like https://<workspace>--nounbox-vlm-serve.modal.run
goes into the VLM labeler as base_url (+ "/v1"), model = MODEL below.

Money: gpu="L4", scale-to-zero (scaledown_window) — we pay only while labeling.
First run: image build + model download (~15GB) takes minutes; after that the
model is cached in the Volume. To stop: modal app stop nounbox-vlm

Production note: the endpoint is public, guarded only by an unguessable URL.
To secure it, add an Authorization check in the wrapper, or use modal ProxyAuth.
"""

import modal

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
GPU = "L4"  # 24GB — enough for 7B bf16 + KV cache

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
    scaledown_window=300,  # 5 minutes idle -> scale to zero
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
