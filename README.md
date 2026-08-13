# AutoLabelUi

**Label Studio, tuned for OCR, with autolabel-first UX.**

Self-hosted platform for building OCR training datasets: drop in files in any
format, get automatic pre-labeling from pluggable engines, review it fast with
a keyboard-driven UI, export in training-ready formats.

> **Status: v0.2 — early stage.** The core loop (ingest → autolabel → review →
> export) works end-to-end, but APIs and the DB schema are still moving,
> there is no auth yet (single-user), and migrations/tests are being
> established. Expect rough edges.

## Key features

- **Ingest anything** — PNG/JPEG/WebP, multi-page TIFF/GIF, HEIC, PDF, ZIP
  archives. Everything is normalized to PNG, deduplicated (SHA-256), and
  blur-scored on the way in.
- **Pre-labeling in one click** — RapidOCR on CPU out of the box, or your own
  GPU on Modal for the hard pages; also any VLM behind an OpenAI-compatible
  API and a generic HTTP labeler for arbitrary backends. Every annotation
  carries provenance: engine name/version, confidence, review status.
- **Review UI built for speed** — lowest-confidence-first queue, bulk accept
  by confidence threshold, polygon overlay, hotkeys for everything
  (↑↓ select, ←→ pages, A accept, R reject, E edit text, D draw box).
- **Export** — `paddleocr_det`, `paddleocr_rec`, `coco`. Only reviewed
  (accepted/edited) annotations are exported, and every ZIP includes a
  snapshot manifest for dataset versioning.
- **Plugin SDK** — a labeler is one method: `predict(image, config) ->
  list[Annotation]`. Register a Python entry point and the platform picks it
  up.

## Quick start

Requires Docker with the compose plugin.

```bash
cp .env.example .env
docker compose -f docker-compose.yml up -d --build
```

Open http://localhost:8080, create a project, upload files, hit **Autolabel**,
review, export. That is the release configuration: the frontend is a static
bundle served by nginx, which also proxies `/api` to the backend — no dev
server, no source mounts.

For development, drop the `-f` flag. Compose then also picks up
`docker-compose.override.yml`, which swaps in the Vite dev server with HMR,
`uvicorn --reload` and live-mounted source:

```bash
docker compose up -d --build
```

- Dev UI: http://localhost:5173 · API docs: http://localhost:8000/docs
- MinIO console (dev only): http://localhost:9001

Both modes bind their ports to `127.0.0.1`. Put a TLS-terminating reverse
proxy in front before exposing anything, and set `S3_PUBLIC_ENDPOINT_URL` to
an address the browser can reach — image previews are presigned URLs opened
directly against MinIO, so they cannot be proxied under a path prefix (the
SigV4 signature covers host and path).

## Two ways to label

AutoLabelUi ships with two modes, and you pick per project — no config files
either way.

### Simple (default): CPU, works out of the box

`docker compose up -d` and hit **Autolabel**. The default engine is **RapidOCR**
(PP-OCRv5 detection + recognition on onnxruntime): line-level 4-point polygons,
real per-line confidence, Russian and Latin scripts, ~1 s per A4 page on a
laptop CPU. No accounts, no API keys, no GPU. Model weights (~13 MB) download
once into a Docker volume and it runs fully offline afterwards.

### Advanced: your own GPU on Modal

For skewed scans, photos, dense layouts or large batches, open **Settings**,
paste a Modal API token (`modal token new`) and press **Connect GPU**. The
platform deploys [`deploy/modal/paddleocr_modal.py`](deploy/modal/paddleocr_modal.py)
into *your* Modal account, secures the endpoint with a generated bearer token,
and the **GPU** engine appears in the engine list. You pay Modal directly for
the GPU seconds you use; it scales to zero when idle.

Both modes produce the **same kind of output** — line-level polygons plus text —
so switching does not change the shape of your dataset.

### Other engines

- **VLM** — any OpenAI-compatible vision endpoint (Modal, OpenRouter, Ollama,
  vLLM). Good at reading text and at KIE; **not** a reliable source of boxes —
  general chat VLMs place line boxes poorly (measured: mean IoU 0.285, table
  cells 0.00). Use it to fill in text, not geometry.
- **HTTP** — any backend behind a tiny convention, for your own inference server.
- **Consensus** — meta-engine: runs several engines and turns their agreement
  (IoU + text similarity) into a real confidence score.
- **PaddleOCR (local, CPU)** — opt-in, rebuild with
  `WITH_PADDLE=1 docker compose up -d --build` (adds several GB to the image).

## Architecture

Monorepo layout:

```
AutoLabelUi/
├── docker-compose.yml    # postgres, minio, redis, api, worker, web
├── sdk/                  # pip package: labeler plugin contract (autolabelui-sdk)
├── server/               # FastAPI + SQLAlchemy (async) + arq worker
│   └── app/services/     #   ingest.py (format converters), export.py (dataset formats)
├── web/                  # React + TypeScript (Vite) — Review UI
├── labelers/             # engine plugins (pip packages, entry points)
│   ├── rapidocr/         #   DEFAULT: PP-OCRv5 det+rec on onnxruntime (CPU)
│   ├── paddleocr/        #   heavyweight local det+rec (CPU, opt-in via WITH_PADDLE=1)
│   ├── vlm/              #   any VLM via OpenAI-compatible API
│   ├── http/             #   generic HTTP labeler (any backend behind one convention)
│   └── consensus/        #   meta-labeler: engine agreement -> real confidence
└── deploy/
    └── modal/            # Modal recipes for serverless GPU inference
        ├── vlm.py              # vLLM + Qwen2.5-VL (OpenAI-compatible endpoint)
        └── paddleocr_modal.py  # PaddleOCR on GPU (HTTP-labeler convention)
```

One universal annotation model covers detection / recognition / layout / KIE:
geometry (bbox/polygon), label, text, attrs, plus provenance
(source, confidence, status). Exporters are adapters over this model, so new
formats are cheap to add.

## Writing a labeler plugin

A labeler is a plain class implementing the SDK protocol — no inheritance
required:

```python
from autolabelui_sdk import Annotation, BBox, Capability

class MyLabeler:
    name = "my-ocr"
    version = "0.1.0"
    capabilities = {Capability.DETECTION, Capability.RECOGNITION}

    def predict(self, image: bytes, config: dict) -> list[Annotation]:
        return [Annotation(geometry=BBox(10, 10, 100, 30), text="hello", confidence=0.95)]
```

```toml
# your plugin's pyproject.toml
[project.entry-points."autolabelui.labelers"]
my_ocr = "my_package:MyLabeler"
```

Install the package next to the worker and it appears in the engine list. See
[`sdk/README.md`](sdk/README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Security notes

- **Change the default credentials.** Everything in `.env.example` is for
  local development only.
- Service ports are bound to localhost by default — put a reverse proxy with
  TLS and auth in front before exposing anything.
- When deploying beyond localhost, set `S3_PUBLIC_ENDPOINT_URL` so presigned
  image URLs point at an address the browser can reach.
- The GPU endpoint deployed from **Settings** is protected with a bearer token
  generated by the platform. Recipes you deploy by hand (`modal deploy ...`)
  are public at unguessable URLs unless you set `AUTOLABELUI_GPU_TOKEN` —
  anyone who learns the URL can spend your GPU budget.
- Your Modal token is encrypted at rest; the encryption key is generated on
  first start and kept in a Docker volume, so back it up with your database.

## License

[AGPL-3.0](LICENSE). If you run a modified version as a network service, you
must offer its source code to the users of that service.
