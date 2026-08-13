# AutoLabelUi

**Name your classes in plain English. Get boxes. Fix what is wrong. Export.**

Self-hosted auto-labeling for object detection. No accounts, no API keys, no
GPU, and nothing leaves your machine.

![AutoLabelUi in action](docs/demo.gif)

```bash
git clone https://github.com/USER/AutoLabelUi && cd AutoLabelUi
cp .env.example .env
docker compose -f docker-compose.yml up -d --build
```

Open http://localhost:8080, create a project, type the classes you care about,
drop in photos, press **Autolabel**.

> **Status: early.** The core loop works end to end and is covered by tests, but
> the API and the database schema still move between releases, there is no
> multi-user support, and migrations are not in place yet. Expect rough edges,
> and pin a commit if you depend on it.

## Why this exists

Every annotation tool assumes you already have a model. This one assumes you do
not: you describe what you are looking for in words, and an open-vocabulary
detector draws the first pass. Your job shrinks from *drawing* thousands of
boxes to *checking* them.

The classes are arbitrary. `microwave oven`, `slow cooker`, `fire hydrant`,
`forklift` — nothing is trained in advance, nothing is fine-tuned, and the model
has no fixed list of categories.

## How it works

**Ingest** — PNG, JPEG, WebP, multi-page TIFF and GIF, HEIC, PDF, ZIP archives.
Everything is normalized, deduplicated by content hash, and blur-scored.

**Label** — [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)
(Apache-2.0) runs on the CPU and returns one box per object with a real
confidence score. About 2 seconds per photo on a laptop; the weights (~620 MB)
download once.

**Review** — the queue puts the least confident boxes first, because that is
where your attention is worth the most. Drag the handles to fix a box, press a
digit to change its class, `A` to accept, `D` to draw a missing one. Accept
everything above a threshold in one click.

**Export** — YOLO (`data.yaml` plus normalized labels, with a deterministic
train/val split) or COCO instances. Only annotations a human has reviewed are
exported.

## Optional: your own GPU

For large batches, open **Settings**, paste a Modal API token and press
**Connect GPU**. The platform deploys the recipe into *your* Modal account,
secures the endpoint with a generated bearer token, and a GPU engine appears in
the engine list. You pay Modal directly for the seconds you use, and it scales
to zero when idle. This is entirely optional — the CPU path is the default and
needs no account at all.

## Choosing an engine

| Engine | Runs on | Notes |
|---|---|---|
| **OWLv2** | CPU | Default. Predictions do not change when you add a class |
| LLMDet | CPU | Slower, sometimes tighter boxes. Max 91 classes per run |
| GPU on Modal | your Modal account | For large batches |
| Consensus | CPU | Runs several engines and scores their agreement |
| VLM / HTTP | any endpoint | Bring your own model behind a small convention |

Engines are plugins. A labeler is one method —
`predict(image, config) -> list[Annotation]` — registered as a Python entry
point, and the core never needs to know about it. See [`sdk/`](sdk).

## What it is not good at

Being honest saves you an afternoon:

- **Attributes do not work.** `carpet` is found; `persian carpet` is not. No
  open-vocabulary detector reliably separates a class from its adjective. Detect
  the broad class, then split it downstream.
- **Crowded scenes are harder than single objects.** On our benchmark, F1 drops
  from 0.89 on product-style photos to 0.78 on cluttered rooms.
- **Zero-shot is a first draft, not a finished dataset.** A model fine-tuned on
  your corrected data will beat it on your domain. That is the point: this tool
  gets you to that data faster.
- Boxes only. No masks, no polygons, no keypoints, no video.

## Benchmark

Measured on 79 photos from LVIS val (323 objects, 114 classes) spanning cluttered
scenes, street shots, phone snaps and product-style images. Each image was
prompted with its own classes, IoU 0.5, greedy matching.

| Engine | F1 | Precision | Recall | Sec/image (CPU) |
|---|---|---|---|---|
| OWLv2 (default) | 0.823 | 0.826 | 0.820 | 2.3 |
| LLMDet base | 0.848 | 0.867 | 0.830 | 4.7 |

Treat these as a starting point, not a verdict: 323 objects is a small sample,
and LVIS is a friendly benchmark for this family of models. What matters is how
an engine does on *your* photos.

One finding worth reusing elsewhere: the stock
`post_process_grounded_object_detection` in `transformers` keeps only the
best-scoring class per box, which silently dropped 12% of detections in our runs
by relabeling them with a neighbouring class (`kitchen sink` → `sink`,
`table lamp` → `lamp`). Emitting boxes per query instead recovers all of them and
makes predictions independent of how many classes you have.

## Thresholds

Defaults come from the benchmark above:

- **0.25** to show a box to a human (precision 0.83, recall 0.82)
- **0.40** to accept in bulk without looking (precision 0.94)

## Architecture

```
docker-compose.yml    postgres, minio, redis, api, worker, web
sdk/                  labeler plugin contract (pip package)
server/               FastAPI + SQLAlchemy (async) + arq worker
web/                  React + TypeScript review UI
labelers/             engine plugins
deploy/modal/         serverless GPU recipes
```

One annotation model covers detection and OCR alike: geometry, label, text,
attributes, plus provenance — which engine produced it, with what confidence,
and what the human did with it. That last part is what makes quality
measurable instead of anecdotal.

OCR is still supported as a second task type (`Project.task_type`), with
PaddleOCR and VLM engines and PaddleOCR export formats.

## Development

`docker compose up -d --build` (without `-f`) picks up
`docker-compose.override.yml` and swaps in the Vite dev server with hot reload,
`uvicorn --reload` and live-mounted source. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Security

- Change the default credentials in `.env` before exposing anything.
- Ports bind to `127.0.0.1`. Put a TLS-terminating proxy in front.
- Set `APP_ACCESS_TOKEN` if anyone else can reach the API port: without it, the
  settings endpoints that accept your Modal token and trigger deploys are open.
- Set `S3_PUBLIC_ENDPOINT_URL` when deploying beyond localhost, or image
  previews break — they are presigned URLs opened directly against MinIO.

## Roadmap

See [ROADMAP.md](ROADMAP.md). Box editing landed recently; next up is quality
calibration on your own data, separating the prompt from the class name, and
importing existing COCO/YOLO datasets.

## License

[AGPL-3.0](LICENSE). Run a modified version as a network service and you owe its
source to your users.

Demo photos come from [COCO](https://cocodataset.org) / Flickr under
Creative Commons Attribution licenses — see [docs/DEMO_IMAGES.md](docs/DEMO_IMAGES.md).
