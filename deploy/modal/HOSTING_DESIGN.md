# Hosting the whole platform on Modal — design document

> The "no server of my own — one command and it all runs on Modal" mode.
> **A complement** to docker-compose (the primary self-hosted path), not a replacement.
> The vision: the core stays light and runs anywhere; heavy GPU inference has already
> been offloaded to Modal (`deploy/modal/vlm.py`, `deploy/modal/paddleocr_modal.py`) —
> what we design here is offloading the core itself (API + worker + DB + files + UI).

Status: design (no code was changed). Research date: 2026-08-11.

---

## 0. Sidebar: the GPU recipe auto-deployed from the UI

A scheme separate from this document, already approved by the owner: labeling runs in
two modes — **SIMPLE** (`rapidocr`, CPU, works right after `docker compose up`) and
**ADVANCED** (`modal_gpu`, optional). The platform brings the second mode up on its
own, and `paddleocr_modal.py` is the very recipe it deploys.

The flow: on the settings page the user pastes a Modal API token
(`ak-…` / `as-…`, what `modal token new` hands out; `PUT /api/v1/settings`) and presses
"Connect GPU" (`POST /api/v1/settings/gpu/deploy` → a Job of type `DEPLOY_GPU`).
The platform imports `deploy/modal/paddleocr_modal.py`, takes the module-level `app`
object and calls `modal.runner.deploy_app(app, name="nounbox-paddleocr",
client=modal.Client.from_credentials(...))`; the URL is fetched separately with
`modal.Function.from_name(app, "fastapi_app", client=…).hydrate(client=…).get_web_url()`
(it is not part of `DeployResult`) and stored in `settings.gpu_endpoint_url`. From
there the worker in `run_autolabel` substitutes the endpoint into the `modal_gpu`
engine config by itself — nobody writes JSON by hand.

What follows from this for the recipe (and is already done):
- `app` at module level, imports free of side effects and of reading local files —
  the deploy runs from the FastAPI process, not from the CLI;
- the file is self-contained: `include_source=True` mounts EXACTLY this one `.py`
  into the Modal container, so any `from ...` reaching into the monorepo will not
  resolve there;
- access token: if the deploying process has `NOUNBOX_GPU_TOKEN` set, it is baked
  into the app's Secret (`Secret.from_dict`, read at import time — meaning the variable
  must be set BEFORE the module is imported) and `/predict` starts requiring
  `Authorization: Bearer <token>`; with no variable set, the endpoint stays open behind
  an unguessable URL, as before;
- the button is idempotent for free: a repeat deploy is a new version (rolling), and
  when nothing changed Modal answers "Deployment skipped". Do not use
  `strategy="recreate"` — it kills live containers.

The post-deploy check: `GET /health` returns `{"auth": "bearer"|"open",
"engines_loaded": N}` — this is how the platform confirms that the container came up
on a GPU and that the token made it through. The first `POST /predict` additionally
pays for downloading the PP-OCRv5 weights (cached in the `nounbox-paddlex-cache`
Volume), so it is better to "warm" the endpoint with a tiny PNG than with the user's
first real page.

---

## 1. What Modal gives us (research summary)

Facts from the current documentation that this design rests on:

**Web endpoints / ASGI.** `@modal.asgi_app()` serves a whole FastAPI application as a
web endpoint (`https://<workspace>--<app>-<fn>.modal.run`); `@modal.concurrent(max_inputs=N)`
lets a single container serve N requests in parallel (the ASGI event loop).
Request body up to 4 GiB. Custom domains are Team/Enterprise plan only; on Starter we
stay on `*.modal.run`. Endpoint protection is Proxy Auth Tokens
(`requires_proxy_auth=True`, the `Modal-Key`/`Modal-Secret` headers or
`Authorization: Bearer <id>.<secret>`), but a browser UI cannot supply such headers —
the UI needs a token mechanism of its own (see §5).

**Cold start.** Container boot is ~1 s; the real latency comes from application
initialization (imports, models). The tools: `scaledown_window` (2 s … 20 min, default
60 s), `min_containers` (never drop to zero — but you pay for idle),
`buffer_containers`, memory snapshots (a snapshot of warmed-up memory — "substantially
reduce cold start latency").

**Volumes.** A distributed filesystem with commits: a manual `.commit()` plus
background autocommits "every few seconds" and a final snapshot on stop.
The semantics are **last write wins, per file**; "concurrent modifications of the same
files should be avoided"; **file locking is not supported** ("nor is distributed
file locking supported"). Throughput up to 2.5 GB/s, degradation past 50,000 files
(v1); Volumes v2 scale better, but the "one writer per file" rule stands.
**The conclusion for a DBMS:** any database on a Volume is acceptable only with
strictly one writer container; atomicity of a commit spanning several files
(`*.db` + `*-wal`, for instance) is not guaranteed — you need your own
checkpoints/backups. This is the key constraint on the entire design.

**Sandboxes.** Containers created at runtime, meant for untrusted code: max lifetime
24 h, ephemeral, with tunnels and volume mounts. Not suitable as a "home" for the
platform (the lifetime cap, the manual lifecycle), but it is a ready-made mechanism
for later running **third-party labeler plugins in isolation** (v0.3+).

**Dict / Queue.** Queue: item ≤ 1 MiB, ≤ 5,000 items per partition, a partition is
cleared 24 h after the last put. Usable as a queue transport, but `Function.spawn()`
is simpler and more reliable: it returns a `FunctionCall`, the result can be polled by
id for up to 7 days, and functions support `retries`/`timeout`. Dict will come in handy
for progress reporting on long jobs (not critical for v1).

**Auto Endpoints (managed).** `modal endpoint create --model <hf-id>` — a managed
OpenAI-compatible endpoint (`/v1`), scale-to-zero, billed for compute only, with the
recipe's source open. The catalog: the Qwen, Kimi, Gemma, DeepSeek, Nemotron, GPT-OSS
and GLM families. **Vision models (Qwen2.5-VL / Qwen3-VL) are not confirmed to be in
the catalog by the documentation** (the blog mentions a "vision-language model
Endpoint", but no specific VL models appear in the list). So for the VLM labeler our
own `deploy/modal/vlm.py` (vLLM + Qwen2.5-VL) remains; the catalog is worth rechecking
periodically — as soon as VL models show up, the recipe can be replaced with a single
command. Modal's official SGLang + Qwen-VL example is the fallback recipe.

**Pricing (current).** CPU $0.0000131/physical core/s (0.125 cores minimum),
RAM $0.00000222/GiB/s, GPU: T4 $0.000164/s (≈$0.59/h), L4 $0.000222/s (≈$0.80/h),
A100-80GB $0.000694/s, H100 $0.001097/s. Volume $0.09/GiB/month (with an included
quota). The Starter plan: **$30/month of free credits**, billed only for the time
containers actually run (idle under scale-to-zero is not billed; with
`min_containers=1` it is).

---

## 2. How the code is currently tied to infrastructure

What exactly keeps the core on docker-compose, file by file:

| Dependency | Where in the code | What blocks Modal mode |
|---|---|---|
| Postgres (asyncpg) | `server/app/db.py`, `config.py: database_url` | needs a running Postgres |
| PG dialect specifics | `server/app/models.py` (`JSONB`, `UUID` from `sqlalchemy.dialects.postgresql`), `workers/tasks.py:70` (`.astext`) | will not start on SQLite |
| MinIO/S3 | `server/app/storage.py` (boto3, presigned URLs) | needs an S3 service |
| Redis + arq | `main.py` (`create_pool`), `api/documents.py:50`, `api/jobs.py:32` (`enqueue_job`), `workers/tasks.py` (`WorkerSettings`) | needs Redis and a separate worker process |
| UI as a separate dev server | `web/` (Vite, proxying to the api) | needs a second HTTP service |

The tasks themselves (`run_ingest`, `run_autolabel`) are ordinary async functions with
no arq ties beyond the `(ctx, job_id)` signature: portable as they are.

---

## 3. Option A — "Modal-native lite": ASGI + SQLite + Volume + spawn

### Architecture

One Modal App `nounbox-platform`, one service class, **one container**:

```python
# deploy/modal/platform.py (sketch)
data_vol = modal.Volume.from_name("nounbox-data", create_if_missing=True)

@app.cls(
    image=image,                      # server + sdk + labelers/http,vlm + web/dist
    volumes={"/data": data_vol},
    max_containers=1,                 # THE ONLY writer of SQLite and of the files
    scaledown_window=1200,            # 20 min — the maximum
    timeout=3600,
)
@modal.concurrent(max_inputs=32)
class Platform:
    @modal.asgi_app()
    def web(self):
        from app.main import app as fastapi_app   # env: MODE=modal
        return fastapi_app

    @modal.method()
    async def run_job(self, task: str, job_id: str):
        from app.workers import tasks
        await getattr(tasks, task)({}, job_id)
```

- **DB**: SQLite on the Volume (`sqlite+aiosqlite:////data/autolabel.db`).
- **Files**: the same Volume (`/data/objects/...`) instead of MinIO.
- **Queue**: `Platform().run_job.spawn("run_ingest", job_id)` instead of
  `arq.enqueue_job`. With `max_containers=1` the spawn lands in **the same container
  and process** → the same SQLite connection, and — importantly — while the job runs it
  counts as an active input: the container will not scale to zero in the middle of an
  ingest.
- **UI**: `web/dist` (a vite build) is served by FastAPI itself through `StaticFiles` —
  one endpoint for everything.
- **GPU inference**: unchanged — the HTTP/VLM labelers call out to separate Modal apps
  (`vlm.py`, `paddleocr_modal.py`) or to an Auto Endpoint.

### Code changes required (all of them useful for compose too)

1. `server/app/models.py` — portable types:
   `JSONB` → `JSON().with_variant(JSONB, "postgresql")`;
   `UUID(as_uuid=True)` (the PG dialect) → `sqlalchemy.Uuid` (SA 2.0 — CHAR(32) on
   SQLite, native uuid on PG). Behavior on Postgres does not change.
2. `server/app/workers/tasks.py:70` — `Annotation.source["name"].astext` →
   `.as_string()` (a portable accessor; compiles to `->>` on PG).
3. `server/app/config.py` — `storage_backend: s3|local`, `local_storage_dir`,
   `queue_backend: arq|modal`, `serve_static: bool`.
4. `server/app/storage.py` — an interface with two implementations: the current S3 one
   and `LocalStorage` (files in a directory on the Volume; `presigned_url` → an
   HMAC-signed route `GET /api/v1/files/{key}?exp=&sig=`). One new route.
5. A new `server/app/queue.py` — `async def enqueue(request, task, job_id)`: an arq
   implementation (the current behavior) and a modal one
   (`modal.Cls.from_name("nounbox-platform", "Platform")().run_job.spawn(...)`).
   `main.py` creates the arq pool only when `queue_backend=arq`.
6. `server/app/db.py` — for SQLite: `PRAGMA journal_mode=WAL`, `busy_timeout`,
   NullPool; `pyproject.toml` + `aiosqlite`.
7. A new `deploy/modal/platform.py` (~100 lines) — the sketch above.

### Properties

- **Cold start**: ~1 s of container plus 2–4 s of FastAPI/SQLAlchemy imports; with a
  memory snapshot, noticeably less. If wanted, `min_containers=1` removes cold start
  entirely (see the price). After 20 min of silence the first request waits a few
  seconds — acceptable for a labeling tool.
- **Persistence**: SQLite and the files survive restarts through Volume commits.
  The risk: a background commit is not atomic across `*.db` and `*-wal` — mitigated by
  (a) `PRAGMA wal_checkpoint(TRUNCATE)` after every job and on a timer,
  (b) a periodic `VACUUM INTO /data/backup/autolabel-<ts>.db` (a consistent copy in a
  single file), (c) the export ZIPs — which also give an offsite backup of the dataset.
  A crash between commits loses the last few seconds of writes — tolerable at low load,
  no longer so for a team (then it is option C).
- **Concurrency**: `@modal.concurrent` gives parallel requests inside one process;
  SQLite in WAL mode comfortably handles 1–5 reviewers. There is no horizontal scaling
  by design (`max_containers=1`) — a deliberate ceiling.
- **Cost/month at low load** (scale-to-zero, ~60 h of activity/month, 0.5 cores,
  1 GiB): CPU 216,000 s × 0.5 × $0.0000131 ≈ $1.4 + RAM ≈ $0.5 ≈ **$2/month**; with
  `min_containers=1` (0.125 cores, ~0.75 GiB around the clock) ≈ $4.2 + $4.3 ≈
  **$8.6/month**. The Volume stays inside the included quota. All of it is covered by
  the $30 of free credits → **effectively $0**. GPU is separate, billed by actual
  labeling ($0.59–0.80/h for T4/L4).
- **Complexity**: medium — 6 edits plus 1 new file, ~2–4 days including an e2e check.

---

## 4. Option B — "everything in one container": supervisor(postgres+minio+redis+api+worker) + Volume

### Architecture

A single image with supervisord/s6: postgres, redis, minio, uvicorn, the arq worker;
`PGDATA`, the MinIO data and the dumps live on the Volume; outward-facing is a
`@modal.web_server` on the API port. `max_containers=1` is mandatory and
`min_containers=1` is effectively mandatory too.

### Code changes

Near zero: the same env variables as in compose (`database_url` on localhost and so
on). The work moves into the image: installing postgres/minio/redis, the supervisor
config, a first-run initialization script, healthchecks, startup ordering. Plus serving
`web/dist` (as in A, the StaticFiles item).

### An honest word about Postgres on Modal Volumes

This is the weakest spot of the option:

- Postgres constantly rewrites a great many files (heap, WAL, pg_control). The Volume's
  background commit captures state "every few seconds" **without atomicity across
  files** — a `PGDATA` restored after a crash may be out of sync (WAL from one moment,
  heap from another). This is the classic "looks fine until it doesn't" scenario:
  Postgres may come up and silently lose or corrupt data, or it may not come up at all.
- Modal's documentation does not outright forbid a DBMS, but its model ("avoid
  concurrent modifications of the same files", no file locking, last-write-wins) is
  aimed at write-once artifacts (model weights, datasets), not at a DBMS data
  directory.
- Scale-to-zero is not allowed (a stop means a final snapshot of a hot PGDATA), which
  means a 24/7 container and the whole serverless economics is lost.
- There is only one mitigation — treat PGDATA as a cache: `pg_dump` every N minutes to
  the Volume or to S3, and be ready to restore from the dump. In which case, why keep a
  live Postgres at all?

### Properties

- **Cold start**: not applicable (24/7); a restart takes 10–30 s (Postgres recovery).
- **Persistence**: formally present, in practice a risk of PGDATA corruption on any
  involuntary restart; only the "PGDATA-as-cache + frequent pg_dump" combination is
  dependable.
- **Concurrency**: a full Postgres inside the container, but the limit is still one
  container.
- **Cost/month**: 24/7, 1 core + 2 GiB: $34 + $11.5 ≈ **$45/month** (0.5 cores +
  1.5 GiB ≈ $26/month) — more expensive than a VPS with the same guarantees.
- **Complexity**: low in code, high in image work and operations; a permanent
  operational risk.

**Verdict**: an anti-pattern for Modal. Useful, if at all, as a throwaway demo.

---

## 5. Option C — hybrid: the core on Modal, state in external managed services

### Architecture

- API + worker — as in A (the same `platform.py`, but `max_containers` may be > 1).
- **Postgres — Neon** (serverless, free tier: 0.5 GB, 100 CU-hours/month, autosuspend
  after 5 min, auto-resume in ~hundreds of ms). `postgresql+asyncpg://…` — **the
  current code works unchanged**, JSONB/UUID stay native.
- **Files — Cloudflare R2** (S3-compatible, free tier 10 GB, **zero egress**) or any
  S3. `storage.py` works as is: only `s3_endpoint_url` and the keys change; presigned
  URLs are supported on R2.
- **Queue — Modal spawn** (as in A; Redis is not needed at all) — or Upstash Redis plus
  the current arq, but that is an extra service and constant polling by the worker.

### Code changes

If option A has been done — **zero**: C is a configuration of A
(`database_url=neon, storage_backend=s3(r2), queue_backend=modal`).
Without A: only the queue abstraction (item 5 of §3) plus serving the static files.

### Properties

- **Cold start**: as in A (plus ~0.5–1 s for a Neon resume after idling).
- **Persistence**: the best of the three — a real Postgres with Neon's backups, objects
  in R2 with R2's durability. No compromises with Volume semantics.
- **Concurrency**: the best — the API is stateless, `max_containers>1` is possible,
  jobs are parallel spawns, and Postgres honestly handles concurrent writes. This is
  also the path to multi-user (v0.4).
- **Cost/month at low load**: compute as in A ($0–9, covered by credits) + Neon Free $0
  + R2 Free $0 ≈ **$0/month**; growth is smooth and predictable.
- **Downsides**: **three accounts instead of one** (Modal + Neon + Cloudflare) — which
  breaks "one command and it all runs"; the data spreads across three providers (for
  private datasets that is a deal-breaker for some); free tiers change, and a dependency
  on somebody else's limits appears.
- **Complexity**: on top of A — trivial (documentation plus an example .env).

---

## 6. Comparison

| Criterion | A: SQLite+Volume | B: supervisor+PG | C: hybrid Neon+R2 |
|---|---|---|---|
| "One command", no other accounts | ✅ `modal deploy` | ✅ but slow | ❌ 3 accounts |
| Code changes | medium (portability) | ~none (all in the image) | 0 on top of A |
| Cold start | 3–5 s (0 with min_containers) | none (24/7) | as A + Neon resume |
| Data integrity | fine with checkpoints; risk on a crash | ⚠️ real PGDATA risk | ✅ the best |
| Concurrency | 1 container, WAL | 1 container, PG | ✅ horizontal |
| Cost/month (low load) | ~$2–9 → $0 with credits | ~$26–45 | ~$0 |
| Path to multi-user (v0.4) | hits the 1-container wall | hits a wall | ✅ ready |
| Complexity | medium | image + operations | +docs on top of A |

---

## 7. Recommendation

**Build option A as the default mode, designed so that C is a configuration of it.**
Reject option B.

The reasoning:

1. Only A delivers what the owner promised — "no server, one command": all the state
   lives in Modal, no third-party sign-ups, and the $30 of credits cover everything.
2. 90% of the work for A (portable types, the storage/queue abstractions, static files)
   is not "code for Modal" — it is removing the core's hard coupling to the compose
   stack. That directly reinforces the "the core runs on any machine" vision: as a side
   effect we get a single-container `docker run` mode with SQLite and no compose at all.
3. C automatically becomes a documented upgrade path for when the user outgrows A
   (a team, large datasets, multi-user in v0.4): change three env variables, move the
   data over — the code is the same. There is no need to choose between A and C now —
   A contains both.
4. B contradicts the semantics of Modal Volumes (no atomic multi-file commits, no
   locks) and the economics of serverless (a 24/7 container costs more than a VPS).

A's limitation is accepted deliberately: one container, small teams. The threshold past
which it is time to move to C is recorded in the docs (roughly: more than 3–5 active
reviewers, or a DB larger than 1–2 GB).

---

## 8. Phased rollout plan (option A → C)

**Phase 0. Core portability (does not change compose behavior).**
- `models.py`: `JSON().with_variant(JSONB, "postgresql")`, `sqlalchemy.Uuid`;
  `tasks.py`: `.as_string()`.
- `storage.py` → a backend interface (S3 + Local with a signed file route).
- A new `queue.py` (arq | modal), `main.py` — conditional creation of the arq pool.
- `config.py`: `storage_backend`, `queue_backend`, `local_storage_dir`,
  `serve_static`; `aiosqlite` added to the dependencies.
- Check: e2e on compose is green (ingest → autolabel → review → export); a smoke test
  on `sqlite+aiosqlite` locally.

**Phase 1. `deploy/modal/platform.py`.**
- The image: server + sdk + `labelers/http` + `labelers/vlm` (without the paddle
  plugin — heavy CPU inference has no place in a light core, and there is a GPU recipe)
  + `web/dist`.
- The `Platform` class (sketched in §3): `max_containers=1`, `@modal.concurrent`,
  volume `/data`, `web()` + `run_job()`.
- SQLite: WAL, `busy_timeout`, `wal_checkpoint(TRUNCATE)` at the end of a job.
- Check: `modal deploy deploy/modal/platform.py` → a full e2e over `*.modal.run`,
  including autolabel through the `paddleocr_modal` endpoint.

**Phase 2. Robustness and UX.**
- Access: `AUTH_TOKEN` in a Modal Secret → a Bearer check in middleware plus a token
  field in the UI (one per installation; a browser-compatible replacement for proxy
  auth).
- Backups: `VACUUM INTO /data/backup/` when jobs finish (rotating N copies); a README
  section on `modal volume get` for evacuating the data.
- Cold start: measure it; enable memory snapshots; document `min_containers=1` as the
  paid "no latency" option.
- README: a "Hosting on Modal" section — literally
  `pip install modal && modal setup && modal deploy deploy/modal/platform.py`.

**Phase 3. The C upgrade path (config and docs only).**
- An example `.env.modal-hybrid` (Neon `database_url`, R2 endpoint/keys,
  `storage_backend=s3`), and instructions for migrating SQLite → Postgres (alembic plus
  a transfer script; synergy with the already planned move to Alembic migrations).
- e2e on the Neon free tier + R2.

**Phase 4. Watching and evolving.**
- Recheck the Auto Endpoints catalog periodically: once Qwen2.5-VL/Qwen3-VL appear →
  replace `vlm.py` with a managed endpoint (less code of ours).
- Job progress via `modal.Dict` (optional).
- Sandboxes — a candidate for isolating third-party labeler plugins (v0.3+).

Rough estimate: phase 0 — 1–2 days, phase 1 — 1–2 days, phase 2 — 1 day,
phase 3 — 0.5–1 day.

---

## 9. Sources

- Volumes (commits, concurrency, absence of locks): <https://modal.com/docs/guide/volumes>
- CPU/RAM/GPU/Volume pricing, the $30 of Starter credits: <https://modal.com/pricing>
- Web endpoints, `@modal.asgi_app`, `@modal.concurrent`: <https://modal.com/docs/guide/webhooks>
- Endpoint URLs, custom domains (Team/Enterprise): <https://modal.com/docs/guide/webhook-urls>
- Proxy Auth Tokens: <https://modal.com/docs/guide/webhook-proxy-auth>
- Cold start: `min_containers`, `buffer_containers`, `scaledown_window`, memory snapshots: <https://modal.com/docs/guide/cold-start>
- A job queue via `spawn`/`FunctionCall` (results kept 7 days): <https://modal.com/docs/guide/job-queue>
- Dict and Queue (patterns): <https://modal.com/docs/guide/dicts-and-queues>; Queue limits: <https://modal.com/docs/reference/modal.Queue>
- Sandboxes (lifecycle, 24 h, tunnels): <https://modal.com/docs/guide/sandboxes>
- Auto Endpoints (the catalog, `modal endpoint create`, OpenAI compatibility): <https://modal.com/docs/guide/endpoints>; the announcement: <https://modal.com/blog/introducing-auto-endpoints>
- A VLM example (SGLang + Qwen-VL) on Modal: <https://modal.com/docs/examples/sglang_vlm>
- Neon free tier (0.5 GB, 100 CU-h, autosuspend after 5 min): <https://neon.com/pricing>
- Cloudflare R2 (free tier 10 GB, zero egress): <https://developers.cloudflare.com/r2/pricing/>
