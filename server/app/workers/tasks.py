"""Background jobs (arq). Run with: arq app.workers.tasks.WorkerSettings"""

import asyncio
import json
import logging
import uuid
from collections import deque
from datetime import datetime, timezone

from arq import func
from arq.connections import RedisSettings
from sqlalchemy import delete, select
from starlette.concurrency import run_in_threadpool

from app import storage
from app.config import settings
from app.db import SessionLocal
from app.models import (
    Annotation,
    AnnotationStatus,
    Document,
    DocumentStatus,
    GpuDeployment,
    GpuStatus,
    Image,
    Job,
    JobStatus,
    Project,
    ProjectClass,
    TaskType,
)
from app.services import settings_store

logger = logging.getLogger(__name__)

NO_CLASSES_ERROR = "Add project classes before starting the labeling run"


def _as_xyxy(geometry: dict) -> tuple[float, float, float, float] | None:
    """Annotation geometry -> (x1, y1, x2, y2); None for an unknown type."""
    if geometry.get("type") == "bbox":
        x, y = float(geometry["x"]), float(geometry["y"])
        return (x, y, x + float(geometry["width"]), y + float(geometry["height"]))
    points = geometry.get("points")
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# above this overlap we treat the engine's box as one already reviewed
REVIEWED_DUPLICATE_IOU = 0.7

# marker for "this engine has already labeled this image" — distinguishable from
# both a predictions list and an exception, which is what else can come back
_ALREADY_LABELED = object()


def _carries_fatal(task) -> bool:
    """A finished task whose result aborts the whole job."""
    if not task.done() or task.cancelled():
        return False
    _, _, outcomes = task.result()
    return any(isinstance(outcome, ValueError) for _, outcome in outcomes)


async def _predict_image(image, labelers: dict, configs: dict, labeled: dict):
    """Fetch one image and run every engine on it. Touches no database session.

    That restriction is the point of the function. It is run concurrently for
    several images, and a SQLAlchemy session is not safe to share between tasks
    — so everything that writes (session.add, the counters, the job result)
    stays in the caller, which consumes these results one at a time.

    Nothing is raised: a failure is returned as the exception object, because
    the caller has to tell the two kinds apart. A ValueError means a config
    error that will repeat on every remaining frame and must fail the job;
    anything else costs one image.
    """
    outcomes: list = []
    try:
        try:
            data = await run_in_threadpool(storage.get_bytes, image.s3_key)
        except Exception as exc:
            return image, exc, []

        for labeler in labelers.values():
            if image.id in labeled[labeler.name]:
                outcomes.append((labeler, _ALREADY_LABELED))
                continue
            try:
                predictions = await run_in_threadpool(
                    labeler.predict, data, configs[labeler.name]
                )
            except ValueError as exc:
                # This aborts the whole job, so stop working on this image at
                # once — the old sequential loop raised here and never reached
                # the next engine. Without the break the remaining engines run
                # a full inference whose result is discarded microseconds later,
                # and on a GPU engine that is billable work after the job has
                # already been decided.
                outcomes.append((labeler, exc))
                break
            except Exception as exc:
                outcomes.append((labeler, exc))
                continue
            outcomes.append((labeler, predictions))
    except Exception as exc:  # the promise above, made structural
        return image, exc, outcomes
    return image, None, outcomes


def _image_concurrency(labelers: dict) -> int:
    """How many images to have in flight at once.

    One, unless every engine in the run waits on another machine. A mixed run
    stays sequential on purpose: the local engine would be the bottleneck
    anyway, and running several copies of it at once makes it slower, not
    faster.
    """
    if not labelers or not all(
        settings_store.is_remote(name) for name in labelers
    ):
        return 1
    return max(1, settings.remote_labeler_concurrency)


async def run_autolabel(ctx: dict, job_id: str) -> None:
    """Run the labelers over every image of the project and store annotations.

    Engines are picked up from entry points (nounbox_sdk.load_labelers).
    If none are installed, the job finishes with an explanation in result.
    """
    from nounbox_sdk import load_labelers

    async with SessionLocal() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.error("Job %s not found", job_id)
            return

        job.status = JobStatus.RUNNING
        await session.commit()

        try:
            labelers = load_labelers()
            wanted = job.payload.get("labeler")
            if wanted and wanted not in labelers:
                available = ", ".join(sorted(labelers)) or "none"
                hint = (
                    " Local PaddleOCR requires an image built with WITH_PADDLE=1."
                    if wanted == "paddleocr"
                    else ""
                )
                raise RuntimeError(
                    f"Labeler {wanted!r} is not installed; available: {available}.{hint}"
                )
            if wanted:
                labelers = {wanted: labelers[wanted]}
            if not labelers:
                raise RuntimeError(
                    "No labelers installed (entry points group 'nounbox.labelers')"
                )

            # the engine config is filled in from the settings (for the GPU
            # engines — their endpoint); the user never writes that JSON by hand
            settings_row = await settings_store.get_row(session)
            deployments = await settings_store.load_deployments(session)
            base_config = dict(job.payload.get("config") or {})

            not_ready: dict[str, str] = {}

            project = await session.get(Project, job.project_id)
            if project is None:
                raise RuntimeError("Project not found")

            # An engine may only run on a task it actually serves. Without this
            # the labeler=null fan-out runs EVERY installed engine, so an OCR
            # engine labels a detection project and returns text_line boxes
            # that no project class matches — export drops them silently and
            # the run reports success with nothing to show for it.
            for name in list(labelers):
                if settings_store.labeler_supports_task(name, project.task_type):
                    continue
                serves = "/".join(settings_store.CATALOG[name]["tasks"])
                message = (
                    f"Engine {name} handles {serves} projects; "
                    f"this project is {project.task_type}"
                )
                if wanted:
                    raise RuntimeError(message)
                not_ready[name] = message
                del labelers[name]
            if not labelers:
                raise RuntimeError("; ".join(not_ready.values()))

            # detection: the engine looks for exactly what is listed in the
            # project classes. An empty list is an error with a hint, not
            # "label whatever you find" — otherwise the job quietly ends with
            # zero annotations.
            if project.task_type == TaskType.DETECTION:
                class_names = (
                    (
                        await session.execute(
                            select(ProjectClass.name)
                            .where(ProjectClass.project_id == project.id)
                            .order_by(ProjectClass.sort_order, ProjectClass.created_at)
                        )
                    )
                    .scalars()
                    .all()
                )
                if not class_names:
                    raise RuntimeError(NO_CLASSES_ERROR)
                base_config["classes"] = list(class_names)

            configs: dict[str, dict] = {}
            for name in list(labelers):
                try:
                    configs[name] = settings_store.resolve_labeler_config(
                        name, base_config, settings_row, deployments
                    )
                except settings_store.LabelerNotReadyError as exc:
                    # engine requested explicitly: fail the job, not a silent skip
                    if wanted:
                        raise
                    not_ready[name] = str(exc)
                    del labelers[name]
            if not labelers:
                raise RuntimeError("; ".join(not_ready.values()))

            images = (
                (
                    await session.execute(
                        select(Image)
                        .join(Document, Image.document_id == Document.id)
                        .where(Document.project_id == job.project_id)
                    )
                )
                .scalars()
                .all()
            )

            # Re-run: the class list may have changed, so we drop this engine's
            # pending boxes and label again. Boxes a human accepted or edited
            # are left alone — that is their work, not the machine's.
            rerun = bool(job.payload.get("rerun"))
            replaced = 0
            if rerun:
                for name in labelers:
                    result = await session.execute(
                        delete(Annotation).where(
                            Annotation.image_id.in_([i.id for i in images]),
                            Annotation.source["name"].as_string() == name,
                            Annotation.status == AnnotationStatus.PENDING,
                        )
                    )
                    replaced += result.rowcount or 0
                await session.flush()

            # human-reviewed boxes: on a re-run the engine proposes them again,
            # and without this check duplicates would pile on top of accepted ones
            reviewed: dict[uuid.UUID, list[tuple[str, tuple]]] = {}
            if rerun:
                rows = (
                    await session.execute(
                        select(Annotation).where(
                            Annotation.image_id.in_([i.id for i in images]),
                            Annotation.status.in_(
                                (AnnotationStatus.ACCEPTED, AnnotationStatus.EDITED)
                            ),
                        )
                    )
                ).scalars().all()
                for row in rows:
                    box = _as_xyxy(row.geometry)
                    if box is not None:
                        reviewed.setdefault(row.image_id, []).append((row.label, box))

            # images already labeled by each of the engines are skipped.
            # On a re-run the skip is lifted: that is the whole point of it.
            labeled: dict[str, set] = {}
            for name in labelers:
                rows = (
                    (
                        await session.execute(
                            select(Annotation.image_id)
                            .join(Image, Annotation.image_id == Image.id)
                            .join(Document, Image.document_id == Document.id)
                            .where(
                                Document.project_id == job.project_id,
                                Annotation.source["name"].as_string() == name,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                labeled[name] = set() if rerun else set(rows)

            created, skipped, failed, duplicates = 0, 0, 0, 0
            # A rolling window, not a barrier: `concurrency` predictions stay in
            # flight, and each completed one is applied before the next starts.
            # Batching in fixed groups would idle the remote engine for the tail
            # of every group while the slowest image finished.
            #
            # Results are consumed in image order on purpose. It costs nothing —
            # the tasks run regardless — and it keeps an annotation provably
            # attached to the image it was predicted from.
            concurrency = _image_concurrency(labelers)
            semaphore = asyncio.Semaphore(concurrency)

            async def predict(image):
                async with semaphore:
                    return await _predict_image(image, labelers, configs, labeled)

            pending: deque = deque()
            queued = 0
            while queued < len(images) and len(pending) < concurrency:
                pending.append(asyncio.create_task(predict(images[queued])))
                queued += 1

            applied = 0
            try:
                while pending:
                    image, fetch_error, outcomes = await pending.popleft()
                    # Do not start another image once a fatal error is already
                    # sitting in a finished task. It is only a peek — the error
                    # is still applied in order below — but it stops the run
                    # from reaching for more work it has already decided to
                    # throw away. What is in flight cannot be un-sent, so the
                    # waste is bounded by the concurrency, not by the project.
                    if queued < len(images) and not any(map(_carries_fatal, pending)):
                        pending.append(asyncio.create_task(predict(images[queued])))
                        queued += 1
                    applied += 1
                    # bound the identity map on a run of thousands of images;
                    # the single commit still happens in the finally below
                    if applied % 50 == 0:
                        await session.flush()
                    if fetch_error is not None:
                        # broken row (file missing from S3) — don't kill the whole job
                        logger.error(
                            "Failed to fetch image %s", image.id, exc_info=fetch_error
                        )
                        failed += 1
                        continue
                    for labeler, outcome in outcomes:
                        if outcome is _ALREADY_LABELED:
                            skipped += 1
                            continue
                        if isinstance(outcome, ValueError):
                            # a config error (no classes, too many of them for the
                            # engine, a wrong model) is the same for every frame:
                            # staying quiet means the job "succeeds" labeling nothing
                            raise RuntimeError(
                                f"{labeler.name}: {outcome}"
                            ) from outcome
                        if isinstance(outcome, Exception):
                            logger.error(
                                "Labeler %s failed on image %s",
                                labeler.name,
                                image.id,
                                exc_info=outcome,
                            )
                            failed += 1
                            continue
                        predictions = outcome
                        source = {
                            "type": "engine",
                            "name": labeler.name,
                            "version": labeler.version,
                        }
                        checked = reviewed.get(image.id, [])
                        for pred in predictions:
                            geom = pred.geometry
                            if checked:
                                box = _as_xyxy(
                                    {
                                        "type": "bbox",
                                        "x": geom.x,
                                        "y": geom.y,
                                        "width": geom.width,
                                        "height": geom.height,
                                    }
                                    if hasattr(geom, "width")
                                    else {"type": "polygon", "points": [list(p) for p in geom]}
                                )
                                if box is not None and any(
                                    label == pred.label
                                    and _iou(box, other) >= REVIEWED_DUPLICATE_IOU
                                    for label, other in checked
                                ):
                                    # already reviewed by a human — don't push it again
                                    duplicates += 1
                                    continue
                            if hasattr(geom, "width"):  # BBox dataclass
                                geometry = {
                                    "type": "bbox",
                                    "x": geom.x,
                                    "y": geom.y,
                                    "width": geom.width,
                                    "height": geom.height,
                                }
                            else:  # Polygon
                                geometry = {"type": "polygon", "points": [list(p) for p in geom]}
                            session.add(
                                Annotation(
                                    image_id=image.id,
                                    geometry=geometry,
                                    label=pred.label,
                                    text=pred.text,
                                    attrs=pred.attrs,
                                    confidence=pred.confidence,
                                    source=source,
                                )
                            )
                            created += 1
            finally:
                # a config error aborts the run — stop the predictions already
                # in flight instead of paying for results nobody will read
                for task in pending:
                    task.cancel()

            job.status = JobStatus.DONE
            job.result = {
                "images": len(images),
                "annotations_created": created,
                "skipped_already_labeled": skipped,
                "failed_images": failed,
                "labelers": list(labelers),
                "skipped_labelers": not_ready,
                "replaced_pending": replaced,
                "skipped_duplicates": duplicates,
            }
        except Exception as exc:
            logger.exception("Autolabel job %s failed", job_id)
            job.status = JobStatus.FAILED
            job.result = {"error": str(exc)}
        finally:
            job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.commit()


async def run_ingest(ctx: dict, job_id: str) -> None:
    """Normalize a document -> images (PDF pages, HEIC, TIFF, ZIP...).

    Duplicates (by hash of the normalized content, within the project) are skipped.
    """
    from app.services import ingest

    async with SessionLocal() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.error("Job %s not found", job_id)
            return

        job.status = JobStatus.RUNNING
        await session.commit()

        doc = None
        try:
            doc = await session.get(Document, uuid.UUID(job.payload["document_id"]))
            if doc is None:
                raise RuntimeError("Document not found")
            doc.status = DocumentStatus.PROCESSING
            await session.flush()

            data = await run_in_threadpool(storage.get_bytes, doc.s3_key)
            extracted = await run_in_threadpool(
                ingest.extract_pages, doc.filename, data
            )

            # hashes of the images already in the project — for deduplication
            existing = set(
                (
                    await session.execute(
                        select(Image.content_hash)
                        .join(Document, Image.document_id == Document.id)
                        .where(
                            Document.project_id == doc.project_id,
                            Image.content_hash.is_not(None),
                        )
                    )
                )
                .scalars()
                .all()
            )

            created, duplicates = 0, 0
            for page in extracted.pages:
                if page.content_hash in existing:
                    duplicates += 1
                    continue
                s3_key = f"images/{doc.project_id}/{uuid.uuid4()}.png"
                await run_in_threadpool(
                    storage.put_bytes, s3_key, page.data, "image/png"
                )
                session.add(
                    Image(
                        document_id=doc.id,
                        page_index=page.page_index,
                        s3_key=s3_key,
                        width=page.width,
                        height=page.height,
                        content_hash=page.content_hash,
                        blur_score=page.blur_score,
                    )
                )
                existing.add(page.content_hash)
                created += 1

            doc.status = DocumentStatus.READY
            job.status = JobStatus.DONE
            job.result = {
                "pages": len(extracted.pages),
                "images_created": created,
                "duplicates_skipped": duplicates,
                "nested_archives_skipped": extracted.nested_archives_skipped,
            }
        except Exception as exc:
            logger.exception("Ingest job %s failed", job_id)
            if doc is not None:
                doc.status = DocumentStatus.FAILED
            job.status = JobStatus.FAILED
            job.result = {"error": str(exc)}
        finally:
            job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.commit()


async def _warm_up(
    endpoint_url: str, path: str, gpu_token: str | None, config: dict
) -> bool:
    """POST a 1x1 PNG so the first real photo does not pay the weight download.

    Best effort by definition: a cold Modal container downloads hundreds of
    megabytes of weights on its first request, and without this the user's
    first image looks like a hang. A failure here costs a slow first image and
    nothing else, so it never fails the deploy job.

    The config header is mandatory, not optional politeness. The detection
    recipe answers 400 to a request with no classes BEFORE it loads the model,
    so an empty warm-up returns in milliseconds having downloaded nothing —
    and a 4xx must therefore not count as warmed, or the job cheerfully reports
    success for a container that is still cold.
    """
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Labeler-Config": json.dumps(config),
    }
    if gpu_token:
        headers["Authorization"] = f"Bearer {gpu_token}"
    try:
        import httpx

        async with httpx.AsyncClient(timeout=180.0) as http:
            response = await http.post(
                settings_store.predict_url(endpoint_url, path),
                content=WARMUP_PNG,
                headers=headers,
            )
        if response.status_code >= 400:
            logger.info(
                "GPU warm-up answered %s — the model may still be cold",
                response.status_code,
            )
            return False
        return True
    except Exception as exc:
        logger.info("GPU warm-up skipped: %s", exc)
        return False


# 1x1 transparent PNG — the smallest thing that boots the model
WARMUP_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00"
    b"\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


async def run_deploy_gpu(ctx: dict, job_id: str) -> None:
    """Deploy one GPU recipe into the user's Modal account.

    Which recipe is in job.payload["engine"] (modal_gpu — OCR, the default for
    a pre-upgrade caller; modal_gpu_detect — detection). The token is taken
    from the settings and decrypted here — it never reaches the job payload or
    the logs. Result: the engine's deployment row gets endpoint_url +
    status=ready; on failure — failed and the message in its error.
    """
    import secrets as secrets_mod

    from app.crypto import decrypt_secret, encrypt_secret
    from app.services import gpu_recipes, modal_deploy

    async with SessionLocal() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.error("Job %s not found", job_id)
            return

        engine = job.payload.get("engine") or gpu_recipes.MODAL_GPU
        if engine not in gpu_recipes.GPU_RECIPES:
            job.status = JobStatus.FAILED
            job.result = {"error": f"Unknown GPU engine {engine!r}"}
            job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.commit()
            return
        recipe = gpu_recipes.GPU_RECIPES[engine]
        legacy = engine == gpu_recipes.MODAL_GPU

        job.status = JobStatus.RUNNING
        row = await settings_store.get_or_create(session)
        # persist: same reason as the deploy endpoint — this row gets written
        deployments = await settings_store.load_deployments(session, persist=True)
        deployment = deployments.get(engine)
        if deployment is None:
            deployment = GpuDeployment(
                engine=engine,
                app_name=job.payload.get("app_name")
                or gpu_recipes.configured_app_name(engine),
            )
            session.add(deployment)
        deployment.status = GpuStatus.DEPLOYING
        deployment.error = None
        if legacy:
            # one release of dual-writing: a rollback to the previous image
            # must still find a working OCR endpoint in the old columns
            row.gpu_status = GpuStatus.DEPLOYING
            row.gpu_error = None
        await session.commit()

        # both are pre-bound because the except block scrubs them out of the
        # error message, and the first raise below happens before either is set
        token_secret = ""
        gpu_token = ""
        warmed = False
        try:
            if not (row.modal_token_id and row.modal_token_secret_encrypted):
                raise RuntimeError(
                    "No Modal token saved — enter it on the settings page"
                )
            token_secret = decrypt_secret(row.modal_token_secret_encrypted)
            # we close the endpoint with a Bearer token: the URL is hard to
            # guess, but it leaks into browser history, proxy logs and screenshots
            gpu_token = settings.nounbox_gpu_token or secrets_mod.token_urlsafe(32)
            deployed = await modal_deploy.deploy_gpu_app(
                row.modal_token_id,
                token_secret,
                engine=engine,
                app_name=job.payload.get("app_name") or None,
                gpu_token=gpu_token,
            )
            if not settings.nounbox_gpu_token:
                deployment.access_token_encrypted = encrypt_secret(gpu_token)

            deployment.status = GpuStatus.READY
            deployment.endpoint_url = deployed.endpoint_url
            deployment.error = None
            if legacy:
                row.gpu_status = GpuStatus.READY
                row.gpu_endpoint_url = deployed.endpoint_url
                row.gpu_error = None
                if not settings.nounbox_gpu_token:
                    row.gpu_access_token_encrypted = deployment.access_token_encrypted

            # Commit BEFORE warming up. The Modal app is already live and
            # already billable at this point, and the generated bearer is baked
            # into its Secret; the warm-up that follows waits up to 180 s on a
            # cold container. If the worker dies in that window and this state
            # is still only in memory, the user is left paying for an endpoint
            # the platform has no record of and cannot address.
            deployment.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.commit()

            warmed = await _warm_up(
                deployed.endpoint_url,
                recipe.predict_path,
                gpu_token,
                recipe.warmup_config,
            )
            job.status = JobStatus.DONE
            job.result = {
                "engine": engine,
                "app_id": deployed.app_id,
                "app_page_url": deployed.app_page_url,
                "endpoint_url": deployed.endpoint_url,
                "warnings": deployed.warnings,
                "warmed": warmed,
            }
        except Exception as exc:
            # the secret must not seep into the DB, the API response or the log;
            # no traceback — it can carry the arguments of the modal calls
            # gpu_token is a live credential in this scope as well — this
            # message goes into the DB, the API response and the log
            message = modal_deploy.scrub_secrets(str(exc), token_secret, gpu_token)
            logger.error("Deploy GPU job %s failed: %s", job_id, message)
            deployment.status = GpuStatus.FAILED
            deployment.error = message
            if legacy:
                row.gpu_status = GpuStatus.FAILED
                row.gpu_error = message
            job.status = JobStatus.FAILED
            job.result = {"engine": engine, "error": message}
        finally:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            deployment.updated_at = now
            row.updated_at = now
            job.finished_at = now
            await session.commit()


class WorkerSettings:
    # arq by default kills a job after 300 s and retries it up to 5 times: for
    # labeling hundreds of pages and for deploying the GPU image (minutes on the
    # first build) that is a guaranteed abort. Retries are pointless: status and
    # error already land in Job, and a repeat autolabel would redo the same images.
    functions = [
        func(run_autolabel, timeout=6 * 3600, max_tries=1),
        func(run_ingest, timeout=3600, max_tries=1),
        func(run_deploy_gpu, timeout=3600, max_tries=1),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)

    # Explicit, because the labeling run is no longer single-threaded. Every
    # image in flight holds a thread from the pool run_in_threadpool draws on —
    # 40 for the whole process — and arq's own default of 10 concurrent jobs
    # times remote_labeler_concurrency would reach exactly that, leaving an
    # ingest job (minutes of rasterizing a PDF) waiting for a labeling thread.
    # Four labeling jobs at the shipped concurrency of 4 use 16 of the 40.
    max_jobs = 4
