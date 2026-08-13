"""Фоновые задачи (arq). Запуск: arq app.workers.tasks.WorkerSettings"""

import logging
import uuid
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

NO_CLASSES_ERROR = "Добавьте классы проекта перед запуском разметки"


def _as_xyxy(geometry: dict) -> tuple[float, float, float, float] | None:
    """Геометрия аннотации -> (x1, y1, x2, y2); None для неизвестного типа."""
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


# выше этого перекрытия считаем, что движок предложил уже проверенную рамку
REVIEWED_DUPLICATE_IOU = 0.7



async def run_autolabel(ctx: dict, job_id: str) -> None:
    """Прогнать labeler'ы по всем изображениям проекта и записать аннотации.

    Движки подхватываются из entry points (autolabelui_sdk.load_labelers).
    Если ни одного не установлено — задача завершается с пояснением в result.
    """
    from autolabelui_sdk import load_labelers

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
                    "No labelers installed (entry points group 'autolabelui.labelers')"
                )

            # конфиг движка дополняется настройками (для modal_gpu — endpoint GPU),
            # руками JSON пользователь не пишет
            settings_row = await settings_store.get_row(session)
            base_config = dict(job.payload.get("config") or {})

            # detection: движок ищет ровно то, что перечислено в классах проекта.
            # Пустой список — не «размечай всё подряд», а ошибка с подсказкой,
            # иначе задача тихо завершится нулём аннотаций.
            project = await session.get(Project, job.project_id)
            if project is None:
                raise RuntimeError("Project not found")
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
            not_ready: dict[str, str] = {}
            for name in list(labelers):
                try:
                    configs[name] = settings_store.resolve_labeler_config(
                        name, base_config, settings_row
                    )
                except settings_store.LabelerNotReadyError as exc:
                    # движок запрошен явно — это ошибка задачи, а не тихий пропуск
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

            # Перезапуск: список классов мог измениться, поэтому непроверенные
            # рамки этого движка убираем и размечаем заново. Принятые и
            # исправленные человеком не трогаем — это его работа, не машинная.
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

            # проверенные человеком рамки: при перезапуске движок предложит их
            # заново, и без этой сверки поверх принятого легли бы дубли
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

            # изображения, уже размеченные каждым из движков, — пропускаем.
            # При перезапуске пропуск снимаем: ради него всё и затевалось.
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
            for image in images:
                try:
                    data = await run_in_threadpool(storage.get_bytes, image.s3_key)
                except Exception:
                    # битая запись (файл отсутствует в S3) — не роняем всю задачу
                    logger.exception("Failed to fetch image %s", image.id)
                    failed += 1
                    continue
                for labeler in labelers.values():
                    if image.id in labeled[labeler.name]:
                        skipped += 1
                        continue
                    try:
                        predictions = await run_in_threadpool(
                            labeler.predict, data, configs[labeler.name]
                        )
                    except ValueError as exc:
                        # ошибка конфигурации (нет классов, их слишком много для
                        # движка, неверная модель) одинакова для всех кадров:
                        # молчать нельзя — иначе задача «успешно» разметит в ноль
                        raise RuntimeError(f"{labeler.name}: {exc}") from exc
                    except Exception:
                        logger.exception(
                            "Labeler %s failed on image %s", labeler.name, image.id
                        )
                        failed += 1
                        continue
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
                                # это уже проверено человеком — не подсовываем заново
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
    """Нормализация документа -> images (PDF-страницы, HEIC, TIFF, ZIP...).

    Дубликаты (по хешу нормализованного содержимого в рамках проекта) пропускаются.
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

            # хеши уже имеющихся в проекте изображений — для дедупликации
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


async def run_deploy_gpu(ctx: dict, job_id: str) -> None:
    """Развернуть GPU-рецепт в аккаунт Modal пользователя.

    Токен берётся из настроек и расшифровывается здесь — в payload задачи и в
    логи он не попадает. Результат: gpu_endpoint_url + gpu_status=ready,
    при ошибке — failed и текст в gpu_error.
    """
    import secrets as secrets_mod

    from app.crypto import decrypt_secret, encrypt_secret
    from app.services import modal_deploy

    async with SessionLocal() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.error("Job %s not found", job_id)
            return

        job.status = JobStatus.RUNNING
        row = await settings_store.get_or_create(session)
        row.gpu_status = GpuStatus.DEPLOYING
        row.gpu_error = None
        await session.commit()

        token_secret = ""
        try:
            if not (row.modal_token_id and row.modal_token_secret_encrypted):
                raise RuntimeError(
                    "Токен Modal не сохранён — введите его на странице настроек"
                )
            token_secret = decrypt_secret(row.modal_token_secret_encrypted)
            # эндпоинт закрываем Bearer-токеном: URL угадать трудно, но он
            # утекает в историю браузера, логи прокси и скриншоты
            gpu_token = settings.autolabelui_gpu_token or secrets_mod.token_urlsafe(32)
            deployed = await modal_deploy.deploy_gpu_app(
                row.modal_token_id,
                token_secret,
                app_name=job.payload.get("app_name") or None,
                gpu_token=gpu_token,
            )
            if not settings.autolabelui_gpu_token:
                row.gpu_access_token_encrypted = encrypt_secret(gpu_token)

            row.gpu_status = GpuStatus.READY
            row.gpu_endpoint_url = deployed.endpoint_url
            row.gpu_error = None
            job.status = JobStatus.DONE
            job.result = {
                "app_id": deployed.app_id,
                "app_page_url": deployed.app_page_url,
                "endpoint_url": deployed.endpoint_url,
                "warnings": deployed.warnings,
            }
        except Exception as exc:
            # секрет не должен просочиться ни в БД, ни в ответ API, ни в лог;
            # traceback не пишем — в нём могут оказаться аргументы вызовов modal
            message = modal_deploy.scrub_secrets(str(exc), token_secret)
            logger.error("Deploy GPU job %s failed: %s", job_id, message)
            row.gpu_status = GpuStatus.FAILED
            row.gpu_error = message
            job.status = JobStatus.FAILED
            job.result = {"error": message}
        finally:
            row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            job.finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
            await session.commit()


class WorkerSettings:
    # arq по умолчанию рвёт задачу через 300 с и повторяет её до 5 раз: для
    # разметки сотен страниц и для деплоя GPU-образа (первая сборка — минуты)
    # это гарантированный обрыв. Повтор не нужен: статус и ошибка уже пишутся
    # в Job, а повторный autolabel лишь заново прошёл бы по тем же картинкам.
    functions = [
        func(run_autolabel, timeout=6 * 3600, max_tries=1),
        func(run_ingest, timeout=3600, max_tries=1),
        func(run_deploy_gpu, timeout=3600, max_tries=1),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
