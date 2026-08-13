"""Detection autolabel: the engine gets the project classes, empty list is an error."""

import uuid

import nounbox_sdk
from nounbox_sdk import Annotation as SdkAnnotation
from nounbox_sdk import BBox
from sqlalchemy import select

from app import storage
from app.models import (
    Annotation,
    Document,
    Image,
    Job,
    JobStatus,
    JobType,
    Project,
    ProjectClass,
    TaskType,
)
from app.workers import tasks


class RecordingDetector:
    """Detection engine: remembers the config and returns a single box."""

    version = "0.1.0"
    name = "owlv2"

    def __init__(self) -> None:
        self.configs: list[dict] = []

    def predict(self, image: bytes, config: dict) -> list[SdkAnnotation]:
        self.configs.append(config)
        label = (config.get("classes") or ["text_line"])[0]
        return [
            SdkAnnotation(
                geometry=BBox(x=10.0, y=20.0, width=30.0, height=40.0),
                label=label,
                confidence=0.71,
            )
        ]


def install(monkeypatch, labeler) -> None:
    monkeypatch.setattr(
        nounbox_sdk, "load_labelers", lambda: {labeler.name: labeler}
    )
    monkeypatch.setattr(storage, "get_bytes", lambda key: b"png-bytes")


async def make_project(
    session_factory, task_type: TaskType, classes: list[str]
) -> uuid.UUID:
    async with session_factory() as session:
        project = Project(name="p", description="", task_type=task_type)
        session.add(project)
        await session.flush()
        for order, name in enumerate(classes):
            session.add(
                ProjectClass(
                    project_id=project.id, name=name, color="#3b82f6", sort_order=order
                )
            )
        document = Document(project_id=project.id, filename="a.jpg", s3_key="doc/a.jpg")
        session.add(document)
        await session.flush()
        session.add(Image(document_id=document.id, s3_key="img/a.jpg"))
        await session.commit()
        return project.id


async def run_job(session_factory, project_id, labeler="owlv2", config=None) -> Job:
    async with session_factory() as session:
        job = Job(
            project_id=project_id,
            type=JobType.AUTOLABEL,
            payload={"labeler": labeler, "config": config or {}},
        )
        session.add(job)
        await session.commit()
        job_id = str(job.id)

    await tasks.run_autolabel({}, job_id)

    async with session_factory() as session:
        return await session.get(Job, uuid.UUID(job_id))


async def test_project_classes_land_in_engine_config(
    session_factory, key_path, monkeypatch
):
    labeler = RecordingDetector()
    install(monkeypatch, labeler)
    project_id = await make_project(
        session_factory, TaskType.DETECTION, ["carpet", "sofa", "chandelier"]
    )

    job = await run_job(session_factory, project_id, config={"score_threshold": 0.25})

    assert job.status == JobStatus.DONE, job.result
    assert labeler.configs == [
        {"score_threshold": 0.25, "classes": ["carpet", "sofa", "chandelier"]}
    ]


async def test_classes_follow_sort_order(session_factory, key_path, monkeypatch):
    labeler = RecordingDetector()
    install(monkeypatch, labeler)
    project_id = await make_project(session_factory, TaskType.DETECTION, ["sofa", "carpet"])
    async with session_factory() as session:
        rows = (await session.execute(select(ProjectClass))).scalars().all()
        for row in rows:
            row.sort_order = 0 if row.name == "carpet" else 1
        await session.commit()

    await run_job(session_factory, project_id)

    assert labeler.configs[0]["classes"] == ["carpet", "sofa"]


async def test_detection_without_classes_fails_with_hint(
    session_factory, key_path, monkeypatch
):
    labeler = RecordingDetector()
    install(monkeypatch, labeler)
    project_id = await make_project(session_factory, TaskType.DETECTION, [])

    job = await run_job(session_factory, project_id)

    assert job.status == JobStatus.FAILED
    assert job.result["error"] == "Add project classes before starting the labeling run"
    assert labeler.configs == []  # the engine was not even called
    async with session_factory() as session:
        assert (await session.execute(select(Annotation))).first() is None


async def test_ocr_project_config_has_no_classes(session_factory, key_path, monkeypatch):
    labeler = RecordingDetector()
    labeler.name = "rapidocr"
    install(monkeypatch, labeler)
    project_id = await make_project(session_factory, TaskType.OCR, [])

    job = await run_job(session_factory, project_id, labeler="rapidocr", config={"lang": "ru"})

    assert job.status == JobStatus.DONE, job.result
    assert labeler.configs == [{"lang": "ru"}]


async def test_project_classes_override_config_classes(
    session_factory, key_path, monkeypatch
):
    """The class list comes from the project, not from the job JSON — otherwise the
    labeling drifts away from what the human sees in the UI and gets in the export."""
    labeler = RecordingDetector()
    install(monkeypatch, labeler)
    project_id = await make_project(session_factory, TaskType.DETECTION, ["carpet"])

    await run_job(session_factory, project_id, config={"classes": ["stale"]})

    assert labeler.configs[0]["classes"] == ["carpet"]


async def test_detection_annotations_keep_class_label_and_score(
    session_factory, key_path, monkeypatch
):
    install(monkeypatch, RecordingDetector())
    project_id = await make_project(session_factory, TaskType.DETECTION, ["carpet"])

    job = await run_job(session_factory, project_id)

    assert job.result["annotations_created"] == 1
    async with session_factory() as session:
        annotation = (await session.execute(select(Annotation))).scalar_one()
    assert annotation.label == "carpet"
    assert annotation.confidence == 0.71
    assert annotation.text is None
    assert annotation.geometry == {
        "type": "bbox",
        "x": 10.0,
        "y": 20.0,
        "width": 30.0,
        "height": 40.0,
    }
    assert annotation.source["name"] == "owlv2"


async def test_job_payload_is_not_polluted_by_classes(
    session_factory, key_path, monkeypatch
):
    install(monkeypatch, RecordingDetector())
    project_id = await make_project(session_factory, TaskType.DETECTION, ["carpet"])

    job = await run_job(session_factory, project_id, config={"nms_iou": 0.4})

    assert job.payload["config"] == {"nms_iou": 0.4}



async def run_job_rerun(session_factory, project_id, rerun: bool) -> Job:
    async with session_factory() as session:
        job = Job(
            project_id=project_id,
            type=JobType.AUTOLABEL,
            payload={"labeler": "owlv2", "config": {}, "rerun": rerun},
        )
        session.add(job)
        await session.commit()
        job_id = str(job.id)
    await tasks.run_autolabel({}, job_id)
    async with session_factory() as session:
        return await session.get(Job, uuid.UUID(job_id))


async def test_rerun_keeps_reviewed_and_does_not_duplicate_it(
    session_factory, key_path, monkeypatch
):
    """A rerun after a class was added must not duplicate the human's work."""
    from app.models import AnnotationStatus

    install(monkeypatch, RecordingDetector())
    project_id = await make_project(session_factory, TaskType.DETECTION, ["car"])
    await run_job(session_factory, project_id)

    async with session_factory() as session:
        ann = (await session.execute(select(Annotation))).scalars().one()
        ann.status = AnnotationStatus.ACCEPTED
        await session.commit()

    # without rerun the image is skipped — the user would get zero for no reason
    job = await run_job_rerun(session_factory, project_id, rerun=False)
    assert job.result["annotations_created"] == 0
    assert job.result["skipped_already_labeled"] == 1

    # with rerun the engine ran again, but did not slip the same box in twice
    job = await run_job_rerun(session_factory, project_id, rerun=True)
    assert job.result["skipped_already_labeled"] == 0, "a rerun must not skip already-labeled images"
    assert job.result["skipped_duplicates"] == 1
    assert job.result["annotations_created"] == 0

    async with session_factory() as session:
        rows = (await session.execute(select(Annotation))).scalars().all()
    assert len(rows) == 1, "the accepted box got duplicated"
    assert rows[0].status == AnnotationStatus.ACCEPTED, "the human's edit was lost"


async def test_rerun_adds_boxes_for_newly_added_class(
    session_factory, key_path, monkeypatch
):
    """The main scenario: add a class -> rerun -> the boxes show up."""
    from app.models import AnnotationStatus

    install(monkeypatch, RecordingDetector())
    project_id = await make_project(session_factory, TaskType.DETECTION, ["car"])
    await run_job(session_factory, project_id)
    async with session_factory() as session:
        ann = (await session.execute(select(Annotation))).scalars().one()
        ann.status = AnnotationStatus.ACCEPTED
        await session.commit()

    # the user added a class: the engine now returns a box with the new label
    async with session_factory() as session:
        session.add(
            ProjectClass(
                project_id=project_id, name="wheel", color="#ef4444", sort_order=1
            )
        )
        await session.commit()

    class WheelDetector(RecordingDetector):
        def predict(self, image: bytes, config: dict):
            out = super().predict(image, config)
            out.append(
                SdkAnnotation(
                    geometry=BBox(x=200.0, y=200.0, width=20.0, height=20.0),
                    label="wheel",
                    confidence=0.6,
                )
            )
            return out

    install(monkeypatch, WheelDetector())
    job = await run_job_rerun(session_factory, project_id, rerun=True)

    assert job.result["annotations_created"] == 1  # only the new class
    async with session_factory() as session:
        rows = (await session.execute(select(Annotation))).scalars().all()
    labels = sorted(a.label for a in rows)
    assert labels == ["car", "wheel"]
    assert any(a.status == AnnotationStatus.ACCEPTED for a in rows)


async def test_rerun_drops_stale_pending_boxes(session_factory, key_path, monkeypatch):
    install(monkeypatch, RecordingDetector())
    project_id = await make_project(session_factory, TaskType.DETECTION, ["car"])
    await run_job(session_factory, project_id)

    job = await run_job_rerun(session_factory, project_id, rerun=True)

    assert job.result["replaced_pending"] == 1
    async with session_factory() as session:
        rows = (await session.execute(select(Annotation))).scalars().all()
    assert len(rows) == 1, "the old pending box had to be replaced, not duplicated"
