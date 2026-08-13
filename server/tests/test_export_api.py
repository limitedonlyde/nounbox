"""Export HTTP layer: formats by task_type, project classes in the labels."""

import io
import json
import uuid
import zipfile

import pytest
from PIL import Image as PILImage

from app import storage
from app.models import Annotation, AnnotationStatus, Document, Image

BASE = "/api/v1"


@pytest.fixture(autouse=True)
def fake_storage(monkeypatch):
    """Image bytes come from S3 — we substitute a real 640x480 PNG."""
    buffer = io.BytesIO()
    PILImage.new("RGB", (640, 480), "white").save(buffer, format="PNG")
    monkeypatch.setattr(storage, "get_bytes", lambda key: buffer.getvalue())


async def make_project(client, task_type: str | None = None) -> str:
    # A non-ASCII project name on purpose: this is the suite's only check that
    # one survives the HTTP layer, the database round-trip and the export ZIP.
    body: dict = {"name": "квартиры"}
    if task_type is not None:
        body["task_type"] = task_type
    response = await client.post(f"{BASE}/projects", json=body)
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def add_classes(client, project_id: str, *names: str) -> None:
    response = await client.put(
        f"{BASE}/projects/{project_id}/classes", json={"names": list(names)}
    )
    assert response.status_code == 200, response.text


async def seed_annotations(
    session_factory, project_id: str, labelled: list[tuple[str, AnnotationStatus]]
) -> None:
    async with session_factory() as session:
        document = Document(
            project_id=uuid.UUID(project_id), filename="a.png", s3_key="doc/a.png"
        )
        session.add(document)
        await session.flush()
        image = Image(
            document_id=document.id, s3_key="img/a.png", width=640, height=480
        )
        session.add(image)
        await session.flush()
        for label, status in labelled:
            session.add(
                Annotation(
                    image_id=image.id,
                    geometry={
                        "type": "bbox",
                        "x": 100,
                        "y": 50,
                        "width": 200,
                        "height": 100,
                    },
                    label=label,
                    text="АКТ",
                    attrs={},
                    confidence=0.9,
                    source={"type": "engine", "name": "owlv2"},
                    status=status,
                )
            )
        await session.commit()


async def seed_image(
    session_factory,
    project_id: str,
    labelled: tuple[tuple[str, AnnotationStatus], ...] = (),
    reviewed: bool = False,
) -> uuid.UUID:
    """A separate project frame with the given annotations; returns its id."""
    async with session_factory() as session:
        document = Document(
            project_id=uuid.UUID(project_id), filename="a.png", s3_key="doc/a.png"
        )
        session.add(document)
        await session.flush()
        image = Image(
            document_id=document.id,
            s3_key="img/a.png",
            width=640,
            height=480,
            reviewed=reviewed,
        )
        session.add(image)
        await session.flush()
        for label, status in labelled:
            session.add(
                Annotation(
                    image_id=image.id,
                    geometry={
                        "type": "bbox",
                        "x": 100,
                        "y": 50,
                        "width": 200,
                        "height": 100,
                    },
                    label=label,
                    attrs={},
                    confidence=0.9,
                    source={"type": "engine", "name": "owlv2"},
                    status=status,
                )
            )
        await session.commit()
        return image.id


def yolo_label(zf: zipfile.ZipFile, image_id: uuid.UUID) -> str:
    """Contents of labels/<split>/<stem>.txt for a frame (split chosen by hash)."""
    names = [n for n in zf.namelist() if n.endswith(f"/{image_id.hex}.txt")]
    assert names, f"no label file for {image_id.hex} in {zf.namelist()}"
    return zf.read(names[0]).decode()


async def export(client, project_id: str, fmt: str | None = None):
    query = f"?format={fmt}" if fmt else ""
    return await client.get(f"{BASE}/projects/{project_id}/export{query}")


def read_zip(response) -> zipfile.ZipFile:
    assert response.status_code == 200, response.text
    return zipfile.ZipFile(io.BytesIO(response.content))


# --- format list ---
async def test_formats_endpoint_follows_task_type(client):
    detection = await make_project(client)
    ocr = await make_project(client, "ocr")

    assert (await client.get(f"{BASE}/projects/{detection}/export/formats")).json() == {
        "task_type": "detection",
        "formats": ["yolo_detect", "coco"],
    }
    assert (await client.get(f"{BASE}/projects/{ocr}/export/formats")).json() == {
        "task_type": "ocr",
        "formats": ["paddleocr_det", "paddleocr_rec", "coco"],
    }


async def test_formats_endpoint_404_for_unknown_project(client):
    response = await client.get(f"{BASE}/projects/{uuid.uuid4()}/export/formats")
    assert response.status_code == 404


async def test_ocr_format_rejected_for_detection_project(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet")
    await seed_annotations(
        session_factory, project_id, [("carpet", AnnotationStatus.ACCEPTED)]
    )

    response = await export(client, project_id, "paddleocr_det")
    assert response.status_code == 400
    assert "task_type=detection" in response.json()["detail"]


async def test_yolo_rejected_for_ocr_project(client, session_factory):
    project_id = await make_project(client, "ocr")
    await seed_annotations(
        session_factory, project_id, [("text_line", AnnotationStatus.ACCEPTED)]
    )

    response = await export(client, project_id, "yolo_detect")
    assert response.status_code == 400
    assert "task_type=ocr" in response.json()["detail"]


# --- export contents ---
async def test_default_format_is_yolo_for_detection(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet", "sofa")
    await seed_annotations(
        session_factory, project_id, [("sofa", AnnotationStatus.ACCEPTED)]
    )

    zf = read_zip(await export(client, project_id))
    names = zf.namelist()
    assert "data.yaml" in names
    labels = [n for n in names if n.startswith("labels/")]
    # class_idx = class position in the project list (sort_order), not alphabet
    assert zf.read(labels[0]).decode() == "1 0.312500 0.208333 0.312500 0.208333"
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format"] == "yolo_detect"
    assert manifest["classes"] == ["carpet", "sofa"]


async def test_default_format_is_paddleocr_det_for_ocr(client, session_factory):
    project_id = await make_project(client, "ocr")
    await seed_annotations(
        session_factory, project_id, [("text_line", AnnotationStatus.ACCEPTED)]
    )

    zf = read_zip(await export(client, project_id))
    assert "label.txt" in zf.namelist()
    assert json.loads(zf.read("manifest.json"))["format"] == "paddleocr_det"


async def test_coco_categories_come_from_project_classes(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet", "sofa", "chandelier")
    await seed_annotations(
        session_factory, project_id, [("chandelier", AnnotationStatus.EDITED)]
    )

    zf = read_zip(await export(client, project_id, "coco"))
    coco = json.loads(zf.read("annotations.json"))
    assert [c["name"] for c in coco["categories"]] == ["carpet", "sofa", "chandelier"]
    assert coco["annotations"][0]["category_id"] == 3
    assert coco["annotations"][0]["bbox"] == [100, 50, 200, 100]


async def test_detection_export_without_classes_fails(client, session_factory):
    project_id = await make_project(client)
    await seed_annotations(
        session_factory, project_id, [("carpet", AnnotationStatus.ACCEPTED)]
    )

    response = await export(client, project_id, "yolo_detect")
    assert response.status_code == 400
    assert "classes" in response.json()["detail"]


async def test_only_verified_annotations_are_exported(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet")
    await seed_annotations(
        session_factory,
        project_id,
        [
            ("carpet", AnnotationStatus.ACCEPTED),
            ("carpet", AnnotationStatus.EDITED),
            ("carpet", AnnotationStatus.PENDING),
            ("carpet", AnnotationStatus.REJECTED),
        ],
    )

    zf = read_zip(await export(client, project_id, "yolo_detect"))
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["annotations"] == 2
    assert manifest["statuses_included"] == ["accepted", "edited"]


# --- background frames: reviewed by a human, no objects ---
async def test_frame_with_only_rejected_annotations_is_exported_as_background(
    client, session_factory
):
    """All boxes rejected — the frame is reviewed and empty; for a detector that
    is a negative example, not a reason to drop it from the dataset."""
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet")
    with_object = await seed_image(
        session_factory, project_id, (("carpet", AnnotationStatus.ACCEPTED),)
    )
    background = await seed_image(
        session_factory, project_id, (("carpet", AnnotationStatus.REJECTED),)
    )

    zf = read_zip(await export(client, project_id, "yolo_detect"))
    assert yolo_label(zf, with_object) == "0 0.312500 0.208333 0.312500 0.208333"
    assert yolo_label(zf, background) == ""  # an empty .txt is legal background
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["images"] == 2
    assert manifest["annotations"] == 1
    assert manifest["background_images"] == 1


async def test_image_marked_reviewed_without_annotations_is_exported(
    client, session_factory
):
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet")
    await seed_image(
        session_factory, project_id, (("carpet", AnnotationStatus.ACCEPTED),)
    )
    background = await seed_image(session_factory, project_id, (), reviewed=True)

    zf = read_zip(await export(client, project_id, "yolo_detect"))
    assert yolo_label(zf, background) == ""
    assert json.loads(zf.read("manifest.json"))["images"] == 2


async def test_unreviewed_images_stay_out_of_the_dataset(client, session_factory):
    """A frame no human has looked at must not go into training as background."""
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet")
    await seed_image(
        session_factory, project_id, (("carpet", AnnotationStatus.ACCEPTED),)
    )
    untouched = await seed_image(session_factory, project_id, ())
    only_pending = await seed_image(
        session_factory, project_id, (("carpet", AnnotationStatus.PENDING),)
    )

    zf = read_zip(await export(client, project_id, "yolo_detect"))
    names = zf.namelist()
    assert not [n for n in names if untouched.hex in n]
    assert not [n for n in names if only_pending.hex in n]
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["images"] == 1
    assert manifest["background_images"] == 0


async def test_coco_keeps_background_image_without_annotations(
    client, session_factory
):
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet")
    await seed_image(
        session_factory, project_id, (("carpet", AnnotationStatus.ACCEPTED),)
    )
    await seed_image(session_factory, project_id, (), reviewed=True)

    zf = read_zip(await export(client, project_id, "coco"))
    coco = json.loads(zf.read("annotations.json"))
    assert len(coco["images"]) == 2
    assert len(coco["annotations"]) == 1
    # the background frame has an images entry and the file, but no annotations
    annotated = {entry["image_id"] for entry in coco["annotations"]}
    background = [entry for entry in coco["images"] if entry["id"] not in annotated]
    assert len(background) == 1
    assert all(entry["file_name"] in zf.namelist() for entry in coco["images"])


async def test_export_without_any_reviewed_image_fails(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "carpet")
    await seed_image(
        session_factory, project_id, (("carpet", AnnotationStatus.PENDING),)
    )

    response = await export(client, project_id, "yolo_detect")
    assert response.status_code == 400
    assert "reviewed" in response.json()["detail"]


async def _labels_in(zip_bytes: bytes) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        return {
            n: zf.read(n).decode()
            for n in zf.namelist()
            if n.startswith("labels/") and n.endswith(".txt")
        }


async def test_partially_reviewed_frame_is_not_exported_as_background(
    client, session_factory
):
    """One rejected box, one unreviewed — the frame is unfinished, NOT background.

    An empty label file here would teach the model to treat the object as background.
    """
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")
    await seed_image(
        session_factory,
        project_id,
        (("sofa", AnnotationStatus.REJECTED), ("sofa", AnnotationStatus.PENDING)),
    )

    response = await client.get(
        f"{BASE}/projects/{project_id}/export", params={"format": "yolo_detect"}
    )

    assert response.status_code == 400, "nothing to export — the frame is unfinished"


async def test_fully_rejected_frame_becomes_background(client, session_factory):
    """All boxes rejected, none left unreviewed — a valid background example."""
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")
    await seed_image(
        session_factory, project_id, (("sofa", AnnotationStatus.REJECTED),)
    )

    response = await client.get(
        f"{BASE}/projects/{project_id}/export", params={"format": "yolo_detect"}
    )

    assert response.status_code == 200, response.text
    labels = await _labels_in(response.content)
    # the lone frame deliberately goes into both train and val — otherwise one
    # of the parts would be left empty
    assert labels, "the frame must land in the dataset"
    assert all(v.strip() == "" for v in labels.values()), "background frame: empty .txt"


async def test_image_marked_reviewed_becomes_background(client, session_factory):
    """A frame without a single box enters the dataset only after it is
    explicitly marked reviewed.
    """
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")
    image_id = await seed_image(session_factory, project_id, ())

    empty = await client.get(
        f"{BASE}/projects/{project_id}/export", params={"format": "yolo_detect"}
    )
    assert empty.status_code == 400, "an unreviewed frame does not enter the dataset"

    marked = await client.patch(f"{BASE}/images/{image_id}", json={"reviewed": True})
    assert marked.status_code == 200, marked.text
    assert marked.json()["reviewed"] is True

    response = await client.get(
        f"{BASE}/projects/{project_id}/export", params={"format": "yolo_detect"}
    )
    assert response.status_code == 200, response.text
    assert next(iter((await _labels_in(response.content)).values())).strip() == ""


async def test_background_frames_stay_out_of_ocr_exports(client, session_factory):
    """A frame with no lines is junk in an OCR dataset: the loader drops it anyway."""
    project_id = await make_project(client, task_type="ocr")
    await seed_image(
        session_factory, project_id, (("text_line", AnnotationStatus.ACCEPTED),)
    )
    background = await seed_image(session_factory, project_id, ())
    await client.patch(f"{BASE}/images/{background}", json={"reviewed": True})

    response = await client.get(
        f"{BASE}/projects/{project_id}/export", params={"format": "paddleocr_det"}
    )

    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        images = [n for n in zf.namelist() if n.startswith("images/")]
        lines = zf.read("label.txt").decode().strip().split("\n")
    assert len(images) == 1, "a background frame must not reach an OCR export"
    assert len(lines) == 1
