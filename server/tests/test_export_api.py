"""HTTP-слой экспорта: форматы по task_type, классы проекта в разметке."""

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
    """Байты картинок берутся из S3 — подменяем на настоящий PNG 640x480."""
    buffer = io.BytesIO()
    PILImage.new("RGB", (640, 480), "white").save(buffer, format="PNG")
    monkeypatch.setattr(storage, "get_bytes", lambda key: buffer.getvalue())


async def make_project(client, task_type: str | None = None) -> str:
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


async def export(client, project_id: str, fmt: str | None = None):
    query = f"?format={fmt}" if fmt else ""
    return await client.get(f"{BASE}/projects/{project_id}/export{query}")


def read_zip(response) -> zipfile.ZipFile:
    assert response.status_code == 200, response.text
    return zipfile.ZipFile(io.BytesIO(response.content))


# --- список форматов ---
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


# --- содержимое экспорта ---
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
    # class_idx = позиция класса в списке проекта (sort_order), не алфавит
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
