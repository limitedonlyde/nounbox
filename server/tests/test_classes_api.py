"""Project classes: CRUD, name uniqueness, deleting a class that is in use."""

import uuid

import pytest
from sqlalchemy import select

from app.api.classes import PALETTE
from app.models import Annotation, Document, Image, ProjectClass

BASE = "/api/v1"


async def make_project(client, task_type: str | None = None) -> str:
    body: dict = {"name": "flats"}
    if task_type is not None:
        body["task_type"] = task_type
    response = await client.post(f"{BASE}/projects", json=body)
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def add_classes(client, project_id: str, *names: str) -> list[dict]:
    response = await client.put(
        f"{BASE}/projects/{project_id}/classes", json={"names": list(names)}
    )
    assert response.status_code == 200, response.text
    return response.json()


async def make_image(session_factory, project_id: str) -> uuid.UUID:
    async with session_factory() as session:
        document = Document(
            project_id=uuid.UUID(project_id), filename="a.png", s3_key="doc/a.png"
        )
        session.add(document)
        await session.flush()
        image = Image(document_id=document.id, s3_key="img/a.png")
        session.add(image)
        await session.commit()
        return image.id


async def add_annotation(client, image_id: uuid.UUID, label: str) -> str:
    response = await client.post(
        f"{BASE}/images/{image_id}/annotations",
        json={
            "geometry": {"type": "bbox", "x": 1, "y": 2, "width": 3, "height": 4},
            "label": label,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# --- project task_type ---
async def test_project_is_detection_by_default(client):
    project_id = await make_project(client)

    project = (await client.get(f"{BASE}/projects/{project_id}")).json()

    assert project["task_type"] == "detection"


async def test_project_task_type_ocr_is_accepted(client):
    project_id = await make_project(client, "ocr")

    assert (await client.get(f"{BASE}/projects/{project_id}")).json()["task_type"] == "ocr"


async def test_unknown_task_type_rejected(client):
    response = await client.post(
        f"{BASE}/projects", json={"name": "p", "task_type": "segmentation"}
    )

    assert response.status_code == 422


# --- POST / GET ---
async def test_new_project_has_no_classes(client):
    project_id = await make_project(client)

    response = await client.get(f"{BASE}/projects/{project_id}/classes")

    assert response.status_code == 200
    assert response.json() == []


async def test_created_classes_get_palette_colors_and_order(client):
    project_id = await make_project(client)

    first = (
        await client.post(f"{BASE}/projects/{project_id}/classes", json={"name": "carpet"})
    ).json()
    second = (
        await client.post(f"{BASE}/projects/{project_id}/classes", json={"name": "sofa"})
    ).json()

    assert (first["name"], first["sort_order"], first["color"]) == (
        "carpet",
        0,
        PALETTE[0],
    )
    assert (second["sort_order"], second["color"]) == (1, PALETTE[1])
    assert first["color"] != second["color"]
    listed = (await client.get(f"{BASE}/projects/{project_id}/classes")).json()
    assert [c["name"] for c in listed] == ["carpet", "sofa"]


async def test_explicit_color_wins(client):
    project_id = await make_project(client)

    item = (
        await client.post(
            f"{BASE}/projects/{project_id}/classes",
            json={"name": "chandelier", "color": "#123ABC"},
        )
    ).json()

    assert item["color"] == "#123ABC"


async def test_palette_repeats_after_full_circle(client):
    project_id = await make_project(client)

    classes = await add_classes(client, project_id, *[f"class{i}" for i in range(14)])

    assert classes[len(PALETTE)]["color"] == PALETTE[0]
    assert len({c["color"] for c in classes[: len(PALETTE)]}) == len(PALETTE)


async def test_name_is_normalized(client):
    project_id = await make_project(client)

    item = (
        await client.post(
            f"{BASE}/projects/{project_id}/classes", json={"name": "  coffee   table "}
        )
    ).json()

    assert item["name"] == "coffee table"


@pytest.mark.parametrize("name", ["ковёр", "", "   ", "2fast", "sofa!"])
async def test_non_english_or_empty_name_rejected(client, name):
    project_id = await make_project(client)

    response = await client.post(
        f"{BASE}/projects/{project_id}/classes", json={"name": name}
    )

    assert response.status_code == 422


async def test_bad_color_rejected(client):
    project_id = await make_project(client)

    response = await client.post(
        f"{BASE}/projects/{project_id}/classes", json={"name": "sofa", "color": "red"}
    )

    assert response.status_code == 422


# --- uniqueness ---
async def test_duplicate_name_rejected(client):
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")

    response = await client.post(
        f"{BASE}/projects/{project_id}/classes", json={"name": "sofa"}
    )

    assert response.status_code == 409
    assert "sofa" in str(response.json()["detail"])


async def test_duplicate_name_is_case_insensitive(client):
    """'Sofa' and 'sofa' are one query to the engine — we do not want doubled boxes."""
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")

    response = await client.post(
        f"{BASE}/projects/{project_id}/classes", json={"name": "Sofa"}
    )

    assert response.status_code == 409


async def test_same_name_in_other_project_is_fine(client):
    first = await make_project(client)
    second = await make_project(client)
    await add_classes(client, first, "sofa")

    response = await client.post(f"{BASE}/projects/{second}/classes", json={"name": "sofa"})

    assert response.status_code == 201


# --- PUT: replacing the whole list ---
async def test_put_replaces_list_and_keeps_surviving_classes(client):
    project_id = await make_project(client)
    before = await add_classes(client, project_id, "carpet", "sofa", "lamp")
    sofa = next(c for c in before if c["name"] == "sofa")

    after = await add_classes(client, project_id, "sofa", "chandelier")

    assert [c["name"] for c in after] == ["sofa", "chandelier"]
    assert [c["sort_order"] for c in after] == [0, 1]
    # a surviving class keeps its id and color — Review UI boxes do not change color
    assert after[0]["id"] == sofa["id"]
    assert after[0]["color"] == sofa["color"]
    assert (await client.get(f"{BASE}/projects/{project_id}/classes")).json() == after


async def test_put_deduplicates_names(client):
    project_id = await make_project(client)

    classes = await add_classes(client, project_id, "sofa", "Sofa", "carpet", "sofa")

    assert [c["name"] for c in classes] == ["sofa", "carpet"]


async def test_put_empty_list_clears_classes(client):
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")

    assert await add_classes(client, project_id) == []


async def test_put_rejects_dropping_class_with_annotations(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa", "carpet")
    image_id = await make_image(session_factory, project_id)
    await add_annotation(client, image_id, "sofa")

    response = await client.put(
        f"{BASE}/projects/{project_id}/classes", json={"names": ["carpet"]}
    )

    assert response.status_code == 409
    assert "sofa" in str(response.json()["detail"])
    names = [c["name"] for c in (await client.get(f"{BASE}/projects/{project_id}/classes")).json()]
    assert names == ["sofa", "carpet"]


async def test_put_case_rename_reaches_annotations(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")
    image_id = await make_image(session_factory, project_id)
    annotation_id = await add_annotation(client, image_id, "sofa")

    assert [c["name"] for c in await add_classes(client, project_id, "Sofa")] == ["Sofa"]
    async with session_factory() as session:
        annotation = await session.get(Annotation, uuid.UUID(annotation_id))
    assert annotation.label == "Sofa"


async def test_put_rejects_non_english_name(client):
    project_id = await make_project(client)

    response = await client.put(
        f"{BASE}/projects/{project_id}/classes", json={"names": ["sofa", "диван"]}
    )

    assert response.status_code == 422


# --- PATCH ---
async def test_patch_updates_color_and_order(client):
    project_id = await make_project(client)
    classes = await add_classes(client, project_id, "sofa", "carpet")

    response = await client.patch(
        f"{BASE}/classes/{classes[1]['id']}", json={"color": "#000000", "sort_order": 0}
    )

    assert response.status_code == 200
    assert response.json()["color"] == "#000000"
    listed = (await client.get(f"{BASE}/projects/{project_id}/classes")).json()
    assert [c["name"] for c in listed] == ["carpet", "sofa"]


async def test_patch_renames_class_and_its_annotations(client, session_factory):
    project_id = await make_project(client)
    classes = await add_classes(client, project_id, "sofa")
    image_id = await make_image(session_factory, project_id)
    annotation_id = await add_annotation(client, image_id, "sofa")

    response = await client.patch(f"{BASE}/classes/{classes[0]['id']}", json={"name": "couch"})

    assert response.status_code == 200
    assert response.json()["name"] == "couch"
    async with session_factory() as session:
        annotation = await session.get(Annotation, uuid.UUID(annotation_id))
    assert annotation.label == "couch"


async def test_patch_to_existing_name_rejected(client):
    project_id = await make_project(client)
    classes = await add_classes(client, project_id, "sofa", "carpet")

    response = await client.patch(f"{BASE}/classes/{classes[1]['id']}", json={"name": "Sofa"})

    assert response.status_code == 409


async def test_patch_keeps_untouched_fields(client):
    project_id = await make_project(client)
    classes = await add_classes(client, project_id, "sofa")

    updated = (
        await client.patch(f"{BASE}/classes/{classes[0]['id']}", json={"name": "couch"})
    ).json()

    assert updated["color"] == classes[0]["color"]
    assert updated["sort_order"] == classes[0]["sort_order"]


# --- DELETE ---
async def test_delete_unused_class(client, session_factory):
    project_id = await make_project(client)
    classes = await add_classes(client, project_id, "sofa")

    response = await client.delete(f"{BASE}/classes/{classes[0]['id']}")

    assert response.status_code == 204
    async with session_factory() as session:
        assert (await session.execute(select(ProjectClass))).first() is None


async def test_delete_used_class_returns_409_with_count(client, session_factory):
    project_id = await make_project(client)
    classes = await add_classes(client, project_id, "sofa")
    image_id = await make_image(session_factory, project_id)
    await add_annotation(client, image_id, "sofa")
    await add_annotation(client, image_id, "sofa")

    response = await client.delete(f"{BASE}/classes/{classes[0]['id']}")

    assert response.status_code == 409
    assert response.json()["detail"]["annotations"] == 2
    assert (await client.get(f"{BASE}/projects/{project_id}/classes")).json()


async def test_force_delete_removes_class_with_annotations(client, session_factory):
    project_id = await make_project(client)
    classes = await add_classes(client, project_id, "sofa", "carpet")
    image_id = await make_image(session_factory, project_id)
    await add_annotation(client, image_id, "sofa")
    kept = await add_annotation(client, image_id, "carpet")

    response = await client.delete(f"{BASE}/classes/{classes[0]['id']}?force=true")

    assert response.status_code == 204
    remaining = (await client.get(f"{BASE}/images/{image_id}/annotations")).json()
    assert [a["id"] for a in remaining] == [kept]
    names = [c["name"] for c in (await client.get(f"{BASE}/projects/{project_id}/classes")).json()]
    assert names == ["carpet"]


async def test_annotations_of_other_project_do_not_block_delete(client, session_factory):
    """A class of project A must not count same-named annotations from project B."""
    first = await make_project(client)
    second = await make_project(client)
    classes = await add_classes(client, first, "sofa")
    await add_classes(client, second, "sofa")
    image_id = await make_image(session_factory, second)
    await add_annotation(client, image_id, "sofa")

    assert (await client.delete(f"{BASE}/classes/{classes[0]['id']}")).status_code == 204


# --- 404 ---
async def test_classes_of_unknown_project(client):
    missing = uuid.uuid4()

    assert (await client.get(f"{BASE}/projects/{missing}/classes")).status_code == 404
    assert (
        await client.post(f"{BASE}/projects/{missing}/classes", json={"name": "sofa"})
    ).status_code == 404
    assert (
        await client.put(f"{BASE}/projects/{missing}/classes", json={"names": []})
    ).status_code == 404


async def test_unknown_class_id(client):
    missing = uuid.uuid4()

    assert (await client.patch(f"{BASE}/classes/{missing}", json={"name": "sofa"})).status_code == 404
    assert (await client.delete(f"{BASE}/classes/{missing}")).status_code == 404


async def test_put_force_drops_class_with_annotations(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa", "carpet")
    image_id = await make_image(session_factory, project_id)
    await add_annotation(client, image_id, "sofa")

    response = await client.put(
        f"{BASE}/projects/{project_id}/classes",
        params={"force": "true"},
        json={"names": ["carpet"]},
    )

    assert response.status_code == 200
    assert [c["name"] for c in response.json()] == ["carpet"]
    left = (await client.get(f"{BASE}/images/{image_id}/annotations")).json()
    assert left == []


async def test_put_409_body_carries_annotation_count(client, session_factory):
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")
    image_id = await make_image(session_factory, project_id)
    await add_annotation(client, image_id, "sofa")
    await add_annotation(client, image_id, "sofa")

    response = await client.put(f"{BASE}/projects/{project_id}/classes", json={"names": []})

    assert response.status_code == 409
    assert response.json()["detail"]["annotations"] == 2


async def test_put_new_class_does_not_reuse_colour_of_survivor(client):
    project_id = await make_project(client)
    await add_classes(client, project_id, "sofa")

    # the new name comes FIRST — by index it would get the survivor's color
    response = await client.put(
        f"{BASE}/projects/{project_id}/classes", json={"names": ["carpet", "sofa"]}
    )

    colors = [c["color"] for c in response.json()]
    assert len(set(colors)) == 2, colors
