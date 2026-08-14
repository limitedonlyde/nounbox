"""POST /projects/{id}/autolabel: an engine may only run on a task it serves.

The worker repeats this check and is the real guard (the API is not the only
way a job gets created). This one exists so the user finds out immediately,
in the UI, instead of watching a job run and fail.
"""


async def make_project(client, task_type: str) -> str:
    response = await client.post(
        "/api/v1/projects", json={"name": "p", "task_type": task_type}
    )
    return response.json()["id"]


async def test_ocr_engine_rejected_for_a_detection_project(client, arq):
    project_id = await make_project(client, "detection")

    response = await client.post(
        f"/api/v1/projects/{project_id}/autolabel", json={"labeler": "modal_gpu"}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "handles ocr projects" in detail
    assert "this project is detection" in detail
    assert arq.enqueued == []


async def test_detection_gpu_rejected_for_an_ocr_project(client, arq):
    project_id = await make_project(client, "ocr")

    response = await client.post(
        f"/api/v1/projects/{project_id}/autolabel", json={"labeler": "modal_gpu_detect"}
    )

    assert response.status_code == 400
    assert arq.enqueued == []


async def test_matching_engine_is_accepted(client, arq):
    project_id = await make_project(client, "detection")

    response = await client.post(
        f"/api/v1/projects/{project_id}/autolabel", json={"labeler": "modal_gpu_detect"}
    )

    assert response.status_code == 202
    assert len(arq.enqueued) == 1


async def test_unknown_engine_is_not_rejected_here(client, arq):
    """A third-party plugin declares no task, and the worker is the place that
    knows what is installed — the API must not guess."""
    project_id = await make_project(client, "detection")

    response = await client.post(
        f"/api/v1/projects/{project_id}/autolabel", json={"labeler": "my_engine"}
    )

    assert response.status_code == 202


async def test_fan_out_is_not_rejected_here(client, arq):
    """labeler=null means "every installed engine"; the worker drops the ones
    that do not fit the task and reports them in skipped_labelers."""
    project_id = await make_project(client, "detection")

    response = await client.post(
        f"/api/v1/projects/{project_id}/autolabel", json={"labeler": None}
    )

    assert response.status_code == 202
