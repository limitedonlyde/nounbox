"""Ручки настроек: секрет не утекает, токен маскируется, deploy ставит Job."""

from sqlalchemy import select

from app.crypto import decrypt_secret
from app.models import GpuStatus, InstanceSettings, Job, JobStatus, JobType


async def test_get_settings_defaults(client):
    response = await client.get("/api/v1/settings")

    assert response.status_code == 200
    assert response.json() == {
        "modal_configured": False,
        "modal_token_id_masked": None,
        "gpu_status": "not_configured",
        "gpu_endpoint_url": None,
        "gpu_error": None,
    }


async def test_put_token_masks_id_and_hides_secret(client, session_factory, token_pair):
    token_id, token_secret = token_pair
    response = await client.put(
        "/api/v1/settings",
        json={"modal_token_id": token_id, "modal_token_secret": token_secret},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["modal_configured"] is True
    assert body["modal_token_id_masked"] == f"{token_id[:7]}...{token_id[-4:]}"
    assert body["gpu_status"] == "not_configured"
    # секрет не должен встречаться в ответе ни в каком виде
    assert token_secret not in response.text
    assert "modal_token_secret" not in body

    async with session_factory() as session:
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert row.modal_token_id == token_id
    assert row.modal_token_secret_encrypted is not None
    assert token_secret not in row.modal_token_secret_encrypted
    assert decrypt_secret(row.modal_token_secret_encrypted) == token_secret


async def test_get_after_put_never_returns_secret(client, token_pair):
    token_id, token_secret = token_pair
    await client.put(
        "/api/v1/settings",
        json={"modal_token_id": token_id, "modal_token_secret": token_secret},
    )

    response = await client.get("/api/v1/settings")

    assert token_secret not in response.text
    assert response.json()["modal_token_id_masked"].endswith(token_id[-4:])


async def test_put_rejects_proxy_token_without_echoing_it(client, session_factory):
    response = await client.put(
        "/api/v1/settings",
        json={
            "modal_token_id": "wk-ppcTj7Fx3henF5Woy3Gpnq",
            "modal_token_secret": "ws-verysecretvalue",
        },
    )

    assert response.status_code == 400
    assert "proxy" in response.json()["detail"].lower()
    assert "ws-verysecretvalue" not in response.text

    async with session_factory() as session:
        assert (await session.execute(select(InstanceSettings))).first() is None


async def test_put_rejects_garbage_token(client):
    response = await client.put(
        "/api/v1/settings",
        json={"modal_token_id": "garbage", "modal_token_secret": "garbage"},
    )

    assert response.status_code == 400
    assert "ak-" in response.json()["detail"]


async def test_put_same_token_keeps_ready_gpu(client, session_factory, token_pair):
    token_id, token_secret = token_pair
    await client.put(
        "/api/v1/settings",
        json={"modal_token_id": token_id, "modal_token_secret": token_secret},
    )
    async with session_factory() as session:
        row = (await session.execute(select(InstanceSettings))).scalar_one()
        row.gpu_status = GpuStatus.READY
        row.gpu_endpoint_url = "https://ws--autolabelui-gpu-fastapi-app.modal.run"
        await session.commit()

    same = await client.put(
        "/api/v1/settings",
        json={"modal_token_id": token_id, "modal_token_secret": token_secret},
    )
    assert same.json()["gpu_status"] == "ready"

    other = await client.put(
        "/api/v1/settings",
        json={"modal_token_id": "ak-other0000000000", "modal_token_secret": token_secret},
    )
    assert other.json()["gpu_status"] == "not_configured"
    assert other.json()["gpu_endpoint_url"] is None


async def test_delete_modal_clears_token(client, session_factory, token_pair):
    token_id, token_secret = token_pair
    await client.put(
        "/api/v1/settings",
        json={"modal_token_id": token_id, "modal_token_secret": token_secret},
    )

    response = await client.delete("/api/v1/settings/modal")

    assert response.status_code == 200
    assert response.json() == {
        "modal_configured": False,
        "modal_token_id_masked": None,
        "gpu_status": "not_configured",
        "gpu_endpoint_url": None,
        "gpu_error": None,
    }
    async with session_factory() as session:
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert row.modal_token_id is None
    assert row.modal_token_secret_encrypted is None


async def test_deploy_without_token_is_rejected(client, arq):
    response = await client.post("/api/v1/settings/gpu/deploy")

    assert response.status_code == 400
    assert arq.enqueued == []


async def test_deploy_creates_job_without_secret_in_payload(
    client, session_factory, arq, token_pair
):
    token_id, token_secret = token_pair
    await client.put(
        "/api/v1/settings",
        json={"modal_token_id": token_id, "modal_token_secret": token_secret},
    )

    response = await client.post("/api/v1/settings/gpu/deploy")

    assert response.status_code == 202
    body = response.json()
    assert body["type"] == "deploy_gpu"
    assert body["status"] == "queued"
    assert body["project_id"] is None
    assert token_secret not in response.text
    assert token_id not in response.text
    assert arq.enqueued == [("run_deploy_gpu", (body["id"],))]

    async with session_factory() as session:
        job = (await session.execute(select(Job))).scalar_one()
        row = (await session.execute(select(InstanceSettings))).scalar_one()
    assert job.type == JobType.DEPLOY_GPU
    assert job.status == JobStatus.QUEUED
    assert token_secret not in str(job.payload)
    assert row.gpu_status == GpuStatus.DEPLOYING

    # прогресс поллится обычной ручкой задач
    polled = await client.get(f"/api/v1/jobs/{body['id']}")
    assert polled.status_code == 200
    assert polled.json()["type"] == "deploy_gpu"
