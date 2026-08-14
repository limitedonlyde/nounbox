import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Job, JobStatus, JobType, Project
from app.schemas import AutolabelRequest, JobOut
from app.services import settings_store

router = APIRouter(tags=["jobs"])


@router.post("/projects/{project_id}/autolabel", response_model=JobOut, status_code=202)
async def start_autolabel(
    project_id: uuid.UUID,
    body: AutolabelRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")

    # An engine that cannot serve this task type is refused right here, so the
    # user sees it in the UI instead of waiting for a job to fail. The worker
    # repeats the check — it is the real guard, this one is only fast feedback.
    if body.labeler and not settings_store.labeler_supports_task(
        body.labeler, project.task_type
    ):
        serves = "/".join(settings_store.CATALOG[body.labeler]["tasks"])
        raise HTTPException(
            400,
            f"Engine {body.labeler} handles {serves} projects; "
            f"this project is {project.task_type}",
        )

    job = Job(
        project_id=project_id,
        type=JobType.AUTOLABEL,
        payload={
            "labeler": body.labeler,
            "config": body.config,
            "rerun": body.rerun,
        },
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    await request.app.state.arq.enqueue_job("run_autolabel", str(job.id))
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/projects/{project_id}/jobs/active", response_model=JobOut | None)
async def get_active_job(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    """The project's unfinished job, if there is one.

    Needed so the page can pick the work back up after a tab reload, a network
    drop or the laptop going to sleep: otherwise polling dies together with the
    tab and the user is stuck on "Autolabel started..." forever.
    """
    result = await session.execute(
        select(Job)
        .where(
            Job.project_id == project_id,
            Job.status.in_((JobStatus.QUEUED, JobStatus.RUNNING)),
        )
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()
