from fastapi import APIRouter

from app.api import (
    annotations,
    classes,
    documents,
    exports,
    images,
    jobs,
    labelers,
    projects,
    settings,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(projects.router)
api_router.include_router(classes.router)
api_router.include_router(documents.router)
api_router.include_router(images.router)
api_router.include_router(annotations.router)
api_router.include_router(jobs.router)
api_router.include_router(exports.router)
api_router.include_router(settings.router)
api_router.include_router(labelers.router)
