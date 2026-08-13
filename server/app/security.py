"""Protection for the endpoints that can spend the owner's money and hold
their secrets.

The platform has no real authentication yet — it is single-user and listens on
localhost. But the settings endpoints are special: they take the Modal API
token and start a deploy into the user's own Modal account. Anyone who reaches
port (a port forward, a reverse proxy, a colleague on the same network) could
spend that person's GPU budget. Hence a shared access token from the environment.

APP_ACCESS_TOKEN empty (the default) — the endpoints stay open, as before, but
the platform warns about it in the log and in the GET /settings response, so
that the missing protection is impossible to overlook.
"""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Request

from app.config import settings

logger = logging.getLogger(__name__)

_warned = False


def access_token_configured() -> bool:
    return bool(settings.app_access_token)


def require_access(request: Request) -> None:
    """FastAPI dependency: lets the request through on a valid Bearer token."""
    global _warned
    expected = settings.app_access_token
    if not expected:
        if not _warned:
            _warned = True
            logger.warning(
                "APP_ACCESS_TOKEN is not set: the settings endpoints (Modal token, "
                "GPU deploy) are open to anyone who can reach the API port. Set the "
                "variable if anyone else can reach this platform."
            )
        return

    scheme, _, value = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        value.strip().encode(), expected.encode()
    ):
        raise HTTPException(
            401,
            "Access token required: send the header "
            "Authorization: Bearer <APP_ACCESS_TOKEN>",
            headers={"WWW-Authenticate": "Bearer"},
        )
