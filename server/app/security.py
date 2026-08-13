"""Защита ручек, которые распоряжаются чужими деньгами и секретами.

Полноценной аутентификации в платформе пока нет — она одно-пользовательская и
слушает localhost. Но ручки настроек особые: через них вводится API-токен
Modal и запускается деплой в чужой аккаунт. Любой, кто дотянулся до порта
(проброс, reverse-proxy, коллега в той же сети), мог бы потратить чужой
GPU-бюджет. Поэтому здесь — общий токен доступа из окружения.

APP_ACCESS_TOKEN пуст (дефолт) — ручки открыты, как раньше, но платформа
предупреждает об этом в логе и в ответе GET /settings, чтобы отсутствие
защиты нельзя было не заметить.
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
    """Зависимость FastAPI: пускает при верном Bearer-токене."""
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
