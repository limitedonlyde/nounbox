"""Шифрование секретов настроек (Fernet).

Ключ пользователь не генерирует руками: при первом обращении платформа создаёт
его сама в файле settings_key_path (volume, права 0600). Переопределяется
переменной окружения SETTINGS_ENCRYPTION_KEY.
"""

from __future__ import annotations

import logging
import os
import stat
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings

logger = logging.getLogger(__name__)


class SecretKeyError(RuntimeError):
    """Ключ шифрования недоступен или испорчен."""


class SecretDecryptError(RuntimeError):
    """Секрет не расшифровывается текущим ключом."""


def _generate_key_file(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    try:
        # O_EXCL: два процесса (api и worker) стартуют одновременно — победит один
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return path.read_bytes().strip()
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    logger.info("Generated settings encryption key at %s", path)
    return key


def load_key() -> bytes:
    if settings.settings_encryption_key:
        return settings.settings_encryption_key.strip().encode()

    path = Path(settings.settings_key_path)
    try:
        if path.exists():
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & 0o077:
                path.chmod(0o600)
                logger.warning("Fixed permissions of %s (were %o)", path, mode)
            key = path.read_bytes().strip()
            if not key:
                raise SecretKeyError(
                    f"Key file {path} is empty — delete it and restart"
                )
            return key
        return _generate_key_file(path)
    except OSError as exc:
        raise SecretKeyError(
            f"Could not read or create the encryption key {path}: {exc}. "
            "Point SETTINGS_KEY_PATH at a writable path, "
            "or set SETTINGS_ENCRYPTION_KEY (a Fernet key)."
        ) from exc


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    try:
        return Fernet(load_key())
    except (ValueError, TypeError) as exc:
        raise SecretKeyError(
            "The settings encryption key is invalid (a base64 Fernet key is expected)"
        ) from exc


def reset_key_cache() -> None:
    _fernet.cache_clear()


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise SecretDecryptError(
            "The stored secret cannot be decrypted — the encryption key has changed. "
            "Enter the Modal token again."
        ) from exc


def mask_token_id(token_id: str | None) -> str | None:
    """ak-1234567890abcdef -> ak-1234...cdef"""
    if not token_id:
        return None
    if len(token_id) <= 11:
        return f"{token_id[:3]}..."
    return f"{token_id[:7]}...{token_id[-4:]}"
