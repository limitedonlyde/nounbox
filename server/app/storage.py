"""S3-совместимое хранилище (MinIO). Синхронный boto3 — вызывать через run_in_threadpool."""

from collections.abc import Sequence

import boto3
from botocore.client import Config

from app.config import settings

# лимит S3 на количество ключей в одном delete_objects
DELETE_BATCH = 1000


def _client(public: bool = False):
    return boto3.client(
        "s3",
        endpoint_url=(
            settings.s3_public_endpoint_url if public else settings.s3_endpoint_url
        ),
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket() -> None:
    s3 = _client()
    buckets = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    if settings.s3_bucket not in buckets:
        s3.create_bucket(Bucket=settings.s3_bucket)


def put_bytes(key: str, data: bytes, content_type: str = "") -> None:
    _client().put_object(
        Bucket=settings.s3_bucket, Key=key, Body=data, ContentType=content_type
    )


def get_bytes(key: str) -> bytes:
    return _client().get_object(Bucket=settings.s3_bucket, Key=key)["Body"].read()


def delete_objects(keys: Sequence[str]) -> None:
    """Удалить объекты пачками (S3 принимает не больше 1000 ключей за запрос).

    Отсутствующий объект ошибкой не считается — S3 удаляет идемпотентно.
    """
    if not keys:
        return
    s3 = _client()
    for start in range(0, len(keys), DELETE_BATCH):
        chunk = keys[start : start + DELETE_BATCH]
        s3.delete_objects(
            Bucket=settings.s3_bucket,
            Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
        )


def presigned_url(key: str, expires: int = 3600) -> str:
    # SigV4 подписывает Host — генерируем через клиент с публичным endpoint
    return _client(public=True).generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires,
    )
