"""S3-compatible storage (MinIO). boto3 is synchronous — call via run_in_threadpool."""

from collections.abc import Sequence

import boto3
from botocore.client import Config

from app.config import settings

# S3 limit on the number of keys in a single delete_objects
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
    """Delete objects in batches (S3 accepts at most 1000 keys per request).

    A missing object is not an error — S3 deletes idempotently.
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
    # SigV4 signs the Host header — generate with a client on the public endpoint
    return _client(public=True).generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires,
    )
