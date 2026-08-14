from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = (
        "postgresql+asyncpg://autolabel:autolabel@localhost:5432/autolabel"
    )
    redis_url: str = "redis://localhost:6379/0"

    # How many images a labeling run may have in flight against a REMOTE engine
    # at once (local engines always run one at a time — they already occupy the
    # worker's cores). Measured against the Modal T4 recipe: its shipped
    # max_containers=4 saturates at ~296 images/min, so sending more than a
    # handful concurrently only grows latency. Keep this at or below the
    # recipe's max_containers.
    # Bounded on purpose: every in-flight image holds a thread from the pool
    # that run_in_threadpool draws on (40 for the whole worker process), so an
    # operator raising this to "as high as it goes" would starve ingest jobs
    # running beside it.
    remote_labeler_concurrency: int = Field(4, ge=1, le=16)

    s3_endpoint_url: str = "http://localhost:9000"
    # public endpoint for presigned URLs (the browser on the host opens them)
    s3_public_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "autolabel"
    s3_secret_key: str = "autolabel123"
    s3_bucket: str = "autolabel"

    cors_origins: str = "http://localhost:5173"

    # Encryption key for settings secrets. Generated automatically on first
    # start into a file (volume, mode 0600); settings_encryption_key wins over it.
    settings_key_path: str = "/data/settings.key"
    settings_encryption_key: str = ""

    # "Hard" mode: the platform deploys the GPU recipe into the user's own
    # Modal account. The default recipe path sits next to the server code.
    # empty name -> deploy under the app name from the recipe itself
    modal_gpu_app_name: str = "nounbox-gpu"
    modal_gpu_recipe_path: str = ""
    # The detection GPU is a SECOND Modal app with its own image and its own
    # name: the OCR and detection recipes share no dependency. Per-engine
    # overrides on purpose — one shared path would deploy one recipe as both.
    modal_gpu_detect_app_name: str = "nounbox-gpu-detect"
    modal_gpu_detect_recipe_path: str = ""
    # if set, the recipe protects /predict with a Bearer header, and the
    # worker puts the same token into the modal_gpu engine config
    nounbox_gpu_token: str = ""
    # Shared access token for the settings endpoints (Modal token, GPU deploy).
    # Empty — the endpoints are open, and the platform warns about it.
    app_access_token: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
