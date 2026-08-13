from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = (
        "postgresql+asyncpg://autolabel:autolabel@localhost:5432/autolabel"
    )
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint_url: str = "http://localhost:9000"
    # публичный endpoint для presigned URL (их открывает браузер с хоста)
    s3_public_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "autolabel"
    s3_secret_key: str = "autolabel123"
    s3_bucket: str = "autolabel"

    cors_origins: str = "http://localhost:5173"

    # Ключ шифрования секретов настроек. Генерируется сам при первом старте
    # в файл (volume, права 0600); settings_encryption_key его переопределяет.
    settings_key_path: str = "/data/settings.key"
    settings_encryption_key: str = ""

    # Режим «сложно»: платформа сама разворачивает GPU-рецепт в аккаунт Modal
    # пользователя. Путь к рецепту по умолчанию — рядом с кодом сервера.
    # пустое имя -> деплой под именем приложения из самого рецепта
    modal_gpu_app_name: str = "autolabelui-gpu"
    modal_gpu_recipe_path: str = ""
    # если задан, рецепт закрывает /predict заголовком Bearer, а worker
    # подставляет тот же токен в конфиг движка modal_gpu
    autolabelui_gpu_token: str = ""
    # Общий токен доступа к ручкам настроек (токен Modal, деплой GPU).
    # Пусто — ручки открыты, платформа предупреждает об этом.
    app_access_token: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
