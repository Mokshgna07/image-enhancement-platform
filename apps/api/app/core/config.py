from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Image Enhancement API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    api_v1_prefix: str = "/api/v1"

    database_url: str = Field(
	default="postgresql+psycopg://app_user:development_password@localhost:5432/image_enhancement",
        validation_alias="DATABASE_URL",
    )

    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        validation_alias="CORS_ORIGINS",
    )

    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
    )

    model_version: str = Field(
        default="edsr_x4_baseline",
        validation_alias="SR_MODEL_VERSION",
    )

    model_device: str = Field(
        default="auto",
        validation_alias="SR_DEVICE",
    )

    sr_config_path: str = Field(
        default="ml/configs/training.yaml",
        validation_alias="SR_CONFIG_PATH",
    )

    sr_checkpoint_4x: str = Field(
        default="runs/edsr_x4_baseline/checkpoints/best.pt",
        validation_alias="SR_CHECKPOINT_4X",
    )

    sr_checkpoint_2x: str | None = Field(
        default=None,
        validation_alias="SR_CHECKPOINT_2X",
    )

    access_token_expire_minutes: int = Field(
        default=15,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    refresh_token_expire_days: int = Field(
        default=30,
        validation_alias="REFRESH_TOKEN_EXPIRE_DAYS",
    )

    jwt_secret_key: str = Field(
        default="CHANGE_ME_IN_ENVIRONMENT",
        validation_alias="JWT_SECRET_KEY",
    )

    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias="JWT_ALGORITHM",
    )

    storage_root: str = Field(
        default=str(Path(__file__).resolve().parents[2] / "storage"),
        validation_alias="STORAGE_ROOT",
    )

    max_upload_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        validation_alias="MAX_UPLOAD_SIZE_BYTES",
    )

    max_image_pixels: int = Field(
        default=25_000_000,
        validation_alias="MAX_IMAGE_PIXELS",
    )

    redis_broker_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="REDIS_BROKER_URL",
    )

    redis_result_backend_url: str = Field(
        default="redis://localhost:6379/1",
        validation_alias="REDIS_RESULT_BACKEND_URL",
    )

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
