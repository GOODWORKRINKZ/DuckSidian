"""Конфигурация через переменные окружения / .env."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str
    telegram_chat_id: int
    telegram_topic_id: int | None = None
    telegram_admins: List[int] = Field(default_factory=list)

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    ingest_cron: str = "0 23 * * *"
    tz: str = "UTC"

    vault_path: Path = Path("/vault")
    data_path: Path = Path("/data")

    git_autocommit: bool = True
    git_remote_url: str = ""
    git_user_name: str = "DuckSidian Bot"
    git_user_email: str = "ducksidian@localhost"

    @field_validator("telegram_admins", mode="before")
    @classmethod
    def _parse_admins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v


settings = Settings()  # type: ignore[call-arg]
