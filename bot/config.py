"""Конфигурация через переменные окружения / .env."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatConfig(BaseModel):
    """Конфигурация одного отслеживаемого чата."""

    chat_id: int
    name: str            # слаг — используется как поддиректория vault
    topic_id: int | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str
    # Первичный чат (backward compat). Используется, если TELEGRAM_CHATS пуст.
    telegram_chat_id: int
    telegram_topic_id: int | None = None
    telegram_admins: List[int] = Field(default_factory=list)

    @field_validator("telegram_topic_id", mode="before")
    @classmethod
    def _parse_topic_id(cls, v):
        if v == "" or v is None:
            return None
        return v

    # Многочатовый режим: "chat_id:name[:topic_id]|chat_id:name[:topic_id]|..."
    # Пример: "-100123:main:456|-100456:sales:"
    # Если пусто — единственный чат из telegram_chat_id / telegram_topic_id.
    telegram_chats: str = ""

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
        if isinstance(v, int):
            return [v]
        return v

    def get_chats(self) -> list[ChatConfig]:
        """Вернуть список конфигураций отслеживаемых чатов.

        Формат TELEGRAM_CHATS: "chat_id:name[:topic_id]|..."
        """
        raw = self.telegram_chats.strip()
        if not raw:
            return [
                ChatConfig(
                    chat_id=self.telegram_chat_id,
                    name="main",
                    topic_id=self.telegram_topic_id,
                )
            ]
        configs: list[ChatConfig] = []
        for part in raw.split("|"):
            part = part.strip()
            if not part:
                continue
            segs = part.split(":")
            if len(segs) < 2:
                continue
            chat_id = int(segs[0])
            name = segs[1].strip() or f"chat{abs(chat_id)}"
            topic_id = int(segs[2]) if len(segs) > 2 and segs[2].strip() else None
            configs.append(ChatConfig(chat_id=chat_id, name=name, topic_id=topic_id))
        return configs or [
            ChatConfig(
                chat_id=self.telegram_chat_id,
                name="main",
                topic_id=self.telegram_topic_id,
            )
        ]


settings = Settings()  # type: ignore[call-arg]
