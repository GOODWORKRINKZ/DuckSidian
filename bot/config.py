"""Конфигурация через переменные окружения / .env."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_id_list(v) -> list[int]:
    """'1, 2;3' | '[1,2]' | 7 | [..] → [1, 2, 3]."""
    if v is None:
        return []
    if isinstance(v, int):
        return [v]
    if isinstance(v, str):
        v = v.strip().strip("[]")
        if not v:
            return []
        return [int(x.strip().strip('"').strip("'")) for x in v.replace(";", ",").split(",") if x.strip()]
    return [int(x) for x in v]


class ChatConfig(BaseModel):
    """Конфигурация одного отслеживаемого чата."""

    chat_id: int
    name: str            # слаг — используется как поддиректория vault
    topic_id: int | None = None


class LLMProvider(BaseModel):
    """Описание одного OpenAI-совместимого вендора."""

    key: str                      # слаг: deepseek | minimax | qwen | ...
    title: str
    base_url: str                 # базовый URL ДО /chat/completions (включая /v1, если нужно)
    api_key_env: str              # имя переменной окружения с ключом
    default_model: str
    models: list[str] = Field(default_factory=list)   # известные модели (подсказка для /model)
    # Цены $/1M токенов: input, output, cached-input. Нужны только для оценки расхода.
    price_in: float = 0.0
    price_out: float = 0.0
    price_cached: float = 0.0
    # DeepSeek-специфичный параметр `thinking` — другие вендоры могут вернуть 400.
    supports_thinking_flag: bool = False
    # Доп. поля тела запроса (например, enable_thinking=False у Qwen: без него
    # reasoning-модели думают минутами и упираются в таймаут клиента).
    extra_payload: dict = Field(default_factory=dict)
    # Применять extra_payload только к моделям с этими префиксами (пусто = ко всем):
    # glm-5.* на том же эндпоинте отвергает enable_thinking=False.
    extra_payload_prefixes: list[str] = Field(default_factory=list)
    # Есть ли endpoint баланса (/user/balance есть только у DeepSeek).
    has_balance_api: bool = False


# Реестр вендоров. Все — OpenAI-compatible chat/completions с tool-calling.
LLM_PROVIDERS: dict[str, LLMProvider] = {
    "deepseek": LLMProvider(
        key="deepseek", title="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-flash",
        models=["deepseek-v4-flash", "deepseek-chat", "deepseek-reasoner"],
        price_in=0.07, price_out=0.28, price_cached=0.014,
        supports_thinking_flag=True, has_balance_api=True,
    ),
    "minimax": LLMProvider(
        key="minimax", title="MiniMax",
        base_url="https://api.minimax.io/v1",
        api_key_env="MINIMAX_API_KEY",
        default_model="MiniMax-M3",
        models=["MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.1"],
        # Ориентировочно (OpenRouter-прайс M3); официальные цены — в кабинете MiniMax.
        price_in=0.23, price_out=0.96, price_cached=0.05,
    ),
    "qwen": LLMProvider(
        key="qwen", title="Qwen (Alibaba DashScope)",
        base_url="https://token-plan.maas.qwencloudapi.com/compatible-mode/v1",
        api_key_env="QWEN_API_KEY",
        default_model="qwen3.7-plus",
        # Модели token-plan ключа (текст/рассуждение); сторонние — тоже через этот эндпоинт.
        models=["qwen3.7-plus", "qwen3.8-max", "qwen3.7-max", "qwen3.8-flash",
                "qwen3.6-flash", "deepseek-v4.1-flash", "deepseek-v4-pro",
                "deepseek-v4-flash-0731", "glm-5.3", "glm-5.2", "auto"],
        price_in=0.40, price_out=1.20, price_cached=0.10,
    ),
    "openai": LLMProvider(
        key="openai", title="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4.1-mini",
        models=["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
        price_in=0.40, price_out=1.60, price_cached=0.10,
    ),
    "openrouter": LLMProvider(
        key="openrouter", title="OpenRouter (любая модель)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_model="deepseek/deepseek-chat",
        models=["deepseek/deepseek-chat", "qwen/qwen3-235b-a22b", "minimax/minimax-m2"],
    ),
    "ollama": LLMProvider(
        key="ollama", title="Ollama (локально)",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
        default_model="qwen3:8b",
        models=["qwen3:8b", "qwen2.5:14b", "llama3.1:8b"],
    ),
    "custom": LLMProvider(
        key="custom", title="Custom OpenAI-compatible",
        base_url="",
        api_key_env="LLM_API_KEY",
        default_model="",
    ),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: str
    # Первичный чат (backward compat). Используется, если TELEGRAM_CHATS пуст.
    telegram_chat_id: int
    telegram_topic_id: int | None = None
    # Списки id читаем как СТРОКИ: pydantic-settings пытается json.loads() любое
    # поле типа list и падает на "123,456". Парсим сами (см. свойства ниже).
    telegram_admins_raw: str = Field(default="", validation_alias="TELEGRAM_ADMINS")
    # Пользователи, которым разрешён личный режим (DM). Админы — всегда.
    # Список можно дополнять в рантайме через /allow (хранится в БД).
    telegram_private_users_raw: str = Field(default="", validation_alias="TELEGRAM_PRIVATE_USERS")

    @property
    def telegram_admins(self) -> List[int]:
        return parse_id_list(self.telegram_admins_raw)

    @property
    def telegram_private_users(self) -> List[int]:
        return parse_id_list(self.telegram_private_users_raw)

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

    # --- LLM (мульти-вендор) ---
    # LLM_PROVIDER выбирает вендора из реестра LLM_PROVIDERS. Ключ берётся из
    # переменной, указанной в реестре (DEEPSEEK_API_KEY, MINIMAX_API_KEY, ...),
    # либо из LLM_API_KEY (перекрывает). LLM_MODEL / LLM_BASE_URL — переопределения.
    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    # Ключи вендоров (все опциональны — нужен хотя бы один для активного провайдера).
    deepseek_api_key: str = ""
    minimax_api_key: str = ""
    qwen_api_key: str = Field(
        default="", validation_alias=AliasChoices("QWEN_API_KEY", "DASHSCOPE_API_KEY"))
    # Эндпоинт Qwen (OpenAI-compatible); QWEN_BASE_URL в .env перекрывает.
    qwen_base_url: str = "https://token-plan.maas.qwencloudapi.com/compatible-mode/v1"
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_api_key: str = "ollama"
    # Backward compat: старые переменные DeepSeek.
    deepseek_base_url: str = ""
    deepseek_model: str = ""

    # --- Медиа-ИИ (MiniMax): речь → текст, текст → речь, зрение, картинки ---
    # Режимы: auto (MiniMax если есть MINIMAX_API_KEY, иначе локальный fallback),
    # minimax (принудительно), off.
    minimax_base_url: str = "https://api.minimax.io"
    stt_provider: str = "auto"        # auto | minimax | whisper | off
    stt_language: str = "ru"          # подсказка языка для ASR ('' = автоопределение)
    vision_provider: str = "auto"     # auto | minimax | ollama | off
    tts_provider: str = "auto"        # auto | minimax | off
    image_provider: str = "auto"      # auto | minimax | off
    minimax_asr_model: str = "asr-1.0"
    minimax_vision_model: str = "MiniMax-M3"
    minimax_vision_detail: str = "default"   # low | default | high
    minimax_tts_model: str = "speech-2.8-hd"
    minimax_tts_voice: str = "Russian_ReliableMan"   # см. /voices — другие голоса
    minimax_tts_speed: float = 1.0
    minimax_tts_language_boost: str = "Russian"
    minimax_image_model: str = "image-01"
    # Голосовые ответы в личке: auto — отвечать голосом, если пользователь прислал голос;
    # on — всегда; off — никогда. Переключается per-user командой /voice.
    personal_voice_reply: str = "auto"
    # Голосовой ответ — короткий устный пересказ сути (символов), полный текст остаётся сообщением.
    voice_reply_max_chars: int = 350

    # Публикация страниц в Telegraph для красивого чтения (Instant View). Страницы
    # публичны по ссылке — выключи, если vault содержит чувствительные данные.
    telegraph_enabled: bool = True

    # --- Личный режим ---
    personal_agent_max_steps: int = 40
    # Сколько последних сообщений диалога подмешивать агенту как контекст.
    personal_history_size: int = 12

    # --- 1688-cli (github.com/superjack2050/1688-cli) — поиск на 1688, только чтение ---
    # Команда запуска CLI ("1688" или "npx 1688-cli"); пусто — инструмент search_1688 выключен.
    # Вход в аккаунт покупателя — один раз вручную на сервере: `1688 login` (QR).
    one688_cli: str = ""
    one688_timeout: int = 180
    one688_profile: str = ""

    # --- STT ---
    # Модель openai-whisper: tiny | base | small | medium | large. Для русского
    # рекомендуется минимум `small` (~460MB), `tiny` часто даёт мусор.
    whisper_model: str = "small"

    ingest_cron: str = "0 23 * * *"
    tz: str = "UTC"

    vault_path: Path = Path("/vault")
    data_path: Path = Path("/data")

    git_autocommit: bool = True
    git_remote_url: str = ""
    git_user_name: str = "DuckSidian Bot"
    git_user_email: str = "ducksidian@localhost"

    @staticmethod
    def _parse_id_list(v) -> List[int]:
        return parse_id_list(v)

    # --- LLM helpers ---

    def provider_api_key(self, provider: LLMProvider) -> str:
        """Ключ для вендора: LLM_API_KEY (если провайдер активный) > *_API_KEY."""
        env_to_attr = {
            "DEEPSEEK_API_KEY": self.deepseek_api_key,
            "MINIMAX_API_KEY": self.minimax_api_key,
            "QWEN_API_KEY": self.qwen_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
            "OLLAMA_API_KEY": self.ollama_api_key,
            "LLM_API_KEY": self.llm_api_key,
        }
        key = env_to_attr.get(provider.api_key_env, "")
        if provider.key == self.llm_provider and self.llm_api_key:
            key = self.llm_api_key
        return key or ""

    def resolve_llm(
        self, provider_key: str | None = None, model: str | None = None,
    ) -> tuple[LLMProvider, str, str, str]:
        """Вернуть (provider, api_key, base_url, model) для указанного или
        активного провайдера. Бросает KeyError для неизвестного провайдера."""
        pkey = (provider_key or self.llm_provider or "deepseek").strip().lower()
        provider = LLM_PROVIDERS[pkey]
        api_key = self.provider_api_key(provider)
        base_url = provider.base_url
        if pkey == "qwen" and self.qwen_base_url:
            base_url = self.qwen_base_url
        mdl = model or provider.default_model
        if pkey == self.llm_provider:
            # Переопределения из .env применяются только к активному провайдеру.
            if self.llm_base_url:
                base_url = self.llm_base_url
            if not model and self.llm_model:
                mdl = self.llm_model
            # Legacy DeepSeek-переменные.
            if pkey == "deepseek":
                if not self.llm_base_url and self.deepseek_base_url:
                    base_url = self.deepseek_base_url
                if not model and not self.llm_model and self.deepseek_model:
                    mdl = self.deepseek_model
        return provider, api_key, base_url.rstrip("/"), mdl

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

    def get_chat_by_name(self, name: str) -> ChatConfig | None:
        for c in self.get_chats():
            if c.name == name:
                return c
        return None


settings = Settings()  # type: ignore[call-arg]
