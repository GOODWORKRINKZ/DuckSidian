"""Медиа-ИИ поверх MiniMax: распознавание речи, синтез голоса, зрение, генерация картинок.

Все вызовы — прямые HTTP к api.minimax.io (не зависят от того, какой вендор
выбран для текстового агента). Каждая возможность включается настройкой
(`STT_PROVIDER`, `VISION_PROVIDER`, `TTS_PROVIDER`, `IMAGE_PROVIDER`); значение
`auto` = MiniMax, если задан MINIMAX_API_KEY, иначе локальный fallback
(Whisper для STT, Ollama moondream для vision, ничего для TTS/картинок).

Endpoints (platform.minimax.io/docs):
  POST /v1/speech_to_text   multipart: model=asr-1.0, file, response_format=json; header language=ru
  POST /v1/chat/completions model=MiniMax-M3, content parts {type:image_url, image_url:{url:data:...}}
  POST /v1/t2a_v2           model=speech-2.8-hd, text, voice_setting, audio_setting, output_format=hex
  POST /v1/image_generation model=image-01, prompt, aspect_ratio, response_format=base64
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

from .config import settings
from .media_refs import strip_images_block

log = logging.getLogger(__name__)
tokens_log = logging.getLogger("bot.tokens")

ASR_FORMATS = {".wav", ".aiff", ".flac", ".m4a", ".mp3", ".aac", ".opus", ".ogg", ".oga"}
ASR_MAX_BYTES = 50 * 1024 * 1024
TTS_MAX_CHARS = 9_000
IMAGE_ASPECTS = {"1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9"}


class MediaAIError(RuntimeError):
    pass


def _strip_think(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.DOTALL | re.IGNORECASE)
    i = s.lower().find("<think>")
    return (s[:i] if i != -1 else s).strip()


def _md_for_tts(text: str) -> str:
    """Убрать markdown/ссылки/служебные блоки, чтобы озвучивать чистый текст."""
    t = text or ""
    t = strip_images_block(t)[0]
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[\[[^\]]*\]\]", " ", t)
    t = re.sub(r"\[\[([^\]|]*)(?:\|([^\]]*))?\]\]", lambda m: m.group(2) or m.group(1), t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[*_#>|]+", " ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


class MediaAI:
    """Клиент медиа-функций MiniMax. Создаётся один раз (см. get_media_ai())."""

    def __init__(self) -> None:
        self.api_key = settings.minimax_api_key or (
            settings.llm_api_key if settings.llm_provider == "minimax" else ""
        )
        self.base_url = (settings.minimax_base_url or "https://api.minimax.io").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        tmo = httpx.Timeout(connect=15.0, read=600.0, write=120.0, pool=10.0)
        self._client = httpx.AsyncClient(timeout=tmo)

    # ---- доступность ----

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _enabled(self, setting: str) -> bool:
        mode = (getattr(settings, setting) or "auto").strip().lower()
        if mode in ("off", "none", "false", "0"):
            return False
        if mode == "minimax":
            return True
        return self.configured  # auto

    @property
    def stt_enabled(self) -> bool:
        return self._enabled("stt_provider")

    @property
    def vision_enabled(self) -> bool:
        return self._enabled("vision_provider")

    @property
    def tts_enabled(self) -> bool:
        return self._enabled("tts_provider")

    @property
    def image_enabled(self) -> bool:
        return self._enabled("image_provider")

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "stt": self.stt_enabled, "vision": self.vision_enabled,
            "tts": self.tts_enabled, "image": self.image_enabled,
            "asr_model": settings.minimax_asr_model,
            "vision_model": settings.minimax_vision_model,
            "tts_model": settings.minimax_tts_model, "tts_voice": settings.minimax_tts_voice,
            "image_model": settings.minimax_image_model,
        }

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, **extra: str) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}"}
        h.update(extra)
        return h

    @staticmethod
    def _check_base_resp(d: dict, what: str) -> None:
        br = d.get("base_resp") or {}
        code = br.get("status_code", 0)
        if code not in (0, None):
            raise MediaAIError(f"{what}: MiniMax status {code}: {br.get('status_msg', '')}")

    # ---- STT ----

    async def transcribe(self, path: Path, language: str | None = None) -> str:
        """Распознать речь из аудиофайла (wav/mp3/ogg/opus/m4a/flac, ≤50MB, ≤500s).
        Возвращает текст (может быть пустым)."""
        if not self.configured:
            raise MediaAIError("MINIMAX_API_KEY не задан")
        size = path.stat().st_size
        if size > ASR_MAX_BYTES:
            raise MediaAIError(f"файл {size // 1024 // 1024}MB > 50MB")
        lang = (language if language is not None else settings.stt_language or "").strip()
        headers = self._headers()
        if lang:
            headers["language"] = lang
        mime = "audio/ogg" if path.suffix.lower() in (".ogg", ".oga", ".opus") else "application/octet-stream"
        resp = await self._client.post(
            f"{self.base_url}/v1/speech_to_text",
            headers=headers,
            data={"model": settings.minimax_asr_model, "response_format": "json"},
            files={"file": (path.name, path.read_bytes(), mime)},
        )
        if resp.status_code >= 400:
            raise MediaAIError(f"ASR HTTP {resp.status_code}: {resp.text[:300]}")
        d = resp.json()
        text = (d.get("text") or "").strip()
        dur = d.get("duration")
        log.info("minimax asr: %s dur=%ss -> %d chars", path.name, dur, len(text))
        tokens_log.info("media asr file=%s duration=%s chars=%d", path.name, dur, len(text))
        return text

    # ---- Vision ----

    async def describe_image_b64(self, b64: str, mime: str = "image/jpeg",
                                 prompt: str | None = None) -> str:
        """Описание изображения через мультимодальную модель (MiniMax-M3)."""
        if not self.configured:
            raise MediaAIError("MINIMAX_API_KEY не задан")
        prompt = prompt or (
            "Опиши изображение кратко и по делу, по-русски (3–6 предложений): что показано, "
            "какие объекты/устройства/детали, есть ли текст, схемы, таблицы, UI — если есть "
            "текст, процитируй ключевое. Без вступлений."
        )
        body = {
            "model": settings.minimax_vision_model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{b64}", "detail": settings.minimax_vision_detail,
                }},
            ]}],
            "max_tokens": 700,
            "temperature": 0.2,
        }
        resp = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(**{"Content-Type": "application/json"}),
            json=body,
        )
        if resp.status_code >= 400:
            raise MediaAIError(f"vision HTTP {resp.status_code}: {resp.text[:300]}")
        d = resp.json()
        self._check_base_resp(d, "vision")
        try:
            msg = d["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise MediaAIError(f"vision: bad response {str(d)[:200]}") from exc
        text = _strip_think(msg.get("content") or "")
        usage = d.get("usage") or {}
        tokens_log.info("media vision model=%s prompt=%s compl=%s",
                        settings.minimax_vision_model,
                        usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return text or "(пустое описание)"

    async def describe_image_file(self, path: Path, prompt: str | None = None) -> str:
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".gif": "image/gif",
        }.get(path.suffix.lower(), "image/jpeg")
        return await self.describe_image_b64(
            base64.b64encode(path.read_bytes()).decode(), mime, prompt,
        )

    # ---- TTS ----

    async def tts(self, text: str, *, voice_id: str | None = None, speed: float | None = None,
                  fmt: str = "mp3") -> bytes:
        """Синтез речи → байты аудио (mp3 по умолчанию)."""
        if not self.configured:
            raise MediaAIError("MINIMAX_API_KEY не задан")
        clean = _md_for_tts(text)
        if not clean:
            raise MediaAIError("нечего озвучивать")
        if len(clean) > TTS_MAX_CHARS:
            clean = clean[:TTS_MAX_CHARS].rsplit(" ", 1)[0] + "…"
        body = {
            "model": settings.minimax_tts_model,
            "text": clean,
            "stream": False,
            "output_format": "hex",
            "language_boost": settings.minimax_tts_language_boost or "auto",
            "voice_setting": {
                "voice_id": voice_id or settings.minimax_tts_voice,
                "speed": float(speed if speed is not None else settings.minimax_tts_speed),
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": fmt, "channel": 1},
        }
        resp = await self._client.post(
            f"{self.base_url}/v1/t2a_v2",
            headers=self._headers(**{"Content-Type": "application/json"}),
            json=body,
        )
        if resp.status_code >= 400:
            raise MediaAIError(f"TTS HTTP {resp.status_code}: {resp.text[:300]}")
        d = resp.json()
        self._check_base_resp(d, "tts")
        hexa = ((d.get("data") or {}).get("audio")) or ""
        if not hexa:
            raise MediaAIError(f"TTS: пустой audio ({str(d)[:200]})")
        extra = d.get("extra_info") or {}
        tokens_log.info("media tts model=%s chars=%s audio_ms=%s",
                        settings.minimax_tts_model, extra.get("usage_characters"),
                        extra.get("audio_length"))
        return bytes.fromhex(hexa)

    async def tts_to_file(self, text: str, dst: Path, *, voice_id: str | None = None,
                          speed: float | None = None, as_voice: bool = True) -> Path:
        """Синтез в файл. as_voice=True → перекодировать ffmpeg'ом в OGG/Opus
        (голосовое сообщение Telegram). Если ffmpeg недоступен — остаётся mp3."""
        audio = await self.tts(text, voice_id=voice_id, speed=speed)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mp3 = dst.with_suffix(".mp3")
        mp3.write_bytes(audio)
        if not as_voice:
            return mp3
        ogg = dst.with_suffix(".ogg")
        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", str(mp3), "-c:a", "libopus", "-b:a", "48k",
                "-vbr", "on", "-application", "voip", str(ogg),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, err = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0 and ogg.exists() and ogg.stat().st_size > 0:
                mp3.unlink(missing_ok=True)
                return ogg
            log.warning("tts ffmpeg->ogg failed rc=%s: %s", proc.returncode, (err or b"")[-200:])
        except Exception as exc:  # noqa: BLE001
            log.warning("tts ffmpeg unavailable: %s", exc)
        return mp3

    async def list_voices(self, lang_prefix: str = "Russian") -> list[dict[str, str]]:
        """Системные голоса (voice_id, name, description), отфильтрованные по префиксу."""
        if not self.configured:
            raise MediaAIError("MINIMAX_API_KEY не задан")
        resp = await self._client.post(
            f"{self.base_url}/v1/get_voice",
            headers=self._headers(**{"Content-Type": "application/json"}),
            json={"voice_type": "system"},
        )
        if resp.status_code >= 400:
            raise MediaAIError(f"get_voice HTTP {resp.status_code}: {resp.text[:200]}")
        d = resp.json()
        out: list[dict[str, str]] = []
        for v in d.get("system_voice") or []:
            vid = str(v.get("voice_id") or "")
            if lang_prefix and not vid.lower().startswith(lang_prefix.lower()):
                continue
            desc = v.get("description")
            if isinstance(desc, list):
                desc = "; ".join(str(x) for x in desc)
            out.append({"voice_id": vid, "name": str(v.get("voice_name") or ""), "description": str(desc or "")})
        return out

    # ---- Image generation ----

    async def generate_images(self, prompt: str, *, aspect_ratio: str = "1:1", n: int = 1,
                              optimize: bool = True) -> list[bytes]:
        """Сгенерировать картинки (jpeg-байты)."""
        if not self.configured:
            raise MediaAIError("MINIMAX_API_KEY не задан")
        prompt = (prompt or "").strip()[:1500]
        if not prompt:
            raise MediaAIError("пустой prompt")
        if aspect_ratio not in IMAGE_ASPECTS:
            aspect_ratio = "1:1"
        body = {
            "model": settings.minimax_image_model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "response_format": "base64",
            "n": max(1, min(int(n), 4)),
            "prompt_optimizer": bool(optimize),
        }
        resp = await self._client.post(
            f"{self.base_url}/v1/image_generation",
            headers=self._headers(**{"Content-Type": "application/json"}),
            json=body,
        )
        if resp.status_code >= 400:
            raise MediaAIError(f"image HTTP {resp.status_code}: {resp.text[:300]}")
        d = resp.json()
        self._check_base_resp(d, "image")
        imgs = (d.get("data") or {}).get("image_base64") or []
        if not imgs:
            meta = d.get("metadata") or {}
            raise MediaAIError(
                f"image: пустой результат (failed_count={meta.get('failed_count')}) — "
                "возможно, prompt отклонён фильтром"
            )
        tokens_log.info("media image model=%s n=%d", settings.minimax_image_model, len(imgs))
        return [base64.b64decode(x) for x in imgs]


_INSTANCE: MediaAI | None = None


def get_media_ai() -> MediaAI:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = MediaAI()
    return _INSTANCE


async def extract_audio_for_asr(src: Path) -> Path | None:
    """Перегнать аудио/видео в wav 16k mono через ffmpeg (для ASR). None — если ffmpeg упал."""
    tmp = Path(tempfile.mkdtemp(prefix="asr_")) / "audio.wav"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(src), "-vn", "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode == 0 and tmp.exists() and tmp.stat().st_size > 44:
            return tmp
        log.warning("ffmpeg audio extract failed rc=%s: %s", proc.returncode, (err or b"")[-200:])
    except Exception as exc:  # noqa: BLE001
        log.warning("ffmpeg unavailable: %s", exc)
    return None
