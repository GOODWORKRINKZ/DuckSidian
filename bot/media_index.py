"""Визуальный индекс медиафайлов.

Хранит vault/.cache/visual_index.json:
  "files":    rel_path → {desc, project, notes, tagged_at}
  "profiles": project_slug → накопленный визуальный профиль (свободный текст)

Используется в два направления:
  1. Агент явно тегирует файл (tag_media) → записывается в files + обновляется profile.
  2. describe_media: до показа описания проверяет файл и запускает identify() —
     если есть совпадение по profile-профилям, добавляет подсказку агенту.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

_INDEX_FILE = ".cache/visual_index.json"
_PROFILE_MAX_CHARS = 1200  # сколько символов держим в одном профиле
_STOP_WORDS = {
    # русские
    "и", "в", "на", "с", "по", "из", "от", "до", "за", "под", "над", "при",
    "для", "это", "что", "как", "не", "но", "а", "или", "же", "бы", "он",
    "она", "они", "оно", "его", "её", "их", "нет", "да", "так", "уже",
    # английские
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "of", "with", "and", "or", "not", "it", "its",
}


class IdentifyHit(NamedTuple):
    project: str
    score: float   # 0.0–1.0 — доля слов профиля найденных в описании
    evidence: list[str]  # слова которые совпали


class MediaIndex:
    def __init__(self, vault_root: Path) -> None:
        self._path = vault_root / _INDEX_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("visual_index load failed: %s", exc)
        return {"files": {}, "profiles": {}}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning("visual_index save failed: %s", exc)

    # --- чтение ---

    def get_file(self, rel_path: str) -> dict | None:
        return self._data["files"].get(rel_path)

    def get_profiles(self) -> dict[str, str]:
        return dict(self._data.get("profiles", {}))

    # --- идентификация по описанию ---

    @staticmethod
    def _keywords(text: str) -> set[str]:
        """Набор значимых слов из текста (lowercased, без стоп-слов, ≥3 букв)."""
        words = re.findall(r"[а-яёa-z]{3,}", text.lower())
        return {w for w in words if w not in _STOP_WORDS}

    def identify(self, desc: str, threshold: float = 0.12) -> list[IdentifyHit]:
        """Сравнить описание с визуальными профилями проектов.

        Возвращает список IdentifyHit отсортированный по убыванию score.
        threshold: минимальный score для включения в результат.
        """
        desc_words = self._keywords(desc)
        if not desc_words:
            return []

        hits: list[IdentifyHit] = []
        for project, profile_text in self._data.get("profiles", {}).items():
            profile_words = self._keywords(profile_text)
            if not profile_words:
                continue
            matched = [w for w in profile_words if w in desc_words]
            score = len(matched) / len(profile_words)
            if score >= threshold:
                hits.append(IdentifyHit(project, round(score, 2), matched[:8]))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    # --- запись ---

    def tag(
        self,
        rel_path: str,
        project: str,
        desc: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Пометить файл как принадлежащий проекту и обновить визуальный профиль."""
        now = datetime.now(timezone.utc).isoformat()
        self._data.setdefault("files", {})[rel_path] = {
            "project": project,
            "desc": (desc or "")[:400],
            "notes": notes or "",
            "tagged_at": now,
        }
        # Добавить notes (и/или фрагмент desc) в профиль проекта.
        profile_addition = ""
        if notes:
            profile_addition = notes.strip()
        elif desc:
            profile_addition = desc.strip()[:200]

        if profile_addition:
            profiles = self._data.setdefault("profiles", {})
            existing = profiles.get(project, "")
            # Не дублировать если уже есть
            if profile_addition[:40] not in existing:
                combined = (existing + " " + profile_addition).strip()
                # Обрезаем до лимита, оставляя конец (последние данные важнее).
                if len(combined) > _PROFILE_MAX_CHARS:
                    combined = combined[-_PROFILE_MAX_CHARS:]
                profiles[project] = combined

        log.info("media_index.tag: %s → %s", rel_path, project)
        self._save()

    def update_profile(self, project: str, visual_text: str) -> None:
        """Перезаписать (или создать) визуальный профиль проекта вручную."""
        self._data.setdefault("profiles", {})[project] = visual_text[:_PROFILE_MAX_CHARS]
        log.info("media_index.update_profile: %s", project)
        self._save()

    def profile_summary(self) -> str:
        """Короткий список всех известных профилей для вставки в промпт."""
        profiles = self._data.get("profiles", {})
        if not profiles:
            return "(визуальные профили проектов пусты)"
        lines = []
        for proj, text in sorted(profiles.items()):
            snippet = text[:120].replace("\n", " ")
            lines.append(f"  • {proj}: {snippet}…" if len(text) > 120 else f"  • {proj}: {text}")
        return "\n".join(lines)
