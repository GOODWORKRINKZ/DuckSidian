"""Auto-commit/push vault'а после операций агента."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import settings

log = logging.getLogger(__name__)


def _git_available() -> bool:
    try:
        import git  # noqa: F401
        return True
    except Exception:
        return False


def ensure_repo(vault_path: Path) -> None:
    if not _git_available():
        log.warning("GitPython not available, skipping git init")
        return
    import git

    if not (vault_path / ".git").exists():
        repo = git.Repo.init(vault_path)
        with repo.config_writer() as cw:
            cw.set_value("user", "name", settings.git_user_name)
            cw.set_value("user", "email", settings.git_user_email)
        log.info("Initialized git repo at %s", vault_path)
    else:
        repo = git.Repo(vault_path)
    if settings.git_remote_url:
        try:
            origin = repo.remote("origin")
            if origin.url != settings.git_remote_url:
                origin.set_url(settings.git_remote_url)
        except ValueError:
            repo.create_remote("origin", settings.git_remote_url)


def commit_and_push(message: str) -> bool:
    if not settings.git_autocommit or not _git_available():
        return False
    import git

    vault = settings.vault_path
    try:
        repo = git.Repo(vault)
    except git.InvalidGitRepositoryError:
        ensure_repo(vault)
        repo = git.Repo(vault)
    repo.git.add(A=True)
    if not repo.is_dirty(untracked_files=True):
        return False
    repo.index.commit(message)
    if settings.git_remote_url:
        try:
            repo.remote("origin").push()
        except Exception as exc:  # noqa: BLE001
            log.warning("git push failed: %s", exc)
            return False
    return True
