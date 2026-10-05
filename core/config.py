"""Shared filesystem paths for clone-voice's own data (not user code paths)."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_root() -> Path:
    override = os.environ.get("CLONE_VOICE_DATA_DIR")
    return Path(override).resolve() if override else REPO_ROOT


DATA_ROOT = resolve_data_root()
PROJECTS_DIR = DATA_ROOT / "projects"
MODELS_DIR = DATA_ROOT / "models"
LOGS_DIR = DATA_ROOT / "logs"


def ensure_app_dirs(root: Path | None = None) -> dict[str, Path]:
    base = root if root is not None else DATA_ROOT
    dirs = {
        "projects": base / "projects",
        "models": base / "models",
        "logs": base / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
