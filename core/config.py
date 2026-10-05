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


def resolve_vendor_dir() -> Path:
    override = os.environ.get("CLONE_VOICE_VENDOR_DIR")
    return Path(override).resolve() if override else REPO_ROOT / "vendor" / "GPT-SoVITS"


# One vendored GPT-SoVITS checkout and version for the whole app (not per-project) -
# matches how scripts/vendor_gpt_sovits.* actually sets it up, and avoids every project
# needing its own copy of multi-GB pretrained checkpoints.
VENDOR_DIR = resolve_vendor_dir()
GPT_SOVITS_VERSION = os.environ.get("CLONE_VOICE_GPT_SOVITS_VERSION", "v2Pro")


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
