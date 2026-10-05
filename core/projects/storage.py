"""JSON persistence + on-disk layout for projects. See DESIGN.md S6."""

import json
import shutil
from pathlib import Path

from core.projects.models import Project, ProjectState

PROJECT_FILENAME = "project.json"


class ProjectNotFoundError(LookupError):
    def __init__(self, project_id: str):
        super().__init__(f"No project found with id {project_id!r}")
        self.project_id = project_id


def project_dir(projects_root, project_id: str) -> Path:
    return Path(projects_root) / project_id


def ensure_project_layout(projects_root, project_id: str) -> dict[str, Path]:
    base = project_dir(projects_root, project_id)
    layout = {
        "root": base,
        "raw": base / "raw",
        "processed": base / "processed",
        "segments": base / "processed" / "segments",
        "training": base / "training",
        "checkpoints": base / "training" / "checkpoints",
        "logs": base / "training" / "logs",
        "output": base / "output",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def save_project(projects_root, project: Project) -> None:
    base = project_dir(projects_root, project.id)
    base.mkdir(parents=True, exist_ok=True)
    target = base / PROJECT_FILENAME
    tmp = base / f".{PROJECT_FILENAME}.tmp"
    tmp.write_text(json.dumps(_to_dict(project), indent=2))
    tmp.replace(target)  # atomic swap - a reader never sees a half-written file


def load_project(projects_root, project_id: str) -> Project:
    path = project_dir(projects_root, project_id) / PROJECT_FILENAME
    if not path.exists():
        raise ProjectNotFoundError(project_id)
    return _from_dict(json.loads(path.read_text()))


def delete_project_dir(projects_root, project_id: str) -> None:
    base = project_dir(projects_root, project_id)
    if base.exists():
        shutil.rmtree(base)


def list_project_ids(projects_root) -> list[str]:
    root = Path(projects_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / PROJECT_FILENAME).exists())


def _to_dict(project: Project) -> dict:
    return {
        "id": project.id,
        "name": project.name,
        "state": project.state.value,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "language": project.language,
        "config": project.config,
        "error_message": project.error_message,
        "error_from_state": project.error_from_state.value if project.error_from_state else None,
    }


def _from_dict(data: dict) -> Project:
    return Project(
        id=data["id"],
        name=data["name"],
        state=ProjectState(data["state"]),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        language=data.get("language", "en"),
        config=data.get("config", {}),
        error_message=data.get("error_message"),
        error_from_state=ProjectState(data["error_from_state"]) if data.get("error_from_state") else None,
    )
