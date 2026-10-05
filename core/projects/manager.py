"""CRUD + state-transition orchestration for projects, backed by core.projects.storage."""

import dataclasses
import uuid

from core.projects import storage
from core.projects.models import Project, ProjectState, now_iso


def create_project(
    projects_root,
    name: str,
    *,
    language: str = "en",
    config: dict | None = None,
    id_factory=None,
    now=None,
) -> Project:
    id_factory = id_factory or (lambda: uuid.uuid4().hex)
    project = Project.new(id=id_factory(), name=name, language=language, config=config, now=now)
    storage.ensure_project_layout(projects_root, project.id)
    storage.save_project(projects_root, project)
    return project


def get_project(projects_root, project_id: str) -> Project:
    return storage.load_project(projects_root, project_id)


def list_projects(projects_root) -> list[Project]:
    return [storage.load_project(projects_root, pid) for pid in storage.list_project_ids(projects_root)]


def rename_project(projects_root, project_id: str, new_name: str, *, now=None) -> Project:
    project = storage.load_project(projects_root, project_id)
    renamed = dataclasses.replace(project, name=new_name, updated_at=now_iso(now))
    storage.save_project(projects_root, renamed)
    return renamed


def delete_project(projects_root, project_id: str) -> None:
    storage.delete_project_dir(projects_root, project_id)


def update_config(projects_root, project_id: str, updates: dict, *, now=None) -> Project:
    project = storage.load_project(projects_root, project_id)
    merged = {**project.config, **updates}
    updated = dataclasses.replace(project, config=merged, updated_at=now_iso(now))
    storage.save_project(projects_root, updated)
    return updated


def transition_project(
    projects_root,
    project_id: str,
    new_state: ProjectState,
    *,
    error_message: str | None = None,
    now=None,
) -> Project:
    project = storage.load_project(projects_root, project_id)
    updated = project.transition_to(new_state, error_message=error_message, now=now)
    storage.save_project(projects_root, updated)
    return updated
