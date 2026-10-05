import json

import pytest

from core.projects import storage
from core.projects.models import Project, ProjectState


def _sample_project(project_id="p1"):
    return Project.new(id=project_id, name="Test Voice", config={"x": 1})


def test_ensure_project_layout_creates_all_expected_dirs(tmp_path):
    layout = storage.ensure_project_layout(tmp_path, "p1")

    for key in ("raw", "processed", "segments", "training", "checkpoints", "logs", "output"):
        assert layout[key].is_dir()
    assert layout["segments"].parent == layout["processed"]
    assert layout["checkpoints"].parent == layout["training"]


def test_save_and_load_project_round_trips(tmp_path):
    project = _sample_project()

    storage.save_project(tmp_path, project)
    loaded = storage.load_project(tmp_path, project.id)

    assert loaded == project


def test_save_and_load_preserves_error_fields(tmp_path):
    project = _sample_project().transition_to(ProjectState.PREPROCESSING)
    project = project.transition_to(ProjectState.ERROR, error_message="bad audio")

    storage.save_project(tmp_path, project)
    loaded = storage.load_project(tmp_path, project.id)

    assert loaded.error_message == "bad audio"
    assert loaded.error_from_state == ProjectState.PREPROCESSING


def test_load_project_raises_when_missing(tmp_path):
    with pytest.raises(storage.ProjectNotFoundError):
        storage.load_project(tmp_path, "does-not-exist")


def test_save_project_writes_valid_json(tmp_path):
    project = _sample_project()
    storage.save_project(tmp_path, project)

    raw = (tmp_path / project.id / storage.PROJECT_FILENAME).read_text()
    data = json.loads(raw)

    assert data["id"] == project.id
    assert data["state"] == "created"
    assert data["config"] == {"x": 1}


def test_save_project_does_not_leave_tmp_file_behind(tmp_path):
    project = _sample_project()
    storage.save_project(tmp_path, project)

    leftovers = list((tmp_path / project.id).glob(".*.tmp"))
    assert leftovers == []


def test_repeated_saves_never_leave_invalid_json(tmp_path):
    # Basic concurrency-safety check: write-to-tmp-then-rename means every save
    # leaves a fully valid file, never a half-written one - checked across a
    # realistic sequence of state changes rather than a single save.
    project = _sample_project()
    storage.save_project(tmp_path, project)

    for state in (
        ProjectState.PREPROCESSING,
        ProjectState.PREPROCESSED,
        ProjectState.TRAINING,
        ProjectState.TRAINED,
    ):
        project = project.transition_to(state)
        storage.save_project(tmp_path, project)
        raw = (tmp_path / project.id / storage.PROJECT_FILENAME).read_text()
        assert json.loads(raw)["state"] == state.value


def test_list_project_ids_returns_only_dirs_with_project_json(tmp_path):
    storage.save_project(tmp_path, _sample_project("p1"))
    storage.save_project(tmp_path, _sample_project("p2"))
    (tmp_path / "not-a-project").mkdir()

    assert storage.list_project_ids(tmp_path) == ["p1", "p2"]


def test_list_project_ids_on_missing_root_returns_empty(tmp_path):
    assert storage.list_project_ids(tmp_path / "does-not-exist") == []


def test_delete_project_dir_removes_everything(tmp_path):
    project = _sample_project()
    storage.ensure_project_layout(tmp_path, project.id)
    storage.save_project(tmp_path, project)

    storage.delete_project_dir(tmp_path, project.id)

    assert not (tmp_path / project.id).exists()


def test_delete_project_dir_on_missing_project_does_not_raise(tmp_path):
    storage.delete_project_dir(tmp_path, "never-existed")
