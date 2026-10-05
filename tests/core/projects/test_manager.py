from datetime import datetime, timezone

import pytest

from core.projects import manager
from core.projects.models import InvalidTransitionError, ProjectState
from core.projects.storage import ProjectNotFoundError


def _fixed_now():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_create_project_persists_and_returns_it(tmp_path):
    project = manager.create_project(tmp_path, "My Voice", id_factory=lambda: "p1", now=_fixed_now)

    assert project.id == "p1"
    assert project.name == "My Voice"
    assert project.state == ProjectState.CREATED
    assert manager.get_project(tmp_path, "p1") == project


def test_create_project_sets_up_directory_layout(tmp_path):
    manager.create_project(tmp_path, "My Voice", id_factory=lambda: "p1")

    assert (tmp_path / "p1" / "raw").is_dir()
    assert (tmp_path / "p1" / "training" / "checkpoints").is_dir()
    assert (tmp_path / "p1" / "output").is_dir()


def test_create_project_uses_default_id_factory_when_none_given(tmp_path):
    project = manager.create_project(tmp_path, "My Voice")

    assert len(project.id) == 32


def test_get_project_raises_for_unknown_id(tmp_path):
    with pytest.raises(ProjectNotFoundError):
        manager.get_project(tmp_path, "nope")


def test_list_projects_returns_all_created_projects_sorted_by_id(tmp_path):
    manager.create_project(tmp_path, "B", id_factory=lambda: "p2")
    manager.create_project(tmp_path, "A", id_factory=lambda: "p1")

    projects = manager.list_projects(tmp_path)

    assert [p.id for p in projects] == ["p1", "p2"]


def test_list_projects_on_empty_root_returns_empty_list(tmp_path):
    assert manager.list_projects(tmp_path) == []


def test_rename_project_updates_name_and_persists(tmp_path):
    manager.create_project(tmp_path, "Old Name", id_factory=lambda: "p1", now=_fixed_now)

    renamed = manager.rename_project(
        tmp_path, "p1", "New Name", now=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc)
    )

    assert renamed.name == "New Name"
    assert manager.get_project(tmp_path, "p1").name == "New Name"
    assert renamed.updated_at != renamed.created_at


def test_rename_project_raises_for_unknown_id(tmp_path):
    with pytest.raises(ProjectNotFoundError):
        manager.rename_project(tmp_path, "nope", "New Name")


def test_transition_project_persists_new_state(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1")

    moved = manager.transition_project(tmp_path, "p1", ProjectState.PREPROCESSING)

    assert moved.state == ProjectState.PREPROCESSING
    assert manager.get_project(tmp_path, "p1").state == ProjectState.PREPROCESSING


def test_transition_project_raises_on_invalid_transition(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1")

    with pytest.raises(InvalidTransitionError):
        manager.transition_project(tmp_path, "p1", ProjectState.TRAINED)


def test_transition_project_does_not_persist_on_invalid_transition(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1")

    with pytest.raises(InvalidTransitionError):
        manager.transition_project(tmp_path, "p1", ProjectState.TRAINED)

    assert manager.get_project(tmp_path, "p1").state == ProjectState.CREATED


def test_update_config_merges_new_keys_without_dropping_existing_ones(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1", config={"a": 1})

    updated = manager.update_config(tmp_path, "p1", {"b": 2})

    assert updated.config == {"a": 1, "b": 2}
    assert manager.get_project(tmp_path, "p1").config == {"a": 1, "b": 2}


def test_update_config_overwrites_an_existing_key(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1", config={"a": 1})

    updated = manager.update_config(tmp_path, "p1", {"a": 99})

    assert updated.config == {"a": 99}


def test_update_config_raises_for_unknown_id(tmp_path):
    with pytest.raises(ProjectNotFoundError):
        manager.update_config(tmp_path, "nope", {"a": 1})


def test_update_config_bumps_updated_at(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1", now=_fixed_now)

    updated = manager.update_config(
        tmp_path, "p1", {"a": 1}, now=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc)
    )

    assert updated.updated_at != updated.created_at


def test_delete_project_removes_it_from_listing(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1")

    manager.delete_project(tmp_path, "p1")

    assert manager.list_projects(tmp_path) == []
    with pytest.raises(ProjectNotFoundError):
        manager.get_project(tmp_path, "p1")


def test_full_lifecycle_create_through_trained_and_retrain(tmp_path):
    manager.create_project(tmp_path, "Lifecycle Voice", id_factory=lambda: "p1")

    manager.transition_project(tmp_path, "p1", ProjectState.PREPROCESSING)
    manager.transition_project(tmp_path, "p1", ProjectState.PREPROCESSED)
    manager.transition_project(tmp_path, "p1", ProjectState.TRAINING)
    trained = manager.transition_project(tmp_path, "p1", ProjectState.TRAINED)
    assert trained.state == ProjectState.TRAINED

    retrained = manager.transition_project(tmp_path, "p1", ProjectState.TRAINING)
    assert retrained.state == ProjectState.TRAINING
    assert manager.get_project(tmp_path, "p1").state == ProjectState.TRAINING


def test_full_lifecycle_error_and_recovery(tmp_path):
    manager.create_project(tmp_path, "x", id_factory=lambda: "p1")
    manager.transition_project(tmp_path, "p1", ProjectState.PREPROCESSING)

    errored = manager.transition_project(tmp_path, "p1", ProjectState.ERROR, error_message="bad input")
    assert errored.state == ProjectState.ERROR
    assert errored.error_from_state == ProjectState.PREPROCESSING

    recovered = manager.transition_project(tmp_path, "p1", ProjectState.PREPROCESSING)
    assert recovered.error_message is None
    assert recovered.error_from_state is None
