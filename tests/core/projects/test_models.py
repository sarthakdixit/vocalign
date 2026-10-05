import dataclasses
from datetime import datetime, timezone

import pytest

from core.projects.models import InvalidTransitionError, Project, ProjectState, VALID_TRANSITIONS


def _fixed_now():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _project_in_state(state, **overrides):
    project = Project.new(id="p1", name="x", now=_fixed_now)
    return dataclasses.replace(project, state=state, **overrides)


def test_new_project_starts_in_created_state():
    project = Project.new(id="p1", name="My Voice", now=_fixed_now)

    assert project.state == ProjectState.CREATED
    assert project.created_at == project.updated_at
    assert project.language == "en"
    assert project.config == {}
    assert project.error_message is None
    assert project.error_from_state is None


def test_new_project_accepts_custom_language_and_config():
    project = Project.new(id="p1", name="x", language="hi", config={"foo": "bar"}, now=_fixed_now)

    assert project.language == "hi"
    assert project.config == {"foo": "bar"}


def test_new_project_copies_config_instead_of_aliasing_caller_dict():
    caller_config = {"foo": "bar"}
    project = Project.new(id="p1", name="x", config=caller_config, now=_fixed_now)

    caller_config["foo"] = "mutated"

    assert project.config == {"foo": "bar"}


def test_every_state_has_an_entry_in_the_transition_table():
    for state in ProjectState:
        assert state in VALID_TRANSITIONS


VALID_CASES = [
    (ProjectState.CREATED, ProjectState.PREPROCESSING),
    (ProjectState.PREPROCESSING, ProjectState.PREPROCESSED),
    (ProjectState.PREPROCESSING, ProjectState.ERROR),
    (ProjectState.PREPROCESSED, ProjectState.PREPROCESSING),
    (ProjectState.PREPROCESSED, ProjectState.TRAINING),
    (ProjectState.TRAINING, ProjectState.TRAINED),
    (ProjectState.TRAINING, ProjectState.ERROR),
    (ProjectState.TRAINED, ProjectState.TRAINING),
    (ProjectState.ERROR, ProjectState.PREPROCESSING),
    (ProjectState.ERROR, ProjectState.TRAINING),
]


@pytest.mark.parametrize("start,target", VALID_CASES)
def test_valid_transitions_are_allowed(start, target):
    project = _project_in_state(start)

    result = project.transition_to(target, now=_fixed_now)

    assert result.state == target


INVALID_CASES = [
    (ProjectState.CREATED, ProjectState.TRAINED),
    (ProjectState.CREATED, ProjectState.ERROR),
    (ProjectState.CREATED, ProjectState.TRAINING),
    (ProjectState.PREPROCESSED, ProjectState.TRAINED),
    (ProjectState.PREPROCESSED, ProjectState.ERROR),
    (ProjectState.TRAINING, ProjectState.PREPROCESSED),
    (ProjectState.TRAINING, ProjectState.CREATED),
    (ProjectState.TRAINED, ProjectState.ERROR),
    (ProjectState.TRAINED, ProjectState.CREATED),
    (ProjectState.TRAINED, ProjectState.PREPROCESSED),
    (ProjectState.ERROR, ProjectState.TRAINED),
    (ProjectState.ERROR, ProjectState.ERROR),
]


@pytest.mark.parametrize("start,target", INVALID_CASES)
def test_invalid_transitions_raise(start, target):
    project = _project_in_state(start)

    with pytest.raises(InvalidTransitionError) as exc_info:
        project.transition_to(target, now=_fixed_now)

    assert exc_info.value.from_state == start
    assert exc_info.value.to_state == target


def test_transition_to_error_captures_message_and_prior_state():
    project = _project_in_state(ProjectState.TRAINING)

    errored = project.transition_to(ProjectState.ERROR, error_message="GPU OOM", now=_fixed_now)

    assert errored.state == ProjectState.ERROR
    assert errored.error_message == "GPU OOM"
    assert errored.error_from_state == ProjectState.TRAINING


def test_transitioning_out_of_error_clears_error_fields():
    project = _project_in_state(
        ProjectState.ERROR, error_message="boom", error_from_state=ProjectState.TRAINING
    )

    recovered = project.transition_to(ProjectState.TRAINING, now=_fixed_now)

    assert recovered.error_message is None
    assert recovered.error_from_state is None


def test_transition_updates_timestamp_but_not_created_at():
    created = Project.new(id="p1", name="x", now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

    moved = created.transition_to(
        ProjectState.PREPROCESSING, now=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc)
    )

    assert moved.updated_at != created.updated_at
    assert moved.created_at == created.created_at


def test_project_is_immutable():
    project = Project.new(id="p1", name="x", now=_fixed_now)

    with pytest.raises(dataclasses.FrozenInstanceError):
        project.name = "changed"
