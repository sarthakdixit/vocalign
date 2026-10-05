import time

from core.infer.quality_signal import QualityBand
from core.projects import manager
from core.projects.models import ProjectState
from core.train.runner import RunResult, RunStatus
from gui.pipeline import training


def _preprocessed_project(projects_root, total_speech_seconds=50.0, dataset_list_name="dataset.list"):
    project = manager.create_project(projects_root, "My Voice", id_factory=lambda: "p1")
    dataset_list_path = projects_root / "p1" / "processed" / dataset_list_name
    dataset_list_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_list_path.write_text("")
    manager.transition_project(projects_root, project.id, ProjectState.PREPROCESSING)
    manager.update_config(
        projects_root, project.id,
        {"total_speech_seconds": total_speech_seconds, "dataset_list_path": str(dataset_list_path)},
    )
    return manager.transition_project(projects_root, project.id, ProjectState.PREPROCESSED)


def _completed(lines=()):
    return RunResult(status=RunStatus.COMPLETED, return_code=0, log_lines=list(lines))


def _write_checkpoint(projects_root, project_id, name="experiment-e4.ckpt"):
    weights_dir = projects_root / project_id / "training" / "s1_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / name).write_bytes(b"fake checkpoint")


def test_run_training_skips_training_at_zero_shot_only_tier(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=5.0)

    outcome = training.run_training(projects_root, project.id)

    assert outcome.ok is True
    assert outcome.skipped_zero_shot is True
    updated = manager.get_project(projects_root, project.id)
    assert updated.state == ProjectState.TRAINED
    assert updated.config["checkpoint_path"] is None


def test_run_training_full_success_path(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=50.0)
    _write_checkpoint(projects_root, project.id)
    band = QualityBand(label="strong match", average_secs=0.8, average_utmos=4.0, sentence_count=5)

    outcome = training.run_training(
        projects_root, project.id,
        run_data_prep_fn=lambda *a, **k: [_completed()],
        run_gpt_stage_fn=lambda *a, **k: _completed(),
        quality_signal_fn=lambda *a, **k: band,
    )

    assert outcome.ok is True
    assert outcome.checkpoint_path.endswith("experiment-e4.ckpt")
    assert outcome.quality_band == band
    updated = manager.get_project(projects_root, project.id)
    assert updated.state == ProjectState.TRAINED
    assert updated.config["checkpoint_path"] == outcome.checkpoint_path
    assert updated.config["quality_band"]["label"] == "strong match"


def test_run_training_fails_project_on_data_prep_failure(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=50.0)

    outcome = training.run_training(
        projects_root, project.id,
        run_data_prep_fn=lambda *a, **k: [RunResult(status=RunStatus.FAILED, return_code=1, error="boom")],
    )

    assert outcome.ok is False
    assert "boom" in outcome.error
    updated = manager.get_project(projects_root, project.id)
    assert updated.state == ProjectState.ERROR


def test_run_training_fails_project_on_gpt_stage_failure(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=50.0)

    outcome = training.run_training(
        projects_root, project.id,
        run_data_prep_fn=lambda *a, **k: [_completed()],
        run_gpt_stage_fn=lambda *a, **k: RunResult(status=RunStatus.FAILED, return_code=1, error="cuda oom"),
    )

    assert outcome.ok is False
    assert "cuda oom" in outcome.error
    updated = manager.get_project(projects_root, project.id)
    assert updated.state == ProjectState.ERROR


def test_run_training_reports_cancellation_distinctly_from_failure(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=50.0)

    outcome = training.run_training(
        projects_root, project.id,
        run_data_prep_fn=lambda *a, **k: [_completed()],
        run_gpt_stage_fn=lambda *a, **k: RunResult(status=RunStatus.CANCELLED, return_code=None),
    )

    assert outcome.ok is False
    assert outcome.cancelled is True
    updated = manager.get_project(projects_root, project.id)
    assert updated.state == ProjectState.ERROR


def test_run_training_fails_if_no_checkpoint_found_after_reported_success(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=50.0)
    # Deliberately NOT writing a checkpoint file - simulates training reporting success
    # without actually leaving a usable checkpoint behind.

    outcome = training.run_training(
        projects_root, project.id,
        run_data_prep_fn=lambda *a, **k: [_completed()],
        run_gpt_stage_fn=lambda *a, **k: _completed(),
    )

    assert outcome.ok is False
    assert "no checkpoint" in outcome.error.lower()
    updated = manager.get_project(projects_root, project.id)
    assert updated.state == ProjectState.ERROR


def test_run_training_passes_progress_lines_through(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=5.0)
    seen = []

    training.run_training(projects_root, project.id, on_progress=seen.append)

    assert any("zero-shot" in line.lower() or "recipe" in line.lower() for line in seen)


# --- TrainingSession (background-thread + cancel mechanics) ---


def test_training_session_reports_result_once_the_thread_finishes(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=5.0)

    session = training.TrainingSession(projects_root, project.id).start()
    for _ in range(100):
        if session.done:
            break
        time.sleep(0.01)

    assert session.done
    assert session.result.ok is True
    assert session.result.skipped_zero_shot is True


def test_training_session_drain_log_returns_accumulated_lines_and_then_empties(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=5.0)

    session = training.TrainingSession(projects_root, project.id).start()
    for _ in range(100):
        if session.done:
            break
        time.sleep(0.01)

    first_drain = session.drain_log()
    second_drain = session.drain_log()

    assert len(first_drain) > 0
    assert second_drain == []


def test_training_session_cancel_sets_the_cancel_check_seen_by_run_training(tmp_path):
    projects_root = tmp_path / "projects"
    project = _preprocessed_project(projects_root, total_speech_seconds=50.0)
    seen_cancel_checks = []

    def _fake_data_prep(*args, cancel_check, **kwargs):
        seen_cancel_checks.append(cancel_check())
        return [_completed()]

    session = training.TrainingSession(
        projects_root, project.id, run_data_prep_fn=_fake_data_prep,
    )
    session.cancel()  # cancel before starting - the fake step should observe it as True
    session.start()
    for _ in range(100):
        if session.done:
            break
        time.sleep(0.01)

    assert seen_cancel_checks == [True]
