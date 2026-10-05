from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from core.projects import manager
from core.projects.models import ProjectState
from gui.pipeline import new_project


def _fake_ffmpeg_run(seconds=6.0, sample_rate=16000):
    def _run(command):
        output_path = command[-1]
        sf.write(str(output_path), np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate)

    return _run


def _whole_clip_detector(samples, sample_rate):
    return [(0.0, len(samples) / sample_rate)]


def _no_speech_detector(samples, sample_rate):
    return []


class _FakeWhisperModel:
    def __init__(self, text="hello there world", start=0.0, end=6.0, avg_logprob=-0.1, no_speech_prob=0.0):
        self._segment = SimpleNamespace(start=start, end=end, text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)

    def transcribe(self, audio_path, word_timestamps=False, **kwargs):
        return iter([self._segment]), {"language": "en"}


def _uploaded_clip(tmp_path, name="upload.wav"):
    path = tmp_path / name
    sf.write(str(path), np.zeros(1000, dtype=np.float32), 16000)
    return path


def test_create_and_preprocess_project_happy_path_reaches_preprocessed(tmp_path):
    projects_root = tmp_path / "projects"
    upload = _uploaded_clip(tmp_path)

    outcome = new_project.create_and_preprocess_project(
        projects_root, "My Voice", upload, "hello there world",
        ffmpeg_run=_fake_ffmpeg_run(), vad_detector=_whole_clip_detector, whisper_model=_FakeWhisperModel(),
    )

    assert outcome.ok is True
    assert outcome.chunk_count > 0
    assert outcome.total_speech_seconds == pytest.approx(6.0)
    project = manager.get_project(projects_root, outcome.project_id)
    assert project.state == ProjectState.PREPROCESSED


def test_create_and_preprocess_project_persists_expected_config_keys(tmp_path):
    projects_root = tmp_path / "projects"
    upload = _uploaded_clip(tmp_path)

    outcome = new_project.create_and_preprocess_project(
        projects_root, "My Voice", upload, "hello there world",
        ffmpeg_run=_fake_ffmpeg_run(), vad_detector=_whole_clip_detector, whisper_model=_FakeWhisperModel(),
    )

    config = manager.get_project(projects_root, outcome.project_id).config
    assert config["reference_text"] == "hello there world"
    assert config["recipe_tier"] is not None
    assert config["transcript_match_ratio"] == pytest.approx(1.0, abs=0.01)
    assert Path(config["dataset_list_path"]).exists()


def test_create_and_preprocess_project_copies_upload_into_raw_dir(tmp_path):
    projects_root = tmp_path / "projects"
    upload = _uploaded_clip(tmp_path)

    outcome = new_project.create_and_preprocess_project(
        projects_root, "My Voice", upload, "hello there world",
        ffmpeg_run=_fake_ffmpeg_run(), vad_detector=_whole_clip_detector, whisper_model=_FakeWhisperModel(),
    )

    raw_dir = projects_root / outcome.project_id / "raw"
    assert any(raw_dir.iterdir())


def test_create_and_preprocess_project_blocks_on_too_little_speech(tmp_path):
    projects_root = tmp_path / "projects"
    upload = _uploaded_clip(tmp_path)

    outcome = new_project.create_and_preprocess_project(
        projects_root, "My Voice", upload, "hello",
        ffmpeg_run=_fake_ffmpeg_run(), vad_detector=_no_speech_detector, whisper_model=_FakeWhisperModel(),
    )

    assert outcome.ok is False
    assert outcome.warnings
    project = manager.get_project(projects_root, outcome.project_id)
    assert project.state == ProjectState.ERROR


def test_create_and_preprocess_project_warns_but_succeeds_on_transcript_mismatch(tmp_path):
    projects_root = tmp_path / "projects"
    upload = _uploaded_clip(tmp_path)

    outcome = new_project.create_and_preprocess_project(
        projects_root, "My Voice", upload, "completely unrelated reference text here",
        ffmpeg_run=_fake_ffmpeg_run(), vad_detector=_whole_clip_detector,
        whisper_model=_FakeWhisperModel(text="totally different audio content entirely"),
    )

    assert outcome.ok is True
    assert any("doesn't look like it matches" in w for w in outcome.warnings)
    project = manager.get_project(projects_root, outcome.project_id)
    assert project.state == ProjectState.PREPROCESSED


def test_create_and_preprocess_project_marks_error_on_unexpected_exception(tmp_path):
    projects_root = tmp_path / "projects"
    upload = _uploaded_clip(tmp_path)

    def _raising_run(command):
        raise RuntimeError("ffmpeg exploded")

    outcome = new_project.create_and_preprocess_project(
        projects_root, "My Voice", upload, "hello", ffmpeg_run=_raising_run,
    )

    assert outcome.ok is False
    assert "ffmpeg exploded" in outcome.error
    project = manager.get_project(projects_root, outcome.project_id)
    assert project.state == ProjectState.ERROR
    assert project.error_message == outcome.error
