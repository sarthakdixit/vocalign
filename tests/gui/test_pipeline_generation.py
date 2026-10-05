from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gui.pipeline import generation


def _fake_segment(text):
    return SimpleNamespace(start=0.0, end=1.0, text=text, avg_logprob=-0.1, no_speech_prob=0.0)


class _FakeWhisperModel:
    """Always transcribes back whatever chunk text it's "given" - here that just means
    a fixed string, since these tests don't care about WER-gate specifics."""

    def transcribe(self, audio_path, word_timestamps=False, **kwargs):
        return iter([_fake_segment("hello there")]), {"language": "en"}


class _FakeTts:
    def __init__(self, sr=16000):
        self._sr = sr

    def run(self, inputs):
        yield self._sr, np.zeros(10, dtype=np.float32)


def _fake_context(project_id="p1"):
    prompt = SimpleNamespace(audio_path=Path("/abs/ref.wav"), text="hello there")
    return generation._ProjectInferenceContext(
        project_id=project_id,
        tts_instance=_FakeTts(),
        prompt=prompt,
        secs_fn=lambda candidate: 0.8,
        utmos_fn=lambda candidate: 4.0,
    )


# --- load_inference_context caching/eviction ---


def test_load_inference_context_reuses_the_cached_entry_for_the_same_project(tmp_path, monkeypatch):
    monkeypatch.setattr(generation, "_cache", None)
    calls = []

    def build_fn(projects_root, project_id):
        calls.append(project_id)
        return _fake_context(project_id)

    first = generation.load_inference_context(tmp_path, "p1", build_fn=build_fn)
    second = generation.load_inference_context(tmp_path, "p1", build_fn=build_fn)

    assert first is second
    assert calls == ["p1"]


def test_load_inference_context_rebuilds_for_a_different_project(tmp_path, monkeypatch):
    monkeypatch.setattr(generation, "_cache", None)
    calls = []

    def build_fn(projects_root, project_id):
        calls.append(project_id)
        return _fake_context(project_id)

    first = generation.load_inference_context(tmp_path, "p1", build_fn=build_fn)
    second = generation.load_inference_context(tmp_path, "p2", build_fn=build_fn)

    assert first.project_id == "p1"
    assert second.project_id == "p2"
    assert calls == ["p1", "p2"]


def test_load_inference_context_does_not_cache_a_build_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(generation, "_cache", None)

    def failing_build_fn(projects_root, project_id):
        raise RuntimeError("no GPU memory")

    try:
        generation.load_inference_context(tmp_path, "p1", build_fn=failing_build_fn)
    except RuntimeError:
        pass

    assert generation._cache is None


# --- run_generation ---


def test_run_generation_writes_audio_and_metadata(tmp_path):
    outcome = generation.run_generation(
        tmp_path, "p1", "Hello there.",
        load_context_fn=lambda root, pid: _fake_context(pid), whisper_model=_FakeWhisperModel(),
    )

    assert outcome.ok is True
    assert Path(outcome.audio_path).exists()
    assert Path(outcome.metadata_path).exists()
    assert outcome.secs_score is not None
    assert outcome.utmos_score is not None


def test_run_generation_writes_under_the_projects_output_dir(tmp_path):
    outcome = generation.run_generation(
        tmp_path, "p1", "Hello there.",
        load_context_fn=lambda root, pid: _fake_context(pid), whisper_model=_FakeWhisperModel(),
    )

    assert Path(outcome.audio_path).parent == tmp_path / "p1" / "output"


def test_run_generation_returns_failure_outcome_when_context_loading_fails(tmp_path):
    def failing_load_context(root, pid):
        raise RuntimeError("checkpoint missing")

    outcome = generation.run_generation(tmp_path, "p1", "Hello.", load_context_fn=failing_load_context)

    assert outcome.ok is False
    assert "checkpoint missing" in outcome.error


def test_run_generation_returns_failure_outcome_when_generation_itself_raises(tmp_path):
    outcome = generation.run_generation(
        tmp_path, "p1", "   ",  # blank text -> generate() raises ValueError after normalization
        load_context_fn=lambda root, pid: _fake_context(pid), whisper_model=_FakeWhisperModel(),
    )

    assert outcome.ok is False
    assert outcome.error


def test_run_generation_each_call_gets_a_distinct_output_file(tmp_path):
    first = generation.run_generation(
        tmp_path, "p1", "Hello there.",
        load_context_fn=lambda root, pid: _fake_context(pid), whisper_model=_FakeWhisperModel(),
    )
    second = generation.run_generation(
        tmp_path, "p1", "Hello there.",
        load_context_fn=lambda root, pid: _fake_context(pid), whisper_model=_FakeWhisperModel(),
    )

    assert first.audio_path != second.audio_path


# --- list_generation_history ---


def test_list_generation_history_returns_generations_newest_first(tmp_path):
    generation.run_generation(
        tmp_path, "p1", "First.", load_context_fn=lambda root, pid: _fake_context(pid),
        whisper_model=_FakeWhisperModel(),
    )
    second = generation.run_generation(
        tmp_path, "p1", "Second.", load_context_fn=lambda root, pid: _fake_context(pid),
        whisper_model=_FakeWhisperModel(),
    )

    history = generation.list_generation_history(tmp_path, "p1")

    assert len(history) == 2
    assert history[0].audio_path == second.audio_path


def test_list_generation_history_is_empty_for_a_project_with_no_generations(tmp_path):
    assert generation.list_generation_history(tmp_path, "p1") == []
