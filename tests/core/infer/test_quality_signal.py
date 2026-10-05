from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.infer import quality_signal


def _fake_segment(text):
    return SimpleNamespace(start=0.0, end=1.0, text=text, avg_logprob=-0.1, no_speech_prob=0.0)


class _SequentialWhisperModel:
    def __init__(self, texts):
        self._texts = list(texts)
        self._index = 0

    def transcribe(self, audio_path, word_timestamps=False, **kwargs):
        text = self._texts[self._index]
        self._index += 1
        return iter([_fake_segment(text)]), {"language": "en"}


class _FakeTts:
    def __init__(self, sr=16000):
        self._sr = sr

    def run(self, inputs):
        yield self._sr, np.zeros(10, dtype=np.float32)


def test_directional_band_strong_match():
    assert quality_signal.directional_band(0.9) == "strong match"


def test_directional_band_likely_good():
    assert quality_signal.directional_band(0.6) == "likely good"


def test_directional_band_uncertain():
    assert quality_signal.directional_band(0.1) == "uncertain - consider more reference audio"


def test_directional_band_unknown_when_no_score():
    assert quality_signal.directional_band(None) == "unknown"


def test_directional_band_boundary_values_are_inclusive():
    assert quality_signal.directional_band(quality_signal.STRONG_MATCH_SECS) == "strong match"
    assert quality_signal.directional_band(quality_signal.LIKELY_GOOD_SECS) == "likely good"


def test_run_quality_signal_averages_secs_across_sentences():
    sentences = ("Hello.", "Goodbye.")
    scores = iter([0.8, 0.6])

    band = quality_signal.run_quality_signal(
        tts_instance=_FakeTts(),
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="ref",
        prompt_lang="en",
        secs_fn=lambda candidate: next(scores),
        whisper_model=_SequentialWhisperModel(list(sentences)),
        test_sentences=sentences,
    )

    assert band.sentence_count == 2
    assert band.average_secs == pytest.approx(0.7)


def test_run_quality_signal_label_reflects_averaged_score():
    sentences = ("Hello.", "Goodbye.")
    scores = iter([0.9, 0.9])

    band = quality_signal.run_quality_signal(
        tts_instance=_FakeTts(),
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="ref",
        prompt_lang="en",
        secs_fn=lambda candidate: next(scores),
        whisper_model=_SequentialWhisperModel(list(sentences)),
        test_sentences=sentences,
    )

    assert band.label == "strong match"


def test_run_quality_signal_uses_one_candidate_per_sentence(monkeypatch):
    import core.infer.quality_signal as quality_signal_mod

    seen_configs = []
    real_generate = quality_signal_mod.generate

    def spying_generate(*args, config=None, **kwargs):
        seen_configs.append(config)
        return real_generate(*args, config=config, **kwargs)

    monkeypatch.setattr(quality_signal_mod, "generate", spying_generate)

    sentences = ("Hi.",)
    quality_signal.run_quality_signal(
        tts_instance=_FakeTts(),
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="ref",
        prompt_lang="en",
        secs_fn=lambda candidate: 0.5,
        whisper_model=_SequentialWhisperModel(list(sentences)),
        test_sentences=sentences,
    )

    assert all(c.candidates_per_chunk == 1 for c in seen_configs)
