from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.infer import generate as generate_mod


def _fake_segment(text):
    return SimpleNamespace(start=0.0, end=1.0, text=text, avg_logprob=-0.1, no_speech_prob=0.0)


class _SequentialWhisperModel:
    """Returns each listed text in call order - lets a test make every candidate's
    transcription match its real chunk text exactly (WER=0), deterministically."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self._index = 0

    def transcribe(self, audio_path, word_timestamps=False, **kwargs):
        text = self._texts[self._index]
        self._index += 1
        return iter([_fake_segment(text)]), {"language": "en"}


class _CountingFakeTts:
    def __init__(self, sr=16000):
        self._sr = sr
        self.call_count = 0

    def run(self, inputs):
        self.call_count += 1
        yield self._sr, np.full(10, float(self.call_count), dtype=np.float32)


def test_generate_full_pipeline_picks_highest_scoring_candidate_per_chunk():
    chunk0_text = "Hello there."
    chunk1_text = "Goodbye now."
    fake_tts = _CountingFakeTts(sr=16000)
    whisper = _SequentialWhisperModel([chunk0_text, chunk0_text, chunk1_text, chunk1_text])
    secs_calls = []

    def secs_fn(candidate):
        secs_calls.append(candidate)
        return float(len(secs_calls))  # each successive call scores strictly higher

    result = generate_mod.generate(
        f"{chunk0_text} {chunk1_text}",
        tts_instance=fake_tts,
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="ref transcript",
        prompt_lang="en",
        whisper_model=whisper,
        secs_fn=secs_fn,
        config=generate_mod.GenerationConfig(candidates_per_chunk=2),
    )

    assert fake_tts.call_count == 4
    assert len(result.chunk_results) == 2
    assert result.chunk_results[0].secs == 2.0  # 2nd of 2 candidates in chunk 0
    assert result.chunk_results[1].secs == 4.0  # 2nd of 2 candidates in chunk 1
    assert result.sample_rate == 16000
    assert result.any_low_confidence is False


def test_generate_stitches_chosen_candidates_into_nonempty_audio():
    whisper = _SequentialWhisperModel(["One."] * 3)

    result = generate_mod.generate(
        "One.",
        tts_instance=_CountingFakeTts(sr=24000),
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="ref",
        prompt_lang="en",
        whisper_model=whisper,
    )

    assert result.audio.size > 0
    assert result.sample_rate == 24000


def test_generate_raises_when_nothing_left_after_normalization():
    with pytest.raises(ValueError):
        generate_mod.generate(
            "   ",
            tts_instance=_CountingFakeTts(),
            text_lang="en",
            ref_audio_path=Path("/abs/ref.wav"),
            prompt_text="ref",
            prompt_lang="en",
            whisper_model=_SequentialWhisperModel([]),
        )


def test_generate_raises_for_zero_candidates_per_chunk_with_no_explicit_seeds():
    with pytest.raises(ValueError):
        generate_mod.generate(
            "Hello.",
            tts_instance=_CountingFakeTts(),
            text_lang="en",
            ref_audio_path=Path("/abs/ref.wav"),
            prompt_text="ref",
            prompt_lang="en",
            whisper_model=_SequentialWhisperModel([]),
            config=generate_mod.GenerationConfig(candidates_per_chunk=0),
        )


def test_generate_average_secs_and_utmos_ignore_missing_scores():
    whisper = _SequentialWhisperModel(["Hi."] * 3)

    result = generate_mod.generate(
        "Hi.",
        tts_instance=_CountingFakeTts(),
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="ref",
        prompt_lang="en",
        whisper_model=whisper,
    )

    assert result.average_secs is None  # no secs_fn was provided
    assert result.average_utmos is None


def test_generate_and_export_writes_audio_and_metadata_files(tmp_path):
    whisper = _SequentialWhisperModel(["Hello there."] * 3)

    audio_path, metadata_path = generate_mod.generate_and_export(
        "Hello there.",
        tmp_path / "out.wav",
        tts_instance=_CountingFakeTts(sr=16000),
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="ref",
        prompt_lang="en",
        whisper_model=whisper,
    )

    assert audio_path.exists()
    assert metadata_path.exists()
