from pathlib import Path

import pytest

from core.infer import prompt_selection


def _candidate(path="clip.wav", text="hello", duration=5.0):
    return prompt_selection.PromptCandidate(audio_path=Path(path), text=text, duration=duration)


def test_select_prompt_clip_picks_the_longest_in_range_candidate():
    candidates = [_candidate(duration=4.0), _candidate(duration=8.0), _candidate(duration=6.0)]

    chosen = prompt_selection.select_prompt_clip(candidates)

    assert chosen.duration == 8.0


def test_select_prompt_clip_ignores_candidates_outside_the_valid_range():
    too_short = _candidate(duration=1.0)
    too_long = _candidate(duration=15.0)
    valid = _candidate(duration=5.0)

    chosen = prompt_selection.select_prompt_clip([too_short, too_long, valid])

    assert chosen is valid


def test_select_prompt_clip_accepts_the_boundary_values():
    at_min = _candidate(duration=prompt_selection.MIN_PROMPT_SECONDS)
    at_max = _candidate(duration=prompt_selection.MAX_PROMPT_SECONDS)

    chosen = prompt_selection.select_prompt_clip([at_min, at_max])

    assert chosen is at_max


def test_select_prompt_clip_raises_a_clear_error_when_nothing_is_in_range():
    with pytest.raises(ValueError, match="3-10"):
        prompt_selection.select_prompt_clip([_candidate(duration=1.0), _candidate(duration=20.0)])


def test_select_prompt_clip_raises_for_an_empty_list():
    with pytest.raises(ValueError):
        prompt_selection.select_prompt_clip([])


def test_load_candidates_from_dataset_list_parses_real_format(tmp_path):
    list_path = tmp_path / "dataset.list"
    list_path.write_text("/a/clip0.wav|spk|en|Hello there\n/a/clip1.wav|spk|en|Goodbye now\n")

    candidates = prompt_selection.load_candidates_from_dataset_list(list_path, duration_fn=lambda path: 7.0)

    assert len(candidates) == 2
    assert candidates[0].audio_path == Path("/a/clip0.wav")
    assert candidates[0].text == "Hello there"
    assert candidates[0].duration == 7.0
    assert candidates[1].text == "Goodbye now"


def test_load_candidates_from_dataset_list_skips_blank_lines(tmp_path):
    list_path = tmp_path / "dataset.list"
    list_path.write_text("/a/clip0.wav|spk|en|Hello\n\n/a/clip1.wav|spk|en|Bye\n")

    candidates = prompt_selection.load_candidates_from_dataset_list(list_path, duration_fn=lambda p: 5.0)

    assert len(candidates) == 2


def test_load_candidates_from_dataset_list_preserves_text_containing_punctuation(tmp_path):
    # dataset.py's _escape() already strips '|' from chunk text before writing, but
    # split(..., 3) here is extra insurance against truncating text if that ever changes.
    list_path = tmp_path / "dataset.list"
    list_path.write_text("/a/clip0.wav|spk|en|Hello, this has a comma and colon: yes\n")

    candidates = prompt_selection.load_candidates_from_dataset_list(list_path, duration_fn=lambda p: 5.0)

    assert candidates[0].text == "Hello, this has a comma and colon: yes"


def test_load_candidates_from_dataset_list_uses_default_duration_fn_via_soundfile(tmp_path):
    import numpy as np
    import soundfile as sf

    audio_path = tmp_path / "clip0.wav"
    sf.write(str(audio_path), np.zeros(16000 * 5, dtype="float32"), 16000)
    list_path = tmp_path / "dataset.list"
    list_path.write_text(f"{audio_path}|spk|en|Hello\n")

    candidates = prompt_selection.load_candidates_from_dataset_list(list_path)

    assert candidates[0].duration == pytest.approx(5.0)
