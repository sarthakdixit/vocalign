from types import SimpleNamespace

import numpy as np
import pytest

from core.infer import ranking


def _fake_segment(text):
    return SimpleNamespace(start=0.0, end=1.0, text=text, avg_logprob=-0.1, no_speech_prob=0.0)


class _FakeWhisperModel:
    def __init__(self, heard_text):
        self._heard_text = heard_text

    def transcribe(self, audio_path, word_timestamps=False, **kwargs):
        return iter([_fake_segment(self._heard_text)]), {"language": "en"}


def _candidate(sample_rate=16000):
    return ranking.Candidate(samples=np.zeros(1000, dtype=np.float32), sample_rate=sample_rate)


# --- compute_wer ---


def test_compute_wer_identical_text_is_zero():
    assert ranking.compute_wer("hello world", "hello world") == 0.0


def test_compute_wer_one_word_different_out_of_four():
    assert ranking.compute_wer("the quick brown dog", "the quick brown fox") == pytest.approx(0.25)


def test_compute_wer_ignores_case_and_punctuation():
    assert ranking.compute_wer("Hello, World!", "hello world") == 0.0


def test_compute_wer_empty_reference_and_empty_hypothesis_is_zero():
    assert ranking.compute_wer("", "") == 0.0


def test_compute_wer_empty_reference_nonempty_hypothesis_is_capped_at_one():
    assert ranking.compute_wer("something", "") == 1.0


def test_compute_wer_completely_different_text_is_high():
    assert ranking.compute_wer("foo bar", "hello world") == pytest.approx(1.0)


# --- transcribe_candidate / score_candidate ---


def test_transcribe_candidate_returns_heard_text():
    model = _FakeWhisperModel("hello there")
    assert ranking.transcribe_candidate(_candidate(), whisper_model=model) == "hello there"


def test_score_candidate_not_rejected_when_wer_is_low():
    model = _FakeWhisperModel("hello world")
    scored = ranking.score_candidate(_candidate(), "hello world", whisper_model=model)
    assert scored.rejected is False
    assert scored.wer == 0.0


def test_score_candidate_rejected_when_wer_exceeds_threshold():
    model = _FakeWhisperModel("completely wrong text here")
    scored = ranking.score_candidate(_candidate(), "hello world", whisper_model=model)
    assert scored.rejected is True


def test_score_candidate_skips_secs_and_utmos_when_rejected():
    model = _FakeWhisperModel("totally different garbled output")
    calls = []
    scored = ranking.score_candidate(
        _candidate(),
        "hello world",
        whisper_model=model,
        secs_fn=lambda c: calls.append("secs") or 0.9,
        utmos_fn=lambda c: calls.append("utmos") or 4.0,
    )
    assert scored.rejected is True
    assert scored.secs is None
    assert scored.utmos is None
    assert calls == []


def test_score_candidate_computes_secs_and_utmos_when_not_rejected():
    model = _FakeWhisperModel("hello world")
    scored = ranking.score_candidate(
        _candidate(), "hello world", whisper_model=model, secs_fn=lambda c: 0.85, utmos_fn=lambda c: 4.1
    )
    assert scored.secs == 0.85
    assert scored.utmos == 4.1


# --- combined_score / pick_best ---


def test_combined_score_weights_secs_higher_than_utmos():
    high_secs = ranking.ScoredCandidate(candidate=_candidate(), wer=0.0, secs=1.0, utmos=0.0, rejected=False)
    high_utmos = ranking.ScoredCandidate(candidate=_candidate(), wer=0.0, secs=0.0, utmos=5.0, rejected=False)
    assert high_secs.combined_score > high_utmos.combined_score


def test_combined_score_treats_missing_scores_as_zero():
    scored = ranking.ScoredCandidate(candidate=_candidate(), wer=0.0, secs=None, utmos=None, rejected=False)
    assert scored.combined_score == 0.0


def test_pick_best_returns_none_for_empty_list():
    assert ranking.pick_best([]) is None


def test_pick_best_picks_highest_combined_score_among_survivors():
    worse = ranking.ScoredCandidate(candidate=_candidate(), wer=0.0, secs=0.5, utmos=3.0, rejected=False)
    better = ranking.ScoredCandidate(candidate=_candidate(), wer=0.0, secs=0.9, utmos=4.5, rejected=False)
    assert ranking.pick_best([worse, better]) is better


def test_pick_best_ignores_rejected_candidates_when_survivors_exist():
    rejected_but_would_score_high = ranking.ScoredCandidate(
        candidate=_candidate(), wer=0.9, secs=0.99, utmos=5.0, rejected=True
    )
    survivor = ranking.ScoredCandidate(candidate=_candidate(), wer=0.1, secs=0.5, utmos=3.0, rejected=False)
    assert ranking.pick_best([rejected_but_would_score_high, survivor]) is survivor


def test_pick_best_falls_back_to_lowest_wer_when_all_rejected():
    worse = ranking.ScoredCandidate(candidate=_candidate(), wer=0.9, secs=None, utmos=None, rejected=True)
    better = ranking.ScoredCandidate(candidate=_candidate(), wer=0.5, secs=None, utmos=None, rejected=True)
    assert ranking.pick_best([worse, better]) is better


# --- default_secs_fn / default_utmos_fn wiring ---


def test_default_secs_fn_wires_compute_embedding_and_cosine_similarity(monkeypatch):
    import core.infer.similarity as similarity_mod

    monkeypatch.setattr(similarity_mod, "compute_embedding", lambda samples, sr, encoder=None: np.array([1.0, 0.0]))
    monkeypatch.setattr(similarity_mod, "cosine_similarity", lambda a, b: 0.77)

    fn = ranking.default_secs_fn(reference_embedding=np.array([1.0, 0.0]))

    assert fn(_candidate()) == 0.77


def test_default_utmos_fn_wires_predict_mos(monkeypatch):
    import core.infer.naturalness as naturalness_mod

    monkeypatch.setattr(naturalness_mod, "predict_mos", lambda samples, sr, scorer=None: 4.4)

    fn = ranking.default_utmos_fn()

    assert fn(_candidate()) == 4.4
