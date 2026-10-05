from types import SimpleNamespace

import numpy as np
import pytest

from core.infer import naturalness


class _FakeScorer:
    def __init__(self, mos=3.8):
        self._mos = mos
        self.seen = None

    def score(self, wav):
        self.seen = wav
        return self._mos


class _FakeTensorResult:
    """Stands in for a real torch.Tensor's .item() without needing torch installed -
    utmos-pytorch's real score() returns a tensor of shape (batch_size,)."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_predict_mos_returns_scorer_result_as_float():
    scorer = _FakeScorer(mos=4.2)

    result = naturalness.predict_mos(
        np.zeros(10, dtype=np.float32), 32000, scorer=scorer, preprocess_fn=lambda s, sr: s
    )

    assert result == 4.2
    assert isinstance(result, float)


def test_predict_mos_passes_the_preprocessed_audio_to_the_scorer():
    scorer = _FakeScorer()
    preprocessed = "sentinel-preprocessed-value"

    naturalness.predict_mos(
        np.ones(5, dtype=np.float32), 16000, scorer=scorer, preprocess_fn=lambda s, sr: preprocessed
    )

    assert scorer.seen == preprocessed


def test_predict_mos_forwards_samples_and_sample_rate_to_preprocess_fn():
    seen = {}

    naturalness.predict_mos(
        np.ones(5, dtype=np.float32),
        16000,
        scorer=_FakeScorer(),
        preprocess_fn=lambda s, sr: seen.update(samples=s, sample_rate=sr),
    )

    assert seen["sample_rate"] == 16000
    np.testing.assert_array_equal(seen["samples"], np.ones(5, dtype=np.float32))


def test_predict_mos_unwraps_a_tensor_like_result_via_item():
    # Confirmed via the real package's README: score() returns a tensor, not a float.
    scorer = _FakeScorer(mos=_FakeTensorResult(4.5))

    result = naturalness.predict_mos(
        np.zeros(10, dtype=np.float32), 16000, scorer=scorer, preprocess_fn=lambda s, sr: s
    )

    assert result == 4.5
    assert isinstance(result, float)


# --- _temporarily_set (the mechanics behind restoring GPT-SoVITS's monkey-patched
# torch.nn.functional.multi_head_attention_forward around a real UTMOS call - see
# naturalness.py's module docstring for the real conflict this works around) ---


def test_temporarily_set_sets_the_value_inside_the_with_block():
    ns = SimpleNamespace(attr="original")

    with naturalness._temporarily_set(ns, "attr", "patched"):
        assert ns.attr == "patched"


def test_temporarily_set_restores_the_original_value_after_the_with_block():
    ns = SimpleNamespace(attr="original")

    with naturalness._temporarily_set(ns, "attr", "patched"):
        pass

    assert ns.attr == "original"


def test_temporarily_set_restores_even_if_the_body_raises():
    ns = SimpleNamespace(attr="original")

    with pytest.raises(ValueError):
        with naturalness._temporarily_set(ns, "attr", "patched"):
            raise ValueError("boom")

    assert ns.attr == "original"


def test_pristine_multihead_attention_is_a_no_op_when_nothing_was_captured(monkeypatch):
    # Importing torch here would be a real dependency this test shouldn't need -
    # confirms the early-return path never touches it when nothing was captured.
    monkeypatch.setattr(naturalness, "_pristine_mha_forward", None)

    with naturalness._pristine_multihead_attention():
        pass
