import numpy as np

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
