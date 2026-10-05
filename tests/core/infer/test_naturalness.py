import numpy as np

from core.infer import naturalness


class _FakeScorer:
    def __init__(self, mos=3.8):
        self._mos = mos
        self.seen = None

    def score(self, samples, sample_rate):
        self.seen = (samples, sample_rate)
        return self._mos


def test_predict_mos_returns_scorer_result_as_float():
    scorer = _FakeScorer(mos=4.2)

    result = naturalness.predict_mos(np.zeros(10, dtype=np.float32), 32000, scorer=scorer)

    assert result == 4.2
    assert isinstance(result, float)


def test_predict_mos_passes_samples_and_sample_rate_through():
    scorer = _FakeScorer()
    samples = np.ones(5, dtype=np.float32)

    naturalness.predict_mos(samples, 16000, scorer=scorer)

    seen_samples, seen_rate = scorer.seen
    np.testing.assert_array_equal(seen_samples, samples)
    assert seen_rate == 16000
