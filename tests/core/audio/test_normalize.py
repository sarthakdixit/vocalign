import numpy as np

from core.audio import normalize

SAMPLE_RATE = 16000


def _sine(duration_s=2.0, freq=440, amplitude=0.1, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_measure_lufs_quiet_sine_is_lower_than_loud_sine():
    quiet = normalize.measure_lufs(_sine(amplitude=0.05), SAMPLE_RATE)
    loud = normalize.measure_lufs(_sine(amplitude=0.5), SAMPLE_RATE)

    assert quiet < loud


def test_normalize_loudness_moves_measured_level_toward_target():
    quiet = _sine(amplitude=0.02)
    before = normalize.measure_lufs(quiet, SAMPLE_RATE)

    normalized = normalize.normalize_loudness(quiet, SAMPLE_RATE, target_lufs=-23.0)
    after = normalize.measure_lufs(normalized, SAMPLE_RATE)

    assert abs(after - (-23.0)) < abs(before - (-23.0))
    assert abs(after - (-23.0)) < 1.0


def test_normalize_loudness_does_not_crash_on_silence():
    silence = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)

    result = normalize.normalize_loudness(silence, SAMPLE_RATE)

    np.testing.assert_array_equal(result, silence)


def test_true_peak_ceiling_limits_a_loud_signal():
    loud = _sine(amplitude=0.99)

    limited = normalize._apply_true_peak_ceiling(loud, ceiling_db=-1.0)

    ceiling_linear = 10 ** (-1.0 / 20)
    assert np.max(np.abs(limited)) <= ceiling_linear + 1e-6


def test_true_peak_ceiling_leaves_quiet_signal_untouched():
    quiet = _sine(amplitude=0.05)

    result = normalize._apply_true_peak_ceiling(quiet, ceiling_db=-1.0)

    np.testing.assert_array_equal(result, quiet)
