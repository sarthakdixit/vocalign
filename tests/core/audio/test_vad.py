import numpy as np

from core.audio import vad


def test_detect_speech_segments_uses_injected_detector():
    calls = []

    def fake_detector(samples, sample_rate):
        calls.append((len(samples), sample_rate))
        return [(0.5, 1.5)]

    result = vad.detect_speech_segments(np.zeros(100), 16000, detector=fake_detector)

    assert result == [(0.5, 1.5)]
    assert calls == [(100, 16000)]


def test_trim_silence_cuts_to_first_and_last_segment():
    samples = np.arange(16000, dtype=np.float32)
    segments = [(0.2, 0.3), (0.7, 0.8)]

    trimmed = vad.trim_silence(samples, 16000, segments)

    assert trimmed[0] == samples[3200]
    assert trimmed[-1] == samples[12800 - 1]


def test_trim_silence_with_no_segments_returns_empty():
    samples = np.arange(100, dtype=np.float32)

    assert vad.trim_silence(samples, 16000, []).size == 0


def test_total_speech_seconds_sums_segment_durations():
    assert vad.total_speech_seconds([(0, 1), (2, 2.5), (5, 5.25)]) == 1.75


def test_speech_ratio_handles_zero_duration():
    assert vad.speech_ratio([(0, 1)], 0) == 0.0


def test_speech_ratio_computes_fraction():
    assert vad.speech_ratio([(0, 3)], 10) == 0.3


def test_evaluate_quality_blocks_below_hard_minimum():
    report = vad.evaluate_quality([(0, 1)], total_duration_seconds=10)

    assert report.blocking is True
    assert report.total_speech_seconds == 1.0
    assert any("minimum" in w for w in report.warnings)


def test_evaluate_quality_warns_but_does_not_block_in_soft_range():
    report = vad.evaluate_quality([(0, 4)], total_duration_seconds=10)

    assert report.blocking is False
    assert any("recommended minimum" in w for w in report.warnings)


def test_evaluate_quality_warns_on_low_speech_ratio_even_with_enough_absolute_speech():
    report = vad.evaluate_quality([(0, 8)], total_duration_seconds=20)

    assert report.blocking is False
    assert any("ratio" in w for w in report.warnings)


def test_evaluate_quality_clean_clip_has_no_warnings():
    report = vad.evaluate_quality([(0, 8)], total_duration_seconds=10)

    assert report.blocking is False
    assert report.warnings == []
