import numpy as np

from core.infer import stitch as stitch_mod


def test_stitch_empty_list_returns_empty_array():
    result = stitch_mod.stitch([], sample_rate=32000)
    assert result.size == 0


def test_stitch_filters_out_empty_chunks():
    chunk = np.ones(100, dtype=np.float32)
    result = stitch_mod.stitch([np.array([]), chunk, np.array([])], sample_rate=32000)
    np.testing.assert_array_equal(result, chunk)


def test_stitch_single_chunk_returned_unchanged():
    chunk = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = stitch_mod.stitch([chunk], sample_rate=32000)
    np.testing.assert_array_equal(result, chunk)


def test_stitch_total_length_accounts_for_overlap():
    sr = 1000
    crossfade_ms = 10  # -> 10 samples at this sample rate
    chunk_a = np.ones(100, dtype=np.float32)
    chunk_b = np.ones(100, dtype=np.float32) * 2
    result = stitch_mod.stitch([chunk_a, chunk_b], sample_rate=sr, crossfade_ms=crossfade_ms)
    assert len(result) == 100 + 100 - 10


def test_stitch_crossfade_blends_smoothly_at_midpoint():
    sr = 1000
    crossfade_ms = 10
    chunk_a = np.zeros(50, dtype=np.float32)
    chunk_b = np.ones(50, dtype=np.float32)
    result = stitch_mod.stitch([chunk_a, chunk_b], sample_rate=sr, crossfade_ms=crossfade_ms)
    fade_len = 10
    overlap_start = len(chunk_a) - fade_len
    midpoint_value = result[overlap_start + fade_len // 2]
    assert 0.0 < midpoint_value < 1.0


def test_stitch_handles_three_chunks():
    chunk_a = np.ones(50, dtype=np.float32)
    chunk_b = np.ones(50, dtype=np.float32) * 2
    chunk_c = np.ones(50, dtype=np.float32) * 3
    result = stitch_mod.stitch([chunk_a, chunk_b, chunk_c], sample_rate=1000, crossfade_ms=10)
    assert len(result) == 150 - 10 - 10


def test_stitch_crossfade_clamped_when_chunk_shorter_than_requested_fade():
    tiny = np.ones(3, dtype=np.float32)
    normal = np.ones(50, dtype=np.float32) * 2
    result = stitch_mod.stitch([tiny, normal], sample_rate=1000, crossfade_ms=10)
    assert len(result) == 3 + 50 - 3  # fade_len clamped to min(10, 3, 50) = 3


def test_stitch_output_is_float32():
    chunk_a = np.ones(50, dtype=np.int16)
    chunk_b = np.ones(50, dtype=np.int16)
    result = stitch_mod.stitch([chunk_a, chunk_b], sample_rate=1000, crossfade_ms=10)
    assert result.dtype == np.float32
