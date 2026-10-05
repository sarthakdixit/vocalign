import numpy as np

from core.align.whisper_align import WhisperSegment
from core.audio import chunk as chunk_mod


def _seg(start, end, text, avg_logprob=-0.1, no_speech_prob=0.0):
    return WhisperSegment(start=start, end=end, text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


def test_merge_short_segments_combines_consecutive_short_ones():
    # seg0 is short, absorbs into seg1 until the running total clears min_seconds;
    # seg2 is independently long enough and is left as its own chunk.
    segments = [_seg(0, 0.5, "hi"), _seg(0.5, 3.0, "there"), _seg(3.0, 6.0, "a long one here")]

    merged = chunk_mod.merge_short_segments(segments, min_seconds=2.0)

    assert len(merged) == 2
    assert merged[0].start == 0 and merged[0].end == 3.0
    assert merged[0].text == "hi there"
    assert merged[1] == segments[2]


def test_merge_short_segments_cascades_through_multiple_short_segments():
    # Each running total is still under min_seconds, so all three keep absorbing
    # into one chunk rather than stopping after the first pairwise merge.
    segments = [_seg(0, 0.5, "hi"), _seg(0.5, 1.0, "there"), _seg(1.0, 5.0, "a long one here")]

    merged = chunk_mod.merge_short_segments(segments, min_seconds=2.0)

    assert len(merged) == 1
    assert merged[0].start == 0 and merged[0].end == 5.0
    assert merged[0].text == "hi there a long one here"


def test_merge_short_segments_merge_takes_worst_confidence():
    segments = [
        _seg(0, 0.5, "hi", avg_logprob=-0.1, no_speech_prob=0.1),
        _seg(0.5, 1.0, "there", avg_logprob=-0.9, no_speech_prob=0.2),
    ]

    merged = chunk_mod.merge_short_segments(segments, min_seconds=2.0)

    assert len(merged) == 1
    assert merged[0].avg_logprob == -0.9
    assert merged[0].no_speech_prob == 0.2


def test_merge_short_segments_empty_input():
    assert chunk_mod.merge_short_segments([]) == []


def test_merge_short_segments_leaves_long_segments_alone():
    segments = [_seg(0, 3.0, "already long"), _seg(3.0, 6.0, "also long")]

    merged = chunk_mod.merge_short_segments(segments, min_seconds=2.0)

    assert merged == segments


def test_split_long_segments_splits_into_pieces_under_max():
    segment = _seg(0, 21.0, "one two three four five six seven")

    split = chunk_mod.split_long_segments([segment], max_seconds=10.0)

    assert len(split) == 3
    assert all(piece.end - piece.start <= 10.0 + 1e-6 for piece in split)
    assert split[0].start == 0
    assert split[-1].end == 21.0


def test_split_long_segments_leaves_short_segments_alone():
    segment = _seg(0, 5.0, "short enough")

    split = chunk_mod.split_long_segments([segment], max_seconds=10.0)

    assert split == [segment]


def test_split_long_segments_distributes_all_words_with_no_loss():
    segment = _seg(0, 20.0, "a b c d e f g h")

    split = chunk_mod.split_long_segments([segment], max_seconds=10.0)

    rejoined = " ".join(piece.text for piece in split if piece.text)
    assert set(rejoined.split()) == set("a b c d e f g h".split())


def test_build_training_chunks_merges_then_splits():
    segments = [
        _seg(0, 0.5, "hi"),
        _seg(0.5, 1.0, "there"),
        _seg(1.0, 25.0, "a very long stretch of speech that needs splitting up"),
    ]

    chunks = chunk_mod.build_training_chunks(segments, min_seconds=2.0, max_seconds=10.0)

    # All three cascade-merge first (each running total is still under 2s) into one
    # 25s chunk, which then gets split back into <=10s pieces - check the structural
    # invariants rather than exact per-chunk text, since the split is word-count-based.
    assert all(c.duration <= 10.0 + 1e-6 for c in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == 25.0
    all_words = " ".join(c.text for c in chunks).split()
    assert all_words == "hi there a very long stretch of speech that needs splitting up".split()


def test_build_training_chunks_flags_low_confidence():
    segments = [_seg(0, 5.0, "shaky audio", avg_logprob=-5.0, no_speech_prob=0.9)]

    chunks = chunk_mod.build_training_chunks(segments)

    assert chunks[0].low_confidence is True


def test_slice_audio_returns_expected_sample_range():
    samples = np.arange(16000, dtype=np.float32)
    c = chunk_mod.TrainingChunk(start=0.5, end=1.0, text="x", low_confidence=False)

    sliced = chunk_mod.slice_audio(samples, 16000, c)

    assert sliced[0] == samples[8000]
    assert len(sliced) == 8000
