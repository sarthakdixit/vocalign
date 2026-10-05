from types import SimpleNamespace

from core.align import whisper_align as wa


def _fake_segment(text, start=0.0, end=1.0, avg_logprob=-0.2, no_speech_prob=0.05):
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=avg_logprob, no_speech_prob=no_speech_prob)


class _FakeModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio_path, word_timestamps=False, **kwargs):
        return iter(self._segments), {"language": "en"}


def test_transcribe_and_segment_converts_model_output():
    fake_segments = [_fake_segment("hello there", 0.0, 1.2), _fake_segment("how are you", 1.2, 2.5)]
    model = _FakeModel(fake_segments)

    result = wa.transcribe_and_segment("unused.wav", model=model)

    assert result == [
        wa.WhisperSegment(start=0.0, end=1.2, text="hello there", avg_logprob=-0.2, no_speech_prob=0.05),
        wa.WhisperSegment(start=1.2, end=2.5, text="how are you", avg_logprob=-0.2, no_speech_prob=0.05),
    ]


def test_transcribe_and_segment_strips_whitespace_in_text():
    model = _FakeModel([_fake_segment("  padded text  ")])

    result = wa.transcribe_and_segment("unused.wav", model=model)

    assert result[0].text == "padded text"


def test_is_low_confidence_flags_low_avg_logprob():
    segment = wa.WhisperSegment(start=0, end=1, text="x", avg_logprob=-2.0, no_speech_prob=0.0)
    assert wa.is_low_confidence(segment) is True


def test_is_low_confidence_flags_high_no_speech_prob():
    segment = wa.WhisperSegment(start=0, end=1, text="x", avg_logprob=0.0, no_speech_prob=0.9)
    assert wa.is_low_confidence(segment) is True


def test_is_low_confidence_false_for_a_clean_segment():
    segment = wa.WhisperSegment(start=0, end=1, text="x", avg_logprob=-0.1, no_speech_prob=0.01)
    assert wa.is_low_confidence(segment) is False


def test_compare_to_reference_identical_text_scores_high():
    segments = [wa.WhisperSegment(start=0, end=1, text="hello world", avg_logprob=0, no_speech_prob=0)]

    report = wa.compare_to_reference(segments, "Hello, world!")

    assert report.similarity_ratio > 0.9
    assert report.likely_mismatched is False


def test_compare_to_reference_unrelated_text_scores_low():
    # Disjoint letter sets (b/a/n vs x/y/z) so the match ratio is unambiguously low.
    segments = [wa.WhisperSegment(start=0, end=1, text="banana banana banana", avg_logprob=0, no_speech_prob=0)]

    report = wa.compare_to_reference(segments, "xyz xyz xyz xyz")

    assert report.likely_mismatched is True


def test_compare_to_reference_empty_reference_is_mismatched():
    segments = [wa.WhisperSegment(start=0, end=1, text="anything", avg_logprob=0, no_speech_prob=0)]

    report = wa.compare_to_reference(segments, "")

    assert report.similarity_ratio == 0.0
    assert report.likely_mismatched is True
