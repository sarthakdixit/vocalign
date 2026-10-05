"""Builds final training chunks (audio slice + text) from Whisper segments."""

from dataclasses import dataclass

import numpy as np

from core.align.whisper_align import WhisperSegment, is_low_confidence

MIN_CHUNK_SECONDS = 2.0
MAX_CHUNK_SECONDS = 10.0


@dataclass(frozen=True)
class TrainingChunk:
    start: float
    end: float
    text: str
    low_confidence: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_short_segments(
    segments: list[WhisperSegment], min_seconds: float = MIN_CHUNK_SECONDS
) -> list[WhisperSegment]:
    if not segments:
        return []

    merged = [segments[0]]
    for segment in segments[1:]:
        last = merged[-1]
        if last.end - last.start < min_seconds:
            merged[-1] = WhisperSegment(
                start=last.start,
                end=segment.end,
                text=f"{last.text} {segment.text}".strip(),
                avg_logprob=min(last.avg_logprob, segment.avg_logprob),
                no_speech_prob=max(last.no_speech_prob, segment.no_speech_prob),
            )
        else:
            merged.append(segment)
    return merged


def split_long_segments(
    segments: list[WhisperSegment], max_seconds: float = MAX_CHUNK_SECONDS
) -> list[WhisperSegment]:
    # Splits evenly by time and distributes words proportionally - word_timestamps
    # is off in transcribe_and_segment for speed, so this is an approximation.
    result: list[WhisperSegment] = []
    for segment in segments:
        duration = segment.end - segment.start
        if duration <= max_seconds:
            result.append(segment)
            continue

        pieces = max(2, int(duration // max_seconds) + 1)
        piece_duration = duration / pieces
        words = segment.text.split()
        words_per_piece = max(1, len(words) // pieces) if words else 0
        for i in range(pieces):
            piece_start = segment.start + i * piece_duration
            piece_end = segment.start + (i + 1) * piece_duration if i < pieces - 1 else segment.end
            word_start = i * words_per_piece
            # Last piece absorbs any remainder instead of silently dropping trailing words.
            word_end = (i + 1) * words_per_piece if i < pieces - 1 else len(words)
            piece_text = " ".join(words[word_start:word_end]) if words else ""
            result.append(
                WhisperSegment(
                    start=piece_start,
                    end=piece_end,
                    text=piece_text,
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                )
            )
    return result


def build_training_chunks(
    segments: list[WhisperSegment],
    min_seconds: float = MIN_CHUNK_SECONDS,
    max_seconds: float = MAX_CHUNK_SECONDS,
) -> list[TrainingChunk]:
    merged = merge_short_segments(segments, min_seconds)
    sized = split_long_segments(merged, max_seconds)
    return [
        TrainingChunk(start=s.start, end=s.end, text=s.text, low_confidence=is_low_confidence(s))
        for s in sized
    ]


def slice_audio(samples: np.ndarray, sample_rate: int, chunk: TrainingChunk) -> np.ndarray:
    start_idx = int(chunk.start * sample_rate)
    end_idx = int(chunk.end * sample_rate)
    return samples[start_idx:end_idx]
