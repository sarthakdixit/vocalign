"""Top-level generation pipeline: normalize target text, split into sentence chunks,
synthesize several candidates per chunk (varied seed), gate+rank them, stitch the
winners into one clip, export with quality metadata. See DESIGN.md S8.3.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.infer.chunk_text import split_into_chunks
from core.infer.export import GenerationMetadata, export_audio
from core.infer.ranking import Candidate, ScoredCandidate, pick_best, score_candidate
from core.infer.stitch import stitch
from core.infer.text_normalize import normalize_text
from core.infer.tts_adapter import SynthesisRequest, synthesize


@dataclass(frozen=True)
class GenerationConfig:
    candidates_per_chunk: int = 3
    seeds: tuple[int, ...] | None = None  # None = auto (0..candidates_per_chunk-1)


@dataclass(frozen=True)
class GenerationResult:
    audio: np.ndarray
    sample_rate: int
    chunk_results: list[ScoredCandidate]

    @property
    def average_secs(self) -> float | None:
        values = [c.secs for c in self.chunk_results if c.secs is not None]
        return sum(values) / len(values) if values else None

    @property
    def average_utmos(self) -> float | None:
        values = [c.utmos for c in self.chunk_results if c.utmos is not None]
        return sum(values) / len(values) if values else None

    @property
    def any_low_confidence(self) -> bool:
        return any(c.rejected for c in self.chunk_results)


def generate(
    target_text: str,
    *,
    tts_instance,
    text_lang: str,
    ref_audio_path: Path,
    prompt_text: str,
    prompt_lang: str,
    whisper_model=None,
    secs_fn=None,
    utmos_fn=None,
    config: GenerationConfig = GenerationConfig(),
    on_progress=None,
) -> GenerationResult:
    if config.candidates_per_chunk < 1 and not config.seeds:
        raise ValueError("candidates_per_chunk must be >= 1 (or pass explicit seeds)")

    on_progress = on_progress or (lambda message: None)
    seeds = config.seeds or tuple(range(config.candidates_per_chunk))

    normalized = normalize_text(target_text)
    chunks = split_into_chunks(normalized)
    if not chunks:
        raise ValueError("No text to synthesize after normalization/chunking.")

    chunk_results: list[ScoredCandidate] = []
    sample_rate = None

    for i, chunk in enumerate(chunks):
        on_progress(f"Synthesizing chunk {i + 1}/{len(chunks)}: {chunk!r}")
        scored_candidates = []
        for seed in seeds:
            request = SynthesisRequest(
                text=chunk,
                text_lang=text_lang,
                ref_audio_path=ref_audio_path,
                prompt_text=prompt_text,
                prompt_lang=prompt_lang,
                seed=seed,
            )
            sr, audio = synthesize(tts_instance, request)
            sample_rate = sr
            candidate = Candidate(samples=audio, sample_rate=sr)
            scored_candidates.append(
                score_candidate(candidate, chunk, whisper_model=whisper_model, secs_fn=secs_fn, utmos_fn=utmos_fn)
            )

        chunk_results.append(pick_best(scored_candidates))

    stitched = stitch([c.candidate.samples for c in chunk_results], sample_rate=sample_rate)
    return GenerationResult(audio=stitched, sample_rate=sample_rate, chunk_results=chunk_results)


def generate_and_export(target_text: str, output_path: Path, **generate_kwargs) -> tuple[Path, Path]:
    result = generate(target_text, **generate_kwargs)
    metadata = GenerationMetadata(
        text=target_text,
        secs_score=result.average_secs,
        utmos_score=result.average_utmos,
        low_confidence=result.any_low_confidence,
    )
    return export_audio(output_path, result.audio, result.sample_rate, metadata)
