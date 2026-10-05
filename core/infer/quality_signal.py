"""Post-training quality signal (DESIGN.md S8.2 step 5 / S8.3): synthesize a handful
of held-out test sentences and report an averaged SECS/UTMOS directional band rather
than a single raw score, since per-utterance noise on either metric is real
(RESEARCH.md Finding 4) - averaging over several tames it. Uses exactly one candidate
per sentence (no best-of-N search) since the point is measuring the model honestly,
not cherry-picking its best possible output.

Band thresholds below are placeholders pending real calibration data from actual
trained projects, not research-backed numbers - same caveat as the recipe-tier epoch
counts elsewhere in this project.
"""

from dataclasses import dataclass

from core.infer.generate import GenerationConfig, generate

DEFAULT_TEST_SENTENCES = (
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "A journey of a thousand miles begins with a single step.",
    "Technology is best when it brings people together.",
    "The weather today is bright and a little cold.",
)

STRONG_MATCH_SECS = 0.75
LIKELY_GOOD_SECS = 0.55


@dataclass(frozen=True)
class QualityBand:
    label: str
    average_secs: float | None
    average_utmos: float | None
    sentence_count: int


def directional_band(average_secs: float | None) -> str:
    if average_secs is None:
        return "unknown"
    if average_secs >= STRONG_MATCH_SECS:
        return "strong match"
    if average_secs >= LIKELY_GOOD_SECS:
        return "likely good"
    return "uncertain - consider more reference audio"


def run_quality_signal(
    *,
    tts_instance,
    text_lang: str,
    ref_audio_path,
    prompt_text: str,
    prompt_lang: str,
    secs_fn,
    utmos_fn=None,
    whisper_model=None,
    test_sentences: tuple[str, ...] = DEFAULT_TEST_SENTENCES,
) -> QualityBand:
    secs_values = []
    utmos_values = []

    for sentence in test_sentences:
        result = generate(
            sentence,
            tts_instance=tts_instance,
            text_lang=text_lang,
            ref_audio_path=ref_audio_path,
            prompt_text=prompt_text,
            prompt_lang=prompt_lang,
            whisper_model=whisper_model,
            secs_fn=secs_fn,
            utmos_fn=utmos_fn,
            config=GenerationConfig(candidates_per_chunk=1),
        )
        if result.average_secs is not None:
            secs_values.append(result.average_secs)
        if result.average_utmos is not None:
            utmos_values.append(result.average_utmos)

    average_secs = sum(secs_values) / len(secs_values) if secs_values else None
    average_utmos = sum(utmos_values) / len(utmos_values) if utmos_values else None
    return QualityBand(
        label=directional_band(average_secs),
        average_secs=average_secs,
        average_utmos=average_utmos,
        sentence_count=len(test_sentences),
    )
