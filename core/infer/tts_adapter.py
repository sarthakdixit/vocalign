"""Direct-import inference adapter for GPT-SoVITS's TTS_infer_pack.TTS class. Config
shape, mixed-checkpoint behavior, and the run() contract confirmed directly against
the real repo (commit 48b1a0169a28582a8984402f82cf438d3bfa6aca) - see DESIGN.md S3/13.
Unlike training, this is a direct Python import, not a subprocess: TTS_infer_pack.TTS
is a real, stable, maintainer-used class (their own api_v2.py imports it the same way).

Confirmed: TTS_Config validates t2s_weights_path (GPT stage) and vits_weights_path
(SoVITS stage) completely independently, with no cross-check between them - supplying
our fine-tuned GPT checkpoint alongside the original pretrained SoVITS checkpoint (the
settled training-scope decision) is exactly what this library supports, not a hack.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TtsCheckpoints:
    version: str
    t2s_weights_path: Path  # our fine-tuned GPT/s1 checkpoint
    vits_weights_path: Path  # base pretrained SoVITS/s2 checkpoint - never fine-tuned
    bert_base_path: Path
    cnhubert_base_path: Path
    device: str = "cuda"
    is_half: bool = True


def build_tts_config_dict(checkpoints: TtsCheckpoints) -> dict:
    """Must be nested under "custom": TTS_Config.__init__ does
    configs_.get("custom", configs_["v2"]), so a flat dict is silently ignored."""
    return {
        "custom": {
            "version": checkpoints.version,
            "device": checkpoints.device,
            "is_half": checkpoints.is_half,
            "t2s_weights_path": str(checkpoints.t2s_weights_path),
            "vits_weights_path": str(checkpoints.vits_weights_path),
            "bert_base_path": str(checkpoints.bert_base_path),
            # "cnhuhbert" (not "cnhubert") - matches the real repo's own field name
            # verbatim; this is not a typo in our code, fixing it would break the load.
            "cnhuhbert_base_path": str(checkpoints.cnhubert_base_path),
        }
    }


@dataclass(frozen=True)
class SynthesisRequest:
    text: str
    text_lang: str
    ref_audio_path: Path
    prompt_text: str  # the REFERENCE clip's own transcript (not the target text)
    prompt_lang: str
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    seed: int = -1
    speed_factor: float = 1.0


def build_run_inputs(request: SynthesisRequest) -> dict:
    return {
        "text": request.text,
        "text_lang": request.text_lang,
        "ref_audio_path": str(request.ref_audio_path),
        "prompt_text": request.prompt_text,
        "prompt_lang": request.prompt_lang,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "temperature": request.temperature,
        "seed": request.seed,
        "speed_factor": request.speed_factor,
    }


def load_tts(checkpoints: TtsCheckpoints, tts_class=None, config_class=None):
    """Lazily imports the real GPT-SoVITS classes unless injected (same pattern as
    torch/whisper/silero-vad elsewhere in this project)."""
    if tts_class is None or config_class is None:
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        tts_class = tts_class or TTS
        config_class = config_class or TTS_Config
    config = config_class(build_tts_config_dict(checkpoints))
    return tts_class(config)


def synthesize(tts_instance, request: SynthesisRequest) -> tuple[int, "numpy.ndarray"]:
    """tts_instance is whatever load_tts() produced (or a test fake with the same
    generator-based .run() contract). run() is confirmed to be a generator - the real
    library's own api_v2.py also does next(tts_pipeline.run(req)) for a single result."""
    generator = tts_instance.run(build_run_inputs(request))
    return next(generator)
