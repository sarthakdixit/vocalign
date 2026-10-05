"""No-reference automatic naturalness/MOS prediction via utmos-pytorch (a fairseq-free
reimplementation - the original `utmos` PyPI package depends on fairseq, a known
problematic install on modern Python/PyTorch, per RESEARCH.md Finding 4's update).
Used only for ranking/reporting, never as a training objective (UTMOS is known to be
adversarially gameable if optimized against directly).

Real API confirmed 2026-10-06 against the actual package README, after a real
`AttributeError` on a real run showed the original guess was wrong on two counts, not
one: the class is `UTMOSScoreTorch(device=...)`, not `Scorer()`, and its `.score()`
takes an already-mono-16kHz torch tensor directly - not a numpy array plus a separate
sample_rate the way most of this project's other scoring functions take audio. This was
the one function in this module explicitly flagged at the time as unconfirmed the way
the GPT-SoVITS integration was, precisely because it hadn't been checked this closely.
"""

import numpy as np


def predict_mos(samples: np.ndarray, sample_rate: int, scorer=None, preprocess_fn=None) -> float:
    scorer = scorer or _default_scorer()
    preprocess_fn = preprocess_fn or _to_mono_16k_tensor
    wav = preprocess_fn(samples, sample_rate)
    return _scalar(scorer.score(wav))


def _to_mono_16k_tensor(samples: np.ndarray, sample_rate: int):
    import torch
    import torchaudio

    wav = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    wav = wav.unsqueeze(0)
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16000)
    return wav


def _default_scorer():
    from utmos_pytorch import UTMOSScoreTorch

    # CPU rather than CUDA: UTMOS is a reporting metric, not on the critical path for
    # voice identity, and this process already juggles GPT+SoVITS+BERT+HuBERT+
    # Resemblyzer on a 3.7GB card - not worth adding more VRAM pressure for this.
    return UTMOSScoreTorch(device="cpu")


def _scalar(score) -> float:
    if hasattr(score, "item"):
        return float(score.item())
    return float(score)
