"""No-reference automatic naturalness/MOS prediction via utmos-pytorch (a fairseq-free
reimplementation - the original `utmos` PyPI package depends on fairseq, a known
problematic install on modern Python/PyTorch, per RESEARCH.md Finding 4's update).
Used only for ranking/reporting, never as a training objective (UTMOS is known to be
adversarially gameable if optimized against directly).

Real API confirmed 2026-10-06 against the actual package README, after a real
`AttributeError` on a real run showed the original guess was wrong on two counts, not
one: the class is `UTMOSScoreTorch(device=...)`, not `Scorer()`, and its `.score()`
takes an already-mono-16kHz torch tensor directly - not a numpy array plus a separate
sample_rate the way most of this project's other scoring functions take audio.

Second real finding, same day: past that fix, scoring hit a real `NameError:
name '_in_projection' is not defined` inside the *vendored GPT-SoVITS repo's own*
GPT_SoVITS/AR/modules/patched_mha_with_cache.py. GPT-SoVITS monkey-patches
`torch.nn.functional.multi_head_attention_forward` globally (process-wide, not scoped to
its own calls) to add KV-caching for its autoregressive decoding - and that patch has its
own gap, hit only by utmos-pytorch's unrelated wav2vec2-based model, apparently never by
GPT-SoVITS's own narrower internal usage. Once GPT-SoVITS's TTS class has been loaded in
this process, nothing here can fix GPT-SoVITS's patch directly - capture_pristine_torch_
state() (called by the caller, early, before any GPT-SoVITS import) saves the original
so predict_mos() can restore it just for the one call that needs it.
"""

import contextlib

import numpy as np

_pristine_mha_forward = None


def capture_pristine_torch_state() -> None:
    """Call this once, as early as possible in the process - specifically before any
    GPT-SoVITS code gets imported - so predict_mos() has something real to restore
    later. A no-op if called more than once (keeps whatever was captured first)."""
    global _pristine_mha_forward
    import torch.nn.functional as F

    if _pristine_mha_forward is None:
        _pristine_mha_forward = F.multi_head_attention_forward


def predict_mos(samples: np.ndarray, sample_rate: int, scorer=None, preprocess_fn=None) -> float:
    scorer = scorer or _default_scorer()
    preprocess_fn = preprocess_fn or _to_mono_16k_tensor
    wav = preprocess_fn(samples, sample_rate)
    with _pristine_multihead_attention():
        return _scalar(scorer.score(wav))


def _pristine_multihead_attention():
    if _pristine_mha_forward is None:
        # capture_pristine_torch_state() was never called early enough to capture
        # anything real - nothing to restore, so just run as-is.
        return contextlib.nullcontext()
    import torch.nn.functional as F

    return _temporarily_set(F, "multi_head_attention_forward", _pristine_mha_forward)


@contextlib.contextmanager
def _temporarily_set(namespace, attr_name: str, value):
    current = getattr(namespace, attr_name)
    setattr(namespace, attr_name, value)
    try:
        yield
    finally:
        setattr(namespace, attr_name, current)


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
