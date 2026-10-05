from pathlib import Path

import numpy as np

from core.infer import tts_adapter


def _checkpoints(**overrides):
    defaults = dict(
        version="v2",
        t2s_weights_path=Path("/abs/finetuned_s1.ckpt"),
        vits_weights_path=Path("/abs/pretrained/s2G.pth"),
        bert_base_path=Path("/abs/pretrained/bert"),
        cnhubert_base_path=Path("/abs/pretrained/hubert"),
    )
    defaults.update(overrides)
    return tts_adapter.TtsCheckpoints(**defaults)


def _request(**overrides):
    defaults = dict(
        text="hello world",
        text_lang="en",
        ref_audio_path=Path("/abs/ref.wav"),
        prompt_text="this is the reference clip's own transcript",
        prompt_lang="en",
    )
    defaults.update(overrides)
    return tts_adapter.SynthesisRequest(**defaults)


class _FakeConfig:
    def __init__(self, configs):
        self.configs = configs


class _FakeTtsClass:
    def __init__(self, config):
        self.config = config


class _FakeTts:
    def __init__(self, sr=32000, audio=None):
        self._sr = sr
        self._audio = audio if audio is not None else np.zeros(100, dtype=np.int16)
        self.seen_inputs = []

    def run(self, inputs):
        self.seen_inputs.append(inputs)
        yield self._sr, self._audio


def test_build_tts_config_dict_nests_under_custom_key():
    config = tts_adapter.build_tts_config_dict(_checkpoints())
    assert set(config.keys()) == {"custom"}


def test_build_tts_config_dict_field_names_match_real_template():
    inner = tts_adapter.build_tts_config_dict(_checkpoints())["custom"]
    assert set(inner.keys()) == {
        "version",
        "device",
        "is_half",
        "t2s_weights_path",
        "vits_weights_path",
        "bert_base_path",
        "cnhuhbert_base_path",
    }


def test_build_tts_config_dict_uses_paths_as_given_without_modification():
    checkpoints = _checkpoints(
        t2s_weights_path=Path("/abs/a.ckpt"),
        vits_weights_path=Path("/abs/b.pth"),
    )
    inner = tts_adapter.build_tts_config_dict(checkpoints)["custom"]
    assert inner["t2s_weights_path"] == "/abs/a.ckpt"
    assert inner["vits_weights_path"] == "/abs/b.pth"


def test_build_tts_config_dict_passes_through_version_and_device():
    inner = tts_adapter.build_tts_config_dict(_checkpoints(version="v2Pro", device="cpu"))["custom"]
    assert inner["version"] == "v2Pro"
    assert inner["device"] == "cpu"


def test_load_tts_constructs_config_then_tts_with_injected_classes():
    checkpoints = _checkpoints()

    tts = tts_adapter.load_tts(checkpoints, tts_class=_FakeTtsClass, config_class=_FakeConfig)

    assert isinstance(tts, _FakeTtsClass)
    assert isinstance(tts.config, _FakeConfig)
    assert tts.config.configs == tts_adapter.build_tts_config_dict(checkpoints)


def test_synthesize_returns_the_sample_rate_yielded_from_run():
    fake_tts = _FakeTts(sr=32000)

    sr, _ = tts_adapter.synthesize(fake_tts, _request())

    assert sr == 32000


def test_synthesize_normalizes_int16_audio_to_float32():
    # Confirmed via a real run: run() yields raw int16 PCM, which Resemblyzer's SECS
    # scoring rejects outright when given an array directly (not loaded from a file).
    audio = np.array([0, 16384, -32768, 32767], dtype=np.int16)
    fake_tts = _FakeTts(audio=audio)

    _, returned_audio = tts_adapter.synthesize(fake_tts, _request())

    assert returned_audio.dtype == np.float32
    np.testing.assert_allclose(returned_audio, [0.0, 0.5, -1.0, 32767 / 32768], atol=1e-6)


def test_synthesize_passes_through_already_float_audio_unchanged():
    audio = np.array([0.1, -0.5, 0.9], dtype=np.float32)
    fake_tts = _FakeTts(audio=audio)

    _, returned_audio = tts_adapter.synthesize(fake_tts, _request())

    assert returned_audio.dtype == np.float32
    np.testing.assert_allclose(returned_audio, audio)


def test_synthesize_passes_core_fields_to_run():
    fake_tts = _FakeTts()
    request = _request(text="hello", text_lang="en", prompt_text="ref text", prompt_lang="en")

    tts_adapter.synthesize(fake_tts, request)

    seen = fake_tts.seen_inputs[0]
    assert seen["text"] == "hello"
    assert seen["text_lang"] == "en"
    assert seen["prompt_text"] == "ref text"
    assert seen["prompt_lang"] == "en"
    assert seen["ref_audio_path"] == str(request.ref_audio_path)


def test_build_run_inputs_defaults_match_library_documented_defaults():
    inputs = tts_adapter.build_run_inputs(_request())

    assert inputs["top_k"] == 15
    assert inputs["top_p"] == 1.0
    assert inputs["temperature"] == 1.0
    assert inputs["seed"] == -1
    assert inputs["speed_factor"] == 1.0


def test_build_run_inputs_respects_overridden_sampling_params():
    request = _request(top_k=5, temperature=0.7, seed=42)

    inputs = tts_adapter.build_run_inputs(request)

    assert inputs["top_k"] == 5
    assert inputs["temperature"] == 0.7
    assert inputs["seed"] == 42
