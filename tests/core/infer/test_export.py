import json

import numpy as np
import soundfile as sf

from core.infer import export as export_mod


def test_export_audio_writes_readable_wav(tmp_path):
    samples = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    metadata = export_mod.GenerationMetadata(text="hello world")

    audio_path, _ = export_mod.export_audio(tmp_path / "out.wav", samples, 32000, metadata)

    read_back, sr = sf.read(str(audio_path), dtype="float32")
    assert sr == 32000
    np.testing.assert_allclose(read_back, samples, atol=1e-4)


def test_export_audio_writes_metadata_json_alongside(tmp_path):
    metadata = export_mod.GenerationMetadata(text="hi", secs_score=0.9, utmos_score=3.5, low_confidence=False)

    _, metadata_path = export_mod.export_audio(tmp_path / "out.wav", np.zeros(100, dtype=np.float32), 16000, metadata)

    data = json.loads(metadata_path.read_text())
    assert data["text"] == "hi"
    assert data["secs_score"] == 0.9
    assert data["utmos_score"] == 3.5
    assert data["low_confidence"] is False


def test_export_audio_metadata_path_is_sidecar_of_audio_path(tmp_path):
    audio_path, metadata_path = export_mod.export_audio(
        tmp_path / "clip.wav", np.zeros(10, dtype=np.float32), 16000, export_mod.GenerationMetadata(text="x")
    )
    assert audio_path.name == "clip.wav"
    assert metadata_path.name == "clip.wav.json"


def test_export_audio_includes_extra_fields_in_metadata(tmp_path):
    metadata = export_mod.GenerationMetadata(text="x", extra={"chunk_index": 2})

    _, metadata_path = export_mod.export_audio(tmp_path / "out.wav", np.zeros(10, dtype=np.float32), 16000, metadata)

    assert json.loads(metadata_path.read_text())["chunk_index"] == 2


def test_export_audio_creates_missing_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "c.wav"

    export_mod.export_audio(nested, np.zeros(10, dtype=np.float32), 16000, export_mod.GenerationMetadata(text="x"))

    assert nested.exists()


def test_export_audio_falls_back_to_plain_writer_if_comment_path_fails(tmp_path, monkeypatch):
    def broken_soundfile(*args, **kwargs):
        raise RuntimeError("simulated SoundFile failure")

    monkeypatch.setattr(export_mod.sf, "SoundFile", broken_soundfile)

    audio_path, _ = export_mod.export_audio(
        tmp_path / "out.wav", np.zeros(50, dtype=np.float32), 16000, export_mod.GenerationMetadata(text="x")
    )

    assert audio_path.exists()
    read_back, sr = sf.read(str(audio_path))
    assert sr == 16000
    assert len(read_back) == 50
