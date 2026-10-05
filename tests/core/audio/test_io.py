import numpy as np
import soundfile as sf

from core.audio import io as audio_io


def test_build_ffmpeg_command_includes_mono_and_sample_rate(tmp_path):
    command = audio_io.build_ffmpeg_command(tmp_path / "in.mp3", tmp_path / "out.wav", 16000)

    assert command[0] == "ffmpeg"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert str(tmp_path / "in.mp3") in command
    assert str(tmp_path / "out.wav") in command


def test_load_and_transcode_reads_back_what_the_runner_produced(tmp_path):
    input_path = tmp_path / "in.wav"
    input_path.write_bytes(b"not real audio - the fake runner ignores this")
    synthetic = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)

    def fake_run(command):
        sf.write(command[-1], synthetic, 16000)

    samples, sample_rate = audio_io.load_and_transcode(input_path, sample_rate=16000, run=fake_run)

    assert sample_rate == 16000
    assert samples.shape[0] == synthetic.shape[0]
    np.testing.assert_allclose(samples, synthetic, atol=1e-4)


def test_load_and_transcode_passes_requested_sample_rate_to_command(tmp_path):
    input_path = tmp_path / "in.wav"
    input_path.write_bytes(b"placeholder")
    seen_commands = []

    def fake_run(command):
        seen_commands.append(command)
        sf.write(command[-1], np.zeros(10, dtype=np.float32), 22050)

    audio_io.load_and_transcode(input_path, sample_rate=22050, run=fake_run)

    assert "22050" in seen_commands[0]


def test_write_wav_round_trips(tmp_path):
    out_path = tmp_path / "out.wav"
    samples = np.linspace(-1.0, 1.0, 4800, dtype=np.float32)

    audio_io.write_wav(out_path, samples, 24000)
    read_back, rate = sf.read(out_path, dtype="float32")

    assert rate == 24000
    np.testing.assert_allclose(read_back, samples, atol=1e-4)
