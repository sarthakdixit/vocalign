"""Transcodes arbitrary input audio to a canonical mono WAV via ffmpeg, then reads it."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def build_ffmpeg_command(input_path, output_path, sample_rate: int) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(output_path),
    ]


def load_and_transcode(input_path, sample_rate: int = 16000, run=None) -> tuple[np.ndarray, int]:
    run = run or _run_ffmpeg
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "transcoded.wav"
        run(build_ffmpeg_command(input_path, output_path, sample_rate))
        samples, actual_rate = sf.read(output_path, dtype="float32", always_2d=False)
        return np.atleast_1d(samples), actual_rate


def write_wav(path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), samples, sample_rate)


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True)
