import pytest

from core.audio.chunk import TrainingChunk
from core.train import dataset


def _chunk(text, low_confidence=False):
    return TrainingChunk(start=0.0, end=1.0, text=text, low_confidence=low_confidence)


def test_write_dataset_list_formats_lines_correctly(tmp_path):
    chunks = [(_chunk("hello there"), tmp_path / "a.wav"), (_chunk("how are you"), tmp_path / "b.wav")]

    list_path, skipped = dataset.write_dataset_list(tmp_path / "out.list", chunks, "myspeaker", language="en")

    lines = list_path.read_text().splitlines()
    assert lines == [
        f"{tmp_path / 'a.wav'}|myspeaker|en|hello there",
        f"{tmp_path / 'b.wav'}|myspeaker|en|how are you",
    ]
    assert skipped == 0


def test_write_dataset_list_excludes_low_confidence_by_default(tmp_path):
    chunks = [
        (_chunk("good one"), tmp_path / "a.wav"),
        (_chunk("shaky one", low_confidence=True), tmp_path / "b.wav"),
    ]

    list_path, skipped = dataset.write_dataset_list(tmp_path / "out.list", chunks, "spk")

    lines = list_path.read_text().splitlines()
    assert len(lines) == 1
    assert "good one" in lines[0]
    assert skipped == 1


def test_write_dataset_list_can_include_low_confidence(tmp_path):
    chunks = [(_chunk("shaky", low_confidence=True), tmp_path / "a.wav")]

    list_path, skipped = dataset.write_dataset_list(
        tmp_path / "out.list", chunks, "spk", include_low_confidence=True
    )

    assert len(list_path.read_text().splitlines()) == 1
    assert skipped == 0


def test_write_dataset_list_escapes_pipe_and_newline_in_text(tmp_path):
    chunks = [(_chunk("a|weird\ntext"), tmp_path / "a.wav")]

    list_path, _ = dataset.write_dataset_list(tmp_path / "out.list", chunks, "spk")

    line = list_path.read_text().splitlines()[0]
    assert line.count("|") == 3


def test_write_dataset_list_rejects_unsupported_language(tmp_path):
    with pytest.raises(ValueError):
        dataset.write_dataset_list(tmp_path / "out.list", [], "spk", language="fr")


def test_write_dataset_list_handles_empty_chunk_list(tmp_path):
    list_path, skipped = dataset.write_dataset_list(tmp_path / "out.list", [], "spk")

    assert list_path.read_text() == ""
    assert skipped == 0
