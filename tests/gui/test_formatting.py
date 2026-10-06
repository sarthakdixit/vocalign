from types import SimpleNamespace

from gui import formatting


def _project(id="p1", name="My Voice", state_value="trained", updated_at="2026-10-06T00:00:00+00:00"):
    return SimpleNamespace(id=id, name=name, state=SimpleNamespace(value=state_value), updated_at=updated_at)


def test_project_dropdown_choices_formats_label_with_state():
    choices = formatting.project_dropdown_choices([_project(name="My Voice", state_value="trained")])

    assert choices == [("My Voice (trained)", "p1")]


def test_project_dropdown_choices_handles_empty_list():
    assert formatting.project_dropdown_choices([]) == []


def test_project_table_rows_includes_id_name_state_and_updated_at():
    rows = formatting.project_table_rows([_project()])

    assert rows == [["p1", "My Voice", "trained", "2026-10-06T00:00:00+00:00"]]


def test_format_duration_handles_none():
    assert formatting.format_duration(None) == "unknown"


def test_format_duration_under_a_minute_omits_minutes():
    assert formatting.format_duration(45.3) == "45.3s"


def test_format_duration_over_a_minute_includes_minutes():
    assert formatting.format_duration(125.0) == "2m 5.0s"


def test_format_preprocessing_summary_success_includes_chunk_count_and_tier():
    outcome = SimpleNamespace(
        ok=True, chunk_count=16, total_speech_seconds=50.9, recipe_tier="minimal", warnings=[], error=None,
    )

    summary = formatting.format_preprocessing_summary(outcome)

    assert "16 chunk" in summary
    assert "minimal" in summary


def test_format_preprocessing_summary_includes_warnings():
    outcome = SimpleNamespace(
        ok=True, chunk_count=1, total_speech_seconds=6.0, recipe_tier="minimal",
        warnings=["Only 6.0s of detected speech"], error=None,
    )

    summary = formatting.format_preprocessing_summary(outcome)

    assert "Only 6.0s of detected speech" in summary


def test_format_preprocessing_summary_failure_shows_error():
    outcome = SimpleNamespace(ok=False, error="Not enough usable speech detected.", warnings=[])

    summary = formatting.format_preprocessing_summary(outcome)

    assert "Not enough usable speech detected." in summary


def test_format_quality_band_none_shows_placeholder():
    assert formatting.format_quality_band(None) == "No quality signal available yet."


def test_format_quality_band_includes_label_secs_and_utmos():
    text = formatting.format_quality_band({"label": "strong match", "average_secs": 0.78, "average_utmos": 4.16})

    assert "strong match" in text
    assert "0.78" in text
    assert "4.16" in text


def test_format_generation_result_failure_shows_error():
    outcome = SimpleNamespace(ok=False, error="checkpoint missing")

    assert formatting.format_generation_result(outcome) == "Generation failed: checkpoint missing"


def test_format_generation_result_failure_includes_progress_log_tail():
    outcome = SimpleNamespace(
        ok=False, error="CUDA out of memory",
        progress_log=("Synthesizing chunk 1/25: 'Hello.'", "Synthesizing chunk 2/25: 'World.'"),
    )

    text = formatting.format_generation_result(outcome)

    assert "CUDA out of memory" in text
    assert "Synthesizing chunk 2/25" in text


def test_format_generation_result_success_includes_scores():
    outcome = SimpleNamespace(ok=True, secs_score=0.78, utmos_score=4.16, low_confidence=False)

    text = formatting.format_generation_result(outcome)

    assert "0.78" in text
    assert "4.16" in text


def test_format_generation_result_flags_low_confidence():
    outcome = SimpleNamespace(ok=True, secs_score=0.5, utmos_score=3.0, low_confidence=True)

    text = formatting.format_generation_result(outcome)

    assert "low confidence" in text


def test_format_generation_result_low_confidence_includes_per_chunk_detail():
    # So a "low confidence" success is diagnosable (which chunk failed, not just that
    # one did) without needing a hard failure to see the progress log at all.
    outcome = SimpleNamespace(
        ok=True, secs_score=0.71, utmos_score=3.98, low_confidence=True,
        progress_log=(
            "Synthesizing chunk 1/2: 'Hello there.'",
            "Chunk 1/2 result: wer=0.00 status=ok",
            "Synthesizing chunk 2/2: 'Goodbye now.'",
            "Chunk 2/2 result: wer=1.00 status=REJECTED (every candidate failed the WER gate)",
        ),
    )

    text = formatting.format_generation_result(outcome)

    assert "Chunk 2/2 result" in text
    assert "REJECTED" in text


def test_format_generation_result_low_confidence_without_progress_log_omits_detail():
    # list_generation_history() reconstructs outcomes from metadata JSON, which never
    # stored progress_log - must not crash or show an empty/garbled detail section.
    outcome = SimpleNamespace(ok=True, secs_score=0.5, utmos_score=3.0, low_confidence=True, progress_log=())

    text = formatting.format_generation_result(outcome)

    assert "Per-chunk detail" not in text
