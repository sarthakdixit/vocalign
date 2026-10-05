"""Small, pure display-formatting helpers shared across gui/app.py's tabs - kept
separate so they're trivially testable without importing Gradio at all.
"""


def project_dropdown_choices(projects: list) -> list[tuple[str, str]]:
    return [(f"{p.name} ({p.state.value})", p.id) for p in projects]


def project_table_rows(projects: list) -> list[list]:
    return [[p.id, p.name, p.state.value, p.updated_at] for p in projects]


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s" if minutes else f"{secs:.1f}s"


def format_preprocessing_summary(outcome) -> str:
    lines = []
    if outcome.ok:
        lines.append(
            f"Preprocessing complete: {outcome.chunk_count} chunk(s), "
            f"{format_duration(outcome.total_speech_seconds)} of usable speech."
        )
        if outcome.recipe_tier:
            lines.append(f"Recipe tier: **{outcome.recipe_tier}**")
    else:
        lines.append(f"Preprocessing failed: {outcome.error}")
    for warning in outcome.warnings:
        lines.append(f"- warning: {warning}")
    return "\n\n".join(lines)


def format_quality_band(quality_band_dict: dict | None) -> str:
    if not quality_band_dict:
        return "No quality signal available yet."
    label = quality_band_dict.get("label", "unknown")
    secs = quality_band_dict.get("average_secs")
    utmos = quality_band_dict.get("average_utmos")
    parts = [f"**{label}**"]
    if secs is not None:
        parts.append(f"SECS={secs:.2f}")
    if utmos is not None:
        parts.append(f"UTMOS={utmos:.2f}")
    return " · ".join(parts)


def format_generation_result(outcome) -> str:
    if not outcome.ok:
        return f"Generation failed: {outcome.error}"
    parts = []
    if outcome.secs_score is not None:
        parts.append(f"SECS={outcome.secs_score:.2f}")
    if outcome.utmos_score is not None:
        parts.append(f"UTMOS={outcome.utmos_score:.2f}")
    if outcome.low_confidence:
        parts.append("low confidence - WER gate rejected every candidate for at least one chunk")
    return " · ".join(parts) if parts else "Done."
