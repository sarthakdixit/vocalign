"""Builds the Gradio app (DESIGN.md S9): Dashboard / New Project / Project Detail /
Generate tabs, each a thin wiring layer over gui/pipeline/* - the actual orchestration
logic lives there and is unit-tested there, not here. This module is deliberately not
unit-tested directly (Gradio wiring isn't meaningfully testable as plain functions);
Batch 5's own test focus is a manual click-through of the real running app instead.
"""

import time

import gradio as gr

from core import config
from core.projects import manager
from gui import formatting
from gui.pipeline import generation, new_project, training

POLL_INTERVAL_SECONDS = 0.5


def build_app(projects_root=None) -> gr.Blocks:
    projects_root = projects_root or config.PROJECTS_DIR
    config.ensure_app_dirs()

    with gr.Blocks(title="clone-voice") as demo:
        current_project_id = gr.State(value=None)
        training_session_state = gr.State(value=None)

        with gr.Tabs() as tabs:
            with gr.Tab("Dashboard", id="dashboard"):
                dash = _build_dashboard_tab()
            with gr.Tab("New Project", id="new_project"):
                new_proj = _build_new_project_tab()
            with gr.Tab("Project Detail", id="project_detail"):
                detail = _build_project_detail_tab()
            with gr.Tab("Generate", id="generate"):
                gen = _build_generate_tab()

        # --- Dashboard wiring ---

        def refresh_dashboard():
            projects = manager.list_projects(projects_root)
            return gr.update(choices=formatting.project_dropdown_choices(projects), value=None), \
                formatting.project_table_rows(projects)

        demo.load(refresh_dashboard, inputs=None, outputs=[dash["dropdown"], dash["table"]])
        dash["refresh_btn"].click(refresh_dashboard, inputs=None, outputs=[dash["dropdown"], dash["table"]])

        def on_rename(project_id, new_name):
            if not project_id:
                return "Select a project first.", *refresh_dashboard()
            if not new_name or not new_name.strip():
                return "Enter a new name first.", *refresh_dashboard()
            try:
                manager.rename_project(projects_root, project_id, new_name.strip())
                status = f"Renamed to {new_name.strip()!r}."
            except Exception as exc:
                status = f"Rename failed: {exc}"
            dropdown_update, table_rows = refresh_dashboard()
            return status, dropdown_update, table_rows

        dash["rename_btn"].click(
            on_rename, inputs=[dash["dropdown"], dash["rename_text"]],
            outputs=[dash["status"], dash["dropdown"], dash["table"]],
        )

        def on_delete(project_id, confirmed):
            if not project_id:
                return "Select a project first.", *refresh_dashboard()
            if not confirmed:
                return "Check 'Confirm delete' first - this permanently removes the project.", *refresh_dashboard()
            try:
                manager.delete_project(projects_root, project_id)
                status = "Project deleted."
            except Exception as exc:
                status = f"Delete failed: {exc}"
            dropdown_update, table_rows = refresh_dashboard()
            return status, dropdown_update, table_rows

        dash["delete_btn"].click(
            on_delete, inputs=[dash["dropdown"], dash["confirm_delete"]],
            outputs=[dash["status"], dash["dropdown"], dash["table"]],
        )

        def on_open(project_id):
            if not project_id:
                return None, "Select a project first.", gr.update()
            return project_id, "", _project_summary(projects_root, project_id)

        dash["open_btn"].click(
            on_open, inputs=[dash["dropdown"]], outputs=[current_project_id, dash["status"], detail["summary"]],
        ).then(
            lambda project_id: gr.Tabs(selected="project_detail") if project_id else gr.Tabs(),
            inputs=[current_project_id], outputs=[tabs],
        )

        # --- New Project wiring ---

        def on_create_project(name, audio_path, reference_text):
            if not name or not name.strip():
                return "Enter a project name first.", None, gr.update()
            if not audio_path:
                return "Upload a reference audio clip first.", None, gr.update()
            outcome = new_project.create_and_preprocess_project(
                projects_root, name.strip(), audio_path, reference_text or "",
            )
            summary = formatting.format_preprocessing_summary(outcome)
            if not outcome.ok:
                return summary, None, gr.update()
            return summary, outcome.project_id, _project_summary(projects_root, outcome.project_id)

        new_proj["submit_btn"].click(
            on_create_project,
            inputs=[new_proj["name"], new_proj["audio"], new_proj["reference_text"]],
            outputs=[new_proj["summary"], current_project_id, detail["summary"]],
        ).then(
            lambda project_id: gr.Tabs(selected="project_detail") if project_id else gr.Tabs(),
            inputs=[current_project_id], outputs=[tabs],
        )

        # --- Project Detail wiring ---

        def on_start_training(project_id):
            if not project_id:
                yield "No project selected - open one from the Dashboard tab first.", None, gr.update()
                return

            session = training.TrainingSession(projects_root, project_id).start()
            log_lines: list[str] = []
            while not session.done:
                log_lines.extend(session.drain_log())
                yield "\n".join(log_lines) or "Starting...", session, gr.update()
                time.sleep(POLL_INTERVAL_SECONDS)
            log_lines.extend(session.drain_log())

            outcome = session.result
            quality_text = formatting.format_quality_band(
                {
                    "label": outcome.quality_band.label,
                    "average_secs": outcome.quality_band.average_secs,
                    "average_utmos": outcome.quality_band.average_utmos,
                }
                if outcome and outcome.quality_band
                else None
            )
            yield "\n".join(log_lines), session, quality_text

        detail["start_btn"].click(
            on_start_training, inputs=[current_project_id],
            outputs=[detail["log"], training_session_state, detail["quality"]],
        )

        def on_cancel_training(session):
            if session is None:
                return "No training is running."
            session.cancel()
            return "Cancelling... (takes effect at the next progress line or step boundary)"

        detail["cancel_btn"].click(
            on_cancel_training, inputs=[training_session_state], outputs=[detail["log"]],
        )

        def on_refresh_detail(project_id):
            if not project_id:
                return "No project selected - open one from the Dashboard tab first."
            return _project_summary(projects_root, project_id)

        detail["refresh_btn"].click(on_refresh_detail, inputs=[current_project_id], outputs=[detail["summary"]])

        # --- Generate wiring ---

        def on_generate(project_id, target_text):
            if not project_id:
                return None, "No project selected - open one from the Dashboard tab first.", gr.update()
            if not target_text or not target_text.strip():
                return None, "Enter some text to synthesize first.", gr.update()
            outcome = generation.run_generation(projects_root, project_id, target_text.strip())
            result_text = formatting.format_generation_result(outcome)
            audio_out = outcome.audio_path if outcome.ok else None
            return audio_out, result_text, _history_rows(projects_root, project_id)

        gen["generate_btn"].click(
            on_generate, inputs=[current_project_id, gen["target_text"]],
            outputs=[gen["audio_out"], gen["result"], gen["history"]],
        )

        def on_refresh_generate(project_id):
            if not project_id:
                return "No project selected.", []
            return f"Project: {manager.get_project(projects_root, project_id).name}", _history_rows(
                projects_root, project_id
            )

        gen["refresh_btn"].click(on_refresh_generate, inputs=[current_project_id], outputs=[gen["project_label"], gen["history"]])

    return demo


def _build_dashboard_tab() -> dict:
    gr.Markdown("List, open, rename, or delete projects.")
    dropdown = gr.Dropdown(label="Project", choices=[], value=None)
    with gr.Row():
        open_btn = gr.Button("Open")
        refresh_btn = gr.Button("Refresh")
    table = gr.Dataframe(headers=["id", "name", "state", "updated_at"], interactive=False)
    with gr.Row():
        rename_text = gr.Textbox(label="New name")
        rename_btn = gr.Button("Rename")
    with gr.Row():
        confirm_delete = gr.Checkbox(label="Confirm delete")
        delete_btn = gr.Button("Delete", variant="stop")
    status = gr.Markdown()
    return {
        "dropdown": dropdown, "open_btn": open_btn, "refresh_btn": refresh_btn, "table": table,
        "rename_text": rename_text, "rename_btn": rename_btn, "confirm_delete": confirm_delete,
        "delete_btn": delete_btn, "status": status,
    }


def _build_new_project_tab() -> dict:
    gr.Markdown("Upload a reference clip (any length) and its transcript to create a new project.")
    name = gr.Textbox(label="Project name")
    audio = gr.Audio(label="Reference audio", type="filepath")
    reference_text = gr.Textbox(label="Reference text (what's said in the clip)", lines=4)
    submit_btn = gr.Button("Create Project", variant="primary")
    summary = gr.Markdown()
    return {"name": name, "audio": audio, "reference_text": reference_text, "submit_btn": submit_btn, "summary": summary}


def _build_project_detail_tab() -> dict:
    summary = gr.Markdown("No project selected - open one from the Dashboard tab.")
    with gr.Row():
        start_btn = gr.Button("Start Training", variant="primary")
        cancel_btn = gr.Button("Cancel")
        refresh_btn = gr.Button("Refresh")
    log = gr.Textbox(label="Training log", lines=15, interactive=False)
    quality = gr.Markdown()
    return {
        "summary": summary, "start_btn": start_btn, "cancel_btn": cancel_btn, "refresh_btn": refresh_btn,
        "log": log, "quality": quality,
    }


def _build_generate_tab() -> dict:
    project_label = gr.Markdown("No project selected - open one from the Dashboard tab.")
    target_text = gr.Textbox(label="Text to synthesize", lines=4)
    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary")
        refresh_btn = gr.Button("Refresh")
    audio_out = gr.Audio(label="Output")
    result = gr.Markdown()
    history = gr.Dataframe(headers=["audio_path", "secs", "utmos"], interactive=False)
    return {
        "project_label": project_label, "target_text": target_text, "generate_btn": generate_btn,
        "refresh_btn": refresh_btn, "audio_out": audio_out, "result": result, "history": history,
    }


def _project_summary(projects_root, project_id: str) -> str:
    project = manager.get_project(projects_root, project_id)
    lines = [f"**{project.name}** - state: `{project.state.value}`"]
    if project.config.get("recipe_tier"):
        lines.append(f"Recipe tier: {project.config['recipe_tier']}")
    if project.config.get("total_speech_seconds") is not None:
        lines.append(f"Usable reference speech: {formatting.format_duration(project.config['total_speech_seconds'])}")
    if project.error_message:
        lines.append(f"Last error: {project.error_message}")
    if project.config.get("quality_band"):
        lines.append(formatting.format_quality_band(project.config["quality_band"]))
    return "\n\n".join(lines)


def _history_rows(projects_root, project_id: str) -> list[list]:
    history = generation.list_generation_history(projects_root, project_id)
    return [[h.audio_path, h.secs_score, h.utmos_score] for h in history]
