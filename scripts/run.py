#!/usr/bin/env python3
"""Launches the clone-voice GUI (DESIGN.md S9).

Windowed (default): a Gradio server wrapped in a native pywebview window. Needs a
display/windowing system.

Headless (--headless): just the Gradio server, bound to 0.0.0.0, URL printed to
console - for a headless GPU box, accessed from a browser on another machine. This is
likely the primary mode on a GPU box with no desktop environment.

Usage:
  python scripts/run.py [--headless] [--port 7860]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.infer.naturalness import capture_pristine_torch_state  # noqa: E402
from core.logging_setup import configure_logging  # noqa: E402


def main() -> int:
    # Must happen before any GPT-SoVITS import (anything under core.train/core.infer
    # that loads real checkpoints) gets a chance to monkey-patch
    # torch.nn.functional.multi_head_attention_forward - see naturalness.py's
    # module docstring for the real conflict this works around.
    capture_pristine_torch_state()
    configure_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Serve on 0.0.0.0 with no pywebview window.")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    from gui.app import build_app

    demo = build_app()

    if args.headless:
        print(f"[run] Headless mode - open http://<this-machine's-ip>:{args.port} in a browser.")
        demo.launch(server_name="0.0.0.0", server_port=args.port)
        return 0

    demo.launch(server_name="127.0.0.1", server_port=args.port, prevent_thread_lock=True, inbrowser=False)
    try:
        import webview

        webview.create_window("clone-voice", f"http://127.0.0.1:{args.port}")
        webview.start()
    finally:
        demo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
