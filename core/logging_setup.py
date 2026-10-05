"""Process-wide logging setup for clone-voice."""

import logging
import logging.handlers
import os
from pathlib import Path

from . import config

LOGGER_NAME = "clone_voice"
DEFAULT_LEVEL = "INFO"
_OWN_HANDLER_ATTR = "_clone_voice_owned"


def configure_logging(level: str | None = None, log_dir: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False

    # Checking "any handlers at all" is not enough: something else (e.g.
    # pytest's own log-capture handler) may have already attached a handler
    # to this same named logger. Only skip if OUR OWN handlers are present.
    already_configured = any(getattr(h, _OWN_HANDLER_ATTR, False) for h in logger.handlers)
    if not already_configured:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        setattr(console_handler, _OWN_HANDLER_ATTR, True)
        logger.addHandler(console_handler)

        target_log_dir = log_dir if log_dir is not None else config.LOGS_DIR
        target_log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            target_log_dir / "clone_voice.log", maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(formatter)
        setattr(file_handler, _OWN_HANDLER_ATTR, True)
        logger.addHandler(file_handler)

    # Resolved after handlers are attached so a "falling back" warning is
    # actually visible through them instead of logging's no-handler fallback.
    logger.setLevel(_resolve_level(logger, level))
    return logger


def _resolve_level(logger: logging.Logger, level: str | None) -> int:
    name = (level or os.environ.get("CLONE_VOICE_LOG_LEVEL") or DEFAULT_LEVEL).upper()
    resolved = getattr(logging, name, None)
    if not isinstance(resolved, int):
        logger.warning("Unknown log level %r, falling back to %s", name, DEFAULT_LEVEL)
        return getattr(logging, DEFAULT_LEVEL)
    return resolved
