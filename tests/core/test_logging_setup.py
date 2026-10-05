import logging

import pytest

from core import logging_setup


@pytest.fixture(autouse=True)
def _reset_clone_voice_logger():
    logger = logging.getLogger(logging_setup.LOGGER_NAME)
    logger.handlers.clear()
    yield
    logger.handlers.clear()


def test_configure_logging_returns_named_logger(tmp_path):
    logger = logging_setup.configure_logging(log_dir=tmp_path)
    assert logger.name == logging_setup.LOGGER_NAME


def test_default_level_is_info(tmp_path, monkeypatch):
    monkeypatch.delenv("CLONE_VOICE_LOG_LEVEL", raising=False)
    logger = logging_setup.configure_logging(log_dir=tmp_path)
    assert logger.level == logging.INFO


def test_explicit_level_overrides_default(tmp_path):
    logger = logging_setup.configure_logging(level="DEBUG", log_dir=tmp_path)
    assert logger.level == logging.DEBUG


def test_env_var_level_used_when_no_explicit_level(tmp_path, monkeypatch):
    monkeypatch.setenv("CLONE_VOICE_LOG_LEVEL", "WARNING")
    logger = logging_setup.configure_logging(log_dir=tmp_path)
    assert logger.level == logging.WARNING


def test_explicit_level_wins_over_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CLONE_VOICE_LOG_LEVEL", "WARNING")
    logger = logging_setup.configure_logging(level="DEBUG", log_dir=tmp_path)
    assert logger.level == logging.DEBUG


def test_invalid_level_falls_back_to_default_without_crashing(tmp_path):
    logger = logging_setup.configure_logging(level="NOT_A_LEVEL", log_dir=tmp_path)
    assert logger.level == logging.INFO


def test_handlers_are_not_duplicated_on_repeated_calls(tmp_path):
    logging_setup.configure_logging(log_dir=tmp_path)
    logging_setup.configure_logging(log_dir=tmp_path)
    logger = logging.getLogger(logging_setup.LOGGER_NAME)
    assert len(logger.handlers) == 2  # console + file, not 4


def test_console_handler_attached(tmp_path):
    logging_setup.configure_logging(log_dir=tmp_path)
    logger = logging.getLogger(logging_setup.LOGGER_NAME)
    plain_stream_handlers = [
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(plain_stream_handlers) == 1


def test_log_file_created_and_receives_messages(tmp_path):
    logger = logging_setup.configure_logging(level="INFO", log_dir=tmp_path)
    logger.info("hello from batch 0")

    for handler in logger.handlers:
        handler.flush()

    log_file = tmp_path / "clone_voice.log"
    assert log_file.exists()
    assert "hello from batch 0" in log_file.read_text()


def test_logger_does_not_propagate_to_root(tmp_path):
    logger = logging_setup.configure_logging(log_dir=tmp_path)
    assert logger.propagate is False
