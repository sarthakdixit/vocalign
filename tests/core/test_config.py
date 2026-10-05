from core import config


def test_resolve_data_root_defaults_to_repo_root(monkeypatch):
    monkeypatch.delenv("CLONE_VOICE_DATA_DIR", raising=False)
    assert config.resolve_data_root() == config.REPO_ROOT


def test_resolve_data_root_honors_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CLONE_VOICE_DATA_DIR", str(tmp_path))
    assert config.resolve_data_root() == tmp_path.resolve()


def test_resolve_data_root_normalizes_relative_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLONE_VOICE_DATA_DIR", "relative_subdir")
    assert config.resolve_data_root() == (tmp_path / "relative_subdir").resolve()


def test_ensure_app_dirs_creates_all_three(tmp_path):
    created = config.ensure_app_dirs(tmp_path)

    assert set(created) == {"projects", "models", "logs"}
    for path in created.values():
        assert path.is_dir()
    assert created["projects"] == tmp_path / "projects"
    assert created["models"] == tmp_path / "models"
    assert created["logs"] == tmp_path / "logs"


def test_ensure_app_dirs_is_idempotent_and_preserves_contents(tmp_path):
    first = config.ensure_app_dirs(tmp_path)
    marker = first["projects"] / "keep-me.txt"
    marker.write_text("do not delete")

    second = config.ensure_app_dirs(tmp_path)

    assert second == first
    assert marker.read_text() == "do not delete"


def test_ensure_app_dirs_creates_missing_parents(tmp_path):
    nested_root = tmp_path / "does" / "not" / "exist" / "yet"

    created = config.ensure_app_dirs(nested_root)

    assert created["logs"].is_dir()
