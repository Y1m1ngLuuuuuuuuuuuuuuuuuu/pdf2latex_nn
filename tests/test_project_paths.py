from pathlib import Path

from src.config import project_paths


def test_env_override_for_data_root(monkeypatch, tmp_path):
    data_root = tmp_path / "data-root"
    monkeypatch.setenv("PDF2LATEX_DATA_ROOT", str(data_root))

    assert project_paths.get_data_root() == data_root.resolve()


def test_explicit_project_root_override(tmp_path):
    root = tmp_path / "repo"

    assert project_paths.get_project_root(root) == root.resolve()


def test_describe_paths_contains_expected_keys():
    values = project_paths.describe_paths()

    assert "project_root" in values
    assert "data_root" in values
    assert all(isinstance(value, str) for value in values.values())

