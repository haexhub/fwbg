import subprocess

import pytest
import yaml

from fwbg.api.workspace import get_research_dir
from fwbg.research.ledger import LEDGER_FILENAME
from fwbg.research.repo import (
    CRITERIA_TEMPLATE,
    DEFAULT_CRITERIA_YAML,
    ResearchRepoError,
    init_research_repo,
)


def commit_count(path) -> int:
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip())


def test_init_creates_layout_template_ledger_and_repo(tmp_path, git_env):
    research = tmp_path / "research"
    assert init_research_repo(research) is True
    for name in ("strategies", "configs", "templates", "reports"):
        assert (research / name).is_dir()
    assert (research / CRITERIA_TEMPLATE).read_text(encoding="utf-8") == DEFAULT_CRITERIA_YAML
    assert (research / LEDGER_FILENAME).read_text(encoding="utf-8") == ""
    assert commit_count(research) == 1


def test_init_is_idempotent_and_keeps_edits(tmp_path, git_env):
    research = tmp_path / "research"
    init_research_repo(research)
    (research / CRITERIA_TEMPLATE).write_text("min_sharpe: 2.0\n", encoding="utf-8")
    assert init_research_repo(research) is False
    assert (research / CRITERIA_TEMPLATE).read_text(encoding="utf-8") == "min_sharpe: 2.0\n"
    assert commit_count(research) == 1


def test_init_without_git_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "no-binaries"))
    with pytest.raises(ResearchRepoError, match="git init"):
        init_research_repo(tmp_path / "research")


def test_default_criteria_are_numbers():
    criteria = yaml.safe_load(DEFAULT_CRITERIA_YAML)
    assert criteria
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in criteria.values())


def test_research_dir_defaults_to_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("FWBG_RESEARCH_DIR", raising=False)
    monkeypatch.setenv("FWBG_WORKSPACE", str(tmp_path))
    assert get_research_dir() == tmp_path / "research"


def test_research_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "elsewhere"))
    assert get_research_dir() == tmp_path / "elsewhere"
