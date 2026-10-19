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

REGISTER_FILES = [".gitignore", "templates/criteria-default.yaml", "trials.jsonl"]


def git(path, *args) -> str:
    result = subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def commit_count(path) -> int:
    return int(git(path, "rev-list", "--count", "HEAD"))


def committed_files(path) -> list[str]:
    return git(path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()


def test_init_creates_layout_template_ledger_and_repo(tmp_path, git_env):
    research = tmp_path / "research"
    assert init_research_repo(research) is True
    for name in ("strategies", "configs", "templates", "reports"):
        assert (research / name).is_dir()
    assert (research / CRITERIA_TEMPLATE).read_text(encoding="utf-8") == DEFAULT_CRITERIA_YAML
    assert (research / LEDGER_FILENAME).read_text(encoding="utf-8") == ""
    assert commit_count(research) == 1
    assert committed_files(research) == REGISTER_FILES
    assert git(research, "branch", "--show-current") == "main"


def test_register_ignores_staging_and_lock(tmp_path, git_env):
    research = tmp_path / "research"
    init_research_repo(research)
    (research / ".staging").mkdir()
    (research / ".staging" / "run.json").write_text("{}", encoding="utf-8")
    (research / ".register.lock").write_text("", encoding="utf-8")
    assert git(research, "status", "--porcelain") == ""


def test_init_commits_only_register_files(tmp_path, git_env):
    research = tmp_path / "research"
    (research / "accounts").mkdir(parents=True)
    (research / "accounts" / "secret.json").write_text('{"api_key": "x"}', encoding="utf-8")
    assert init_research_repo(research) is True
    assert committed_files(research) == REGISTER_FILES


def test_failed_first_commit_is_retried(tmp_path, git_env, monkeypatch):
    research = tmp_path / "research"
    config = tmp_path / "no-identity.gitconfig"
    config.write_text("[user]\n\tuseConfigOnly = true\n", encoding="utf-8")
    with monkeypatch.context() as no_identity:
        no_identity.setenv("GIT_CONFIG_GLOBAL", str(config))
        for role in ("AUTHOR", "COMMITTER"):
            no_identity.delenv(f"GIT_{role}_NAME")
            no_identity.delenv(f"GIT_{role}_EMAIL")
        with pytest.raises(ResearchRepoError, match="git commit"):
            init_research_repo(research)
    assert init_research_repo(research) is True
    assert commit_count(research) == 1


def test_init_completes_a_repository_without_commit(tmp_path, git_env):
    research = tmp_path / "research"
    research.mkdir()
    git(research, "init", "--quiet")
    assert init_research_repo(research) is True
    assert commit_count(research) == 1


def test_missing_ledger_in_initialised_register_is_not_recreated(tmp_path, git_env):
    research = tmp_path / "research"
    init_research_repo(research)
    (research / LEDGER_FILENAME).unlink()
    with pytest.raises(ResearchRepoError, match="restore"):
        init_research_repo(research)
    assert not (research / LEDGER_FILENAME).exists()


def make_repo_with_commit(path) -> None:
    path.mkdir()
    git(path, "init", "--quiet")
    (path / "a.txt").write_text("a", encoding="utf-8")
    git(path, "add", "a.txt")
    git(path, "commit", "--quiet", "--message", "outer")


def repo_state(path) -> tuple[bytes, str]:
    return (path / ".git" / "index").read_bytes(), git(path, "log", "--format=%H")


@pytest.mark.parametrize(
    ("variable", "target"), [("GIT_DIR", ".git"), ("GIT_INDEX_FILE", ".git/index")]
)
def test_inherited_git_variables_cannot_redirect_init(
    tmp_path, git_env, monkeypatch, variable, target
):
    outer = tmp_path / "outer"
    make_repo_with_commit(outer)
    before = repo_state(outer)
    research = tmp_path / "research"

    monkeypatch.setenv(variable, str(outer / target))
    assert init_research_repo(research) is True
    monkeypatch.delenv(variable)

    assert (research / ".git").is_dir()
    assert committed_files(research) == REGISTER_FILES
    assert repo_state(outer) == before


def test_damaged_register_repository_raises_without_using_enclosing_repo(tmp_path, git_env):
    outer = tmp_path / "outer"
    make_repo_with_commit(outer)
    research = outer / "research"
    init_research_repo(research)
    (research / ".git" / "HEAD").unlink()
    before = repo_state(outer)
    with pytest.raises(ResearchRepoError, match="git rev-parse"):
        init_research_repo(research)
    assert repo_state(outer) == before


def test_init_commits_register_files_despite_global_excludes(tmp_path, git_env, monkeypatch):
    excludes = tmp_path / "global-excludes"
    excludes.write_text("*\n", encoding="utf-8")
    config = tmp_path / "excludes.gitconfig"
    config.write_text(f'[core]\n\texcludesFile = "{excludes}"\n', encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    research = tmp_path / "research"
    assert init_research_repo(research) is True
    assert committed_files(research) == REGISTER_FILES


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
