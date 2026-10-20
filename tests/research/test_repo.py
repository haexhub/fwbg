import fcntl
import os
import shutil
import subprocess

import pytest
import yaml

from fwbg.api.workspace import get_research_dir
from fwbg.research.fingerprint import canonical_config, canonical_json, digest
from fwbg.research.ledger import LEDGER_FILENAME
from fwbg.research.repo import (
    CRITERIA_TEMPLATE,
    DEFAULT_CRITERIA_YAML,
    REGISTER_LOCK,
    ResearchRepoError,
    commit_register,
    init_research_repo,
    register_lock,
    store_config,
)

REGISTER_FILES = [".gitignore", "templates/criteria-default.yaml", "trials.jsonl"]


def git(path, *args) -> str:
    result = subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def commit_count(path) -> int:
    return int(git(path, "rev-list", "--count", "HEAD"))


def committed_files(path) -> list[str]:
    return git(path, "ls-tree", "-r", "--name-only", "HEAD").splitlines()


def staged_files(path) -> list[str]:
    """List the paths currently staged in the fixture repository."""
    return git(path, "diff", "--cached", "--name-only").splitlines()


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


def test_commit_register_commits_only_changes_under_named_paths(tmp_path, git_env):
    """Commit selected register paths while preserving unrelated staged work."""
    research = tmp_path / "research"
    init_research_repo(research)
    (research / "reports" / "x.md").write_text("operator draft\n", encoding="utf-8")
    git(research, "add", "reports/x.md")
    (research / "strategies" / "S-0001-x.md").write_text("card\n", encoding="utf-8")
    (research / "notes.txt").write_text("private scratch\n", encoding="utf-8")
    assert commit_register(research, ["strategies"], "chore: add card") is True
    assert set(committed_files(research)) == {*REGISTER_FILES, "strategies/S-0001-x.md"}
    assert staged_files(research) == ["reports/x.md"]
    assert git(research, "ls-files", "notes.txt") == ""
    assert commit_register(research, ["strategies"], "chore: nothing") is False


def test_commit_register_accepts_empty_and_missing_named_paths(tmp_path, git_env):
    """Ignore empty or missing directories when collecting paths to commit."""
    research = tmp_path / "research"
    init_research_repo(research)
    (research / "reports").rmdir()
    (research / "strategies" / "S-0001-x.md").write_text("card\n", encoding="utf-8")
    paths = ["strategies", "configs", "reports", LEDGER_FILENAME]
    assert commit_register(research, paths, "chore: add card") is True
    assert "strategies/S-0001-x.md" in committed_files(research)


def test_commit_register_commits_renames_and_deletions(tmp_path, git_env):
    """Commit both sides of renames and include deleted register files."""
    research = tmp_path / "research"
    init_research_repo(research)
    cards = research / "strategies"
    (cards / "S-0001-old.md").write_text("card\n", encoding="utf-8")
    (cards / "S-0002-gone.md").write_text("other card\n", encoding="utf-8")
    commit_register(research, ["strategies"], "chore: add cards")
    (cards / "S-0001-old.md").rename(cards / "S-0001-new.md")
    (cards / "S-0002-gone.md").unlink()
    assert commit_register(research, ["strategies"], "chore: rename card") is True
    assert set(committed_files(research)) == {*REGISTER_FILES, "strategies/S-0001-new.md"}
    assert staged_files(research) == []


def test_commit_register_commits_a_deleted_tracked_directory(tmp_path, git_env):
    """Commit deletions when a selected tracked directory no longer exists."""
    research = tmp_path / "research"
    init_research_repo(research)
    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))
    store_config(research, digest(config), config)
    commit_register(research, ["configs"], "chore: add config")
    shutil.rmtree(research / "configs")
    assert commit_register(research, ["configs"], "chore: drop configs") is True
    assert set(committed_files(research)) == set(REGISTER_FILES)
    assert git(research, "status", "--porcelain") == ""


def test_temporary_files_are_never_committed(tmp_path, git_env, monkeypatch):
    """Keep stray and atomic-write temporary config files out of register commits."""
    research = tmp_path / "research"
    init_research_repo(research)
    (research / "configs" / "stray.json.tmp").write_text("{", encoding="utf-8")
    temp_names = []
    replace = os.replace

    def record(source, target):
        """Record the temporary filename before performing the real atomic replacement."""
        temp_names.append(os.path.basename(source))
        replace(source, target)

    monkeypatch.setattr("fwbg.research.repo.os.replace", record)
    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))
    path = store_config(research, digest(config), config)
    assert [name.endswith(".tmp") for name in temp_names] == [True]
    assert commit_register(research, ["configs"], "chore: add config") is True
    assert set(committed_files(research)) == {*REGISTER_FILES, f"configs/{path.name}"}


def test_commit_register_keeps_register_ignored_files_out(tmp_path, git_env):
    """Respect register ignore rules when staging selected paths."""
    research = tmp_path / "research"
    init_research_repo(research)
    for staging in (research / ".staging" / "run-1", research / "strategies" / ".staging"):
        staging.mkdir(parents=True)
        (staging / "x.json").write_text("{}", encoding="utf-8")
    (research / "strategies" / "S-0001-x.md").write_text("card\n", encoding="utf-8")
    assert commit_register(research, ["strategies"], "chore: add card") is True
    assert set(committed_files(research)) == {*REGISTER_FILES, "strategies/S-0001-x.md"}
    with pytest.raises(ResearchRepoError, match="git add"):
        commit_register(research, [".staging"], "chore: add staging")
    assert set(committed_files(research)) == {*REGISTER_FILES, "strategies/S-0001-x.md"}


def test_commit_register_ignores_global_excludes(tmp_path, git_env):
    """Commit register configs even when global Git excludes hide JSON files."""
    excludes = tmp_path / "global-excludes"
    excludes.write_text("*.json\n", encoding="utf-8")
    git(tmp_path, "config", "--global", "core.excludesFile", str(excludes))
    research = tmp_path / "research"
    init_research_repo(research)
    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))
    path = store_config(research, digest(config), config)
    assert commit_register(research, ["configs"], "chore: add config") is True
    assert f"configs/{path.name}" in committed_files(research)


def test_commit_register_without_paths_raises(tmp_path, git_env):
    """Reject an empty path list before committing unrelated staged work."""
    research = tmp_path / "research"
    init_research_repo(research)
    (research / "reports" / "x.md").write_text("operator draft\n", encoding="utf-8")
    git(research, "add", "reports/x.md")
    with pytest.raises(ValueError, match="path"):
        commit_register(research, [], "chore: everything")
    assert commit_count(research) == 1


def test_git_timeout_leaves_no_lock_behind(tmp_path, git_env, monkeypatch):
    """Ensure terminating a timed-out Git commit removes its index lock."""
    research = tmp_path / "research"
    init_research_repo(research)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    (hooks / "pre-commit").chmod(0o755)
    git(tmp_path, "config", "--global", "core.hooksPath", str(hooks))
    monkeypatch.setattr("fwbg.research.repo._GIT_TIMEOUT", 1.0)
    (research / "strategies" / "S-0001-x.md").write_text("card\n", encoding="utf-8")
    with pytest.raises(ResearchRepoError, match="git commit"):
        commit_register(research, ["strategies"], "chore: add card")
    assert list((research / ".git").glob("*.lock")) == []


def flock_shared_nonblocking(path) -> None:
    """Take and release a shared lock through a second open file.

    Raises ``BlockingIOError`` only while another holder has an exclusive lock.
    """
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)


def test_register_lock_is_exclusive_until_released(tmp_path):
    """Verify the register lock excludes other holders and releases after exceptions."""
    lock = tmp_path / REGISTER_LOCK
    with register_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            flock_shared_nonblocking(lock)
    flock_shared_nonblocking(lock)
    with pytest.raises(RuntimeError, match="boom"), register_lock(tmp_path):
        raise RuntimeError("boom")
    flock_shared_nonblocking(lock)


def test_register_lock_file_is_ignored_by_git(tmp_path, git_env):
    """Keep the register lock file outside tracked register changes."""
    research = tmp_path / "research"
    init_research_repo(research)
    with register_lock(research):
        assert (research / REGISTER_LOCK).exists()
        assert git(research, "status", "--porcelain") == ""


def config_path(research_dir, fingerprint):
    """Return the stored config path for a SHA-256 fingerprint."""
    return research_dir / "configs" / f"{fingerprint.removeprefix('sha256:')}.json"


def test_store_config_writes_canonical_json_under_its_digest(tmp_path):
    """Store canonical config bytes at their digest path and reuse identical content."""
    (tmp_path / "configs").mkdir()
    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))
    fingerprint = digest(config)
    path = store_config(tmp_path, fingerprint, config)
    assert path == config_path(tmp_path, fingerprint)
    assert path.read_text(encoding="utf-8") == canonical_json(config) + "\n"
    assert store_config(tmp_path, fingerprint, config) == path
    assert list((tmp_path / "configs").iterdir()) == [path]


def test_store_config_refuses_a_config_that_is_not_the_fingerprint(tmp_path):
    """Reject malformed or mismatched fingerprints before storing a config."""
    (tmp_path / "configs").mkdir()
    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))
    other = canonical_config({"timeframe": "HOUR_1"}, ("GOLD",))
    for fingerprint in (digest(other), "sha256:not-a-digest"):
        with pytest.raises(ValueError, match="not the fingerprint"):
            store_config(tmp_path, fingerprint, config)
    assert list((tmp_path / "configs").iterdir()) == []


@pytest.mark.parametrize("existing", [b"tampered\n", b"\xff\xfe not utf-8\n"])
def test_store_config_refuses_a_different_existing_file(tmp_path, existing):
    """Preserve conflicting existing bytes and report a storage error."""
    (tmp_path / "configs").mkdir()
    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))
    path = config_path(tmp_path, digest(config))
    path.write_bytes(existing)
    with pytest.raises(ResearchRepoError, match="different content"):
        store_config(tmp_path, digest(config), config)
    assert path.read_bytes() == existing


def test_store_config_leaves_nothing_behind_when_the_write_fails(tmp_path, monkeypatch):
    """Remove temporary config files when atomic replacement fails."""
    (tmp_path / "configs").mkdir()
    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))

    def fail(*_args):
        """Simulate a failed atomic replacement caused by a full disk."""
        raise OSError("disk full")

    monkeypatch.setattr("fwbg.research.repo.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        store_config(tmp_path, digest(config), config)
    assert list((tmp_path / "configs").iterdir()) == []
