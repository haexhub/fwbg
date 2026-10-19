import json
import subprocess
from pathlib import Path

import pytest

from fwbg.results import storage
from fwbg.results.code_patch import (
    PATCH_FILENAME,
    CodePatchError,
    capture_code_patch,
    write_code_patch,
)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path, git_env):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(root, "init", "--quiet")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", "initial")
    return root


@pytest.fixture
def no_repo(tmp_path, git_env, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    return outside


def make_dirty(root: Path) -> None:
    (root / "src" / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "src" / "new_module.py").write_text("NEW = True\n", encoding="utf-8")
    (root / "notes.txt").write_text("scratch\n", encoding="utf-8")


def test_clean_repo_has_empty_patch(repo):
    assert capture_code_patch(repo) == ""


def test_patch_covers_tracked_changes_and_untracked_sources_only(repo):
    make_dirty(repo)
    patch = capture_code_patch(repo)
    assert "src/tracked.py" in patch
    assert "src/new_module.py" in patch
    assert "notes.txt" not in patch


def test_git_local_env_vars_do_not_distort_the_capture(repo, tmp_path, monkeypatch):
    make_dirty(repo)
    expected = capture_code_patch(repo)
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "bogus" / "index"))
    patch = capture_code_patch(repo)
    assert "src/new_module.py" in patch
    assert "-VALUE = 1\n+VALUE = 2\n" in patch
    assert patch == expected


def test_patch_reproduces_the_working_tree(repo, tmp_path):
    make_dirty(repo)
    patch_file = tmp_path / PATCH_FILENAME
    patch_file.write_text(capture_code_patch(repo / "src"), encoding="utf-8")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "--quiet", str(repo), str(clone))
    git(clone, "apply", str(patch_file))
    for name in ("src/tracked.py", "src/new_module.py"):
        assert (clone / name).read_text(encoding="utf-8") == (repo / name).read_text(
            encoding="utf-8"
        )


def test_non_utf8_changes_are_kept_byte_for_byte(repo, tmp_path, monkeypatch):
    legacy = b"NAME = '\xe9'\n"
    (repo / "src" / "legacy.py").write_bytes(legacy)
    monkeypatch.chdir(repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert write_code_patch(run_dir) == PATCH_FILENAME
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "--quiet", str(repo), str(clone))
    git(clone, "apply", str(run_dir / PATCH_FILENAME))
    assert (clone / "src" / "legacy.py").read_bytes() == legacy


def test_outside_a_repository_raises(no_repo):
    with pytest.raises(CodePatchError):
        capture_code_patch(no_repo)


def test_write_code_patch_skips_clean_trees(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert write_code_patch(run_dir) is None
    assert not (run_dir / PATCH_FILENAME).exists()


def test_write_code_patch_outside_a_repository_returns_none(no_repo, monkeypatch):
    monkeypatch.chdir(no_repo)
    assert write_code_patch(no_repo) is None


def test_dirty_run_directory_records_the_patch(repo, tmp_path, monkeypatch):
    make_dirty(repo)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(tmp_path / "results"))
    run_path = Path(storage.create_run_directory("run-1"))
    config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    assert config["git"]["dirty"] is True
    assert config["git"]["patch"] == PATCH_FILENAME
    assert "src/new_module.py" in (run_path / PATCH_FILENAME).read_text(encoding="utf-8")
