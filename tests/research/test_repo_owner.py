"""Register writes from a root process (the API container) on a host-owned register."""

import logging
import os
from pathlib import Path

import pytest

from fwbg.research import repo
from fwbg.research.repo import (
    LEDGER_FILENAME,
    REGISTER_LOCK,
    commit_register,
    init_research_repo,
    register_lock,
    restore_owner,
)


def _pretend_root(monkeypatch, created_by_root):
    """Run as root on a register owned by 4242:4343; *created_by_root(path)* marks root's files.

    Returns the recorded ``chown`` calls as ``(path, uid, gid, kwargs)``.
    """
    calls = []
    real_lstat = os.lstat

    # A real stat result with st_uid 0 for root's files and the owner's uid for the rest,
    # whoever runs the tests: os.walk itself reads st_mode through os.lstat.
    def lstat(path, **kwargs):
        fields = list(real_lstat(path, **kwargs))
        fields[4] = 0 if created_by_root(Path(path)) else 4242  # st_uid
        return os.stat_result(fields)

    monkeypatch.setattr(repo.os, "geteuid", lambda: 0)
    monkeypatch.setattr(repo, "_owner", lambda path: (4242, 4343))
    monkeypatch.setattr(repo.os, "lstat", lstat)
    monkeypatch.setattr(
        repo.os,
        "chown",
        lambda path, uid, gid, **k: calls.append((Path(path), uid, gid, k)),
    )
    return calls


def test_restore_owner_is_a_no_op_for_a_normal_user(tmp_path, git_env, monkeypatch):
    calls = []
    monkeypatch.setattr(repo.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(repo.os, "chown", lambda *a, **k: calls.append(a))
    init_research_repo(tmp_path)
    restore_owner(tmp_path)
    assert calls == []


def test_restore_owner_chowns_only_what_root_created(tmp_path, git_env, monkeypatch):
    init_research_repo(tmp_path)

    def created_by_root(path):
        name = path.relative_to(tmp_path).as_posix()
        return name == "trials.jsonl" or name.startswith(".git/objects/")

    calls = _pretend_root(monkeypatch, created_by_root)
    restore_owner(tmp_path)
    changed = {path.relative_to(tmp_path).as_posix() for path, _, _, _ in calls}
    assert "trials.jsonl" in changed
    assert any(name.startswith(".git/objects/") for name in changed)
    assert all(name == "trials.jsonl" or name.startswith(".git/objects/") for name in changed)
    assert {(uid, gid) for _, uid, gid, _ in calls} == {(4242, 4343)}


def test_restore_owner_never_follows_symlinks(tmp_path, git_env, monkeypatch):
    research = tmp_path / "research"
    init_research_repo(research)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("", encoding="utf-8")
    (research / "file-link").symlink_to(outside / "secret")
    (research / "dir-link").symlink_to(outside)
    calls = _pretend_root(monkeypatch, lambda path: True)
    restore_owner(research)
    chowned = {path for path, _, _, _ in calls}
    assert research / "file-link" in chowned
    assert research / "dir-link" in chowned
    assert not any(path.is_relative_to(outside) for path in chowned)
    assert all(kwargs == {"follow_symlinks": False} for _, _, _, kwargs in calls)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a directory with mode 0")
def test_restore_owner_logs_an_unreadable_directory(tmp_path, git_env, monkeypatch, caplog):
    init_research_repo(tmp_path)
    locked = tmp_path / "reports" / "locked"
    locked.mkdir()
    locked.chmod(0)
    _pretend_root(monkeypatch, lambda path: False)
    try:
        with caplog.at_level(logging.WARNING, logger=repo.__name__):
            restore_owner(tmp_path)
    finally:
        locked.chmod(0o755)
    assert str(locked) in caplog.text


def test_restore_owner_skips_a_root_owned_register(tmp_path, git_env, monkeypatch):
    init_research_repo(tmp_path)
    calls = []
    monkeypatch.setattr(repo.os, "geteuid", lambda: 0)
    monkeypatch.setattr(repo, "_owner", lambda path: (0, 0))
    monkeypatch.setattr(repo.os, "chown", lambda *a, **k: calls.append(a))
    restore_owner(tmp_path)
    assert calls == []


def test_register_lock_restores_the_owner_on_exit(tmp_path, git_env, monkeypatch):
    init_research_repo(tmp_path)
    seen = []
    monkeypatch.setattr(repo, "restore_owner", lambda path: seen.append(path))
    with register_lock(tmp_path):
        assert seen == []
    assert seen == [tmp_path]


def test_register_lock_restores_the_owner_after_an_error(tmp_path, git_env, monkeypatch):
    init_research_repo(tmp_path)
    seen = []
    monkeypatch.setattr(repo, "restore_owner", lambda path: seen.append(path))
    try:
        with register_lock(tmp_path):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert seen == [tmp_path]


def test_register_lock_works_on_a_lock_file_it_cannot_write(tmp_path, git_env):
    """A lock file root created stays read-only for the host user, who must still lock it."""
    init_research_repo(tmp_path)
    lock = tmp_path / REGISTER_LOCK
    lock.touch()
    lock.chmod(0o444)
    with register_lock(tmp_path):
        pass


@pytest.mark.skipif(os.geteuid() != 0, reason="requires root to simulate a foreign-owned register")
def test_commit_register_accepts_a_foreign_owned_register(tmp_path, git_env):
    init_research_repo(tmp_path)
    os.chown(tmp_path, 65534, 65534)
    assert commit_register(tmp_path, [LEDGER_FILENAME], "test: no changes") is False
