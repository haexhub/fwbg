"""Register writes from a root process (the API container) on a host-owned register."""

import subprocess
from pathlib import Path

from fwbg.research import repo
from fwbg.research.repo import init_research_repo, register_lock, restore_owner


def test_restore_owner_is_a_no_op_for_a_normal_user(tmp_path, git_env, monkeypatch):
    calls = []
    monkeypatch.setattr(repo.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(repo.os, "chown", lambda *a, **k: calls.append(a))
    init_research_repo(tmp_path)
    restore_owner(tmp_path)
    assert calls == []


def test_restore_owner_chowns_everything_root_created(tmp_path, git_env, monkeypatch):
    init_research_repo(tmp_path)
    calls = []
    monkeypatch.setattr(repo.os, "geteuid", lambda: 0)
    monkeypatch.setattr(repo, "_owner", lambda path: (4242, 4343))
    monkeypatch.setattr(
        repo.os, "chown", lambda path, uid, gid, **k: calls.append((Path(path), uid, gid))
    )
    restore_owner(tmp_path)
    changed = {path.relative_to(tmp_path).as_posix() for path, _, _ in calls}
    assert "trials.jsonl" in changed
    assert any(name.startswith(".git/objects/") for name in changed)
    assert {(uid, gid) for _, uid, gid in calls} == {(4242, 4343)}


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


def test_git_trusts_the_register_directory(tmp_path, git_env, monkeypatch):
    init_research_repo(tmp_path)
    seen = []
    real = subprocess.Popen

    def spy(args, **kwargs):
        seen.append(args)
        return real(args, **kwargs)

    monkeypatch.setattr(repo.subprocess, "Popen", spy)
    repo._git(tmp_path, "status")
    assert f"safe.directory={tmp_path.resolve()}" in seen[0]
