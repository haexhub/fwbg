from fwbg.core.git_env import isolated_git_env


def test_isolated_git_env_drops_repository_variables_and_optional_locks(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
    monkeypatch.setenv("GIT_INDEX_FILE", "/elsewhere/index")
    monkeypatch.setenv("FWBG_X", "kept")
    env = isolated_git_env()
    assert "GIT_DIR" not in env
    assert "GIT_INDEX_FILE" not in env
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["FWBG_X"] == "kept"
