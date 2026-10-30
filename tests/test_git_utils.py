"""file_at_commit reads a file at a ref and never treats the ref as an option."""
import subprocess

import pytest

from fwbg.api.git_utils import file_at_commit


@pytest.fixture
def repo(tmp_path, git_env):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "t")
    git("config", "user.email", "t@example.invalid")
    (tmp_path / "s.json").write_text('{"v": 1}')
    git("add", "s.json")
    git("commit", "-q", "-m", "init")
    return tmp_path


def test_reads_file_at_ref(repo):
    assert file_at_commit(repo, "s.json", "HEAD") == '{"v": 1}'


def test_option_like_ref_is_not_an_option(repo):
    with pytest.raises(RuntimeError):
        file_at_commit(repo, "s.json", f"--output={repo / 'pwned'}")

    assert not (repo / "pwned").exists()
