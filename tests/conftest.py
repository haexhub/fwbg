"""Globale Test-Fixtures."""
import glob
import os

import pytest


def pytest_configure(config):
    """Set up test workspace + clear stale Numba caches.

    The strategy/pipeline JSONs that several tests load live under
    ``tests/_fixtures/workspace/`` so CI can find them without depending on
    the developer's home directory. We point ``FWBG_WORKSPACE`` at that
    directory unless the caller explicitly set one.

    Then drops project-level Numba caches to avoid stale-cache errors after
    signature changes on @njit-compiled functions.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fixture_ws = os.path.join(project_root, "tests", "_fixtures", "workspace")
    if os.path.isdir(fixture_ws):
        os.environ.setdefault("FWBG_WORKSPACE", fixture_ws)

    # create_app() fails closed when no FWBG_API_KEY is set, and most API tests
    # build an app without one, so the dev bypass is set here — before any test
    # module import — rather than in a per-test fixture. Tests that exercise
    # auth override this explicitly via monkeypatch or a subprocess.
    os.environ.setdefault("FWBG_ALLOW_UNAUTHENTICATED_API", "1")

    patterns = [
        os.path.join(project_root, "src", "**", "*.nbi"),
        os.path.join(project_root, "src", "**", "*.nbc"),
        os.path.join(project_root, "packages", "**", "*.nbi"),
        os.path.join(project_root, "packages", "**", "*.nbc"),
    ]
    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            os.remove(path)
            removed += 1
    if removed > 0:
        print(f"\n[conftest] {removed} stale Numba cache files removed")


@pytest.fixture
def git_env(tmp_path_factory, monkeypatch):
    """Deterministic git: fixed identity, no user or system configuration."""
    # Lazy: importing fwbg before pytest_configure sets FWBG_WORKSPACE pins the real data dir.
    from fwbg.research.repo import GIT_LOCAL_ENV_VARS

    config = tmp_path_factory.mktemp("gitconfig") / "config"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in GIT_LOCAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "fwbg tests")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "tests@fwbg.invalid")
