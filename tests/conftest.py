"""Globale Test-Fixtures."""
import glob
import os


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
