"""Globale Test-Fixtures."""
import glob
import os


def pytest_configure(config):
    """Numba-Cache für Projektdateien löschen vor dem Test-Run.

    Verhindert stale-cache Fehler nach Signaturänderungen an @njit-Funktionen.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
