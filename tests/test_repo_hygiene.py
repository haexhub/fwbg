"""Tracked repository files stay portable and discoverable."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_tracked_claude_settings_hold_no_personal_paths_or_permissions():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())

    assert "permissions" not in settings
    assert "/home/" not in json.dumps(settings)


def test_no_test_modules_outside_the_test_paths():
    """A test_*.py at the repository root is never collected (see testpaths)."""
    assert sorted(p.name for p in ROOT.glob("test_*.py")) == []
