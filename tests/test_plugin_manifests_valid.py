"""Every bundled plugin manifest must be valid JSON with the required keys."""

import json
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1] / "src" / "fwbg" / "plugins"
MANIFESTS = sorted(PLUGIN_ROOT.rglob("manifest.json"))


def test_manifests_are_found():
    assert len(MANIFESTS) > 10


@pytest.mark.parametrize(
    "path", MANIFESTS, ids=lambda p: str(p.relative_to(PLUGIN_ROOT))
)
def test_manifest_is_valid_json_with_name_and_version(path):
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    if "plugins" in data:
        return
    assert data.get("name"), "manifest needs a name"
    assert data.get("version"), "manifest needs a version"
