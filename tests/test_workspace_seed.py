"""Tests for seed_workspace_presets() — packaged defaults land in fresh
workspaces, existing files are never overwritten."""
import json

import pytest

from fwbg.api.workspace import _SEED_ROOT, seed_workspace_presets


@pytest.fixture
def fresh_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("FWBG_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("FWBG_STRATEGIES_DIR", raising=False)
    return tmp_path


def test_seed_populates_fresh_workspace(fresh_workspace):
    seeded = seed_workspace_presets()

    seed_files = sorted(p.relative_to(_SEED_ROOT) for p in _SEED_ROOT.glob("*/*.json"))
    assert seeded == len(seed_files) > 0
    for rel in seed_files:
        target = fresh_workspace / "strategies" / rel
        assert target.is_file(), f"missing seeded preset {rel}"
        # Seeds are already in the versioned _meta format migrate_presets expects.
        assert "_meta" in json.loads(target.read_text())


def test_seed_is_idempotent_and_never_overwrites(fresh_workspace):
    seed_workspace_presets()

    edited = fresh_workspace / "strategies" / "pipelines" / "orb_simple_v1.json"
    edited.write_text('{"_meta": {"name": "user-edited"}}')

    assert seed_workspace_presets() == 0
    assert json.loads(edited.read_text())["_meta"]["name"] == "user-edited"
