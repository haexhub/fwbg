"""Seed and fixture strategy configs only pass parameters their plugins declare.

A parameter that a plugin dropped (for example the volatility scaling of the
deprecated ``vol_targeted_kelly``) is otherwise ignored without a trace.
"""

import glob
import json
import pathlib

import pytest

from fwbg.core import get_exit_modifier, get_exit_strategy

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIRS = [
    ROOT / "tests" / "_fixtures" / "workspace" / "strategies",
    ROOT / "src" / "fwbg" / "workspace_seed" / "strategies",
]
FILES = sorted(
    f for d in CONFIG_DIRS for f in glob.glob(str(d / "**" / "*.json"), recursive=True)
)


def _schema(getter, name):
    try:
        return getter(name).get_param_schema()
    except (KeyError, ValueError, ImportError):
        pytest.skip(f"plugin {name!r} is not installed (fwbg-premium)")


@pytest.mark.parametrize("path", FILES, ids=lambda p: pathlib.Path(p).name)
def test_exit_params_are_declared(path):
    config = json.loads(pathlib.Path(path).read_text())
    for entry in config.get("exit_strategies", []):
        schema = _schema(get_exit_strategy, entry["name"])
        undeclared = set(entry.get("params", {})) - set(schema)
        assert not undeclared, f"{entry['name']}: undeclared params {undeclared}"
        modifier = entry.get("exit_modifier")
        if modifier:
            mod_schema = _schema(get_exit_modifier, modifier)
            undeclared = set(entry.get("exit_modifier_params", {})) - set(mod_schema)
            assert not undeclared, f"{modifier}: undeclared params {undeclared}"


def test_no_config_names_the_deprecated_risk_alias():
    for path in FILES:
        assert "vol_targeted_kelly" not in pathlib.Path(path).read_text(), path
