"""A strategy config key the loader no longer reads must not pass silently."""

import logging

import pytest

from fwbg.core.config import StrategyConfig


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        (
            "grids",
            "optimization.model_hyperparameters_grid, exit_strategies and model.required_features",
        ),
        ("exit_strategy", "exit_strategies"),
        ("exit_params", "exit_strategies"),
    ],
)
def test_removed_key_is_rejected_with_its_replacement(key, replacement):
    with pytest.raises(ValueError) as exc_info:
        StrategyConfig.from_dict({"name": "x", key: {}})
    assert f"'{key}' (use {replacement})" in str(exc_info.value)


def test_unknown_key_is_warned_about(caplog):
    with caplog.at_level(logging.WARNING, logger="fwbg.core.config"):
        cfg = StrategyConfig.from_dict({"name": "x", "optimisation": {}})
    assert cfg.name == "x"
    assert "'optimisation'" in caplog.text


def test_underscore_keys_are_metadata(caplog):
    with caplog.at_level(logging.WARNING, logger="fwbg.core.config"):
        cfg = StrategyConfig.from_dict({"name": "x", "_refs": ["a"], "_strategy_dir": "/tmp"})
    assert cfg.name == "x"
    assert caplog.text == ""


def test_every_key_the_loader_reads_is_known(caplog):
    data = {
        "name": "x",
        "description": "d",
        "tags": [],
        "pipeline": {},
        "exit_strategies": [],
        "risk_management": "kelly",
        "risk_params": {},
        "optimization": {},
        "assets": {},
        "model": {},
        "validation": {},
        "filters": {},
        "resources": {},
        "datasource": "d",
        "timeframe": "HOUR",
        "start_date": None,
        "end_date": None,
        "holdout_cutoff": None,
        "cost_multiplier": 1.0,
        "financing_pct": 0.0,
        "data_checks": "strict",
        "signal_rules": None,
        "hypothesis": "",
        "expected_outcome": "",
    }
    assert set(data) | {"regime_filter"} == StrategyConfig.KNOWN_KEYS
    with caplog.at_level(logging.WARNING, logger="fwbg.core.config"):
        StrategyConfig.from_dict(data)
    assert caplog.text == ""
