"""Tests for benefits_from_stationary indicator attribute."""

import pytest
from fwbg.pipeline import get_registry
from fwbg_sdk import PluginPhase


def test_all_indicators_have_benefits_from_stationary():
    """Every indicator plugin must declare benefits_from_stationary."""
    registry = get_registry()
    registry.auto_discover()
    indicators = registry.list_plugins(phase=PluginPhase.INDICATORS)
    assert len(indicators) >= 15

    for fqn in indicators:
        plugin_cls = registry.get(fqn)
        assert hasattr(plugin_cls, "benefits_from_stationary"), (
            f"{fqn} missing benefits_from_stationary"
        )
        assert isinstance(plugin_cls.benefits_from_stationary, bool), (
            f"{fqn}.benefits_from_stationary must be bool"
        )


def test_stationary_indicators_correct():
    """Verify the correct indicators are marked as benefiting from stationary."""
    registry = get_registry()
    registry.auto_discover()
    indicators = registry.list_plugins(phase=PluginPhase.INDICATORS)

    expected_true = {"price_action", "efficiency_ratio"}
    expected_false = {
        "adx",
        "aroon",
        "cci",
        "ema",
        "macd",
        "momentum",
        "sma",
        "supertrend",
        "supertrend_mtf",
        "time_season",
        "volatility",
        "zscore",
    }

    for fqn in indicators:
        plugin_cls = registry.get(fqn)
        short_name = fqn.split(":")[1]
        if short_name in expected_true:
            assert plugin_cls.benefits_from_stationary is True, f"{fqn} should be True"
        elif short_name in expected_false:
            assert plugin_cls.benefits_from_stationary is False, (
                f"{fqn} should be False"
            )


def test_split_indicators_by_stationarity():
    from fwbg.pipeline.features import split_indicators_by_stationarity

    indicators = [
        {"name": "ema", "params": {}},
        {"name": "momentum", "params": {"rsi_periods": [14]}},
        {"name": "cci", "params": {}},
        {"name": "volatility", "params": {}},
    ]

    stationary, raw = split_indicators_by_stationarity(indicators)

    stat_names = [i["name"] for i in stationary]
    raw_names = [i["name"] for i in raw]

    assert "ema" in raw_names
    assert "cci" in raw_names
    assert "momentum" in raw_names
    assert "volatility" in raw_names


def test_split_no_preprocessing_all_raw():
    from fwbg.pipeline.features import split_indicators_by_stationarity

    indicators = [
        {"name": "trend", "params": {}},
        {"name": "momentum", "params": {}},
    ]
    stationary, raw = split_indicators_by_stationarity(
        indicators, has_preprocessing=False
    )
    assert len(stationary) == 0
    assert len(raw) == 2
