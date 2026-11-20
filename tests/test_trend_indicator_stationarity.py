"""Trend indicators work on price levels, so they must not get differenced prices."""

import pytest

from fwbg.pipeline import get_registry
from fwbg.pipeline.features import split_indicators_by_stationarity

PRICE_LEVEL_INDICATORS = [
    "adx",
    "aroon",
    "cci",
    "ema",
    "macd",
    "sma",
    "supertrend",
    "supertrend_mtf",
]


@pytest.mark.parametrize("name", PRICE_LEVEL_INDICATORS)
def test_price_level_indicator_does_not_request_stationary_input(name):
    registry = get_registry()
    registry.auto_discover()
    plugin_cls = registry.get(registry.resolve_name(name))
    assert plugin_cls.benefits_from_stationary is False


def test_price_level_indicators_are_computed_on_raw_prices_with_preprocessing():
    configs = [{"name": name, "params": {}} for name in PRICE_LEVEL_INDICATORS]
    stationary, raw = split_indicators_by_stationarity(configs, has_preprocessing=True)
    assert stationary == []
    assert [c["name"] for c in raw] == PRICE_LEVEL_INDICATORS
