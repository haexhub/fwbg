"""Tests for Supertrend Indicator plugin (standard + MTF)."""
import numpy as np
import pandas as pd
import pytest


def create_ohlc(close, high_factor=1.005, low_factor=0.995):
    n = len(close)
    return pd.DataFrame({
        'O': close * 0.999,
        'H': close * high_factor,
        'L': close * low_factor,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))


@pytest.fixture
def indicator():
    from fwbg.plugins import import_plugin_module
    mod = import_plugin_module("fwbg-core", "indicators", "supertrend")
    return mod.SupertrendIndicator()


@pytest.fixture
def mtf_indicator():
    from fwbg.plugins import import_plugin_module
    mod = import_plugin_module("fwbg-core", "indicators", "supertrend")
    return mod.SupertrendMTFIndicator()


class TestSupertrend:
    def test_direction_binary(self, indicator):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        result = indicator.compute(create_ohlc(close))
        valid = result["st_direction"].dropna()
        assert valid.isin([1, -1]).all()

    def test_positive_in_uptrend(self, indicator):
        close = np.linspace(100, 200, 300)
        result = indicator.compute(create_ohlc(close))
        valid = result["st_direction"].dropna()
        pct_positive = (valid.iloc[-100:] == 1).mean()
        assert pct_positive > 0.6

    def test_flip_at_reversal(self, indicator):
        close = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 100, 200)])
        result = indicator.compute(create_ohlc(close))
        flips = result["st_flip"].dropna()
        assert flips.sum() > 0

    def test_name(self, indicator):
        assert indicator.name == "supertrend"


class TestSupertrendMTF:
    def test_direction_exists(self, mtf_indicator):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        result = mtf_indicator.compute(create_ohlc(close))
        assert "st_d1_direction" in result.columns

    def test_name(self, mtf_indicator):
        assert mtf_indicator.name == "supertrend_mtf"
