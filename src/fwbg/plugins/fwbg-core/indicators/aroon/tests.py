"""Tests for Aroon Indicator plugin."""
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
    mod = import_plugin_module("fwbg-core", "indicators", "aroon")
    return mod.AroonIndicator()


class TestAroon:
    def test_aroon_up_near_100_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["aroon_up"].dropna()
        assert valid.iloc[-50:].mean() > 80

    def test_aroon_down_near_100_in_downtrend(self, indicator):
        n = 300
        close = np.linspace(200, 100, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["aroon_down"].dropna()
        assert valid.iloc[-50:].mean() > 80

    def test_aroon_bounded_0_to_100(self, indicator):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        result = indicator.compute(create_ohlc(close))
        for col in ["aroon_up", "aroon_down"]:
            valid = result[col].dropna()
            assert (valid >= 0).all() and (valid <= 100).all()

    def test_name(self, indicator):
        assert indicator.name == "aroon"
