"""Tests for MACD Indicator plugin."""
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
    mod = import_plugin_module("fwbg-core", "indicators", "macd")
    return mod.MACDIndicator()


class TestMACD:
    def test_macd_line_positive_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["macd_line"].dropna()
        assert (valid.iloc[-50:] > 0).all()

    def test_macd_line_negative_in_downtrend(self, indicator):
        n = 300
        close = np.linspace(200, 100, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["macd_line"].dropna()
        assert (valid.iloc[-50:] < 0).all()

    def test_macd_hist_flip_at_reversal(self, indicator):
        close = np.concatenate([np.linspace(100, 80, 200), np.linspace(80, 120, 200)])
        result = indicator.compute(create_ohlc(close))
        flips = result["macd_hist_flip"].dropna()
        assert flips.sum() > 0

    def test_macd_above_zero_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["macd_above_zero"].dropna()
        assert (valid.iloc[-50:] == 1).all()

    def test_feature_columns(self, indicator):
        cols = indicator.get_feature_columns()
        assert "macd_line" in cols
        assert "macd_hist" in cols
        assert "macd_above_zero" in cols

    def test_name(self, indicator):
        assert indicator.name == "macd"
