"""Tests for CCI Indicator plugin."""
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
    mod = import_plugin_module("fwbg-core", "indicators", "cci")
    return mod.CCIIndicator()


class TestCCI:
    def test_cci_high_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        cci_cols = [c for c in result.columns if c.startswith("cci_")]
        for col in cci_cols:
            valid = result[col].dropna()
            assert valid.iloc[-50:].mean() > 50

    def test_cci_low_in_downtrend(self, indicator):
        n = 300
        close = np.linspace(200, 100, n)
        result = indicator.compute(create_ohlc(close))
        cci_cols = [c for c in result.columns if c.startswith("cci_")]
        for col in cci_cols:
            valid = result[col].dropna()
            assert valid.iloc[-50:].mean() < -50

    def test_name(self, indicator):
        assert indicator.name == "cci"
