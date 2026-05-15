"""Tests for ADX Indicator plugin."""
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
    mod = import_plugin_module("fwbg-core", "indicators", "adx")
    return mod.ADXIndicator()


class TestADX:
    def test_adx_high_in_strong_trend(self, indicator):
        n = 200
        close = 100 * np.cumprod(1 + np.full(n, 0.01))
        df = create_ohlc(close)
        result = indicator.compute(df)
        adx = result["adx_14"].dropna()
        assert adx.mean() > 25, f"Expected ADX > 25 in trend, got {adx.mean():.1f}"

    def test_adx_lower_in_sideways(self, indicator):
        n = 200
        trending = 100 * np.cumprod(1 + np.full(n, 0.01))
        np.random.seed(42)
        sideways = 100 + np.random.randn(n) * 0.5

        result_trend = indicator.compute(create_ohlc(trending))
        result_sideways = indicator.compute(create_ohlc(sideways))

        adx_trend = result_trend["adx_14"].dropna().mean()
        adx_sideways = result_sideways["adx_14"].dropna().mean()
        assert adx_trend > adx_sideways

    def test_feature_columns(self, indicator):
        columns = indicator.get_feature_columns()
        assert "adx_7" in columns
        assert "adx_14" in columns
        assert "adx_21" in columns

    def test_name(self, indicator):
        assert indicator.name == "adx"
