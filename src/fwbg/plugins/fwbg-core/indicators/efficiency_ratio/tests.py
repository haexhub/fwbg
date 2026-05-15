"""Tests for Efficiency Ratio Indicator plugin."""
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
    mod = import_plugin_module("fwbg-core", "indicators", "efficiency_ratio")
    return mod.EfficiencyRatioIndicator()


class TestEfficiencyRatio:
    def test_high_er_in_linear_trend(self, indicator):
        close = np.linspace(100, 150, 200)
        result = indicator.compute(create_ohlc(close))
        er = result["er_20"].dropna()
        assert er.mean() > 0.8

    def test_low_er_in_noisy_sideways(self, indicator):
        np.random.seed(42)
        close = 100 + np.random.randn(200) * 3
        result = indicator.compute(create_ohlc(close))
        er = result["er_20"].dropna()
        assert er.mean() < 0.3

    def test_change_features_exist(self, indicator):
        cols = indicator.get_feature_columns()
        assert "er_10_chg" in cols
        assert "er_20_chg" in cols

    def test_name(self, indicator):
        assert indicator.name == "efficiency_ratio"
