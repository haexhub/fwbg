"""Tests for EMA Indicator plugin."""
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
    mod = import_plugin_module("fwbg-core", "indicators", "ema")
    return mod.EMAIndicator()


class TestEMADistance:
    def test_positive_distance_above_ema(self, indicator):
        n = 200
        close = 100 * np.cumprod(1 + np.full(n, 0.005))
        df = create_ohlc(close)
        result = indicator.compute(df)
        ema_dist = result["ema_dist_21"].dropna()
        assert ema_dist.iloc[-50:].mean() > 0

    def test_negative_distance_below_ema(self, indicator):
        n = 200
        close = 100 * np.cumprod(1 - np.full(n, 0.005))
        df = create_ohlc(close)
        result = indicator.compute(df)
        ema_dist = result["ema_dist_21"].dropna()
        assert ema_dist.iloc[-50:].mean() < 0


class TestEMASources:
    def test_high_low_sources_produce_channel(self, indicator):
        n = 200
        close = 100 * np.cumprod(1 + np.full(n, 0.002))
        df = create_ohlc(close)
        result = indicator.compute(df, lines=[
            {"period": 20, "source": "H"},
            {"period": 20, "source": "L"},
        ])
        assert "ema_dist_20_h" in result.columns
        assert "ema_dist_20_l" in result.columns
        assert "_ema_20_h" in result.columns
        assert "_ema_20_l" in result.columns

    def test_high_ema_above_low_ema(self, indicator):
        n = 200
        close = 100 + np.random.default_rng(42).normal(0, 0.5, n).cumsum()
        df = create_ohlc(close)
        result = indicator.compute(df, lines=[
            {"period": 20, "source": "H"},
            {"period": 20, "source": "L"},
        ])
        h_ema = result["_ema_20_h"].dropna()
        l_ema = result["_ema_20_l"].dropna()
        assert (h_ema > l_ema).all(), "EMA(H) should always be above EMA(L)"

    def test_open_source(self, indicator):
        n = 200
        close = np.linspace(100, 150, n)
        df = create_ohlc(close)
        result = indicator.compute(df, lines=[{"period": 20, "source": "O"}])
        assert "ema_dist_20_o" in result.columns

    def test_invalid_source_raises(self, indicator):
        with pytest.raises(ValueError, match="Invalid EMA source"):
            indicator.compute(
                create_ohlc(np.ones(50)),
                lines=[{"period": 20, "source": "X"}],
            )


class TestEMACrossings:
    def test_cross_source_crossings(self, indicator):
        n = 200
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = indicator.compute(df, lines=[
            {"period": 5, "source": "H"},
            {"period": 5, "source": "L"},
            {"period": 200, "source": "C"},
        ])
        assert "ema_5_h_above_5_l" in result.columns
        assert "ema_5_h_above_200" in result.columns
        assert "ema_5_l_above_200" in result.columns

    def test_crossings_disabled(self, indicator):
        n = 200
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = indicator.compute(df, lines=[
            {"period": 5, "source": "C"},
            {"period": 200, "source": "C"},
        ], crossings=False)
        crossing_cols = [c for c in result.columns if "_above_" in c]
        assert len(crossing_cols) == 0

    def test_crossing_values_binary(self, indicator):
        n = 200
        close = 100 + np.random.default_rng(42).normal(0, 1, n).cumsum()
        df = create_ohlc(close)
        result = indicator.compute(df)
        crossing_cols = [c for c in result.columns if "_above_" in c]
        for col in crossing_cols:
            valid = result[col].dropna()
            assert valid.isin([0.0, 1.0]).all(), f"{col} should be binary"


class TestFeatureColumns:
    def test_get_feature_columns_matches_compute(self, indicator):
        n = 200
        close = 100 + np.random.default_rng(42).normal(0, 0.5, n).cumsum()
        df = create_ohlc(close)
        result = indicator.compute(df)
        columns = indicator.get_feature_columns()
        for c in columns:
            assert c in result.columns, f"Feature {c} not in computed result"

    def test_custom_params_columns(self, indicator):
        params = {"lines": [{"period": 10, "source": "H"}, {"period": 50, "source": "C"}]}
        columns = indicator.get_feature_columns(params)
        assert "ema_dist_10_h" in columns
        assert "ema_dist_50" in columns
        assert "ema_10_h_above_50" in columns

    def test_all_features_have_prefix(self, indicator):
        columns = indicator.get_feature_columns()
        for col in columns:
            assert col.startswith("ema_"), f"Feature {col} missing ema_ prefix"


class TestPluginAttributes:
    def test_name(self, indicator):
        assert indicator.name == "ema"

    def test_version(self, indicator):
        assert isinstance(indicator.version, str)

    def test_validate(self, indicator):
        assert indicator.validate() is True
