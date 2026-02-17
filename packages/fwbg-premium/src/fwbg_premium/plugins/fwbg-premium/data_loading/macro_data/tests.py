"""Tests for macro_data DataLoader plugin - Yield Curve Shape features."""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.registry import get_data_loader


def _make_ohlc_with_macro(n=2000):
    """Create OHLC DataFrame with macro columns for testing."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
        "macro_us2y": 4.5 + rng.normal(0, 0.02, n).cumsum() * 0.01,
        "macro_us5y": 4.2 + rng.normal(0, 0.02, n).cumsum() * 0.01,
        "macro_tnx": 4.0 + rng.normal(0, 0.02, n).cumsum() * 0.01,  # 10Y
        "macro_us30y": 4.3 + rng.normal(0, 0.02, n).cumsum() * 0.01,
        "macro_vix": 20.0 + rng.normal(0, 0.5, n).cumsum() * 0.1,
        "macro_cot_eurusd": (50000 + rng.normal(0, 5000, n).cumsum()).round(),
    }, index=idx)
    return df


class TestYieldCurveShape:
    """Tests for yield curve slope derived features."""

    def test_curve_10y_2y_computed(self):
        """10y-2y yield curve spread should be computed."""
        cls = get_data_loader("macro_data")
        loader = cls()
        df = _make_ohlc_with_macro()

        from fwbg_sdk import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        assert "macro_yield_curve_10y_2y" in ctx.df.columns
        expected = df["macro_tnx"] - df["macro_us2y"]
        actual = ctx.df["macro_yield_curve_10y_2y"]
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_curve_30y_5y_computed(self):
        """30y-5y yield curve spread should be computed."""
        cls = get_data_loader("macro_data")
        loader = cls()
        df = _make_ohlc_with_macro()

        from fwbg_sdk import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        assert "macro_yield_curve_30y_5y" in ctx.df.columns
        expected = df["macro_us30y"] - df["macro_us5y"]
        actual = ctx.df["macro_yield_curve_30y_5y"]
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_curve_lookbacks_computed(self):
        """Yield curve spreads should have daily lookback changes."""
        cls = get_data_loader("macro_data")
        loader = cls()
        df = _make_ohlc_with_macro()

        from fwbg_sdk import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        for lb in [2, 5, 10, 20, 60]:
            assert f"macro_yield_curve_10y_2y_chg_{lb}d" in ctx.df.columns
            assert f"macro_yield_curve_30y_5y_chg_{lb}d" in ctx.df.columns

    def test_us_yield_levels_in_feature_columns(self):
        """US2Y, US5Y, US30Y should appear in get_feature_columns."""
        cls = get_data_loader("macro_data")
        loader = cls()
        cols = loader.get_feature_columns()

        assert "macro_us2y" in cols
        assert "macro_us5y" in cols
        assert "macro_us30y" in cols
        assert "macro_yield_curve_10y_2y" in cols
        assert "macro_yield_curve_30y_5y" in cols
