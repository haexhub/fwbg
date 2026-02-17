"""
Tests for new macro/vol feature enhancements:
- Yield Curve Shape (10y-2y, 30y-5y spreads + lookbacks)
- Realized Vol vs Implied Vol (RV/VIX ratio)
- COT Positioning × Vol Interaction + Price Divergence
"""
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# HELPERS
# =============================================================================

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
        # US Treasury yields (as basis points)
        "macro_us2y": 4.5 + rng.normal(0, 0.02, n).cumsum() * 0.01,
        "macro_us5y": 4.2 + rng.normal(0, 0.02, n).cumsum() * 0.01,
        "macro_tnx": 4.0 + rng.normal(0, 0.02, n).cumsum() * 0.01,  # 10Y
        "macro_us30y": 4.3 + rng.normal(0, 0.02, n).cumsum() * 0.01,
        # VIX for RV/IV tests
        "macro_vix": 20.0 + rng.normal(0, 0.5, n).cumsum() * 0.1,
        # COT positioning (weekly forward-filled to hourly)
        "macro_cot_eurusd": (50000 + rng.normal(0, 5000, n).cumsum()).round(),
    }, index=idx)
    return df


# =============================================================================
# YIELD CURVE SHAPE TESTS
# =============================================================================


class TestYieldCurveShape:
    """Tests for yield curve slope derived features."""

    def test_curve_10y_2y_computed(self):
        """10y-2y yield curve spread should be computed."""
        from fwbg.core.registry import get_data_loader
        cls = get_data_loader("macro_data")
        loader = cls()
        df = _make_ohlc_with_macro()

        from fwbg_sdk import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        assert "macro_yield_curve_10y_2y" in ctx.df.columns
        # Should be TNX - US2Y (negative when inverted)
        expected = df["macro_tnx"] - df["macro_us2y"]
        actual = ctx.df["macro_yield_curve_10y_2y"]
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_curve_30y_5y_computed(self):
        """30y-5y yield curve spread should be computed."""
        from fwbg.core.registry import get_data_loader
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
        from fwbg.core.registry import get_data_loader
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
        from fwbg.core.registry import get_data_loader
        cls = get_data_loader("macro_data")
        loader = cls()
        cols = loader.get_feature_columns()

        assert "macro_us2y" in cols
        assert "macro_us5y" in cols
        assert "macro_us30y" in cols
        assert "macro_yield_curve_10y_2y" in cols
        assert "macro_yield_curve_30y_5y" in cols


# =============================================================================
# REALIZED VOL vs IMPLIED VOL TESTS
# =============================================================================


class TestRealizedVsImpliedVol:
    """Tests for RV/VIX features in volatility plugin."""

    def _get_indicator(self):
        from fwbg.core.registry import INDICATOR_REGISTRY
        from fwbg.plugins import import_plugin_module
        if "volatility" not in INDICATOR_REGISTRY:
            import_plugin_module("fwbg-core", "indicators", "volatility")
        return INDICATOR_REGISTRY["volatility"]()

    def test_rv_computed(self):
        """Realized vol should be computed from close-to-close returns."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "vol_rv_20" in result.columns
        rv = result["vol_rv_20"].dropna()
        assert len(rv) > 0
        # RV should be positive
        assert rv.mean() > 0

    def test_rv_iv_ratio_with_vix(self):
        """RV/VIX ratio should be computed when macro_vix present."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "vol_rv_iv_ratio" in result.columns
        assert "vol_rv_iv_spread" in result.columns
        ratio = result["vol_rv_iv_ratio"].dropna()
        assert len(ratio) > 0
        # Ratio should be positive (both RV and VIX are positive)
        assert ratio.mean() > 0

    def test_rv_iv_not_computed_without_vix(self):
        """RV/VIX should NOT be computed when macro_vix absent."""
        ind = self._get_indicator()
        n = 2000
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        result = ind.compute(df)

        assert "vol_rv_20" in result.columns  # RV always computed
        assert "vol_rv_iv_ratio" not in result.columns
        assert "vol_rv_iv_spread" not in result.columns

    def test_rv_no_lookahead(self):
        """RV features should be shifted by 1 bar."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert pd.isna(result["vol_rv_20"].iloc[0])
        assert pd.isna(result["vol_rv_iv_ratio"].iloc[0])


# =============================================================================
# COT × VOLATILITY INTERACTION TESTS
# =============================================================================


class TestCOTVolInteraction:
    """Tests for COT × Vol interaction in cross_features plugin."""

    def _get_indicator(self):
        from fwbg.core.registry import INDICATOR_REGISTRY
        from fwbg.plugins import import_plugin_module
        if "cross_features" not in INDICATOR_REGISTRY:
            import_plugin_module("fwbg-premium", "indicators", "cross_features")
        return INDICATOR_REGISTRY["cross_features"]()

    def test_interaction_columns_present(self):
        """COT × Vol interaction columns should be created."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "cross_cot_eurusd_vol_interaction" in result.columns
        assert "cross_cot_eurusd_price_divergence" in result.columns

    def test_no_cot_columns_no_crash(self):
        """Should work gracefully without COT columns."""
        ind = self._get_indicator()
        n = 500
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        result = ind.compute(df)

        # Standard cross features should still work
        assert "cross_vol_trend" in result.columns
        # No COT columns = no interaction features
        cot_cols = [c for c in result.columns if "cot_" in c]
        assert len(cot_cols) == 0

    def test_interaction_sign_logic(self):
        """Extreme COT + low vol should produce high absolute interaction."""
        ind = self._get_indicator()
        n = 5000  # Need enough data for 52-week z-score window
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        idx = pd.date_range("2024-01-01", periods=n, freq="h")

        # Create extreme COT positioning (very high)
        cot = np.full(n, 100000.0)
        cot[:4000] = rng.normal(50000, 5000, 4000)  # Normal range first
        cot[4000:] = 100000  # Then extreme high

        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
            "macro_cot_eurusd": cot,
        }, index=idx)

        result = ind.compute(df)
        interaction = result["cross_cot_eurusd_vol_interaction"].dropna()
        # With extreme COT and varying vol, interaction should have variance
        assert interaction.std() > 0

    def test_divergence_detects_mismatch(self):
        """Price divergence should detect when price and COT move opposite."""
        ind = self._get_indicator()
        n = 5000
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="h")

        # Price trending up
        close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))

        # COT trending down (divergence!)
        cot = 50000 - np.arange(n) * 10.0 + rng.normal(0, 100, n)

        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
            "macro_cot_eurusd": cot,
        }, index=idx)

        result = ind.compute(df)
        divergence = result["cross_cot_eurusd_price_divergence"].dropna()
        # With price up and COT down, divergence should be positive
        # (price_z > 0, cot_z < 0, so price_z - cot_z > 0)
        late_values = divergence.iloc[-500:]
        assert late_values.mean() > 0, f"Expected positive divergence, got {late_values.mean()}"

    def test_no_lookahead(self):
        """All interaction features should be shifted."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert pd.isna(result["cross_cot_eurusd_vol_interaction"].iloc[0])
        assert pd.isna(result["cross_cot_eurusd_price_divergence"].iloc[0])

    def test_feature_columns_list(self):
        """get_feature_columns should include COT interaction columns."""
        ind = self._get_indicator()
        cols = ind.get_feature_columns()

        assert "cross_cot_eurusd_vol_interaction" in cols
        assert "cross_cot_eurusd_price_divergence" in cols
        assert "cross_cot_usdjpy_vol_interaction" in cols
