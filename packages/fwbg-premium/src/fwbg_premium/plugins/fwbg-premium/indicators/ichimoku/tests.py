"""
Tests for the Ichimoku Cloud indicator plugin.

Ichimoku is a complete trading system that shows:
- Tenkan-sen (Conversion Line): (9-period High + 9-period Low) / 2
- Kijun-sen (Base Line): (26-period High + 26-period Low) / 2
- Senkou Span A (Cloud A): (Tenkan + Kijun) / 2, shifted 26 bars ahead in source
- Senkou Span B (Cloud B): (52-period High + 52-period Low) / 2
- Chikou Span: current close vs close 26 bars ago

Derived features:
- ichi_cloud_thick: |SenkouA - SenkouB| / close  (always >= 0)
- ichi_above_cloud / ichi_below_cloud / ichi_in_cloud  (mutually exclusive triplet)
- ichi_tk_cross: (Tenkan - Kijun) / close
- ichi_tk_bullish_cross / ichi_tk_bearish_cross: edge-triggered signals
- ichi_strong_bullish: above_cloud AND tk_cross>0 AND bullish_cloud
- ichi_strong_bearish: below_cloud AND tk_cross<0 AND NOT bullish_cloud

NOTE: The plugin calls shift_features() which shifts all outputs by 1 bar to
prevent lookahead bias.
"""
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_ichi = import_plugin_module("fwbg-premium", "indicators", "ichimoku")
if _ichi is None:
    pytest.skip("ichimoku plugin not available", allow_module_level=True)


def _find_indicator_class(module):
    import inspect
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and not inspect.isabstract(obj)
            and hasattr(obj, "compute")
            and hasattr(obj, "get_feature_columns")
        ):
            return obj
    raise RuntimeError(f"Could not find indicator class in {module}")


def _make_ohlc(close, freq="h"):
    n = len(close)
    return pd.DataFrame(
        {
            "O": close * 0.998,
            "H": close * 1.005,
            "L": close * 0.995,
            "C": close,
        },
        index=pd.date_range("2022-01-03", periods=n, freq=freq),
    )


def _get_ind():
    return _find_indicator_class(_ichi)()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestPluginAttributes:
    def test_class_name(self):
        cls = _find_indicator_class(_ichi)
        assert cls.__name__ == "IchimokuIndicators"

    def test_feature_columns_declared(self):
        cols = _get_ind().get_feature_columns()
        assert len(cols) > 0

    def test_tenkan_column_declared(self):
        cols = _get_ind().get_feature_columns()
        tenkan_cols = [c for c in cols if "tenkan" in c]
        assert len(tenkan_cols) > 0, "Tenkan column should be declared"

    def test_kijun_column_declared(self):
        cols = _get_ind().get_feature_columns()
        kijun_cols = [c for c in cols if "kijun" in c]
        assert len(kijun_cols) > 0, "Kijun column should be declared"

    def test_cloud_columns_declared(self):
        cols = _get_ind().get_feature_columns()
        expected = {"ichi_above_cloud", "ichi_below_cloud", "ichi_in_cloud"}
        missing = expected - set(cols)
        assert not missing, f"Missing cloud columns: {missing}"

    def test_composite_signals_declared(self):
        cols = _get_ind().get_feature_columns()
        assert "ichi_strong_bullish" in cols
        assert "ichi_strong_bearish" in cols

    def test_compute_returns_dataframe(self):
        close = np.linspace(100, 200, 300)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_preserves_original_columns(self):
        close = np.linspace(100, 200, 300)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        for col in df.columns:
            assert col in result.columns, f"Original column {col!r} missing"

    def test_all_declared_features_present(self):
        close = np.linspace(100, 200, 500)
        df = _make_ohlc(close)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Declared feature {col!r} not in result"

    def test_no_inf_values(self):
        close = np.linspace(100, 200, 500)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                bad = result[col].isin([float("inf"), float("-inf")])
                assert not bad.any(), f"{col} contains inf"

    def test_first_row_nan(self):
        """All feature columns should be NaN in row 0 (shift_features adds a leading NaN)."""
        close = np.linspace(100, 200, 500)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN"

    def test_only_declared_features_added(self):
        close = np.linspace(100, 200, 500)
        df = _make_ohlc(close)
        ind = _get_ind()
        result = ind.compute(df)
        original = set(df.columns)
        new_cols = set(result.columns) - original
        declared = set(ind.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared columns added to result: {undeclared}"


# ---------------------------------------------------------------------------
# Cloud geometry invariants
# ---------------------------------------------------------------------------


class TestCloudGeometry:
    """Mathematical invariants that must hold regardless of price path."""

    def test_cloud_thickness_non_negative(self):
        """Cloud thickness = |SenkouA - SenkouB| / price -> always >= 0."""
        np.random.seed(7)
        close = 100 + np.cumsum(np.random.randn(500) * 0.5)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_cloud_thick" in result.columns:
            valid = result["ichi_cloud_thick"].dropna()
            assert (valid >= 0).all(), (
                f"Cloud thickness must be >= 0, got min={valid.min():.6f}"
            )

    def test_mutually_exclusive_above_below_cloud(self):
        """Price cannot be simultaneously above AND below the cloud."""
        np.random.seed(13)
        close = 100 + np.cumsum(np.random.randn(400) * 0.5)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_above_cloud" in result.columns and "ichi_below_cloud" in result.columns:
            both = (result["ichi_above_cloud"] == 1) & (result["ichi_below_cloud"] == 1)
            assert not both.any(), "Cannot be simultaneously above and below cloud"

    def test_above_below_in_cloud_exhaustive(self):
        """Every row should be exactly one of: above, below, or in cloud."""
        np.random.seed(21)
        close = 100 + np.cumsum(np.random.randn(400) * 0.5)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        cols = ["ichi_above_cloud", "ichi_below_cloud", "ichi_in_cloud"]
        if all(c in result.columns for c in cols):
            valid = result[cols].dropna()
            row_sums = valid.sum(axis=1)
            # Each row sums to exactly 1 (mutually exclusive, exhaustive)
            assert (row_sums == 1).all(), (
                f"above+below+in_cloud should sum to 1 per row, got: {row_sums.value_counts()}"
            )

    def test_tk_cross_binary_signals_not_simultaneous(self):
        """Bullish and bearish TK crosses cannot fire on the same bar."""
        np.random.seed(5)
        close = 100 + np.cumsum(np.random.randn(500) * 0.5)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_tk_bullish_cross" in result.columns and "ichi_tk_bearish_cross" in result.columns:
            both = (result["ichi_tk_bullish_cross"] == 1) & (result["ichi_tk_bearish_cross"] == 1)
            assert not both.any(), "Bullish and bearish TK cross cannot fire simultaneously"


# ---------------------------------------------------------------------------
# Tenkan / Kijun relationship in trends
# ---------------------------------------------------------------------------


class TestTKRelationship:
    """Verify Tenkan vs Kijun ordering makes economic sense in clear trends."""

    def test_tenkan_above_kijun_in_sustained_uptrend(self):
        """
        In a sustained uptrend, Tenkan (9-bar midpoint) > Kijun (26-bar midpoint)
        because recent highs/lows are higher than the longer-term average.
        """
        n = 500
        close = np.linspace(100, 300, n)  # strong linear uptrend
        df = _make_ohlc(close)
        result = _get_ind().compute(df)

        if "ichi_tenkan" in result.columns and "ichi_kijun" in result.columns:
            valid = result[["ichi_tenkan", "ichi_kijun"]].dropna()
            if len(valid) >= 100:
                last_100 = valid.iloc[-100:]
                pct_tenkan_above = (last_100["ichi_tenkan"] >= last_100["ichi_kijun"]).mean()
                assert pct_tenkan_above > 0.6, (
                    f"In uptrend, Tenkan should be >= Kijun most of the time "
                    f"(got {pct_tenkan_above:.1%})"
                )

    def test_tenkan_below_kijun_in_sustained_downtrend(self):
        """In a sustained downtrend, Tenkan < Kijun."""
        n = 500
        close = np.linspace(300, 100, n)  # strong linear downtrend
        df = _make_ohlc(close)
        result = _get_ind().compute(df)

        if "ichi_tenkan" in result.columns and "ichi_kijun" in result.columns:
            valid = result[["ichi_tenkan", "ichi_kijun"]].dropna()
            if len(valid) >= 100:
                last_100 = valid.iloc[-100:]
                pct_tenkan_below = (last_100["ichi_tenkan"] <= last_100["ichi_kijun"]).mean()
                assert pct_tenkan_below > 0.6, (
                    f"In downtrend, Tenkan should be <= Kijun most of the time "
                    f"(got {pct_tenkan_below:.1%})"
                )

    def test_tk_cross_fires_after_trend_change(self):
        """After flat -> uptrend transition, a bullish TK cross should fire."""
        n = 400
        close = np.concatenate([np.full(200, 100.0), np.linspace(100, 200, 200)])
        df = _make_ohlc(close)
        result = _get_ind().compute(df)
        if "ichi_tk_bullish_cross" in result.columns:
            crosses = result["ichi_tk_bullish_cross"].dropna()
            assert crosses.sum() >= 1, (
                "At least one bullish TK cross should occur after flat->uptrend transition"
            )


# ---------------------------------------------------------------------------
# Cloud position in trends
# ---------------------------------------------------------------------------


class TestCloudPositioning:
    """Verify price-cloud relationship in strong directional moves."""

    def test_price_above_cloud_in_long_uptrend(self):
        """
        In a long sustained uptrend, price should spend the majority of time
        above the cloud (Senkou spans lag behind, so price has risen above them).
        """
        n = 600
        close = np.linspace(100, 400, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)

        if "ichi_above_cloud" in result.columns:
            valid = result["ichi_above_cloud"].dropna()
            if len(valid) >= 200:
                above_rate = valid.iloc[-200:].mean()
                assert above_rate > 0.5, (
                    f"In strong uptrend, price should be above cloud >50% of the time "
                    f"(got {above_rate:.1%})"
                )

    def test_price_below_cloud_in_long_downtrend(self):
        """
        In a long sustained downtrend, price should spend the majority of time
        below the cloud.
        """
        n = 600
        close = np.linspace(400, 100, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)

        if "ichi_below_cloud" in result.columns:
            valid = result["ichi_below_cloud"].dropna()
            if len(valid) >= 200:
                below_rate = valid.iloc[-200:].mean()
                assert below_rate > 0.5, (
                    f"In strong downtrend, price should be below cloud >50% of the time "
                    f"(got {below_rate:.1%})"
                )


# ---------------------------------------------------------------------------
# Composite signal consistency
# ---------------------------------------------------------------------------


class TestCompositeSignals:
    """Verify that composite signals are consistent with their component conditions."""

    def test_strong_bullish_implies_above_cloud(self):
        """
        ichi_strong_bullish = above_cloud & tk_cross>0 & bullish_cloud
        So whenever strong_bullish fires, above_cloud must also be 1.
        """
        n = 600
        close = np.linspace(100, 400, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)

        if "ichi_strong_bullish" in result.columns and "ichi_above_cloud" in result.columns:
            strong_bull_rows = result[result["ichi_strong_bullish"] == 1]
            if len(strong_bull_rows) > 0:
                assert (strong_bull_rows["ichi_above_cloud"] == 1).all(), (
                    "Strong bullish signal should only fire when price is above cloud"
                )

    def test_strong_bearish_implies_below_cloud(self):
        """
        ichi_strong_bearish = below_cloud & tk_cross<0 & NOT bullish_cloud
        So whenever strong_bearish fires, below_cloud must also be 1.
        """
        n = 600
        close = np.linspace(400, 100, n)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)

        if "ichi_strong_bearish" in result.columns and "ichi_below_cloud" in result.columns:
            strong_bear_rows = result[result["ichi_strong_bearish"] == 1]
            if len(strong_bear_rows) > 0:
                assert (strong_bear_rows["ichi_below_cloud"] == 1).all(), (
                    "Strong bearish signal should only fire when price is below cloud"
                )

    def test_strong_bullish_and_bearish_not_simultaneous(self):
        """Strong bullish and bearish cannot both be 1 at the same time."""
        np.random.seed(3)
        close = 100 + np.cumsum(np.random.randn(500) * 0.5)
        df = _make_ohlc(close)
        result = _get_ind().compute(df)

        if "ichi_strong_bullish" in result.columns and "ichi_strong_bearish" in result.columns:
            both = (result["ichi_strong_bullish"] == 1) & (result["ichi_strong_bearish"] == 1)
            assert not both.any(), "Cannot be simultaneously strong bullish and strong bearish"
