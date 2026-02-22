import inspect
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_pa = import_plugin_module("fwbg-core", "indicators", "price_action")
if _pa is None:
    pytest.skip("price_action plugin not available", allow_module_level=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlc(close, freq="h"):
    n = len(close)
    return pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
    }, index=pd.date_range("2022-01-03", periods=n, freq=freq))


def _get_ind(**params):
    for name in dir(_pa):
        obj = getattr(_pa, name)
        if (
            isinstance(obj, type)
            and not inspect.isabstract(obj)
            and hasattr(obj, "compute")
            and hasattr(obj, "get_feature_columns")
        ):
            return obj(**params) if params else obj()
    raise RuntimeError("Could not find indicator class in price_action module")


# ── body ratio ───────────────────────────────────────────────────────────────

class TestBodyRatio:
    def test_marubozu_body_ratio_near_one(self):
        """Candle where H=C and L=O (full body, no wicks) → body_ratio ≈ 1."""
        n = 100
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close - 1.0,
            "C": close,
            "H": close,
            "L": close - 1.0,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        if "pa_body_ratio" in result.columns:
            valid = result["pa_body_ratio"].dropna()
            assert (valid > 0.9).all(), f"Marubozu should have body_ratio near 1, got min={valid.min():.3f}"

    def test_doji_body_ratio_near_zero(self):
        """Candle where O==C but H-L is wide → body_ratio ≈ 0."""
        n = 100
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close,
            "C": close,
            "H": close + 2.0,
            "L": close - 2.0,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        if "pa_body_ratio" in result.columns:
            valid = result["pa_body_ratio"].dropna()
            assert (valid < 0.1).all(), f"Doji should have body_ratio near 0, got max={valid.max():.3f}"

    def test_range_pos_above_half_when_close_in_upper_half(self):
        """When close is in the upper half of the bar range, pa_range_pos > 0.5."""
        n = 100
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close - 0.5,
            "C": close + 0.8,
            "H": close + 1.0,
            "L": close - 1.0,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        if "pa_range_pos" in result.columns:
            valid = result["pa_range_pos"].dropna()
            assert (valid > 0.5).all(), f"Close in upper half → range_pos > 0.5, got min={valid.min():.3f}"

    def test_range_pos_below_half_when_close_in_lower_half(self):
        """When close is in the lower half of the bar range, pa_range_pos < 0.5."""
        n = 100
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close + 0.5,
            "C": close - 0.8,
            "H": close + 1.0,
            "L": close - 1.0,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        if "pa_range_pos" in result.columns:
            valid = result["pa_range_pos"].dropna()
            assert (valid < 0.5).all(), f"Close in lower half → range_pos < 0.5, got max={valid.max():.3f}"


# ── streaks ───────────────────────────────────────────────────────────────────

class TestBullishStreak:
    def test_bullish_streak_increments_on_consecutive_bullish_candles(self):
        """10 consecutive bullish candles → bullish_streak increases monotonically."""
        n = 50
        open_ = np.concatenate([np.full(40, 100.0), np.linspace(100, 109, 10)])
        close = np.concatenate([np.full(40, 100.0), np.linspace(101, 110, 10)])
        df = pd.DataFrame({
            "O": open_,
            "C": close,
            "H": close + 0.3,
            "L": open_ - 0.3,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        if "pa_bullish_streak" in result.columns:
            streak = result["pa_bullish_streak"].dropna().iloc[-5:]
            assert streak.iloc[-1] > streak.iloc[0],                 f"Streak should grow during bullish run, got {streak.values}"

    def test_streak_reset_after_direction_change(self):
        """After bearish candles following bullish run, bearish_streak > 0."""
        n = 100
        open_ = np.concatenate([np.linspace(100, 109, 90), np.full(10, 110.0)])
        close = np.concatenate([np.linspace(101, 110, 90), np.full(10, 108.0)])
        df = pd.DataFrame({
            "O": open_,
            "C": close,
            "H": np.maximum(open_, close) + 0.5,
            "L": np.minimum(open_, close) - 0.5,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = _get_ind().compute(df)
        bull_ok = "pa_bullish_streak" not in result.columns or result["pa_bullish_streak"].dropna().iloc[-1] == 0
        bear_ok = "pa_bearish_streak" in result.columns and result["pa_bearish_streak"].dropna().iloc[-1] > 0
        assert bull_ok or bear_ok, "After bearish candles, streak should change"


# ── inside / outside bars ────────────────────────────────────────────────────

class TestInsideOutsideBar:
    def test_inside_bar_detected(self):
        """Bar fully within previous bar H-L → pa_inside_bar = 1.

        Due to shift_features(shift=1), the pattern at raw index -2 appears
        at result.iloc[-1].
        """
        n = 50
        df = _make_ohlc(np.full(n, 100.0))
        # Wide bar at raw index -3, narrow (inside) bar at raw index -2
        df.iloc[-3, df.columns.get_loc("H")] = 105.0
        df.iloc[-3, df.columns.get_loc("L")] = 95.0
        df.iloc[-2, df.columns.get_loc("H")] = 101.0
        df.iloc[-2, df.columns.get_loc("L")] = 99.0
        df.iloc[-2, df.columns.get_loc("O")] = 99.5
        df.iloc[-2, df.columns.get_loc("C")] = 100.5
        result = _get_ind().compute(df)
        if "pa_inside_bar" in result.columns:
            assert result["pa_inside_bar"].iloc[-1] == 1, "Inside bar not detected"

    def test_outside_bar_detected(self):
        """Bar that engulfs previous bar H-L → pa_outside_bar = 1.

        Due to shift_features(shift=1), the pattern at raw index -2 appears
        at result.iloc[-1].
        """
        n = 50
        df = _make_ohlc(np.full(n, 100.0))
        # Narrow bar at raw index -3, wide (outside) bar at raw index -2
        df.iloc[-3, df.columns.get_loc("H")] = 101.0
        df.iloc[-3, df.columns.get_loc("L")] = 99.0
        df.iloc[-2, df.columns.get_loc("H")] = 105.0
        df.iloc[-2, df.columns.get_loc("L")] = 95.0
        result = _get_ind().compute(df)
        if "pa_outside_bar" in result.columns:
            assert result["pa_outside_bar"].iloc[-1] == 1, "Outside bar not detected"


# ── plugin attributes ─────────────────────────────────────────────────────────

class TestPluginAttributes:
    def test_feature_columns_declared(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        assert len(cols) > 0, "No feature columns declared"

    def test_no_inf_values(self):
        close = np.linspace(100, 110, 300)
        df = _make_ohlc(close)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            if col in result.columns:
                bad = result[col].isin([float("inf"), float("-inf")])
                assert not bad.any(), f"{col} contains inf"

    def test_first_row_nan(self):
        """All features must be shifted (no lookahead bias): row 0 should be NaN."""
        close = np.linspace(100, 110, 200)
        df = _make_ohlc(close)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN (shift required)"

    def test_only_declared_features_added(self):
        """No extra columns beyond what get_feature_columns() declares."""
        close = np.linspace(100, 110, 200)
        df = _make_ohlc(close)
        ind = _get_ind()
        result = ind.compute(df)
        original_cols = set(df.columns)
        new_cols = set(result.columns) - original_cols
        declared = set(ind.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared columns added: {undeclared}"
