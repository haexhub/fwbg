"""Tests for previous_day_levels indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_pdl = import_plugin_module("fwbg-core", "indicators", "previous_day_levels")
if _pdl is None:
    pytest.skip("previous_day_levels plugin not available", allow_module_level=True)


def _make_ohlc_15min(n=2000, seed=42):
    """Create OHLCV DataFrame with 15-minute frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.001, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.003, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.003, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_ohlc_hourly(n=2000, seed=42):
    """Create OHLCV DataFrame with hourly frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.001, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_ohlc_daily(n=500, seed=42):
    """Create OHLCV DataFrame with daily frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.005, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _get_indicator():
    return _pdl.PreviousDayLevelsIndicator()


class TestPDLFeatures:
    """Tests for previous day level features."""

    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        late = result.iloc[200:]
        for col in ind.get_feature_columns():
            non_null = late[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN after warmup"

    def test_position_reasonable_range(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        pos = result["pdl_position"].dropna()
        within = ((pos >= -1) & (pos <= 2)).mean()
        assert within > 0.8, "Most position values should be in [-1, 2]"

    def test_binary_features(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        for col in ["pdl_above_high", "pdl_below_low",
                     "pdl_high_break", "pdl_low_break",
                     "pdl_post_bull", "pdl_post_bear",
                     "pdl_retest_bull", "pdl_retest_bear",
                     "pdl_day_range_expanding"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert set(vals.unique()).issubset({0.0, 1.0}), f"{col} not binary"

    def test_distances_atr_normalized(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        for col in ["pdl_high_dist", "pdl_low_dist"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.abs().median() < 50, f"{col} too large"

    def test_range_vs_atr_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        vals = result["pdl_range_vs_atr"].dropna()
        assert (vals >= 0).all(), "Range vs ATR should be non-negative"


class TestPDLShiftAndInf:
    """Lookahead prevention and inf checks."""

    def test_shift_applied(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min())
        for col in ind.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted"

    def test_no_inf_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        for col in ind.get_feature_columns():
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} has inf values"

    def test_no_undeclared_features(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        original_cols = set(df.columns)
        result = ind.compute(df)
        new_cols = set(result.columns) - original_cols
        declared = set(ind.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared: {undeclared}"

    def test_feature_count(self):
        ind = _get_indicator()
        assert len(ind.get_feature_columns()) == 15


class TestPDLDailySkip:
    """Daily data should not produce PDL features."""

    def test_daily_returns_unchanged(self):
        ind = _get_indicator()
        df = _make_ohlc_daily()
        result = ind.compute(df)
        pdl_cols = [c for c in result.columns if c.startswith("pdl_")]
        assert len(pdl_cols) == 0

    def test_hourly_data_works(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_hourly(n=500))
        assert "pdl_position" in result.columns


def _make_deterministic_retest_data():
    """Build a controlled 3-day OHLC dataset where we know exactly when
    breakout and retest should fire.

    Day 0 (2024-01-01): range 90-110  → PDH=110, PDL=90, midpoint=100
    Day 1 (2024-01-02): price starts at 100, breaks above 110 at 09:00,
        then retraces to 100 at 12:00 → retest_bull should fire at 12:00
    Day 2 (2024-01-03): price starts at 100, breaks below 90 at 09:00,
        then retraces to 100 at 12:00 → retest_bear should fire at 12:00

    Hourly bars, so each day has 24 bars.
    """
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    close = np.full(72, 100.0)
    high = np.full(72, 100.5)
    low = np.full(72, 99.5)
    opn = np.full(72, 100.0)

    # Day 0: range 90-110 (sets PDH=110, PDL=90 for day 1)
    for i in range(24):
        close[i] = 100.0
        high[i] = 110.0
        low[i] = 90.0
        opn[i] = 100.0

    # Day 1: breakout above PDH=110, then retrace to midpoint=100
    for i in range(24, 48):
        h = idx[i].hour
        if h < 9:
            close[i] = 105.0  # below PDH, no breakout yet
            high[i] = 106.0
            low[i] = 104.0
        elif h == 9:
            close[i] = 112.0  # breakout above PDH=110
            high[i] = 113.0
            low[i] = 108.0
        elif h < 12:
            close[i] = 107.0  # post-breakout, still above midpoint
            high[i] = 108.0
            low[i] = 106.0
        elif h == 12:
            close[i] = 100.0  # retrace to midpoint=100
            high[i] = 101.0
            low[i] = 99.0
        else:
            close[i] = 102.0  # stays near midpoint but retest already fired
            high[i] = 103.0
            low[i] = 101.0
        opn[i] = close[i]

    # Day 2: breakout below PDL (PDL for day 2 is min of day 1)
    # Day 1 range: low=99, high=113 → PDH=113, PDL=99, midpoint=106
    # For simplicity, set day 1 H/L to make PDH=113, PDL=99
    # Already correct from above: day 1 has low[12:00]=99, high[9:00]=113
    for i in range(48, 72):
        h = idx[i].hour
        if h < 9:
            close[i] = 105.0  # between PDL=99 and PDH=113
            high[i] = 106.0
            low[i] = 104.0
        elif h == 9:
            close[i] = 97.0  # breakout below PDL=99
            high[i] = 100.0
            low[i] = 96.0
        elif h < 12:
            close[i] = 98.0  # post-breakout, still below midpoint=106
            high[i] = 99.0
            low[i] = 97.0
        elif h == 12:
            close[i] = 106.0  # retrace to midpoint=106
            high[i] = 107.0
            low[i] = 105.0
        else:
            close[i] = 106.0  # near midpoint but retest already fired
            high[i] = 107.0
            low[i] = 105.0
        opn[i] = close[i]

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


class TestPDLRetestSignals:
    """Deterministic tests for PDH/PDL retest entry signals."""

    def test_retest_bull_fires_on_correct_bar(self):
        """After PDH breakout on day 1, retest_bull fires exactly once
        when price returns to the midpoint."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        # Use large retest_atr_width to ensure midpoint is within band
        result = ind.compute(df, retest_atr_width=1.0)

        # retest_bull is shifted by 1 bar (shift_features), so the signal
        # computed at 12:00 appears at 13:00
        day1 = result.loc["2024-01-02"]
        retest_vals = day1["pdl_retest_bull"].dropna()
        assert retest_vals.sum() == 1.0, (
            f"Expected exactly 1 retest_bull on day 1, got {retest_vals.sum()}"
        )
        # The retest fires at bar 12:00, shifted to 13:00
        fired_at = retest_vals[retest_vals == 1.0].index
        assert len(fired_at) == 1
        assert fired_at[0].hour == 13, (
            f"Expected retest_bull at 13:00 (shifted from 12:00), "
            f"got {fired_at[0].hour}:00"
        )

    def test_retest_bear_fires_on_correct_bar(self):
        """After PDL breakout on day 2, retest_bear fires exactly once
        when price returns to the midpoint."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retest_atr_width=1.0)

        day2 = result.loc["2024-01-03"]
        retest_vals = day2["pdl_retest_bear"].dropna()
        assert retest_vals.sum() == 1.0, (
            f"Expected exactly 1 retest_bear on day 2, got {retest_vals.sum()}"
        )
        fired_at = retest_vals[retest_vals == 1.0].index
        assert len(fired_at) == 1
        assert fired_at[0].hour == 13, (
            f"Expected retest_bear at 13:00 (shifted from 12:00), "
            f"got {fired_at[0].hour}:00"
        )

    def test_retest_does_not_fire_twice(self):
        """Retest signals fire at most once per day per direction."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retest_atr_width=1.0)

        for day in ["2024-01-02", "2024-01-03"]:
            day_data = result.loc[day]
            for col in ["pdl_retest_bull", "pdl_retest_bear"]:
                vals = day_data[col].dropna()
                assert vals.sum() <= 1.0, (
                    f"{col} fired {vals.sum()} times on {day}"
                )

    def test_retest_does_not_fire_without_breakout(self):
        """If price never breaks PDH/PDL, no retest signals fire."""
        idx = pd.date_range("2024-01-01", periods=96, freq="h")
        # Day 0: range 90-110
        # Day 1-3: price stays at 100 (between 90 and 110)
        close = np.full(96, 100.0)
        high = np.full(96, 100.5)
        low = np.full(96, 99.5)
        # Day 0: set the range
        for i in range(24):
            high[i] = 110.0
            low[i] = 90.0

        df = pd.DataFrame({
            "O": close.copy(), "H": high, "L": low, "C": close,
        }, index=idx)

        ind = _get_indicator()
        result = ind.compute(df, retest_atr_width=1.0)

        for day in ["2024-01-02", "2024-01-03", "2024-01-04"]:
            try:
                day_data = result.loc[day]
            except KeyError:
                continue
            for col in ["pdl_retest_bull", "pdl_retest_bear"]:
                vals = day_data[col].dropna()
                assert vals.sum() == 0, (
                    f"{col} fired without breakout on {day}"
                )

    def test_post_bull_state_after_breakout(self):
        """pdl_post_bull is 1 for all bars after the PDH breakout."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retest_atr_width=1.0)

        day1 = result.loc["2024-01-02"]
        post = day1["pdl_post_bull"].dropna()
        # Post-bull should be 0 before breakout (shifted by 1 bar),
        # and 1 from breakout bar+1 onward.
        # Breakout at 09:00, shifted → first 1 at 10:00
        at_08 = post.loc[post.index.hour == 8]
        if len(at_08) > 0:
            assert at_08.iloc[0] == 0.0, "Should not be post_bull before breakout"
        at_10 = post.loc[post.index.hour == 10]
        if len(at_10) > 0:
            assert at_10.iloc[0] == 1.0, "Should be post_bull at 10:00 (after breakout at 09:00)"

    def test_retest_disabled(self):
        """When enable_retest=False, retest columns are not present."""
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, enable_retest=False)
        assert "pdl_retest_bull" not in result.columns
        assert "pdl_retest_bear" not in result.columns
        # Other features should still be present
        assert "pdl_high_break" in result.columns


class TestPDLParameters:
    """Test parameter methods."""

    def test_get_default_params(self):
        params = _pdl.PreviousDayLevelsIndicator.get_default_params()
        assert params["atr_period"] == 14
        assert params["ma_period"] == 20
        assert params["enable_retest"] is True
        assert params["retest_atr_width"] == 0.3

    def test_get_param_schema(self):
        schema = _pdl.PreviousDayLevelsIndicator.get_param_schema()
        assert "atr_period" in schema
        assert "ma_period" in schema
        assert "enable_retest" in schema
        assert "retest_atr_width" in schema
        assert schema["atr_period"]["type"] == "int"

    def test_custom_atr_period(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(), atr_period=7)
        assert "pdl_high_dist" in result.columns


class TestPDLDiscovery:
    """Plugin discovery tests."""

    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("previous_day_levels")
        assert cls is not None
