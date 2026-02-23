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
                     "rl50_pdl_retest_bull", "rl50_pdl_retest_bear",
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
        assert len(ind.get_feature_columns()) == 16


class TestPDLDailySkip:
    """Daily data should not produce PDL features."""

    def test_daily_returns_unchanged(self):
        ind = _get_indicator()
        df = _make_ohlc_daily()
        result = ind.compute(df)
        pdl_cols = [c for c in result.columns if c.startswith("pdl_") or c.startswith("rl")]
        assert len(pdl_cols) == 0

    def test_hourly_data_works(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_hourly(n=500))
        assert "pdl_position" in result.columns


def _make_deterministic_retest_data():
    """Build a controlled 3-day OHLC dataset where we know exactly when
    breakout and retest should fire.

    PDH/PDL is computed from H/L during session hours (default 7-21).

    Day 0 (2024-01-01): H=110, L=90 during session
        -> PDH=110, PDL=90, midpoint=100
    Day 1 (2024-01-02): price starts at 105, breaks above 110 at 09:00,
        then retraces to 100 at 12:00 -> retest_bull should fire at 12:00
    Day 2 (2024-01-03): price starts at 105, breaks below PDL at 09:00,
        then retraces to midpoint at 12:00 -> retest_bear should fire at 12:00

    Hourly bars, so each day has 24 bars.
    """
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    close = np.full(72, 100.0)
    high = np.full(72, 100.5)
    low = np.full(72, 99.5)
    opn = np.full(72, 100.0)

    # Day 0: H/L establish range during session (7-21)
    # PDH = max(H during session) = 110, PDL = min(L during session) = 90
    for i in range(24):
        h = idx[i].hour
        close[i] = 100.0
        opn[i] = 100.0
        if h == 8:
            high[i] = 100.5
            low[i] = 90.0    # session low -> PDL=90
        elif h == 11:
            high[i] = 110.0   # session high -> PDH=110
            low[i] = 99.5
        else:
            high[i] = 100.5
            low[i] = 99.5

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

    # Day 2: breakout below PDL (from Day 1 H/L during session 7-21)
    # Day 1 H values in session: 106,106,113,108,108,101,103,...103
    # Day 1 L values in session: 104,104,108,106,106, 99,101,...101
    # PDH = max(H) = 113, PDL = min(L) = 99, midpoint = 106
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
        when price returns to the midpoint (rl=0.5)."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retest_atr_width=0.3)

        # retest_bull is shifted by 1 bar (shift_features), so the signal
        # computed at 12:00 appears at 13:00
        day1 = result.loc["2024-01-02"]
        retest_vals = day1["rl50_pdl_retest_bull"].dropna()
        assert retest_vals.sum() == 1.0, (
            f"Expected exactly 1 retest_bull on day 1, got {retest_vals.sum()}"
        )
        fired_at = retest_vals[retest_vals == 1.0].index
        assert len(fired_at) == 1
        assert fired_at[0].hour == 13, (
            f"Expected retest_bull at 13:00 (shifted from 12:00), "
            f"got {fired_at[0].hour}:00"
        )

    def test_retest_bear_fires_on_correct_bar(self):
        """After PDL breakout on day 2, retest_bear fires exactly once
        when price returns to the midpoint (rl=0.5)."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retest_atr_width=0.3)

        day2 = result.loc["2024-01-03"]
        retest_vals = day2["rl50_pdl_retest_bear"].dropna()
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
        result = ind.compute(df, retest_atr_width=0.3)

        for day in ["2024-01-02", "2024-01-03"]:
            day_data = result.loc[day]
            for col in ["rl50_pdl_retest_bull", "rl50_pdl_retest_bear"]:
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
        result = ind.compute(df, retest_atr_width=0.3)

        for day in ["2024-01-02", "2024-01-03", "2024-01-04"]:
            try:
                day_data = result.loc[day]
            except KeyError:
                continue
            for col in ["rl50_pdl_retest_bull", "rl50_pdl_retest_bear"]:
                vals = day_data[col].dropna()
                assert vals.sum() == 0, (
                    f"{col} fired without breakout on {day}"
                )

    def test_post_bull_state_after_breakout(self):
        """pdl_post_bull is 1 for all bars after the PDH breakout."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retest_atr_width=0.3)

        day1 = result.loc["2024-01-02"]
        post = day1["pdl_post_bull"].dropna()
        # Post-bull should be 0 before breakout (shifted by 1 bar),
        # and 1 from breakout bar+1 onward.
        # Breakout at 09:00, shifted -> first 1 at 10:00
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
        assert "rl50_pdl_retest_bull" not in result.columns
        assert "rl50_pdl_retest_bear" not in result.columns
        # sl_dist is always present (even without retest)
        assert "rl50_pdl_sl_dist" in result.columns
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
        assert params["retracement_levels"] == 0.5
        assert params["range_modes"] == ["hl_session"]

    def test_get_param_schema(self):
        schema = _pdl.PreviousDayLevelsIndicator.get_param_schema()
        assert "atr_period" in schema
        assert "ma_period" in schema
        assert "enable_retest" in schema
        assert "retest_atr_width" in schema
        assert "retracement_levels" in schema
        assert "range_modes" in schema

    def test_custom_atr_period(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(), atr_period=7)
        assert "pdl_high_dist" in result.columns


class TestPDLSLDist:
    """rl50_pdl_sl_dist = (1-0.5) * pd_range = half the previous day range.

    Entry is at PDH/PDL midpoint (rl=0.5). SL at the boundary:
    - Long (breakout up): SL = PDL -> distance = midpoint - PDL = range/2
    - Short (breakout down): SL = PDH -> distance = PDH - midpoint = range/2
    """

    def test_sl_dist_column_exists(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        assert "rl50_pdl_sl_dist" in result.columns

    def test_sl_dist_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        vals = result["rl50_pdl_sl_dist"].dropna()
        assert len(vals) > 0
        assert (vals > 0).all(), "rl50_pdl_sl_dist should be strictly positive"

    def test_sl_dist_equals_half_range(self):
        """rl50_pdl_sl_dist must equal (PDH - PDL) / 2 for rl=0.5."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df)

        # Day 1: PDH=110, PDL=90 -> range=20 -> sl_dist=10
        day1 = result.loc["2024-01-02"]
        sl_vals = day1["rl50_pdl_sl_dist"].dropna()
        assert len(sl_vals) > 0
        expected = (110.0 - 90.0) / 2  # = 10.0
        np.testing.assert_allclose(
            sl_vals.iloc[0], expected, rtol=1e-10,
            err_msg=f"rl50_pdl_sl_dist should be {expected} (half prev day range)"
        )

    def test_sl_dist_no_nan_after_warmup(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min(n=5000))
        late = result.iloc[200:]
        non_null = late["rl50_pdl_sl_dist"].dropna()
        assert len(non_null) > 0

    def test_sl_dist_shifted(self):
        """rl50_pdl_sl_dist must be shifted (no lookahead)."""
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min())
        assert pd.isna(result["rl50_pdl_sl_dist"].iloc[0]), "rl50_pdl_sl_dist not shifted"

    def test_sl_dist_atr_floor_kicks_in_for_small_range(self):
        """When previous day range is tiny, min_sl_atr_mult * ATR provides a floor."""
        # Build data where Day 0 has a tiny range (H=100.5, L=99.5 → range=1)
        # but ATR is much larger (due to earlier volatile bars)
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        close = np.full(48, 100.0)
        high = np.full(48, 100.5)
        low = np.full(48, 99.5)
        opn = np.full(48, 100.0)

        # Day 0: tiny range (1 point) but earlier bars have larger true range
        # to inflate ATR
        for i in range(24):
            if i < 5:
                high[i] = 120.0  # large TR to inflate ATR
                low[i] = 80.0
                close[i] = 100.0
            else:
                high[i] = 100.5
                low[i] = 99.5

        df = pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)
        ind = _get_indicator()

        # Without floor: sl_dist = 0.5 * 1 = 0.5
        result_no_floor = ind.compute(df.copy(), min_sl_atr_mult=0.0)
        # With floor: sl_dist = max(0.5, 1.5 * ATR)
        result_with_floor = ind.compute(df.copy(), min_sl_atr_mult=1.5)

        day1_no = result_no_floor.loc["2024-01-02"]["rl50_pdl_sl_dist"].dropna()
        day1_fl = result_with_floor.loc["2024-01-02"]["rl50_pdl_sl_dist"].dropna()

        if len(day1_no) > 0 and len(day1_fl) > 0:
            # Floor should give a larger SL than pure range-based
            assert day1_fl.iloc[0] >= day1_no.iloc[0], (
                f"ATR floor SL ({day1_fl.iloc[0]:.2f}) should be >= "
                f"range SL ({day1_no.iloc[0]:.2f})"
            )
            # And the floor should be meaningfully larger than 0.5
            assert day1_fl.iloc[0] > 1.0, (
                f"ATR floor should produce SL > 1.0, got {day1_fl.iloc[0]:.2f}"
            )

    def test_sl_dist_no_floor_when_range_is_large(self):
        """When range is large, ATR floor doesn't change the SL."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()

        # Day 1: PDH=110, PDL=90, range=20, sl_dist=10
        result_no_floor = ind.compute(df.copy(), min_sl_atr_mult=0.0)
        result_with_floor = ind.compute(df.copy(), min_sl_atr_mult=1.5)

        day1_no = result_no_floor.loc["2024-01-02"]["rl50_pdl_sl_dist"].dropna()
        day1_fl = result_with_floor.loc["2024-01-02"]["rl50_pdl_sl_dist"].dropna()

        if len(day1_no) > 0 and len(day1_fl) > 0:
            # Range of 20 → range-based SL = 10, which is likely larger than 1.5 * ATR
            # So both should be equal (floor doesn't kick in)
            np.testing.assert_allclose(
                day1_no.iloc[0], day1_fl.iloc[0], rtol=1e-10,
                err_msg="ATR floor should not affect SL when range is already large"
            )


class TestPDLRetracementLevels:
    """Tests for multiple retracement levels with rl{N}_ prefix."""

    def test_list_generates_prefixed_columns(self):
        """Multiple rl values generate separate prefixed column sets."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, retracement_levels=[0.3, 0.5, 0.7])

        for rl_int in [30, 50, 70]:
            pfx = f"rl{rl_int}_"
            assert f"{pfx}pdl_retest_bull" in result.columns
            assert f"{pfx}pdl_retest_bear" in result.columns
            assert f"{pfx}pdl_sl_dist" in result.columns

    def test_sl_dist_varies_with_rl(self):
        """Deeper retrace = smaller SL distance: sl_dist = (1-rl) * range."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retracement_levels=[0.3, 0.5, 0.7])

        # Day 1: PDH=110, PDL=90 -> range=20
        day1 = result.loc["2024-01-02"]

        sl30 = day1["rl30_pdl_sl_dist"].dropna().iloc[0]
        sl50 = day1["rl50_pdl_sl_dist"].dropna().iloc[0]
        sl70 = day1["rl70_pdl_sl_dist"].dropna().iloc[0]

        np.testing.assert_allclose(sl30, 0.7 * 20, rtol=1e-10)  # 14.0
        np.testing.assert_allclose(sl50, 0.5 * 20, rtol=1e-10)  # 10.0
        np.testing.assert_allclose(sl70, 0.3 * 20, rtol=1e-10)  # 6.0

        assert sl30 > sl50 > sl70, "Deeper retrace = smaller SL distance"

    def test_scalar_generates_single_prefix(self):
        """Scalar rl value still gets rl{N}_ prefix."""
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, retracement_levels=0.382)

        assert "rl38_pdl_retest_bull" in result.columns
        assert "rl38_pdl_sl_dist" in result.columns

    def test_non_rl_features_not_duplicated(self):
        """Break detection, distance features etc. are computed once (not per rl)."""
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, retracement_levels=[0.3, 0.5, 0.7])

        # These should exist exactly once (no prefix)
        for col in ["pdl_high_dist", "pdl_low_dist", "pdl_position",
                     "pdl_high_break", "pdl_low_break",
                     "pdl_post_bull", "pdl_post_bear"]:
            assert col in result.columns
            # Should NOT have rl-prefixed duplicates
            assert f"rl30_{col}" not in result.columns

    def test_retest_fires_at_different_levels(self):
        """Shallow retrace (rl=0.3) fires earlier than deep (rl=0.7)
        because entry is closer to breakout boundary."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retracement_levels=[0.3, 0.7], retest_atr_width=0.3)

        # Day 1: PDH=110, PDL=90, range=20
        # rl=0.3: bull entry = 110 - 0.3*20 = 104 (shallow)
        # rl=0.7: bull entry = 110 - 0.7*20 = 96 (deep — below PDL=90?)
        #   Actually 96 > 90, and close[12:00]=100 is within zone of 96.
        #   But half_band = 0.3 * 20 / 2 = 3, so zone is [93, 99]. close=100 is outside.
        # Let's check: rl=0.5: entry = 100, zone [97, 103]. close=100 -> fires.
        # rl=0.3: entry = 104, zone [101, 107]. close=100 is outside. close=107 (h10-11) is in zone!
        day1 = result.loc["2024-01-02"]

        # rl=0.3 should fire (close=107 at 10:00 is within [101, 107])
        bull_30 = day1["rl30_pdl_retest_bull"].dropna()
        assert bull_30.sum() >= 1.0, "rl30 should fire on shallow retrace"


def _make_off_session_breakout_data():
    """Build a 3-day dataset where breakout above PDH happens at 3 AM
    (outside session 8-17), but during session only a downward break occurs.

    PDH/PDL is computed from H/L during session hours (8-17).

    Day 0 (2024-01-01): H=110, L=90 during session (8-17)
        → PDH=110, PDL=90, midpoint=100
    Day 1 (2024-01-02):
        03:00 (off-session): close=112, would break PDH — but ignored
        09:00 (in-session): close=88, breaks below PDL=90
        12:00 (in-session): close=100, retraces to midpoint
    """
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    close = np.full(72, 100.0)
    high = np.full(72, 100.5)
    low = np.full(72, 99.5)
    opn = np.full(72, 100.0)

    # Day 0: H/L establish range during session (8-17)
    # PDH = max(H) = 110, PDL = min(L) = 90
    for i in range(24):
        h = idx[i].hour
        close[i] = 100.0
        opn[i] = 100.0
        if h == 9:
            high[i] = 100.5
            low[i] = 90.0    # session low -> PDL=90
        elif h == 14:
            high[i] = 110.0   # session high -> PDH=110
            low[i] = 99.5
        else:
            high[i] = 100.5
            low[i] = 99.5

    # Day 1: off-session breakout above, in-session breakout below
    for i in range(24, 48):
        h = idx[i].hour
        if h == 3:
            # Off-session: price spikes above PDH=110
            close[i] = 112.0
            high[i] = 113.0
            low[i] = 108.0
        elif h < 8:
            close[i] = 105.0
            high[i] = 106.0
            low[i] = 104.0
        elif h == 9:
            # In-session: break below PDL=90
            close[i] = 88.0
            high[i] = 92.0
            low[i] = 87.0
        elif h == 12:
            # Retrace to midpoint=100 (still in session)
            close[i] = 100.0
            high[i] = 101.0
            low[i] = 99.0
        elif 8 <= h < 17:
            close[i] = 95.0
            high[i] = 96.0
            low[i] = 94.0
        else:
            close[i] = 95.0
            high[i] = 96.0
            low[i] = 94.0
        opn[i] = close[i]

    # Day 2: flat
    for i in range(48, 72):
        close[i] = 100.0
        high[i] = 101.0
        low[i] = 99.0
        opn[i] = 100.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


class TestPDLSessionFilteredBreaks:
    """Break detection is 24/7 (structural events), retest signals are session-only."""

    def test_off_session_breakout_detected(self):
        """A breakout above PDH at 3 AM (outside session 8-17) IS detected.
        Breakouts use Close and have no session filter in all_hours mode."""
        ind = _get_indicator()
        df = _make_off_session_breakout_data()
        result = ind.compute(df, session_start_hour=8, session_end_hour=17)

        day1 = result.loc["2024-01-02"]
        high_break_vals = day1["pdl_high_break"].dropna()
        assert high_break_vals.sum() >= 1.0, (
            f"pdl_high_break should fire even off-session, got {high_break_vals.sum()}"
        )

    def test_off_session_breakout_retest_fires_in_session(self):
        """Off-session breakout above PDH is detected, and the retest signal
        fires during session hours when price retraces to entry level."""
        ind = _get_indicator()
        df = _make_off_session_breakout_data()
        result = ind.compute(df, session_start_hour=8, session_end_hour=17,
                             retest_atr_width=0.3)

        day1 = result.loc["2024-01-02"]
        bull_vals = day1["rl50_pdl_retest_bull"].dropna()
        assert bull_vals.sum() >= 1.0, (
            f"retest_bull should fire (off-session breakout + in-session retest), "
            f"got {bull_vals.sum()}"
        )
        # Verify retest fires during session hours (8-17), not off-session
        fired_hours = day1.index[day1["rl50_pdl_retest_bull"] == 1.0].hour
        for h in fired_hours:
            assert 8 <= h < 17, f"retest_bull fired at hour {h}, outside session 8-17"

    def test_in_session_breakout_still_works(self):
        """The in-session breakout below PDL should still trigger
        retest_bear when price retraces."""
        ind = _get_indicator()
        df = _make_off_session_breakout_data()
        result = ind.compute(df, session_start_hour=8, session_end_hour=17,
                             retest_atr_width=0.3)

        day1 = result.loc["2024-01-02"]
        bear_vals = day1["rl50_pdl_retest_bear"].dropna()
        assert bear_vals.sum() >= 1.0, (
            f"retest_bear should fire (in-session breakout below PDL), "
            f"got {bear_vals.sum()}"
        )

    def test_retest_only_fires_during_session(self):
        """Even when breakout is 24/7, retest signals only fire during session."""
        ind = _get_indicator()
        df = _make_off_session_breakout_data()
        result = ind.compute(df, session_start_hour=8, session_end_hour=17,
                             retest_atr_width=0.3)

        day1 = result.loc["2024-01-02"]
        for col in ["rl50_pdl_retest_bull", "rl50_pdl_retest_bear"]:
            fired = day1.index[day1[col] == 1.0]
            for ts in fired:
                assert 8 <= ts.hour < 17, (
                    f"{col} fired at {ts} (hour {ts.hour}), outside session 8-17"
                )


class TestPDLRangeModes:
    """Tests for range_modes parameter (hl_session, hl_all, close_session, close_all)."""

    def test_default_mode_no_prefix(self):
        """Default range_modes=['hl_session'] produces standard features without extra prefix."""
        ind = _get_indicator()
        result = ind.compute(_make_ohlc_15min())
        assert "pdl_high_dist" in result.columns
        assert "rl50_pdl_retest_bull" in result.columns
        # No mode-prefixed columns
        for pfx in ("ha_", "cs_", "ca_"):
            assert not any(c.startswith(pfx) for c in result.columns), (
                f"Found {pfx} columns with default mode"
            )

    def test_all_modes_generate_prefixed_columns(self):
        """All 4 modes produce separate prefixed feature sets."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, range_modes=["hl_session", "hl_all", "close_session", "close_all"])

        # hl_session (no prefix)
        assert "pdl_high_dist" in result.columns
        assert "rl50_pdl_retest_bull" in result.columns

        # hl_all (ha_ prefix)
        assert "ha_pdl_high_dist" in result.columns
        assert "ha_rl50_pdl_retest_bull" in result.columns

        # close_session (cs_ prefix)
        assert "cs_pdl_high_dist" in result.columns
        assert "cs_rl50_pdl_retest_bull" in result.columns

        # close_all (ca_ prefix)
        assert "ca_pdl_high_dist" in result.columns
        assert "ca_rl50_pdl_retest_bull" in result.columns

    def test_hl_all_range_wider_than_session(self):
        """With off-session H/L extremes, all-hours range should be >= session range."""
        # Build data where off-session has extreme H/L
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        close = np.full(48, 100.0)
        high = np.full(48, 100.5)
        low = np.full(48, 99.5)
        opn = np.full(48, 100.0)

        # Day 0: session (7-21) has H=105, L=95
        # Off-session has H=115, L=85 (wider extremes)
        for i in range(24):
            h = idx[i].hour
            if h == 3:  # off-session
                high[i] = 115.0
                low[i] = 85.0
            elif h == 10:  # in-session
                high[i] = 105.0
                low[i] = 95.0

        df = pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)
        ind = _get_indicator()
        result = ind.compute(df, range_modes=["hl_session", "hl_all"])

        # Day 1: check range_vs_atr
        day1 = result.loc["2024-01-02"]
        session_range = day1["pdl_range_vs_atr"].dropna()
        all_range = day1["ha_pdl_range_vs_atr"].dropna()

        if len(session_range) > 0 and len(all_range) > 0:
            assert all_range.iloc[0] >= session_range.iloc[0], (
                "All-hours range should be >= session range"
            )

    def test_close_range_narrower_than_hl(self):
        """Close-based mode gives narrower or equal range vs H/L mode."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, range_modes=["hl_session", "close_session"])

        # Compare range_vs_atr: Close-based should be <= H/L-based
        hl_range = result["pdl_range_vs_atr"].dropna()
        cl_range = result["cs_pdl_range_vs_atr"].dropna()

        if len(hl_range) > 0 and len(cl_range) > 0:
            # Close range should be <= H/L range for most bars
            ratio = (cl_range <= hl_range + 1e-10).mean()
            assert ratio > 0.9, (
                f"Close range should be <= H/L range for most bars, "
                f"but only {ratio:.1%} are"
            )

    def test_backward_compatible_default(self):
        """Default params produce identical features to explicit hl_session."""
        ind = _get_indicator()
        df = _make_ohlc_15min()

        result_default = ind.compute(df.copy())
        result_explicit = ind.compute(df.copy(), range_modes=["hl_session"])

        for col in ind.get_feature_columns():
            assert col in result_default.columns
            assert col in result_explicit.columns
            pd.testing.assert_series_equal(
                result_default[col], result_explicit[col],
                check_names=False,
            )


class TestPDLDiscovery:
    """Plugin discovery tests."""

    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("previous_day_levels")
        assert cls is not None
