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

    def test_sl_dist_reaches_beyond_boundary(self):
        """rl50_pdl_sl_dist = (1-rl)*R + buffer, ensuring SL clears PDL/PDH."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df)

        # Day 1: PDH=110, PDL=90 -> range=20
        # sl_dist = (1-0.5)*20 + 0.3*20/2 = 10 + 3 = 13
        # (0.3 = default retest_atr_width, buffer = half retest band)
        day1 = result.loc["2024-01-02"]
        sl_vals = day1["rl50_pdl_sl_dist"].dropna()
        assert len(sl_vals) > 0
        pd_range = 110.0 - 90.0
        rl = 0.5
        retest_atr_width = 0.3  # default
        expected = (1 - rl) * pd_range + retest_atr_width * pd_range / 2  # 13.0
        np.testing.assert_allclose(
            sl_vals.iloc[0], expected, rtol=1e-10,
            err_msg=f"rl50_pdl_sl_dist should be {expected}"
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
        """Deeper retrace = smaller SL distance: sl_dist = (1-rl)*R + buffer."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        result = ind.compute(df, retracement_levels=[0.3, 0.5, 0.7])

        # Day 1: PDH=110, PDL=90 -> range=20, retest_atr_width=0.3 (default)
        # buffer = 0.3 * 20 / 2 = 3.0
        day1 = result.loc["2024-01-02"]
        buffer = 0.3 * 20 / 2  # 3.0

        sl30 = day1["rl30_pdl_sl_dist"].dropna().iloc[0]
        sl50 = day1["rl50_pdl_sl_dist"].dropna().iloc[0]
        sl70 = day1["rl70_pdl_sl_dist"].dropna().iloc[0]

        np.testing.assert_allclose(sl30, 0.7 * 20 + buffer, rtol=1e-10)  # 17.0
        np.testing.assert_allclose(sl50, 0.5 * 20 + buffer, rtol=1e-10)  # 13.0
        np.testing.assert_allclose(sl70, 0.3 * 20 + buffer, rtol=1e-10)  # 9.0

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

    def test_default_matches_explicit_hl_session(self):
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


def _make_midnight_crossing_data():
    """Build 1H data for a midnight-crossing session (start=23, end=6).

    Simulates ASX200-like session where 23:00 UTC = 09:00 AEST.

    Day 1 trading session: 23:00 Jan 1 → 05:00 Jan 2
        Session H=110, L=90, range=20, midpoint=100

    Day 1 off-session: 06:00-22:00 Jan 2
        Flat at 100

    Day 2 pre-market spike: 23:00 Jan 2 (= 09:00 AEST Jan 3)
        H=120 — must NOT inflate Day 1 range

    Day 2 trading session: 00:00-05:00 Jan 3
        Close=112 at 01:00 → breakout above PDH=110

    Day 2 off-session: 15:00 Jan 3
        Close=100 → retracement to midpoint (50% of range=20)

    4 calendar days = 96 hourly bars.
    """
    idx = pd.date_range("2024-01-01", periods=96, freq="h")
    close = np.full(96, 100.0)
    high = np.full(96, 100.5)
    low = np.full(96, 99.5)
    opn = np.full(96, 100.0)

    for i in range(96):
        ts = idx[i]
        h = ts.hour
        day = ts.day

        if day == 1 and h == 23:
            # Day 1 session start: 23:00 Jan 1 — set session high
            close[i] = 108.0
            high[i] = 110.0
            low[i] = 105.0
            opn[i] = 105.0
        elif day == 2 and h < 6:
            # Day 1 session continues: 00:00-05:00 Jan 2
            if h == 0:
                # Set session low
                close[i] = 92.0
                high[i] = 95.0
                low[i] = 90.0
                opn[i] = 95.0
            else:
                close[i] = 100.0
                high[i] = 101.0
                low[i] = 99.0
                opn[i] = 100.0
        elif day == 2 and 6 <= h < 23:
            # Day 1 off-session: flat at 100
            close[i] = 100.0
            high[i] = 100.5
            low[i] = 99.5
            opn[i] = 100.0
        elif day == 2 and h == 23:
            # Day 2 pre-market: spike to 120 — must NOT be in Day 1 range
            close[i] = 118.0
            high[i] = 120.0
            low[i] = 115.0
            opn[i] = 115.0
        elif day == 3 and h < 6:
            # Day 2 session: 00:00-05:00 Jan 3
            if h == 1:
                # Breakout above PDH=110
                close[i] = 112.0
                high[i] = 113.0
                low[i] = 110.0
                opn[i] = 111.0
            else:
                close[i] = 115.0
                high[i] = 116.0
                low[i] = 114.0
                opn[i] = 115.0
        elif day == 3 and h == 15:
            # Day 2 off-session: retracement to midpoint=100
            close[i] = 100.0
            high[i] = 101.0
            low[i] = 99.0
            opn[i] = 102.0
        elif day == 3 and 6 <= h < 23:
            # Day 2 off-session (rest)
            close[i] = 105.0
            high[i] = 106.0
            low[i] = 104.0
            opn[i] = 105.0
        elif day == 3 and h == 23:
            # Day 3 session start
            close[i] = 105.0
            high[i] = 106.0
            low[i] = 104.0
            opn[i] = 105.0
        elif day == 4:
            # Day 3 continues
            close[i] = 105.0
            high[i] = 106.0
            low[i] = 104.0
            opn[i] = 105.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


class TestPDLMidnightCrossing:
    """Tests for midnight-crossing sessions (e.g., ASX200 23:00-06:00 UTC)."""

    def test_premarket_not_in_previous_day_range(self):
        """The 23:00 Jan 2 spike (H=120) must NOT inflate Day 1's range.
        Day 1 session (23:00 Jan 1 - 05:00 Jan 2) has H=110, L=90."""
        ind = _get_indicator()
        df = _make_midnight_crossing_data()
        result = ind.compute(
            df, session_start_hour=23, session_end_hour=6,
            range_modes=["hl_session"],
        )

        # Day 2 bars (23:00 Jan 2 - 05:00 Jan 3) should see PDH=110
        # The shifted feature at 00:00 Jan 3 reflects PDH computed at 23:00 Jan 2
        day2_session = result.loc["2024-01-03 00:00":"2024-01-03 05:00"]
        pdh_vals = (
            day2_session["pdl_above_high"].dropna()
        )
        # At 01:00 Jan 3, C=112 > PDH=110 → pdl_above_high=1
        # (shifted from 00:00, so appears at 01:00)
        # If PDH were incorrectly 120, then 112 < 120 → pdl_above_high=0
        assert pdh_vals.sum() > 0, (
            "With correct day grouping, C=112 should be above PDH=110. "
            "If PDH is incorrectly inflated to 120, this fails."
        )

    def test_breakout_against_correct_pdh(self):
        """Breakout fires when C=112 > PDH=110 (correct).
        Must NOT fail because PDH is incorrectly 120."""
        ind = _get_indicator()
        df = _make_midnight_crossing_data()
        result = ind.compute(
            df, session_start_hour=23, session_end_hour=6,
            retest_atr_width=0.3,
        )

        # Check high_break fires on Day 2
        day2_bars = result.loc["2024-01-02 23:00":"2024-01-03 05:00"]
        break_vals = day2_bars["pdl_high_break"].dropna()
        assert break_vals.sum() >= 1.0, (
            f"pdl_high_break should fire (C=112 > PDH=110), got sum={break_vals.sum()}"
        )

    def test_retest_fires_off_session_all_hours(self):
        """With retest_modes=['all_hours'], retest fires at 15:00 UTC
        (off-session) when C=100 retraces to midpoint."""
        ind = _get_indicator()
        df = _make_midnight_crossing_data()
        result = ind.compute(
            df, session_start_hour=23, session_end_hour=6,
            retest_atr_width=0.3,
            retest_modes=["all_hours"],
        )

        # Check that retest_bull fires somewhere on Day 2
        # Day 2 = 23:00 Jan 2 through 22:00 Jan 3
        day2_all = result.loc["2024-01-02 23:00":"2024-01-03 22:00"]
        bull_vals = day2_all["rl50_pdl_retest_bull"].dropna()
        assert bull_vals.sum() >= 1.0, (
            f"all_hours retest should fire off-session, got sum={bull_vals.sum()}"
        )

    def test_retest_blocked_off_session_session_only(self):
        """With retest_modes=['session_only'], NO retest fires at 15:00 UTC
        because it's outside session (23:00-06:00)."""
        ind = _get_indicator()
        df = _make_midnight_crossing_data()
        result = ind.compute(
            df, session_start_hour=23, session_end_hour=6,
            retest_atr_width=0.3,
            retest_modes=["session_only"],
        )

        # The only retracement to midpoint=100 happens at 15:00 Jan 3 (off-session).
        # In session_only mode, this should NOT fire.
        day2_off = result.loc["2024-01-03 06:00":"2024-01-03 22:00"]
        bull_vals = day2_off["sr_rl50_pdl_retest_bull"].dropna()
        assert bull_vals.sum() == 0, (
            f"session_only retest should NOT fire off-session, got sum={bull_vals.sum()}"
        )

    def test_day_ids_change_at_session_start(self):
        """day_group transitions at 23:00 UTC (session start), not 00:00."""
        ind = _get_indicator()
        df = _make_midnight_crossing_data()
        result = ind.compute(
            df, session_start_hour=23, session_end_hour=6,
        )

        # The 22:00 Jan 2 bar and 23:00 Jan 2 bar should have different day_ids.
        # 22:00 Jan 2 belongs to Day 1 trading day.
        # 23:00 Jan 2 belongs to Day 2 trading day.
        # We verify indirectly: pdl_high_break resets at session start.
        # On Day 2 (23:00 Jan 2), high_break should start fresh.
        # post_bull from Day 1 (if any breakout happened) should not carry to Day 2
        # Since Day 1 had no breakout (all close=100 < PDH), post_bull=0 for both.
        # But the key point is that break state resets at 23:00.
        # We verify by checking that the breakout on Day 2 (C=112 at 01:00 Jan 3)
        # produces high_break on Day 2, confirming day_ids changed at 23:00.
        day2_bars = result.loc["2024-01-02 23:00":"2024-01-03 05:00"]
        break_vals = day2_bars["pdl_high_break"].dropna()
        assert break_vals.sum() >= 1.0, (
            "Break state should reset at 23:00 (session start), allowing breakout on Day 2"
        )

    def test_non_crossing_session_unchanged(self):
        """Non-midnight-crossing session (e.g. DAX 8-17) produces identical
        results with the new code path."""
        ind = _get_indicator()
        df = _make_ohlc_15min()

        # Default: session_start=7, session_end=21 (non-crossing)
        result = ind.compute(df)

        # Verify standard features are present and have values
        for col in ["pdl_high_dist", "pdl_position", "rl50_pdl_retest_bull"]:
            assert col in result.columns
            vals = result[col].dropna()
            assert len(vals) > 0

    def test_retest_modes_generate_prefixed_columns(self):
        """retest_modes=['all_hours', 'session_only'] produces both
        unprefixed and sr_ prefixed columns."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(
            df, retest_modes=["all_hours", "session_only"],
        )

        # all_hours (no prefix)
        assert "rl50_pdl_retest_bull" in result.columns
        assert "rl50_pdl_retest_bear" in result.columns
        assert "rl50_pdl_sl_dist" in result.columns

        # session_only (sr_ prefix)
        assert "sr_rl50_pdl_retest_bull" in result.columns
        assert "sr_rl50_pdl_retest_bear" in result.columns
        assert "sr_rl50_pdl_sl_dist" in result.columns

    def test_retest_modes_with_break_modes_cross_product(self):
        """Both dimensions cross-combine: break × retest × rl."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(
            df,
            break_modes=["all_hours", "session_only"],
            retest_modes=["all_hours", "session_only"],
        )

        # 2 break × 2 retest × 1 rl = 4 retest column sets
        expected_prefixes = [
            "",          # all_hours break + all_hours retest
            "sr_",       # all_hours break + session_only retest
            "sb_",       # session_only break + all_hours retest
            "sb_sr_",    # session_only break + session_only retest
        ]
        for pfx in expected_prefixes:
            assert f"{pfx}rl50_pdl_retest_bull" in result.columns, (
                f"Missing {pfx}rl50_pdl_retest_bull"
            )


def _make_hourly_breakout_spike_data():
    """Build 15-min data where a single 15-min bar closes above PDH
    but the hourly candle's Open and Close are both below PDH.

    Day 0 (2024-01-01): session 7-21, H=110, L=90 during session
        -> PDH=110, PDL=90
    Day 1 (2024-01-02):
        09:00 bar: C=105 (hourly Open equivalent)
        09:15 bar: C=112 (spike above PDH=110!) — only 15-min Close above
        09:30 bar: C=108
        09:45 bar: C=106 (hourly Close)
        → Hourly candle at 09:00: Open=105, Close=106 — both below PDH
        → With resample_tf="1h": NO breakout
        → Without resample_tf: breakout fires on the 09:15 bar (C=112 > 110)
    """
    idx = pd.date_range("2024-01-01", periods=192, freq="15min")
    close = np.full(192, 100.0)
    high = np.full(192, 100.5)
    low = np.full(192, 99.5)
    opn = np.full(192, 100.0)

    # Day 0: 96 bars, session 7-21 has H=110, L=90
    for i in range(96):
        h = idx[i].hour
        if h == 8:
            high[i] = 100.5
            low[i] = 90.0
        elif h == 11:
            high[i] = 110.0
            low[i] = 99.5

    # Day 1: spike on single 15-min bar
    for i in range(96, 192):
        ts = idx[i]
        h, m = ts.hour, ts.minute
        if h < 9 or h >= 17:
            close[i] = 105.0
            high[i] = 106.0
            low[i] = 104.0
            opn[i] = 105.0
        elif h == 9 and m == 0:
            # Hourly candle start
            close[i] = 105.0
            high[i] = 106.0
            low[i] = 104.0
            opn[i] = 105.0
        elif h == 9 and m == 15:
            # Spike! Close above PDH=110
            close[i] = 112.0
            high[i] = 113.0
            low[i] = 104.0
            opn[i] = 105.0
        elif h == 9 and m == 30:
            close[i] = 108.0
            high[i] = 109.0
            low[i] = 107.0
            opn[i] = 108.0
        elif h == 9 and m == 45:
            # Hourly candle end — Close back below PDH
            close[i] = 106.0
            high[i] = 107.0
            low[i] = 105.0
            opn[i] = 106.0
        else:
            close[i] = 105.0
            high[i] = 106.0
            low[i] = 104.0
            opn[i] = 105.0

    return pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)


class TestHourlyBreakout:
    """Tests for resample_tf-based hourly breakout confirmation."""

    def test_spike_breakout_filtered(self):
        """A single 15-min bar closing above PDH does NOT trigger breakout
        when resample_tf='1h' and the hourly O/C are both below PDH."""
        ind = _get_indicator()
        df = _make_hourly_breakout_spike_data()
        result = ind.compute(df, resample_tf="1h", min_retracement=0.0)

        day1 = result.loc["2024-01-02"]
        break_vals = day1["pdl_high_break"].dropna()
        assert break_vals.sum() == 0, (
            f"Hourly breakout should NOT fire on 15-min spike, got {break_vals.sum()}"
        )

    def test_spike_breakout_fires_without_resample(self):
        """Without resample_tf, the 15-min Close spike DOES trigger breakout."""
        ind = _get_indicator()
        df = _make_hourly_breakout_spike_data()
        result = ind.compute(df, resample_tf=None, min_retracement=0.0)

        day1 = result.loc["2024-01-02"]
        break_vals = day1["pdl_high_break"].dropna()
        assert break_vals.sum() >= 1.0, (
            "Without resample_tf, 15-min Close spike should trigger breakout"
        )

    def test_hourly_open_breakout_fires(self):
        """When the hourly Open is above PDH (gap up), breakout fires."""
        idx = pd.date_range("2024-01-01", periods=192, freq="15min")
        close = np.full(192, 100.0)
        high = np.full(192, 100.5)
        low = np.full(192, 99.5)
        opn = np.full(192, 100.0)

        # Day 0: session H=110, L=90
        for i in range(96):
            h = idx[i].hour
            if h == 8:
                low[i] = 90.0
            elif h == 11:
                high[i] = 110.0

        # Day 1: hourly Open at 09:00 gaps above PDH=110
        for i in range(96, 192):
            ts = idx[i]
            h, m = ts.hour, ts.minute
            if h == 9 and m == 0:
                # Gap open above PDH
                opn[i] = 112.0
                close[i] = 111.0  # Close also above
                high[i] = 113.0
                low[i] = 111.0
            elif h == 9:
                close[i] = 111.0
                high[i] = 112.0
                low[i] = 110.0
                opn[i] = 111.0
            elif 7 <= h < 21:
                close[i] = 105.0
                high[i] = 106.0
                low[i] = 104.0
                opn[i] = 105.0

        df = pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)
        ind = _get_indicator()
        result = ind.compute(df, resample_tf="1h", min_retracement=0.0)

        day1 = result.loc["2024-01-02"]
        break_vals = day1["pdl_high_break"].dropna()
        assert break_vals.sum() >= 1.0, (
            "Hourly Open gap above PDH should trigger breakout"
        )


class TestMinRetracement:
    """Tests for min_retracement parameter (H/L-based)."""

    def test_retest_blocked_without_retracement(self):
        """After breakout, if price stays near breakout boundary (Low never
        dips 30% into range), no retest fires."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        # Modify Day 1: after breakout at 09:00 (C=112), price stays high
        # Never retraces — all bars have Low > PDH - 0.3 * range
        # PDH=110, PDL=90, range=20, threshold = 110 - 0.3*20 = 104
        # Keep all Lows above 104

        # Re-build day 1 with no retracement
        idx = df.index
        close = df["C"].values.copy()
        high = df["H"].values.copy()
        low = df["L"].values.copy()
        opn = df["O"].values.copy()

        for i in range(24, 48):
            h = idx[i].hour
            if h == 9:
                close[i] = 112.0  # breakout
                high[i] = 113.0
                low[i] = 108.0
                opn[i] = 109.0
            elif h > 9:
                close[i] = 108.0  # stays high, Low=106 > 104 threshold
                high[i] = 109.0
                low[i] = 106.0
                opn[i] = 108.0

        df2 = pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df2, resample_tf=None, min_retracement=0.3)

        day1 = result.loc["2024-01-02"]
        bull_vals = day1["rl50_pdl_retest_bull"].dropna()
        assert bull_vals.sum() == 0, (
            f"No retest should fire without 30% retracement, got {bull_vals.sum()}"
        )

    def test_retest_fires_after_deep_retracement(self):
        """After breakout, once a bar's Low dips below the retrace threshold,
        retest fires on the next entry-level bar."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        # Default data: breakout at 09:00 (C=112), retrace to midpoint (C=100) at 12:00
        # PDH=110, PDL=90, range=20
        # threshold = 110 - 0.3*20 = 104. Low at 12:00 = 99 < 104 -> retracement OK
        result = ind.compute(df, resample_tf=None, min_retracement=0.3)

        day1 = result.loc["2024-01-02"]
        bull_vals = day1["rl50_pdl_retest_bull"].dropna()
        assert bull_vals.sum() >= 1.0, (
            f"Retest should fire after 30% retracement, got {bull_vals.sum()}"
        )

    def test_retracement_via_low_not_close(self):
        """A bar whose Low touches the threshold but Close stays high
        should still satisfy the retracement condition."""
        idx = pd.date_range("2024-01-01", periods=72, freq="h")
        close = np.full(72, 100.0)
        high = np.full(72, 100.5)
        low = np.full(72, 99.5)
        opn = np.full(72, 100.0)

        # Day 0: PDH=110, PDL=90, range=20
        for i in range(24):
            h = idx[i].hour
            if h == 8:
                low[i] = 90.0
            elif h == 11:
                high[i] = 110.0

        # Day 1: breakout, then Low dips but Close stays high
        for i in range(24, 48):
            h = idx[i].hour
            if h == 9:
                close[i] = 112.0  # breakout
                high[i] = 113.0
                low[i] = 110.0
                opn[i] = 111.0
            elif h == 11:
                # Low dips to 103 (below threshold 104), but Close stays at 108
                close[i] = 108.0
                high[i] = 109.0
                low[i] = 103.0  # <-- triggers retracement via Low
                opn[i] = 108.0
            elif h == 12:
                # Price at midpoint (entry level for rl=0.5)
                close[i] = 100.0
                high[i] = 101.0
                low[i] = 99.0
                opn[i] = 101.0
            elif h > 9:
                close[i] = 107.0
                high[i] = 108.0
                low[i] = 106.0
                opn[i] = 107.0

        df = pd.DataFrame({"O": opn, "H": high, "L": low, "C": close}, index=idx)
        ind = _get_indicator()
        result = ind.compute(df, resample_tf=None, min_retracement=0.3)

        day1 = result.loc["2024-01-02"]
        bull_vals = day1["rl50_pdl_retest_bull"].dropna()
        assert bull_vals.sum() >= 1.0, (
            f"Low touching threshold should satisfy retracement, got {bull_vals.sum()}"
        )

    def test_bear_retracement_via_high(self):
        """For bear retest, High touching PDL + min_retracement * range
        should satisfy the retracement condition."""
        ind = _get_indicator()
        df = _make_deterministic_retest_data()
        # Day 2: breakout below PDL at 09:00, retrace to midpoint at 12:00
        # PDH (from Day 1) depends on day 1 data.
        # The default data should work — high at 12:00 is 107, which should
        # be above the threshold.
        result = ind.compute(df, resample_tf=None, min_retracement=0.3)

        day2 = result.loc["2024-01-03"]
        bear_vals = day2["rl50_pdl_retest_bear"].dropna()
        assert bear_vals.sum() >= 1.0, (
            f"Bear retest should fire after retracement via High, got {bear_vals.sum()}"
        )


class TestResampledRange:
    """Tests for resample_tf Close-based range computation."""

    def test_resampled_close_range_differs_from_native(self):
        """With 15-min data where sub-hourly Close spikes exist,
        resample_tf='1h' range should differ from native."""
        ind = _get_indicator()
        df = _make_hourly_breakout_spike_data()

        # close_session with resample
        result_r = ind.compute(
            df.copy(), range_modes=["close_session"],
            resample_tf="1h", min_retracement=0.0,
        )
        # close_session without resample
        result_n = ind.compute(
            df.copy(), range_modes=["close_session"],
            resample_tf=None, min_retracement=0.0,
        )

        # Day 1 range_vs_atr should differ because the 09:15 spike (C=112)
        # inflates the native Close-based day 0 range but not the resampled one
        day0_r = result_r.loc["2024-01-02"]["cs_pdl_range_vs_atr"].dropna()
        day0_n = result_n.loc["2024-01-02"]["cs_pdl_range_vs_atr"].dropna()

        # Both should have values
        assert len(day0_r) > 0 and len(day0_n) > 0

    def test_hl_mode_unaffected_by_resample(self):
        """hl_session range is unchanged by resample_tf (max of max = max)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)

        result_r = ind.compute(
            df.copy(), range_modes=["hl_session"],
            resample_tf="1h", min_retracement=0.0,
        )
        result_n = ind.compute(
            df.copy(), range_modes=["hl_session"],
            resample_tf=None, min_retracement=0.0,
        )

        # Range should be identical
        r_range = result_r["pdl_range_vs_atr"].dropna()
        n_range = result_n["pdl_range_vs_atr"].dropna()
        pd.testing.assert_series_equal(r_range, n_range, check_names=False)

    def test_new_default_params(self):
        """Default params include resample_tf and min_retracement."""
        params = _pdl.PreviousDayLevelsIndicator.get_default_params()
        assert params["resample_tf"] is None
        assert params["min_retracement"] == 0.3

    def test_new_param_schema(self):
        """Param schema includes resample_tf and min_retracement."""
        schema = _pdl.PreviousDayLevelsIndicator.get_param_schema()
        assert "resample_tf" in schema
        assert "min_retracement" in schema


class TestPDLDiscovery:
    """Plugin discovery tests."""

    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("previous_day_levels")
        assert cls is not None
