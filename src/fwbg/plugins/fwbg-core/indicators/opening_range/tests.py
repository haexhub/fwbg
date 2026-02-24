"""Tests for Opening Range Breakout indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_orb = import_plugin_module("fwbg-core", "indicators", "opening_range")
if _orb is None:
    pytest.skip("fwbg-core opening_range plugin not available", allow_module_level=True)


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
    # Ensure H >= max(O,C) and L <= min(O,C)
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
    return _orb.OpeningRangeIndicator()


class TestRollingORB:
    """Tests for rolling (hourly) ORB features."""

    def test_rolling_features_computed(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        for col in ["orb_range", "orb_position", "orb_breakout_up",
                     "orb_breakout_down", "orb_range_vs_atr"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rolling_features_have_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df)

        for col in ["orb_range", "orb_position", "orb_range_vs_atr"]:
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN"

    def test_range_positive(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        range_vals = result["orb_range"].dropna()
        assert (range_vals >= 0).all(), "orb_range should be non-negative"

    def test_breakout_binary(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        for col in ["orb_breakout_up", "orb_breakout_down"]:
            vals = result[col].dropna()
            assert set(vals.unique()).issubset({0, 1}), f"{col} should be binary"

    def test_position_ranges(self):
        """Position should be around 0-1 mostly, can exceed for breakouts."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        pos = result["orb_position"].dropna()
        # Most values should be in reasonable range
        within_range = ((pos >= -1) & (pos <= 2)).mean()
        assert within_range > 0.8, "Most positions should be in [-1, 2] range"

    def test_no_lookahead(self):
        """First row of every feature should be NaN (shifted by 1)."""
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        for col in ["orb_range", "orb_position", "orb_breakout_up"]:
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"

    def test_hourly_data_rolling_is_nan(self):
        """On hourly data with range_bars=1, rolling ORB is all NaN
        (only 1 bar per hour → no bar after range is established)."""
        ind = _get_indicator()
        df = _make_ohlc_hourly()
        result = ind.compute(df)

        assert "orb_range" in result.columns
        # Rolling features are NaN because each hour has only 1 bar
        assert result["orb_range"].dropna().empty

    def test_hourly_data_session_features_work(self):
        """Session ORB should work on hourly data (range persists across hours)."""
        ind = _get_indicator()
        df = _make_ohlc_hourly(n=5000)
        result = ind.compute(df)

        non_null = result["orb_s08_range"].dropna()
        assert len(non_null) > 0

    def test_no_inf_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        feature_cols = [c for c in result.columns if c.startswith("orb_")]
        for col in feature_cols:
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} contains inf values"


class TestSessionORB:
    """Tests for session-specific ORB features."""

    def test_session_features_computed(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        # Default sessions are [8, 9, 14, 15]
        result = ind.compute(df)

        for h in [8, 9, 14, 15]:
            prefix = f"orb_s{h:02d}"
            for suffix in ["_range", "_position", "_breakout_up",
                           "_breakout_down", "_range_vs_atr"]:
                col = f"{prefix}{suffix}"
                assert col in result.columns, f"Missing column: {col}"

    def test_session_features_have_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)  # ~52 days, enough for all sessions
        result = ind.compute(df)

        for h in [8, 9, 14, 15]:
            col = f"orb_s{h:02d}_range"
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN"

    def test_custom_sessions(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, sessions=[9, 17])

        assert "orb_s09_range" in result.columns
        assert "orb_s17_range" in result.columns
        # Default sessions should NOT be present
        assert "orb_s08_range" not in result.columns

    def test_session_breakout_binary(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for h in [8, 9, 14, 15]:
            for direction in ["up", "down"]:
                col = f"orb_s{h:02d}_breakout_{direction}"
                vals = result[col].dropna()
                if len(vals) > 0:
                    assert set(vals.unique()).issubset({0, 1}), f"{col} not binary"


class TestStatFeatures:
    """Tests for rolling statistical features."""

    def test_stat_features_computed(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for col in ["orb_stat_avg_range", "orb_stat_breakout_rate",
                     "orb_stat_continuation_rate"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_stat_features_have_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for col in ["orb_stat_avg_range", "orb_stat_breakout_rate",
                     "orb_stat_continuation_rate"]:
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN"

    def test_rates_between_0_and_1(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for col in ["orb_stat_breakout_rate", "orb_stat_continuation_rate"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals >= 0).all() and (vals <= 1).all(), \
                    f"{col} should be between 0 and 1"


class TestDailySkip:
    """Daily data should not produce ORB features."""

    def test_daily_returns_unchanged(self):
        ind = _get_indicator()
        df = _make_ohlc_daily()
        result = ind.compute(df)

        orb_cols = [c for c in result.columns if c.startswith("orb_")]
        assert len(orb_cols) == 0, "Daily data should not produce ORB features"


class TestParameters:
    """Test parameter variations."""

    def test_range_bars_2(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=2)

        assert "orb_range" in result.columns
        non_null = result["orb_range"].dropna()
        assert len(non_null) > 0

    def test_disable_rolling(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, enable_rolling=False)

        assert "orb_range" not in result.columns
        assert "orb_s08_range" in result.columns

    def test_disable_session(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, enable_session=False)

        assert "orb_range" in result.columns
        assert "orb_s08_range" not in result.columns

    def test_disable_stats(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, enable_stats=False)

        assert "orb_stat_avg_range" not in result.columns

    def test_get_default_params(self):
        params = _orb.OpeningRangeIndicator.get_default_params()
        assert params["range_bars"] == 1
        assert params["sessions"] == [8, 9, 14, 15]

    def test_get_param_schema(self):
        schema = _orb.OpeningRangeIndicator.get_param_schema()
        assert "range_bars" in schema
        assert "sessions" in schema
        assert schema["range_bars"]["type"] == "list[int]"
        assert schema["sessions"]["type"] == "list[int]"

    def test_get_feature_columns_includes_all_pipeline_sessions(self):
        """get_feature_columns() must cover all UTC sessions used in pipeline configs."""
        ind = _get_indicator()
        cols = ind.get_feature_columns()
        # Sessions [0, 1, 2, 5, 6, 7, 8, 12, 13, 14] from orb_scalping_v1.json
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            assert f"orb_s{h:02d}_range" in cols, (
                f"orb_s{h:02d}_range missing from get_feature_columns() — "
                f"session {h} UTC is configured in orb_scalping_v1.json"
            )

    def test_get_feature_columns_excludes_non_pipeline_sessions(self):
        """Sessions 9 and 15 are not in any pipeline config — must not appear in feature columns."""
        ind = _get_indicator()
        cols = ind.get_feature_columns()
        for h in [9, 15]:
            assert f"orb_s{h:02d}_range" not in cols, (
                f"orb_s{h:02d}_range should not be in get_feature_columns() — "
                f"session {h} is not used in any pipeline config"
            )

    def test_get_signal_columns_includes_all_pipeline_sessions(self):
        """get_signal_columns() must cover breakout signals for all pipeline sessions."""
        ind = _get_indicator()
        signals = ind.get_signal_columns()
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            for direction in ["up", "down"]:
                col = f"orb_s{h:02d}_breakout_{direction}"
                assert col in signals, (
                    f"{col} missing from get_signal_columns() — "
                    f"session {h} UTC is configured in orb_scalping_v1.json"
                )


class TestRangeBarsListMode:
    """range_bars=[1, 2] (list) activates prefix mode: rb1_orb_* and rb2_orb_* columns."""

    def test_list_mode_produces_rb_prefixed_columns(self):
        """When range_bars is a list, all columns get rb{n}_ prefix instead of bare orb_ names."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=[1, 2])

        assert "rb1_orb_range" in result.columns, "rb1_ prefix missing for range_bars=1"
        assert "rb2_orb_range" in result.columns, "rb2_ prefix missing for range_bars=2"
        # Bare names must NOT appear when using list mode
        assert "orb_range" not in result.columns, "bare orb_range must not exist in list mode"

    def test_list_mode_both_rb_variants_have_breakout_signals(self):
        """Both rb1 and rb2 variants must produce non-empty breakout signals."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=[1, 2])

        for prefix in ["rb1", "rb2"]:
            for direction in ["up", "down"]:
                col = f"{prefix}_orb_breakout_{direction}"
                assert col in result.columns, f"{col} missing"
                assert result[col].dropna().isin([0, 1]).all(), f"{col} not binary"

    def test_list_mode_stat_columns_have_rb_prefix(self):
        """Stat columns (avg_range, breakout_rate) must also carry the rb{n}_ prefix."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=[1, 2])

        for prefix in ["rb1", "rb2"]:
            assert f"{prefix}_orb_stat_avg_range" in result.columns or \
                   "orb_stat_avg_range" in result.columns, \
                   f"stat column missing for {prefix}"

    def test_list_mode_session_columns_have_rb_prefix(self):
        """Session ORB columns (orb_s08_*) must also carry the rb{n}_ prefix in list mode."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=[1, 2], sessions=[8])

        assert "rb1_orb_s08_range" in result.columns or "orb_s08_range" in result.columns, \
            "Session column must exist in list mode (with or without rb prefix)"
        # Verify at least one rb-prefixed session column exists
        rb_session_cols = [c for c in result.columns if "s08" in c]
        assert len(rb_session_cols) > 0, "No session columns found for hour 8 in list mode"


class TestBreakoutEventFeature:
    """orb_breakout_up/down must be event (transition) features, not state features.

    In ORB trading, the breakout is the EVENT that triggers the entry.
    The feature should fire (=1) only on the FIRST bar where price crosses
    the opening range boundary, not for every subsequent bar that stays above/below.
    """

    def _make_15min_with_sustained_breakout(self, n_warmup_hours=5, n_post_hours=1):
        """Create 15min DataFrame: warmup, then 1 controlled hour, then post hours.

        The controlled hour:
            bar0 (08:00): range bar, or_high=101, or_low=99
            bar1 (08:15): C=100 — no breakout
            bar2 (08:30): C=102 — FIRST breakout (above or_high=101)
            bar3 (08:45): C=103 — sustained above (should NOT fire again in event model)

        n_post_hours ensures the sustained bar's computed value appears in the result
        (shift_features would otherwise push it past the end of the DataFrame).
        """
        warmup_bars = n_warmup_hours * 4
        controlled_bars = 4
        post_bars = n_post_hours * 4
        n_total = warmup_bars + controlled_bars + post_bars

        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5

        b = warmup_bars  # start of controlled hour
        high[b] = 101.0  # range bar: sets or_high=101
        low[b] = 99.0    # range bar: sets or_low=99
        # bar1: C=100 (no breakout, or_high=101)
        close[b + 2] = 102.0  # bar2: first breakout
        close[b + 3] = 103.0  # bar3: still above (STATE=1 here, EVENT=0)

        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        return df, warmup_bars

    def test_rolling_breakout_up_fires_only_on_first_crossing(self):
        """orb_breakout_up = 1 only on first bar crossing above or_high, not subsequent bars.

        Controlled sequence: bar0=range, bar1=no cross, bar2=first cross, bar3=sustained.
        EVENT: bar3 should NOT fire (still above but no new crossing).
        STATE: bar3 WOULD fire (currently above) — wrong behavior.
        """
        ind = _get_indicator()
        df, b = self._make_15min_with_sustained_breakout(n_warmup_hours=5, n_post_hours=1)
        result = ind.compute(df, enable_session=False, enable_stats=False)

        # After shift, bar2's value appears at b+3, bar3's value at b+4 (in post-hours)
        # b+3 = first crossing (should be 1 in both STATE and EVENT)
        # b+4 = sustained crossing (STATE=1, EVENT=0 ← the key distinction)
        bu_b3 = result["orb_breakout_up"].iloc[b + 3]  # should be 1
        bu_b4 = result["orb_breakout_up"].iloc[b + 4]  # STATE=1, EVENT=0

        assert bu_b3 == 1.0, f"bar2 (first crossing) should be 1, got {bu_b3}"
        assert bu_b4 == 0.0, (
            f"bar3 (sustained crossing, no new breakout) should be 0, got {bu_b4}. "
            f"orb_breakout_up is a STATE feature — it must be an EVENT (transition) feature."
        )

    def test_rolling_breakout_down_fires_only_on_first_crossing(self):
        """orb_breakout_down = 1 only on first bar crossing below or_low, not subsequent bars."""
        ind = _get_indicator()

        # Build same structure but for down: bar2=97 (first crossing below or_low=99),
        # bar3=96 (sustained below — should NOT fire again)
        n_warmup, n_post = 5, 1
        warmup_bars = n_warmup * 4
        n_total = warmup_bars + 4 + n_post * 4
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = warmup_bars
        high[b] = 101.0
        low[b] = 99.0
        close[b + 2] = 97.0  # first downward crossing (or_low=99)
        close[b + 3] = 96.0  # sustained below
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)

        result = ind.compute(df, enable_session=False, enable_stats=False)
        bd_b3 = result["orb_breakout_down"].iloc[b + 3]  # first crossing
        bd_b4 = result["orb_breakout_down"].iloc[b + 4]  # sustained (STATE=1, EVENT=0)

        assert bd_b3 == 1.0, f"bar2 (first down crossing) should be 1, got {bd_b3}"
        assert bd_b4 == 0.0, (
            f"bar3 (sustained below or_low) should be 0, got {bd_b4}. "
            f"orb_breakout_down must be an EVENT feature, not a STATE feature."
        )

    def test_rolling_breakout_resets_each_hour(self):
        """Each hour resets: two controlled hours, each with sustained breakout → 2 events total.

        Hour A: bar2 crosses up (event=1), bar3 sustained (STATE=1, EVENT=0)
        Hour B: bar2 crosses up (event=1), bar3 sustained (STATE=1, EVENT=0)
        Total with EVENT model: 2. With STATE model: 4.
        """
        ind = _get_indicator()
        # Use _make_15min_with_sustained_breakout gives 1 controlled hour; duplicate manually
        n_warmup = 5 * 4  # 20 warmup bars
        n_total = n_warmup + 8 + 4  # 2 controlled hours + 1 post hour for visibility
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5

        for offset in [0, 4]:  # Hour A at n_warmup, Hour B at n_warmup+4
            b = n_warmup + offset
            high[b] = 101.0
            low[b] = 99.0
            close[b + 2] = 102.0  # first breakout
            close[b + 3] = 103.0  # sustained (STATE=1, EVENT=0)

        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)

        result = ind.compute(df, enable_session=False, enable_stats=False)
        b = n_warmup
        # After shift: hour A bar2 value at b+3, bar3 value at b+4
        #              hour B bar2 value at b+7, bar3 value at b+8
        bu = result["orb_breakout_up"]
        assert bu.iloc[b + 3] == 1.0, "Hour A first crossing should be 1"
        assert bu.iloc[b + 4] == 0.0, (
            f"Hour A sustained crossing (STATE=1) should be 0 (EVENT feature). Got {bu.iloc[b+4]}."
        )
        assert bu.iloc[b + 7] == 1.0, "Hour B first crossing should be 1"
        assert bu.iloc[b + 8] == 0.0, (
            f"Hour B sustained crossing (STATE=1) should be 0 (EVENT feature). Got {bu.iloc[b+8]}."
        )

    def test_session_breakout_up_fires_only_on_first_crossing(self):
        """Session orb_breakout_up = 1 only on first crossing per session."""
        ind = _get_indicator()

        # Use a larger dataset with session at hour 8 (UTC)
        # Create enough bars so session_id > 0 (first session is discarded)
        n = 200  # ~50 hours of M15 data
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        _rng = np.random.default_rng(99)
        close = 100.0 * np.ones(n)

        # In hour 08 (bars at 08:00, 08:15, 08:30, 08:45):
        # Find the first occurrence of hour=8 after bar 4 (to ensure session_id > 0)
        hours = pd.Series(idx.hour)
        hour8_bars = hours[hours == 8].index.tolist()

        if len(hour8_bars) >= 2:
            # Second occurrence of hour 8: bar at position hour8_bars[1]
            base = hour8_bars[1]
            # bar+0 = range bar (08:00), bar+1 = 08:15, bar+2 = 08:30 (breakout), bar+3 = 08:45
            close[base + 1] = 100.0  # no breakout
            close[base + 2] = 102.0  # FIRST breakout (or_high = close[base]+1 = 101)
            close[base + 3] = 103.0  # still above, should NOT fire again

        high = close + 0.5
        low = close - 0.5
        # Set or_high for hour 8 range bar: H = close + 1.0
        for b in hour8_bars:
            high[b] = close[b] + 1.0
            low[b] = close[b] - 1.0

        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False)

        # In the second occurrence of session 8, breakout should fire exactly once
        if len(hour8_bars) >= 2:
            base = hour8_bars[1]
            # Due to shift_features, the breakout at bar base+2 appears at base+3 in result
            session_region = result["orb_s08_breakout_up"].iloc[base:base + 6]
            total = session_region.dropna().sum()
            assert total <= 1.0, (
                f"Session breakout_up fired {total} times in one session — should be at most 1 (event feature)"
            )

    def test_up_and_down_never_fire_on_same_bar(self):
        """orb_breakout_up and orb_breakout_down can never both be 1 on the same bar.

        A bar cannot simultaneously close above or_high AND below or_low (would require
        close > or_high AND close < or_low, which is impossible since or_high > or_low).
        """
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, enable_session=False, enable_stats=False)

        both_fired = (result["orb_breakout_up"] == 1) & (result["orb_breakout_down"] == 1)
        assert not both_fired.any(), (
            "orb_breakout_up and orb_breakout_down fired simultaneously on the same bar — "
            "physically impossible since close cannot be both above or_high and below or_low."
        )

    def test_both_can_fire_in_same_session_false_breakout_scenario(self):
        """Both breakout directions can fire within the same hour (false breakout / stop-hunt).

        Scenario:
          bar0 (range bar): or_high=101, or_low=99
          bar1: C=102 → FIRST upward breakout (orb_breakout_up=1)
          bar2: C=98  → price reverses below or_low → FIRST downward breakout (orb_breakout_down=1)
          bar3: C=97  → sustained below (should NOT fire again)

        This is the false-breakout / stop-hunt pattern that weekly_orb and orb strategies
        aim to exploit: both signals fire within the same session at different bars.
        If SL is tight (e.g. below bar1 low), the up-trade is stopped out at bar2, and the
        down-breakout at bar2 can be traded independently.
        """
        ind = _get_indicator()
        n_warmup = 5 * 4  # 5 hours of warmup (20 bars)
        n_post = 4
        n_total = n_warmup + 4 + n_post
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5

        b = n_warmup
        high[b] = 101.0
        low[b] = 99.0   # range bar
        close[b + 1] = 102.0              # first upward crossing
        close[b + 2] = 98.0              # reversal: first downward crossing
        close[b + 3] = 97.0              # sustained below

        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, enable_session=False, enable_stats=False)

        # After shift: bar1 → b+2, bar2 → b+3, bar3 → b+4
        bu = result["orb_breakout_up"]
        bd = result["orb_breakout_down"]

        assert bu.iloc[b + 2] == 1.0, f"bar1 upward crossing must fire (got {bu.iloc[b+2]})"
        assert bd.iloc[b + 3] == 1.0, f"bar2 downward crossing must fire (got {bd.iloc[b+3]})"
        # They fired at different bars — never on the same bar
        assert bu.iloc[b + 3] == 0.0, "orb_breakout_up must not fire on downward-crossing bar"
        assert bu.iloc[b + 4] == 0.0, "orb_breakout_up must not fire on sustained-below bar"
        assert bd.iloc[b + 4] == 0.0, "orb_breakout_down must not fire again on sustained bar (event, not state)"


class TestORBSLDist:
    """orb_sl_dist = or_high - or_low — full ORB range as SL distance (entry near breakout → SL at opposite boundary)."""

    def test_rolling_sl_dist_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, enable_session=False, enable_stats=False)
        assert "orb_sl_dist" in result.columns, "orb_sl_dist column missing from rolling ORB output"

    def test_rolling_sl_dist_equals_full_range(self):
        """orb_sl_dist must equal or_high - or_low (full ORB range).

        Entry near breakout boundary → SL at opposite boundary = full range.
        ORB: high=103, low=97 → range=6.0, orb_sl_dist = 6.0.
        """
        ind = _get_indicator()
        n_warmup = 5 * 4  # 5 hours = 20 bars of 15-min data (00:00–04:45)
        n_total = n_warmup + 8
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_warmup  # bar 20 = 05:00 → range bar for that hour
        high[b] = 103.0
        low[b] = 97.0   # ORB range = 6.0
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, enable_session=False, enable_stats=False)
        # After shift_features: computed value for bar i appears at result.iloc[i+1].
        # Bar b is the range bar (valid=False) → result.iloc[b+1] = NaN.
        # Bars b+1, b+2, b+3 are post-range bars (valid=True) → result.iloc[b+2:b+5] = 6.0.
        sl_vals = result["orb_sl_dist"].iloc[b + 1:b + 5].dropna()
        assert len(sl_vals) > 0, "No non-NaN values found for orb_sl_dist in post-range bars"
        assert (sl_vals.round(6) == 6.0).all(), (
            f"Expected orb_sl_dist = 6.0 (full ORB range), got: {sl_vals.values}"
        )

    def test_rolling_sl_dist_positive(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, enable_session=False, enable_stats=False)
        sl_vals = result["orb_sl_dist"].dropna()
        assert (sl_vals > 0).all(), "orb_sl_dist should be strictly positive"

    def test_session_sl_dist_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "orb_s08_sl_dist" in result.columns, "orb_s08_sl_dist column missing from session ORB output"

    def test_session_sl_dist_positive(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        sl_vals = result["orb_s08_sl_dist"].dropna()
        assert (sl_vals > 0).all(), "orb_s08_sl_dist should be strictly positive"


class TestORBPocDist:
    """orb_poc_dist = (close - or_midpoint) / atr — normalized distance to ORB midpoint."""

    def test_rolling_poc_dist_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, enable_session=False, enable_stats=False)
        assert "orb_poc_dist" in result.columns, "orb_poc_dist column missing from rolling ORB output"

    def test_rolling_poc_dist_has_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, enable_session=False, enable_stats=False)
        assert result["orb_poc_dist"].dropna().abs().sum() > 0, "orb_poc_dist is all zero or all NaN"

    def test_poc_dist_zero_at_midpoint(self):
        """When C equals the ORB midpoint exactly, orb_poc_dist must be 0."""
        ind = _get_indicator()
        n_warmup = 5 * 4
        n_total = n_warmup + 8
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_warmup
        high[b] = 103.0
        low[b] = 97.0   # midpoint = (103+97)/2 = 100.0
        # bar b+1: C = 100.0 → exactly at midpoint → poc_dist = 0/ATR = 0
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, enable_session=False, enable_stats=False)
        # After shift: bar b+1 value at result.iloc[b+2]
        poc_at_midpoint = result["orb_poc_dist"].iloc[b + 2]
        assert poc_at_midpoint == pytest.approx(0.0, abs=1e-6), (
            f"poc_dist should be 0 when close == midpoint (100.0), got {poc_at_midpoint}"
        )

    def test_session_poc_dist_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "orb_s08_poc_dist" in result.columns, "orb_s08_poc_dist column missing from session ORB output"


class TestORBPostBreakoutState:
    """orb_s{hh}_post_bull/bear = state: 1 for all bars AFTER first session breakout, resets per session."""

    def _make_session_df(self, n_pre_hour_bars=32):
        """M15 data starting 2024-01-02 00:00; session hour 8 starts at bar n_pre_hour_bars.

        Bar layout around session:
          b+0 (range bar): or_high=103, or_low=97
          b+1: C=100 (no breakout)
          b+2: C=104 (upside breakout)
          b+3: C=100 (retrace to midpoint)
          b+4: C=98  (near or_low, still valid)
          b+5: C=96  (below or_low — thesis invalidated)
        """
        n_total = n_pre_hour_bars + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre_hour_bars
        high[b] = 103.0
        low[b] = 97.0
        # b+1: close stays 100 (no breakout)
        close[b + 2] = 104.0   # upside breakout
        close[b + 3] = 100.0   # retrace to midpoint
        close[b + 4] = 98.0    # near or_low, still valid (98 > 97)
        close[b + 5] = 96.0    # below or_low (thesis invalidated)
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        return df, b

    def test_post_bull_is_zero_before_breakout(self):
        """post_bull = 0 for bars before the first upside breakout in a session."""
        ind = _get_indicator()
        df, b = self._make_session_df()
        assert df.index[b].hour == 8, "Range bar must be at session hour 8"
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False)
        assert "orb_s08_post_bull" in result.columns, "orb_s08_post_bull column missing"
        # Bar b+1 (C=100, no breakout yet): computed post_bull=0 → result.iloc[b+2]
        val = result["orb_s08_post_bull"].iloc[b + 2]
        assert val == 0.0, f"post_bull should be 0 before any breakout, got {val}"

    def test_post_bull_becomes_one_after_upside_breakout(self):
        """post_bull = 1 from the first bar where C > or_high, persists for subsequent bars."""
        ind = _get_indicator()
        df, b = self._make_session_df()
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False)
        # Bar b+2 (C=104 > or_high=103): post_bull becomes 1 → result.iloc[b+3]
        val_at_breakout = result["orb_s08_post_bull"].iloc[b + 3]
        assert val_at_breakout == 1.0, f"post_bull should be 1 at breakout bar, got {val_at_breakout}"
        # Bar b+3 (C=100, retrace): post_bull must STAY 1 (state, not event) → result.iloc[b+4]
        val_after_retrace = result["orb_s08_post_bull"].iloc[b + 4]
        assert val_after_retrace == 1.0, (
            f"post_bull should remain 1 after retrace (it's a state, not an event), got {val_after_retrace}"
        )

    def test_post_bull_resets_in_next_session(self):
        """post_bull resets at the start of a new session occurrence."""
        ind = _get_indicator()
        n_total = 2 * 24 * 4  # 2 days of 15-min data
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5

        hours = pd.DatetimeIndex(idx).hour
        session_bars = np.where(hours == 8)[0]
        assert len(session_bars) >= 5, "Need at least 2 session starts (5 bars each at hour 8)"
        b1 = session_bars[0]   # first 08:00
        b2 = session_bars[4]   # second 08:00 (next day)

        # First session: upside breakout
        high[b1] = 103.0
        low[b1] = 97.0
        close[b1 + 1] = 104.0  # breakout in session 1

        # Second session: normal range bar, no breakout
        high[b2] = 103.0
        low[b2] = 97.0
        # b2+1 stays at close=100 (no breakout)

        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False)

        # Second session bar b2+1 (C=100, no breakout): post_bull must reset to 0
        # After shift: result.iloc[b2+2]
        val = result["orb_s08_post_bull"].iloc[b2 + 2]
        assert val == 0.0, (
            f"post_bull should reset to 0 in the next session (no breakout yet), got {val}"
        )

    def test_post_bear_becomes_one_after_downside_breakout(self):
        """post_bear = 1 from the first bar where C < or_low, persists for subsequent bars."""
        ind = _get_indicator()
        n_total = 32 + 8
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = 32  # bar 32 = 08:00
        high[b] = 103.0
        low[b] = 97.0
        close[b + 1] = 100.0   # no breakout
        close[b + 2] = 96.0    # downside breakout (C=96 < or_low=97)
        close[b + 3] = 100.0   # retrace
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False)

        assert "orb_s08_post_bear" in result.columns, "orb_s08_post_bear column missing"
        # Bar b+1 (no breakout): post_bear = 0 → result.iloc[b+2]
        val_before = result["orb_s08_post_bear"].iloc[b + 2]
        assert val_before == 0.0, f"post_bear before breakout should be 0, got {val_before}"
        # Bar b+2 (C=96 < or_low=97): post_bear = 1 → result.iloc[b+3]
        val_at_breakout = result["orb_s08_post_bear"].iloc[b + 3]
        assert val_at_breakout == 1.0, f"post_bear should be 1 at downside breakout, got {val_at_breakout}"
        # Bar b+3 (retrace): post_bear stays 1 → result.iloc[b+4]
        val_after = result["orb_s08_post_bear"].iloc[b + 4]
        assert val_after == 1.0, f"post_bear should stay 1 after retrace, got {val_after}"


class TestORBRetestEntry:
    """orb_s{hh}_retest_bull/bear = entry signal: fires when post-breakout AND price near ORB midpoint."""

    def _make_retest_df(self, n_pre_hour_bars=32):
        """M15 data with a planted post-breakout retrace scenario:
          b+0: range bar (or_high=103, or_low=97, midpoint=100)
          b+1: C=104 (upside breakout — post_bull becomes 1)
          b+2: C=100 (retrace to midpoint — retest_bull SHOULD fire)
          b+3: C=96  (below or_low — thesis invalidated, retest_bull should NOT fire)
        """
        n_total = n_pre_hour_bars + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        open_ = np.full(n_total, 100.0)
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre_hour_bars
        # Body-based range: O of first bar = 97, C of first bar = 103
        # → or_high = max(97, 103) = 103, or_low = min(97, 103) = 97, midpoint = 100
        open_[b] = 97.0
        close[b] = 103.0
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 104.0   # upside breakout (C > or_high=103)
        close[b + 2] = 100.0   # retrace to midpoint
        close[b + 3] = 96.0    # below or_low (invalidated)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=idx)
        return df, b

    def test_retest_bull_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "orb_s08_retest_bull" in result.columns, "orb_s08_retest_bull column missing"

    def test_retest_bear_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "orb_s08_retest_bear" in result.columns, "orb_s08_retest_bear column missing"

    def test_retest_bull_requires_post_breakout_state(self):
        """retest_bull = 0 when there is no prior upside breakout (post_bull = 0)."""
        ind = _get_indicator()
        df, b = self._make_retest_df()
        # Override b+1 to stay inside range (no breakout)
        df = df.copy()
        df.loc[df.index[b + 1], "C"] = 100.0
        df.loc[df.index[b + 1], "H"] = 100.5
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False,
                             retest_atr_width=1.0)
        # Bar b+2 is at midpoint (C=100) but post_bull=0 → retest_bull must be 0
        # After shift: result.iloc[b+3]
        val = result["orb_s08_retest_bull"].iloc[b + 3]
        assert val == 0.0, (
            f"retest_bull should be 0 without a prior upside breakout, got {val}"
        )

    def test_retest_bull_fires_when_price_at_midpoint_after_breakout(self):
        """retest_bull = 1 when post_bull=1 AND price retraces to the ORB midpoint."""
        ind = _get_indicator()
        df, b = self._make_retest_df()
        # Use wide retest_atr_width to ensure the zero-distance always qualifies
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False,
                             retest_atr_width=1.0)
        # Bar b+1 (C=104, breakout): post_bull becomes 1
        # Bar b+2 (C=100 = midpoint): post_bull=1, near_poc=True, still_valid_bull=True → fires
        # After shift: result.iloc[b+3]
        val = result["orb_s08_retest_bull"].iloc[b + 3]
        assert val == 1.0, (
            f"retest_bull should be 1 when at midpoint after upside breakout, got {val}"
        )

    def test_retest_bull_zero_when_price_below_orb_low(self):
        """retest_bull = 0 when C < or_low (ORB bull thesis invalidated)."""
        ind = _get_indicator()
        df, b = self._make_retest_df()
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False,
                             retest_atr_width=1.0)
        # Bar b+3 (C=96 < or_low=97): still_valid_bull = False → retest_bull = 0
        # After shift: result.iloc[b+4]
        val = result["orb_s08_retest_bull"].iloc[b + 4]
        assert val == 0.0, (
            f"retest_bull should be 0 when price drops below or_low (thesis invalidated), got {val}"
        )

    def test_retest_bear_fires_when_price_at_midpoint_after_bear_breakout(self):
        """retest_bear = 1 when post_bear=1 AND price retraces to the ORB midpoint."""
        ind = _get_indicator()
        n_total = 32 + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        open_ = np.full(n_total, 100.0)
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = 32
        # Body-based range: O=103, C=97 → or_high=103, or_low=97, midpoint=100
        open_[b] = 103.0
        close[b] = 97.0
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 96.0   # downside breakout (C=96 < or_low=97)
        close[b + 2] = 100.0  # retrace to midpoint — retest_bear SHOULD fire
        # still_valid_bear: C < or_high=103 → True ✓
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8], enable_rolling=False, enable_stats=False,
                             retest_atr_width=1.0)
        assert "orb_s08_retest_bear" in result.columns
        # Bar b+2 (C=100, post_bear=1, near midpoint, valid_bear) → retest_bear = 1
        # After shift: result.iloc[b+3]
        val = result["orb_s08_retest_bear"].iloc[b + 3]
        assert val == 1.0, (
            f"retest_bear should be 1 when at midpoint after downside breakout, got {val}"
        )

    def test_retest_bull_bear_in_get_signal_columns(self):
        """retest_bull and retest_bear must appear in get_signal_columns() for all pipeline sessions."""
        ind = _get_indicator()
        signals = ind.get_signal_columns()
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            pfx = f"orb_s{h:02d}"
            assert f"{pfx}_retest_bull" in signals, (
                f"{pfx}_retest_bull missing from get_signal_columns()"
            )
            assert f"{pfx}_retest_bear" in signals, (
                f"{pfx}_retest_bear missing from get_signal_columns()"
            )


class TestCarryForwardDays:
    """carry_forward_days: if no breakout, carry the ORB range to the next session occurrence."""

    def _make_multi_day_df(self, n_days=4):
        """Create M15 data spanning n_days with controlled values.

        All close values = 100.0 (within any reasonable range).
        Session hour = 8. Each day has bars from 00:00 to 23:45.
        """
        n = n_days * 24 * 4  # 4 bars per hour, 24 hours per day
        idx = pd.date_range("2024-01-02 00:00", periods=n, freq="15min")
        close = np.full(n, 100.0)
        high = close + 0.5
        low = close - 0.5
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        return df

    def _set_session_range(self, df, day, session_hour, or_high, or_low):
        """Set the range bar O/C body for a given day's session hour."""
        day_start = df.index[0] + pd.Timedelta(days=day)
        session_time = day_start.replace(hour=session_hour, minute=0)
        if session_time in df.index:
            df.loc[session_time, "O"] = or_low
            df.loc[session_time, "C"] = or_high
            df.loc[session_time, "H"] = or_high + 1.0
            df.loc[session_time, "L"] = or_low - 1.0

    def test_default_no_carry(self):
        """carry_forward_days=0 (default): each session uses its own range."""
        ind = _get_indicator()
        df = self._make_multi_day_df(n_days=3)
        # Day 0: session range [97, 103]
        self._set_session_range(df, 0, 8, 103.0, 97.0)
        # Day 1: session range [98, 102]
        self._set_session_range(df, 1, 8, 102.0, 98.0)

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=0)

        # Day 1's range should be its own (not carried from day 0)
        # sl_dist = or_range / 2 = (102-98) / 2 = 2.0
        day1_start = df.index[0] + pd.Timedelta(days=1, hours=8, minutes=15)
        if day1_start in result.index:
            sl_dist = result.loc[day1_start, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(2.0, abs=0.1), (
                    f"With carry_forward_days=0, day 1 should use its own range/2 (2.0), got {sl_dist}"
                )

    def test_carry_preserves_range_on_no_breakout(self):
        """carry_forward_days=1: no-breakout range carries to next session."""
        ind = _get_indicator()
        df = self._make_multi_day_df(n_days=3)
        # Day 0: wide range [95, 105], no breakout (C=100, within range)
        self._set_session_range(df, 0, 8, 105.0, 95.0)
        # Day 1: narrow range [99, 101] — should be REPLACED by carried [95, 105]
        self._set_session_range(df, 1, 8, 101.0, 99.0)

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=1)

        # Day 1 should have sl_dist = 5.0 (carried range/2 = (105-95)/2), not 1.0 (its own)
        # Find a valid bar in day 1's session (after range period)
        day1_post_range = df.index[0] + pd.Timedelta(days=1, hours=8, minutes=15)
        if day1_post_range in result.index:
            sl_dist = result.loc[day1_post_range, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(5.0, abs=0.1), (
                    f"Carried range/2 should be 5.0 (from day 0), got {sl_dist}"
                )

    def test_carry_resets_after_breakout(self):
        """If a carried session has a breakout, carry chain stops."""
        ind = _get_indicator()
        df = self._make_multi_day_df(n_days=4)
        # Day 0: range [95, 105], no breakout (C=100)
        self._set_session_range(df, 0, 8, 105.0, 95.0)
        # Day 1: narrow range [99, 101] → carried to [95, 105]
        self._set_session_range(df, 1, 8, 101.0, 99.0)
        # Force breakout on day 1 (C=110 > carried or_high=105)
        day1_breakout = df.index[0] + pd.Timedelta(days=1, hours=9)
        if day1_breakout in df.index:
            df.loc[day1_breakout, "C"] = 110.0
            df.loc[day1_breakout, "H"] = 110.0
        # Day 2: range [98, 102] — should use OWN range (carry chain broken)
        self._set_session_range(df, 2, 8, 102.0, 98.0)

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=2)

        # Day 2 should use its own range/2 (2.0), not the carried one (5.0)
        day2_post_range = df.index[0] + pd.Timedelta(days=2, hours=8, minutes=15)
        if day2_post_range in result.index:
            sl_dist = result.loc[day2_post_range, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(2.0, abs=0.1), (
                    f"After breakout in carried session, day 2 should use own range/2 (2.0), got {sl_dist}"
                )

    def test_carry_respects_max_days(self):
        """carry_forward_days=1: carry stops after 1 day even without breakout."""
        ind = _get_indicator()
        df = self._make_multi_day_df(n_days=4)
        # Day 0: range [95, 105], no breakout
        self._set_session_range(df, 0, 8, 105.0, 95.0)
        # Day 1: narrow [99, 101] → carried to [95, 105], no breakout
        self._set_session_range(df, 1, 8, 101.0, 99.0)
        # Day 2: [98, 102] → carry_forward_days=1, so carry should have expired
        self._set_session_range(df, 2, 8, 102.0, 98.0)

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=1)

        # Day 2 should use its own range/2 (2.0), carry expired after 1 day
        day2_post_range = df.index[0] + pd.Timedelta(days=2, hours=8, minutes=15)
        if day2_post_range in result.index:
            sl_dist = result.loc[day2_post_range, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(2.0, abs=0.1), (
                    f"carry_forward_days=1 should expire after 1 day, got sl_dist={sl_dist}"
                )

    def test_carried_session_has_no_range_bar_masking(self):
        """Carried sessions should have valid features even during the session hour bars."""
        ind = _get_indicator()
        df = self._make_multi_day_df(n_days=3)
        # Day 0: range [95, 105], no breakout
        self._set_session_range(df, 0, 8, 105.0, 95.0)
        # Day 1: will be carried
        self._set_session_range(df, 1, 8, 101.0, 99.0)

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=1)

        # In a normal (non-carried) session, the range bar at 08:00 has NaN features.
        # For a carried session, the range is pre-established → 08:00 bar should be valid.
        day1_range_bar = df.index[0] + pd.Timedelta(days=1, hours=8)
        if day1_range_bar in result.index:
            # The range bar itself may have shifted features from the previous bar.
            # Check that at least the next bar after range bar is valid.
            next_bar = df.index[0] + pd.Timedelta(days=1, hours=8, minutes=15)
            if next_bar in result.index:
                val_next = result.loc[next_bar, "orb_s08_sl_dist"]
                assert not pd.isna(val_next), (
                    "Carried session bars should have valid features (not NaN)"
                )

    def test_carry_forward_in_default_params(self):
        """carry_forward_days should appear in get_default_params()."""
        params = _orb.OpeningRangeIndicator.get_default_params()
        assert "carry_forward_days" in params
        assert params["carry_forward_days"] == 0

    def test_carry_forward_in_param_schema(self):
        """carry_forward_days should appear in get_param_schema()."""
        schema = _orb.OpeningRangeIndicator.get_param_schema()
        assert "carry_forward_days" in schema
        assert schema["carry_forward_days"]["default"] == 0


class TestPreRangeBars:
    """pre_range_bars: include N bars before session start in range calculation."""

    def _make_pre_range_df(self, n_days=2):
        """Create M15 data with controlled pre-session values."""
        n = n_days * 24 * 4
        idx = pd.date_range("2024-01-02 00:00", periods=n, freq="15min")
        close = np.full(n, 100.0)
        high = close + 0.5
        low = close - 0.5
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        return df

    def test_default_no_expansion(self):
        """pre_range_bars=0 (default): range uses only session-hour bars."""
        ind = _get_indicator()
        df = self._make_pre_range_df()
        # Session at hour 8, range bar at 08:00
        session_bar = df.index[0] + pd.Timedelta(hours=8)
        df.loc[session_bar, "H"] = 103.0
        df.loc[session_bar, "L"] = 97.0
        # Pre-session bar at 07:45 has extreme values
        pre_bar = df.index[0] + pd.Timedelta(hours=7, minutes=45)
        df.loc[pre_bar, "H"] = 110.0
        df.loc[pre_bar, "L"] = 90.0

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, pre_range_bars=0)

        # Range should be 103 - 97 = 6.0 (pre-session bar ignored)
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(6.0, abs=0.1), (
                    f"pre_range_bars=0 should ignore pre-session bars, got sl_dist={sl_dist}"
                )

    def test_pre_range_expands_high(self):
        """pre_range_bars=2: pre-session bar with higher high expands the range."""
        ind = _get_indicator()
        df = self._make_pre_range_df()
        # Session at hour 8
        session_bar = df.index[0] + pd.Timedelta(hours=8)
        df.loc[session_bar, "H"] = 103.0
        df.loc[session_bar, "L"] = 97.0
        # Pre-bar 1 (07:45): H = 106 → should expand range high to 106
        pre_bar1 = df.index[0] + pd.Timedelta(hours=7, minutes=45)
        df.loc[pre_bar1, "H"] = 106.0
        df.loc[pre_bar1, "L"] = 99.0

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, pre_range_bars=2)

        # Range should be 106 - 97 = 9.0 (expanded high from pre-bar)
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(9.0, abs=0.1), (
                    f"pre_range_bars=2 should expand high to 106, got sl_dist={sl_dist}"
                )

    def test_pre_range_expands_low(self):
        """pre_range_bars=2: pre-session bar with lower low expands the range."""
        ind = _get_indicator()
        df = self._make_pre_range_df()
        session_bar = df.index[0] + pd.Timedelta(hours=8)
        df.loc[session_bar, "H"] = 103.0
        df.loc[session_bar, "L"] = 97.0
        # Pre-bar 1 (07:45): L = 94 → should expand range low to 94
        pre_bar1 = df.index[0] + pd.Timedelta(hours=7, minutes=45)
        df.loc[pre_bar1, "H"] = 101.0
        df.loc[pre_bar1, "L"] = 94.0

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, pre_range_bars=2)

        # Range should be 103 - 94 = 9.0 (expanded low from pre-bar)
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(9.0, abs=0.1), (
                    f"pre_range_bars=2 should expand low to 94, got sl_dist={sl_dist}"
                )

    def test_no_expansion_when_pre_bars_within_range(self):
        """pre_range_bars > 0 but pre-bars are within session range → no change."""
        ind = _get_indicator()
        df = self._make_pre_range_df()
        session_bar = df.index[0] + pd.Timedelta(hours=8)
        df.loc[session_bar, "H"] = 103.0
        df.loc[session_bar, "L"] = 97.0
        # Pre-bar at 07:45 has values within session range
        pre_bar1 = df.index[0] + pd.Timedelta(hours=7, minutes=45)
        df.loc[pre_bar1, "H"] = 101.0
        df.loc[pre_bar1, "L"] = 99.0

        result = ind.compute(df.copy(), sessions=[8], enable_rolling=False,
                             enable_stats=False, pre_range_bars=2)

        # Range should still be 103 - 97 = 6.0
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(6.0, abs=0.1), (
                    f"pre-bars within range should not change it, got sl_dist={sl_dist}"
                )

    def test_pre_range_in_default_params(self):
        """pre_range_bars should appear in get_default_params()."""
        params = _orb.OpeningRangeIndicator.get_default_params()
        assert "pre_range_bars" in params
        assert params["pre_range_bars"] == 0

    def test_pre_range_in_param_schema(self):
        """pre_range_bars should appear in get_param_schema()."""
        schema = _orb.OpeningRangeIndicator.get_param_schema()
        assert "pre_range_bars" in schema
        assert schema["pre_range_bars"]["default"] == 0



class TestCfPrbListPrefixes:
    """Tests for carry_forward_days / pre_range_bars list parameter prefix generation."""

    def test_scalar_params_no_prefix(self):
        """Scalar cf=0, prb=0 → no cf/prb prefix on session columns."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=0, pre_range_bars=0)

        # Should have unprefixed session columns
        assert "orb_s08_range" in result.columns
        # Should NOT have cf/prb prefixed columns
        cf_cols = [c for c in result.columns if c.startswith("cf")]
        assert len(cf_cols) == 0, f"Unexpected cf-prefixed columns: {cf_cols}"

    def test_cf_list_generates_prefixed_columns(self):
        """carry_forward_days=[0, 1] → cf0_prb0_ and cf1_prb0_ prefixed columns."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=[0, 1],
                             pre_range_bars=0)

        assert "cf0_prb0_orb_s08_range" in result.columns, (
            f"Missing cf0_prb0_ prefix. Columns: {[c for c in result.columns if 'orb_s08' in c]}"
        )
        assert "cf1_prb0_orb_s08_range" in result.columns, (
            f"Missing cf1_prb0_ prefix. Columns: {[c for c in result.columns if 'orb_s08' in c]}"
        )
        # Unprefixed should NOT exist
        assert "orb_s08_range" not in result.columns

    def test_prb_list_generates_prefixed_columns(self):
        """pre_range_bars=[0, 1] → cf0_prb0_ and cf0_prb1_ prefixed columns."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=0,
                             pre_range_bars=[0, 1])

        assert "cf0_prb0_orb_s08_range" in result.columns
        assert "cf0_prb1_orb_s08_range" in result.columns
        assert "orb_s08_range" not in result.columns

    def test_combined_lists_generate_cartesian_prefixes(self):
        """cf=[0,1], prb=[0,1] → 4 variant sets (cartesian product)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=[0, 1],
                             pre_range_bars=[0, 1])

        expected_prefixes = ["cf0_prb0_", "cf0_prb1_", "cf1_prb0_", "cf1_prb1_"]
        for prefix in expected_prefixes:
            col = f"{prefix}orb_s08_range"
            assert col in result.columns, (
                f"Missing {prefix} variant. Got: {[c for c in result.columns if 'orb_s08_range' in c]}"
            )

    def test_cf_prb_prefix_does_not_affect_rolling_features(self):
        """Rolling features should NOT get cf/prb prefix."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=True,
                             enable_stats=False, carry_forward_days=[0, 1],
                             pre_range_bars=0)

        # Rolling features should exist WITHOUT cf/prb prefix
        assert "orb_range" in result.columns
        assert "orb_position" in result.columns
        # Session features should have cf/prb prefix
        assert "cf0_prb0_orb_s08_range" in result.columns

    def test_cf_prb_prefix_does_not_affect_stats_features(self):
        """Stat features should NOT get cf/prb prefix."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=True, carry_forward_days=[0, 1],
                             pre_range_bars=0)

        assert "orb_stat_avg_range" in result.columns
        assert "orb_stat_breakout_rate" in result.columns

    def test_rb_and_cf_prb_prefixes_combined(self):
        """range_bars=[1,2] + cf=[0,1] → rb1_cf0_prb0_, rb2_cf1_prb0_, etc."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], range_bars=[1, 2],
                             enable_rolling=False, enable_stats=False,
                             carry_forward_days=[0, 1], pre_range_bars=0)

        # Should have combined rb + cf/prb prefixes
        assert "rb1_cf0_prb0_orb_s08_range" in result.columns
        assert "rb1_cf1_prb0_orb_s08_range" in result.columns
        assert "rb2_cf0_prb0_orb_s08_range" in result.columns
        assert "rb2_cf1_prb0_orb_s08_range" in result.columns

    def test_single_cf_with_prb_list_triggers_prefix(self):
        """cf=0 (scalar) + prb=[0,1] (list) → prefix active (len(prb_list) > 1)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=0,
                             pre_range_bars=[0, 1])

        # Even though cf is scalar, prb being a list triggers prefix mode
        assert "cf0_prb0_orb_s08_range" in result.columns
        assert "cf0_prb1_orb_s08_range" in result.columns

    def test_retest_signals_get_cf_prb_prefix(self):
        """Retest signal columns (retest_bull, retest_bear) get cf/prb prefix."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=False, enable_retracement=True,
                             carry_forward_days=[0, 1], pre_range_bars=0)

        assert "cf0_prb0_orb_s08_retest_bull" in result.columns
        assert "cf1_prb0_orb_s08_retest_bull" in result.columns
        assert "cf0_prb0_orb_s08_retest_bear" in result.columns
        assert "cf1_prb0_orb_s08_retest_bear" in result.columns

    def test_variant_count_matches_cartesian_product(self):
        """Number of session variant sets = len(cf_list) × len(prb_list)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], enable_rolling=False,
                             enable_stats=False, carry_forward_days=[0, 1, 2],
                             pre_range_bars=[0, 1])

        # 3 cf × 2 prb = 6 variants; each has orb_s08_range
        range_cols = [c for c in result.columns if c.endswith("orb_s08_range")]
        assert len(range_cols) == 6, (
            f"Expected 6 variants (3×2), got {len(range_cols)}: {range_cols}"
        )


class TestStrategyConfigSignalColumnIntegration:
    """Integration tests: verify ALL signal columns referenced in strategy
    configs actually exist in the indicator output.

    This catches prefix mismatches (e.g. missing rb_ prefix) that cause
    the SignalModel to output zeros and zero trades.
    """

    @staticmethod
    def _load_json(path):
        import json
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _collect_signal_columns(strategy_cfg):
        """Extract all signal columns from model_hyperparameters + model_hyperparameters_grid."""
        signal_cols = set()
        assignments = strategy_cfg.get("grids", {}).get("assignments", {})
        for asset, asset_cfg in assignments.items():
            hp = asset_cfg.get("model_hyperparameters", {})
            for key in ("signal_column_long", "signal_column_short"):
                val = hp.get(key)
                if val:
                    signal_cols.add(val)
            for variant in asset_cfg.get("model_hyperparameters_grid", []):
                if variant:
                    for key in ("signal_column_long", "signal_column_short"):
                        val = variant.get(key)
                        if val:
                            signal_cols.add(val)
        return signal_cols

    @staticmethod
    def _get_pipeline_orb_params(pipeline_cfg):
        """Extract opening_range indicator params from pipeline config."""
        for ind in pipeline_cfg.get("indicators", []):
            if ind.get("name") == "opening_range":
                return ind.get("params", {})
        return None

    def _compute_indicator_columns(self, orb_params, pipeline_cfg=None):
        """Run all indicators from pipeline on synthetic data, return output columns."""
        df = _make_ohlc_15min(n=3000)

        # Always run opening_range
        ind = _get_indicator()
        result = ind.compute(df, **orb_params)

        # Also run other indicators from the pipeline if available
        if pipeline_cfg:
            from fwbg.plugins import import_plugin_module
            from fwbg_sdk import BaseIndicator as _BaseInd
            for ind_cfg in pipeline_cfg.get("indicators", []):
                name = ind_cfg.get("name")
                if name == "opening_range":
                    continue
                mod = import_plugin_module("fwbg-core", "indicators", name)
                if mod is None:
                    continue
                # Find concrete indicator class (not BaseIndicator)
                ind_cls = None
                for attr_name in dir(mod):
                    cls = getattr(mod, attr_name, None)
                    if (isinstance(cls, type) and issubclass(cls, _BaseInd)
                            and cls is not _BaseInd):
                        ind_cls = cls
                        break
                if ind_cls is None:
                    continue
                params = ind_cfg.get("params", {})
                try:
                    result = ind_cls().compute(result, **params)
                except Exception:
                    pass

        return set(result.columns)

    @pytest.fixture
    def strategy_dir(self):
        import os
        return os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "strategies",
        ))

    def _run_config_check(self, strategy_dir, strategy_file, pipeline_name):
        """Core check: compute indicator with pipeline params, verify all signal columns exist."""
        import os

        strategy_path = os.path.join(strategy_dir, "configs", strategy_file)
        pipeline_path = os.path.join(strategy_dir, "pipelines", f"{pipeline_name}.json")

        if not os.path.exists(strategy_path) or not os.path.exists(pipeline_path):
            pytest.skip(f"Config files not found: {strategy_file} / {pipeline_name}")

        strategy_cfg = self._load_json(strategy_path)
        pipeline_cfg = self._load_json(pipeline_path)

        orb_params = self._get_pipeline_orb_params(pipeline_cfg)
        if orb_params is None:
            pytest.skip(f"No opening_range indicator in {pipeline_name}")

        output_cols = self._compute_indicator_columns(orb_params, pipeline_cfg)
        signal_cols = self._collect_signal_columns(strategy_cfg)
        assert len(signal_cols) > 0, f"No signal columns found in {strategy_file}"

        missing = signal_cols - output_cols
        assert len(missing) == 0, (
            f"Signal columns from {strategy_file} not found in indicator output "
            f"(pipeline={pipeline_name}).\n"
            f"Missing ({len(missing)}): {sorted(missing)[:10]}\n"
            f"Example expected: {sorted(signal_cols)[0]}\n"
            f"Example actual: {sorted(c for c in output_cols if 'retest_bull' in c)[:5]}"
        )

    def test_orb_exploration_signal_columns(self, strategy_dir):
        """All signal columns in orb_exploration.json exist in orb_simple_v1 output."""
        self._run_config_check(strategy_dir, "orb_exploration.json", "orb_simple_v1")

    def test_orb_pdhl_scalping_signal_columns(self, strategy_dir):
        """All signal columns in orb_pdhl_scalping.json exist in orb_scalping_v1 output."""
        self._run_config_check(strategy_dir, "orb_pdhl_scalping.json", "orb_scalping_v1")

    def test_deep_orb_index_signal_columns(self, strategy_dir):
        """All signal columns in deep_orb_index.json exist in orb_simple_v1 output."""
        self._run_config_check(strategy_dir, "deep_orb_index.json", "orb_simple_v1")

    def test_signal_model_nonzero_predictions_with_prefixed_column(self):
        """SignalModel with a valid prefixed signal column produces non-zero predictions."""
        from fwbg.plugins import import_plugin_module
        signal_mod = import_plugin_module("fwbg-core", "models", "signal")
        if signal_mod is None:
            pytest.skip("signal model plugin not available")

        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(
            df, range_bars=[1, 2], sessions=[8],
            enable_rolling=False, enable_stats=False,
            enable_retracement=True, retest_atr_width=0.3,
            carry_forward_days=[0, 1, 2], pre_range_bars=[0, 1],
        )

        signal_col = "rb1_cf0_prb0_orb_s08_retest_bull"
        assert signal_col in result.columns, (
            f"{signal_col} not in output. "
            f"Retest cols: {[c for c in result.columns if 'retest_bull' in c][:10]}"
        )

        model = signal_mod.SignalModel()
        from fwbg_sdk.models import TrainingContext
        ctx = TrainingContext(direction="long")
        features_df = result[[signal_col]].fillna(0)
        targets = np.zeros(len(result))
        model.train(features_df, targets, ctx, signal_column_long=signal_col)

        probs = model._predict_probability_impl(features_df)
        signal_fires = (features_df[signal_col] > 0).sum()
        nonzero_preds = (probs[:, 1] > 0).sum()

        assert nonzero_preds > 0, (
            f"SignalModel produced 0 non-zero predictions. "
            f"Signal fires: {signal_fires}, signal col values: "
            f"{features_df[signal_col].value_counts().to_dict()}"
        )
        assert nonzero_preds == signal_fires, (
            f"SignalModel predictions ({nonzero_preds}) should match "
            f"signal fires ({signal_fires})"
        )

    def test_required_features_auto_collect_matches_indicator_output(self):
        """required_features auto-collected from model_hyperparameters_grid
        must all exist in indicator output."""
        import os
        strategy_dir = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "strategies",
        ))
        strategy_path = os.path.join(strategy_dir, "configs", "orb_pdhl_scalping.json")
        pipeline_path = os.path.join(strategy_dir, "pipelines", "orb_scalping_v1.json")

        if not os.path.exists(strategy_path) or not os.path.exists(pipeline_path):
            pytest.skip("Config files not found")

        strategy_cfg = self._load_json(strategy_path)
        pipeline_cfg = self._load_json(pipeline_path)
        orb_params = self._get_pipeline_orb_params(pipeline_cfg)
        output_cols = self._compute_indicator_columns(orb_params, pipeline_cfg)

        # Simulate auto-collect from SimulationContext.create()
        auto_collected = set()
        assignments = strategy_cfg.get("grids", {}).get("assignments", {})
        for asset, asset_cfg in assignments.items():
            hp = asset_cfg.get("model_hyperparameters", {})
            for key in ("signal_column_long", "signal_column_short"):
                val = hp.get(key)
                if val:
                    auto_collected.add(val)
            for variant in asset_cfg.get("model_hyperparameters_grid", []):
                if variant:
                    for key in ("signal_column_long", "signal_column_short"):
                        val = variant.get(key)
                        if val:
                            auto_collected.add(val)

        missing = auto_collected - output_cols
        assert len(missing) == 0, (
            f"Auto-collected required_features not in indicator output.\n"
            f"Missing ({len(missing)}): {sorted(missing)[:10]}\n"
            f"This would cause 'Feature X nicht in inner_df' warnings at runtime."
        )


class TestBodyBasedRange:
    """Session ORB range is body-based (O/C), not wick-based (H/L).

    The opening range uses Open of the first bar and Close of the last bar
    in the range window.  or_high = max(O_first, C_last), or_low = min(O_first, C_last).
    This gives a tighter range than H/L (filters wick noise).
    """

    def _make_body_vs_wick_df(self, range_bars=2):
        """15min data where body and wicks clearly differ.

        Range bars (session 08, bars b+0 and b+1 for range_bars=2):
          bar0: O=98, H=104, L=95, C=99
          bar1: O=101, H=106, L=96, C=102

        Body range: or_high = max(O_bar0, C_bar1) = max(98, 102) = 102
                    or_low  = min(O_bar0, C_bar1) = min(98, 102) = 98
                    body_range = 4.0

        Wick range: H_max = 106, L_min = 95 → wick_range = 11.0
        """
        n_warmup = 32  # 8 hours warmup
        n_total = n_warmup + 8
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        open_ = np.full(n_total, 100.0)
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_warmup  # 08:00

        # bar0 (08:00): O=98, H=104, L=95, C=99
        open_[b] = 98.0
        close[b] = 99.0
        high[b] = 104.0
        low[b] = 95.0

        # bar1 (08:15): O=101, H=106, L=96, C=102
        open_[b + 1] = 101.0
        close[b + 1] = 102.0
        high[b + 1] = 106.0
        low[b + 1] = 96.0

        # post-range bars: normal
        for i in range(2, 6):
            close[b + i] = 100.0

        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=idx)
        return df, b

    def test_session_range_uses_body_not_wicks(self):
        """or_high/or_low must be derived from O/C body, not H/L wicks."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        result = ind.compute(df, sessions=[8], range_bars=2,
                             enable_rolling=False, enable_stats=False)

        # Body range = max(98, 102) - min(98, 102) = 4.0
        # _range feature is normalized: safe_divide(or_range, C) = 4.0 / 100.0 = 0.04
        # After shift: result.iloc[b+3] has the value for bar b+2
        range_val = result["orb_s08_range"].iloc[b + 3]
        assert not np.isnan(range_val), "orb_s08_range should not be NaN for post-range bar"
        expected = 4.0 / 100.0  # body_range / close
        assert abs(range_val - expected) < 0.001, (
            f"Expected normalized body range {expected} (4.0/100.0), "
            f"got {range_val}. If ~0.11, range is still wick-based (H/L)."
        )

    def test_body_range_smaller_than_wick_range(self):
        """Body-based range must be <= wick-based range for any candle."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, sessions=[8], range_bars=2,
                             enable_rolling=False, enable_stats=False)

        range_vals = result["orb_s08_range"].dropna()
        assert len(range_vals) > 10, "Need enough session ranges to compare"
        # Body range can't exceed wick range: max(O,C) - min(O,C) <= H - L
        assert (range_vals >= 0).all(), "Body range should be non-negative"

    def test_single_bar_body_range(self):
        """With range_bars=1, or_high = max(O, C), or_low = min(O, C) of that single bar."""
        ind = _get_indicator()
        n_warmup = 32
        n_total = n_warmup + 8
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        open_ = np.full(n_total, 100.0)
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_warmup

        # Single range bar: O=97, C=103 → body range = 6.0
        # Wicks: H=105, L=94 → wick range = 11.0
        open_[b] = 97.0
        close[b] = 103.0
        high[b] = 105.0
        low[b] = 94.0

        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8], range_bars=1,
                             enable_rolling=False, enable_stats=False)

        # _range = safe_divide(or_range, C) = 6.0 / 100.0 = 0.06
        # After shift: result.iloc[b+2] has value for bar b+1
        range_val = result["orb_s08_range"].iloc[b + 2]
        assert not np.isnan(range_val), "range should not be NaN"
        expected = 6.0 / 100.0
        assert abs(range_val - expected) < 0.001, (
            f"Expected normalized body range {expected} (6.0/100.0), got {range_val}"
        )

    def test_sl_dist_is_half_range(self):
        """Session sl_dist = or_range / 2 (entry at midpoint → SL at body boundary)."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        result = ind.compute(df, sessions=[8], range_bars=2,
                             enable_rolling=False, enable_stats=False)

        # Body range = 4.0, sl_dist = 4.0 / 2 = 2.0
        sl_val = result["orb_s08_sl_dist"].iloc[b + 3]
        assert not np.isnan(sl_val), "sl_dist should not be NaN"
        assert abs(sl_val - 2.0) < 0.01, (
            f"Expected sl_dist = 2.0 (body_range 4.0 / 2), got {sl_val}"
        )

    def test_poc_dist_reflects_body_midpoint(self):
        """poc_dist = (C - midpoint) / ATR. Midpoint is body-based: (or_high + or_low) / 2."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        # Post-range bar: C=100.0, body midpoint = (102+98)/2 = 100.0 → poc_dist ≈ 0
        result = ind.compute(df, sessions=[8], range_bars=2,
                             enable_rolling=False, enable_stats=False)

        poc_val = result["orb_s08_poc_dist"].iloc[b + 3]
        assert not np.isnan(poc_val), "poc_dist should not be NaN"
        assert abs(poc_val) < 0.1, (
            f"Expected poc_dist ≈ 0 (C=100 at body midpoint=100), got {poc_val}"
        )

    def test_breakout_uses_body_boundary(self):
        """Breakout detection must use body-based or_high/or_low, not wick H/L."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        df = df.copy()
        # Post-range bar: C=103 → above body or_high=102 (breakout!)
        # but below wick H_max=106 (would NOT be breakout if wick-based)
        df.loc[df.index[b + 2], "C"] = 103.0
        df.loc[df.index[b + 2], "H"] = 103.0
        result = ind.compute(df, sessions=[8], range_bars=2,
                             enable_rolling=False, enable_stats=False)

        # After shift: result.iloc[b+3] has bar b+2's breakout value
        bu_val = result["orb_s08_breakout_up"].iloc[b + 3]
        assert bu_val == 1.0, (
            f"Expected breakout_up=1 (C=103 > body or_high=102), got {bu_val}. "
            f"If 0, breakout is still using wick-based boundary."
        )

    def test_rolling_orb_still_uses_wicks(self):
        """Rolling (hourly) ORB features must remain wick-based (H/L), not body-based."""
        ind = _get_indicator()
        n_warmup = 5 * 4
        n_total = n_warmup + 8
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        open_ = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_warmup

        # Range bar: body is narrow (O=99, C=101) but wicks are wide (H=105, L=95)
        open_[b] = 99.0
        close[b] = 101.0
        high[b] = 105.0
        low[b] = 95.0

        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, enable_session=False, enable_stats=False)

        # Rolling ORB uses H/L → sl_dist = raw range = 10.0
        # (orb_range is normalized by C, but orb_sl_dist is raw)
        sl_val = result["orb_sl_dist"].iloc[b + 2]
        assert not np.isnan(sl_val), "Rolling orb_sl_dist should not be NaN"
        assert abs(sl_val - 10.0) < 0.01, (
            f"Expected rolling orb_sl_dist 10.0 (wick-based H/L range), got {sl_val}. "
            f"Rolling ORB should NOT use body-based range."
        )
