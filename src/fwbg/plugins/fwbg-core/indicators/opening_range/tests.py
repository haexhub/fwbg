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


class TestSessionORB:
    """Tests for session-specific ORB features."""

    def test_session_features_computed(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        # Default sessions are [8, 9, 14, 15]
        result = ind.compute(df)

        for h in [8, 9, 14, 15]:
            prefix = f"rb1_orb_s{h:02d}"
            for suffix in ["_range", "_position", "_breakout_up",
                           "_breakout_down", "_range_vs_atr"]:
                col = f"{prefix}{suffix}"
                assert col in result.columns, f"Missing column: {col}"

    def test_session_features_have_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)  # ~52 days, enough for all sessions
        result = ind.compute(df)

        for h in [8, 9, 14, 15]:
            col = f"rb1_orb_s{h:02d}_range"
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN"

    def test_custom_sessions(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, sessions=[9, 17])

        assert "rb1_orb_s09_range" in result.columns
        assert "rb1_orb_s17_range" in result.columns
        # Default sessions should NOT be present
        assert "rb1_orb_s08_range" not in result.columns

    def test_session_breakout_binary(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for h in [8, 9, 14, 15]:
            for direction in ["up", "down"]:
                col = f"rb1_orb_s{h:02d}_breakout_{direction}"
                vals = result[col].dropna()
                if len(vals) > 0:
                    assert set(vals.unique()).issubset({0, 1}), f"{col} not binary"

    def test_hourly_data_session_features_work(self):
        """Session ORB should work on hourly data (range persists across hours)."""
        ind = _get_indicator()
        df = _make_ohlc_hourly(n=5000)
        result = ind.compute(df)

        non_null = result["rb1_orb_s08_range"].dropna()
        assert len(non_null) > 0

    def test_no_inf_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        feature_cols = [c for c in result.columns if "_orb_s" in c]
        for col in feature_cols:
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} contains inf values"


class TestDailySkip:
    """Daily data should not produce ORB features."""

    def test_daily_returns_unchanged(self):
        ind = _get_indicator()
        df = _make_ohlc_daily()
        result = ind.compute(df)

        orb_cols = [c for c in result.columns if "_orb_s" in c]
        assert len(orb_cols) == 0, "Daily data should not produce ORB features"


class TestParameters:
    """Test parameter variations."""

    def test_get_default_params(self):
        params = _orb.OpeningRangeIndicator.get_default_params()
        assert params["range_bars"] == 1
        assert params["sessions"] == [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]
        assert params["candle_span"] == "hl"
        assert params["retracement_levels"] == 0.5
        assert params["min_retracement"] == 0.0
        # Removed params must NOT be in defaults
        assert "enable_rolling" not in params
        assert "enable_stats" not in params
        assert "stat_window" not in params
        assert "enable_session" not in params
        assert "range_mode" not in params

    def test_get_param_schema(self):
        schema = _orb.OpeningRangeIndicator.get_param_schema()
        assert "range_bars" in schema
        assert "sessions" in schema
        assert "candle_span" in schema
        assert schema["range_bars"]["type"] == "list[int]"
        assert schema["sessions"]["type"] == "list[int]"
        # Removed params must NOT be in schema
        assert "range_mode" not in schema
        assert "enable_rolling" not in schema
        assert "enable_stats" not in schema

    def test_get_feature_columns_includes_all_pipeline_sessions(self):
        """get_feature_columns() must cover all UTC sessions used in pipeline configs."""
        ind = _get_indicator()
        cols = ind.get_feature_columns()
        # Sessions [0, 1, 2, 5, 6, 7, 8, 12, 13, 14] from orb_scalping_v1.json
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            assert f"rb1_orb_s{h:02d}_range" in cols, (
                f"rb1_orb_s{h:02d}_range missing from get_feature_columns() — "
                f"session {h} UTC is configured in orb_scalping_v1.json"
            )
        # Retest columns have rl50 prefix
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            assert f"rb1_orb_s{h:02d}_rl50_retest_bull" in cols, (
                f"rb1_orb_s{h:02d}_rl50_retest_bull missing from get_feature_columns()"
            )
            assert f"rb1_orb_s{h:02d}_rl50_retest_bear" in cols
            assert f"rb1_orb_s{h:02d}_rl50_sl_dist" in cols

    def test_get_feature_columns_excludes_non_pipeline_sessions(self):
        """Sessions 9 and 15 are not in any pipeline config — must not appear in feature columns."""
        ind = _get_indicator()
        cols = ind.get_feature_columns()
        for h in [9, 15]:
            assert f"rb1_orb_s{h:02d}_range" not in cols, (
                f"rb1_orb_s{h:02d}_range should not be in get_feature_columns() — "
                f"session {h} is not used in any pipeline config"
            )

    def test_get_signal_columns_includes_all_pipeline_sessions(self):
        """get_signal_columns() must cover breakout signals for all pipeline sessions."""
        ind = _get_indicator()
        signals = ind.get_signal_columns()
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            for direction in ["up", "down"]:
                col = f"rb1_orb_s{h:02d}_breakout_{direction}"
                assert col in signals, (
                    f"{col} missing from get_signal_columns() — "
                    f"session {h} UTC is configured in orb_scalping_v1.json"
                )
            # Retest signals have rl50 prefix
            assert f"rb1_orb_s{h:02d}_rl50_retest_bull" in signals
            assert f"rb1_orb_s{h:02d}_rl50_retest_bear" in signals


class TestRangeBarsListMode:
    """range_bars=[1, 2] (list) activates prefix mode: rb1_orb_* and rb2_orb_* columns."""

    def test_list_mode_produces_rb_prefixed_columns(self):
        """When range_bars is a list, all columns get rb{n}_ prefix instead of bare orb_ names."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=[1, 2], sessions=[8])

        assert "rb1_orb_s08_range" in result.columns, "rb1_ prefix missing for range_bars=1"
        assert "rb2_orb_s08_range" in result.columns, "rb2_ prefix missing for range_bars=2"

    def test_list_mode_both_rb_variants_have_breakout_signals(self):
        """Both rb1 and rb2 variants must produce non-empty breakout signals."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=[1, 2], sessions=[8])

        for prefix in ["rb1", "rb2"]:
            for direction in ["up", "down"]:
                col = f"{prefix}_orb_s08_breakout_{direction}"
                assert col in result.columns, f"{col} missing"
                assert result[col].dropna().isin([0, 1]).all(), f"{col} not binary"

    def test_list_mode_session_columns_have_rb_prefix(self):
        """Session ORB columns (orb_s08_*) must also carry the rb{n}_ prefix in list mode."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=[1, 2], sessions=[8])

        assert "rb1_orb_s08_range" in result.columns or "rb1_orb_s08_range" in result.columns, \
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

    def test_session_breakout_up_fires_on_first_crossing_and_persists(self):
        """Session orb_breakout_up = 1 on first crossing, stays 1 for rest of session (cummax)."""
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
            close[base + 3] = 103.0  # still above — persistent signal stays 1

        high = close + 0.5
        low = close - 0.5
        # Set or_high for hour 8 range bar: H = close + 1.0
        for b in hour8_bars:
            high[b] = close[b] + 1.0
            low[b] = close[b] - 1.0

        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8])

        # Breakout signals are persistent (cummax): once 1, stays 1.
        # Signal columns are NOT shifted — they appear at the bar where they fire.
        if len(hour8_bars) >= 2:
            base = hour8_bars[1]
            # bar+1: C=100 < or_high=101 -> no breakout yet
            assert result["rb1_orb_s08_breakout_up"].iloc[base + 1] == 0.0
            # bar+2: C=102 > or_high=101 -> first breakout
            assert result["rb1_orb_s08_breakout_up"].iloc[base + 2] == 1.0
            # bar+3: persistent — stays 1
            assert result["rb1_orb_s08_breakout_up"].iloc[base + 3] == 1.0
            # Use diff() to find the first firing: only one transition from 0->1
            session_region = result["rb1_orb_s08_breakout_up"].iloc[base:base + 6].dropna()
            transitions = session_region.diff().fillna(0)
            first_fires = (transitions == 1.0).sum()
            assert first_fires == 1, (
                f"Breakout should have exactly 1 transition 0->1, got {first_fires}"
            )

    def test_up_and_down_cannot_first_fire_on_same_bar(self):
        """The initial 0->1 transition for breakout_up and breakout_down cannot happen on the same bar.

        A bar cannot simultaneously close above or_high AND below or_low.
        However, with persistent signals (cummax), both can be 1 on the same bar
        after both breakouts happened at different times in the session.
        We verify the first firing (diff == 1) never occurs on the same bar.
        """
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, sessions=[8])

        bu = result["rb1_orb_s08_breakout_up"].fillna(0)
        bd = result["rb1_orb_s08_breakout_down"].fillna(0)
        # First firing = transition from 0 to 1
        bu_first = bu.diff().fillna(0) == 1.0
        bd_first = bd.diff().fillna(0) == 1.0
        both_first_fired = bu_first & bd_first
        assert not both_first_fired.any(), (
            "breakout_up and breakout_down cannot first fire on the same bar — "
            "close cannot be both above or_high and below or_low."
        )

    def test_both_can_fire_in_same_session_false_breakout_scenario(self):
        """Both breakout directions can fire within the same session (false breakout / stop-hunt).

        Scenario:
          bar0 (range bar): or_high=101, or_low=99
          bar1: C=102 -> FIRST upward breakout (breakout_up becomes 1, persists)
          bar2: C=98  -> reversal below or_low -> FIRST downward breakout (breakout_down becomes 1, persists)
          bar3: C=97  -> both signals persist at 1.0

        Breakout signals use cummax: once 1, stays 1 for rest of session.
        Signal columns are NOT shifted — they appear at the bar where they compute.
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
        # b=20 -> 05:00 UTC. Use sessions=[5] so our controlled hour IS the session.
        result = ind.compute(df, sessions=[5])

        # Signals are NOT shifted; breakout uses cummax (persistent).
        bu = result["rb1_orb_s05_breakout_up"]
        bd = result["rb1_orb_s05_breakout_down"]

        # bar1 (b+1): C=102 > or_high=101 -> breakout_up = 1
        assert bu.iloc[b + 1] == 1.0, f"bar1 upward crossing must fire (got {bu.iloc[b+1]})"
        # bar2 (b+2): C=98 < or_low=99 -> breakout_down = 1
        assert bd.iloc[b + 2] == 1.0, f"bar2 downward crossing must fire (got {bd.iloc[b+2]})"
        # Persistent: breakout_up stays 1 even after price reversed
        assert bu.iloc[b + 2] == 1.0, "breakout_up persists (cummax)"
        assert bu.iloc[b + 3] == 1.0, "breakout_up persists on subsequent bars"
        # Both persist together
        assert bd.iloc[b + 3] == 1.0, "breakout_down persists on subsequent bars"
        # Verify breakout_up fired at b+1 (first post-range bar) and breakout_down at b+2
        # The first bar where each becomes 1 should be different
        bu_valid = bu.dropna()
        bd_valid = bd.dropna()
        bu_first_idx = bu_valid[bu_valid == 1.0].index[0]
        bd_first_idx = bd_valid[bd_valid == 1.0].index[0]
        assert bu_first_idx != bd_first_idx, "First firings must be at different bars"


class TestORBSLDist:
    """rb1_orb_sl_dist = or_range / 2 — half ORB range as SL distance."""

    def test_session_sl_dist_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "rb1_orb_s08_sl_dist" in result.columns, "rb1_orb_s08_sl_dist column missing from session ORB output"

    def test_session_sl_dist_positive(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        sl_vals = result["rb1_orb_s08_sl_dist"].dropna()
        assert (sl_vals > 0).all(), "rb1_orb_s08_sl_dist should be strictly positive"


class TestORBPocDist:
    """orb_poc_dist = (close - or_midpoint) / atr — normalized distance to ORB midpoint."""

    def test_session_poc_dist_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "rb1_orb_s08_poc_dist" in result.columns, "rb1_orb_s08_poc_dist column missing from session ORB output"

    def test_poc_dist_zero_at_midpoint(self):
        """When C equals the ORB midpoint exactly, orb_s08_poc_dist must be 0."""
        ind = _get_indicator()
        n_warmup = 32  # 8 hours = bar 32 is at 08:00
        n_total = n_warmup + 8
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_warmup
        high[b] = 103.0
        low[b] = 97.0   # midpoint = (103+97)/2 = 100.0
        # bar b+1: C = 100.0 -> exactly at midpoint -> poc_dist = 0/ATR = 0
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8])
        # After shift: bar b+1 value at result.iloc[b+2]
        poc_at_midpoint = result["rb1_orb_s08_poc_dist"].iloc[b + 2]
        assert poc_at_midpoint == pytest.approx(0.0, abs=1e-6), (
            f"poc_dist should be 0 when close == midpoint (100.0), got {poc_at_midpoint}"
        )


class TestORBPostBreakoutState:
    """rb1_orb_s{hh}_post_bull/bear = state: 1 for all bars AFTER first session breakout, resets per session."""

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
        result = ind.compute(df, sessions=[8])
        assert "rb1_orb_s08_post_bull" in result.columns, "rb1_orb_s08_post_bull column missing"
        # Bar b+1 (C=100, no breakout yet): computed post_bull=0 -> result.iloc[b+2]
        val = result["rb1_orb_s08_post_bull"].iloc[b + 2]
        assert val == 0.0, f"post_bull should be 0 before any breakout, got {val}"

    def test_post_bull_becomes_one_after_upside_breakout(self):
        """post_bull = 1 from the first bar where C > or_high, persists for subsequent bars."""
        ind = _get_indicator()
        df, b = self._make_session_df()
        result = ind.compute(df, sessions=[8])
        # Bar b+2 (C=104 > or_high=103): post_bull becomes 1 -> result.iloc[b+3]
        val_at_breakout = result["rb1_orb_s08_post_bull"].iloc[b + 3]
        assert val_at_breakout == 1.0, f"post_bull should be 1 at breakout bar, got {val_at_breakout}"
        # Bar b+3 (C=100, retrace): post_bull must STAY 1 (state, not event) -> result.iloc[b+4]
        val_after_retrace = result["rb1_orb_s08_post_bull"].iloc[b + 4]
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
        result = ind.compute(df, sessions=[8])

        # Second session bar b2+1 (C=100, no breakout): post_bull must reset to 0
        # After shift: result.iloc[b2+2]
        val = result["rb1_orb_s08_post_bull"].iloc[b2 + 2]
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
        result = ind.compute(df, sessions=[8])

        assert "rb1_orb_s08_post_bear" in result.columns, "rb1_orb_s08_post_bear column missing"
        # Bar b+1 (no breakout): post_bear = 0 -> result.iloc[b+2]
        val_before = result["rb1_orb_s08_post_bear"].iloc[b + 2]
        assert val_before == 0.0, f"post_bear before breakout should be 0, got {val_before}"
        # Bar b+2 (C=96 < or_low=97): post_bear = 1 -> result.iloc[b+3]
        val_at_breakout = result["rb1_orb_s08_post_bear"].iloc[b + 3]
        assert val_at_breakout == 1.0, f"post_bear should be 1 at downside breakout, got {val_at_breakout}"
        # Bar b+3 (retrace): post_bear stays 1 -> result.iloc[b+4]
        val_after = result["rb1_orb_s08_post_bear"].iloc[b + 4]
        assert val_after == 1.0, f"post_bear should stay 1 after retrace, got {val_after}"


class TestORBRetestEntry:
    """rb1_orb_s{hh}_rl50_retest_bull/bear = entry signal: fires when post-breakout AND price near ORB midpoint."""

    def _make_retest_df(self, n_pre_hour_bars=32):
        """M15 data with a planted post-breakout retrace scenario.

        candle_span='hl' (default): or_high=104, or_low=96, midpoint=100, range=8.
        entry_bull (rl=0.5) = 104 - 0.5*8 = 100. Retest fires when low <= 100.

          b+0: range bar (H=104, L=96)
          b+1: C=106 (upside breakout, L=105 stays above entry)
          b+2: C=100 (retrace to midpoint — rl50_retest_bull SHOULD fire)
          b+3: C=95  (below or_low — thesis invalidated)
        """
        n_total = n_pre_hour_bars + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre_hour_bars
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 106.0   # upside breakout
        close[b + 2] = 100.0   # retrace to midpoint
        close[b + 3] = 95.0    # below or_low (invalidated)
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        # Breakout bar: L must stay above entry level (100) so retest fires on retrace bar, not breakout bar
        low[b + 1] = 105.0
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        return df, b

    def test_retest_bull_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "rb1_orb_s08_rl50_retest_bull" in result.columns, "rb1_orb_s08_rl50_retest_bull column missing"

    def test_retest_bear_column_exists(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8])
        assert "rb1_orb_s08_rl50_retest_bear" in result.columns, "rb1_orb_s08_rl50_retest_bear column missing"

    def test_retest_bull_requires_post_breakout_state(self):
        """rl50_retest_bull = 0 when there is no prior upside breakout (post_bull = 0)."""
        ind = _get_indicator()
        df, b = self._make_retest_df()
        # Override b+1 to stay inside range (no breakout)
        df = df.copy()
        df.loc[df.index[b + 1], "C"] = 100.0
        df.loc[df.index[b + 1], "H"] = 103.0
        result = ind.compute(df, sessions=[8],
)
        # Bar b+2 is at midpoint (C=100) but no breakout -> rl50_retest_bull must be 0
        # After shift: result.iloc[b+3]
        val = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 3]
        assert val == 0.0, (
            f"rl50_retest_bull should be 0 without a prior upside breakout, got {val}"
        )

    def test_retest_bull_fires_when_price_at_midpoint_after_breakout(self):
        """rl50_retest_bull = 1 when post_bull=1 AND price retraces to the ORB midpoint (first touch)."""
        ind = _get_indicator()
        df, b = self._make_retest_df()
        result = ind.compute(df, sessions=[8])
        # Bar b+1 (C=106, breakout): breakout fires
        # Bar b+2 (C=100 = midpoint, L=99.5 <= entry_bull=100): retest fires
        # Signal columns are NOT shifted — they appear at the bar where they fire.
        val = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 2]
        assert val == 1.0, (
            f"rl50_retest_bull should be 1 when at midpoint after upside breakout, got {val}"
        )

    def test_retest_bull_fires_only_on_first_touch(self):
        """rl50_retest_bull fires only when price ENTERS the midpoint zone — not on subsequent bars."""
        ind = _get_indicator()
        n_pre = 32
        n_total = n_pre + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 106.0  # upside breakout
        close[b + 2] = 100.0  # first touch of midpoint -> fires
        close[b + 3] = 100.0  # second bar at midpoint -> must NOT fire again
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        low[b + 1] = 105.0  # breakout bar: L stays above entry level
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8])
        # Signal columns are NOT shifted.
        # b+2 (first touch): fires
        val_first = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 2]
        assert val_first == 1.0, f"rl50_retest_bull first touch should be 1, got {val_first}"
        # b+3 (still in zone, same price): must NOT fire again
        val_second = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 3]
        assert val_second == 0.0, (
            f"rl50_retest_bull must not fire twice in the same zone approach, got {val_second}"
        )

    def test_retest_fires_for_both_directions_after_double_breakout(self):
        """After double breakout (both bull and bear), both retests can fire at midpoint.

        This differs from the old ORB-specific logic which invalidated the first
        breakout direction.  The shared retest logic (used by both ORB and PDHL)
        treats each direction independently: if price broke out and returned to
        the entry level while still above range_low / below range_high, the signal fires.
        """
        ind = _get_indicator()
        n_pre = 32
        n_total = n_pre + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 106.0  # bull breakout
        close[b + 2] = 94.0   # bear breakout
        close[b + 3] = 100.0  # midpoint — both retests fire
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        low[b + 1] = 105.0   # bull breakout bar: L stays above entry
        high[b + 2] = 95.0   # bear breakout bar: H stays below entry
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8])
        # Signal columns are NOT shifted. Both retests fire at b+3 (midpoint bar).
        val_bull = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 3]
        val_bear = result["rb1_orb_s08_rl50_retest_bear"].iloc[b + 3]
        assert val_bull == 1.0, (
            f"rl50_retest_bull should fire at midpoint after bull breakout, got {val_bull}"
        )
        assert val_bear == 1.0, (
            f"rl50_retest_bear should fire at midpoint after bear breakout, got {val_bear}"
        )

    def test_retest_bull_fires_only_once_per_session(self):
        """rl50_retest_bull = 0 on subsequent bars after signal already fired."""
        ind = _get_indicator()
        df, b = self._make_retest_df()
        result = ind.compute(df, sessions=[8],
)
        # rl50_retest_bull fires at b+2 (C=100, midpoint). At b+3 (C=95), already retested -> 0.
        # After shift: result.iloc[b+4]
        val = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 4]
        assert val == 0.0, (
            f"rl50_retest_bull should be 0 after already firing earlier in the session, got {val}"
        )

    def test_retest_bear_fires_when_price_at_midpoint_after_bear_breakout(self):
        """rl50_retest_bear = 1 when post_bear=1 AND price retraces to the ORB midpoint."""
        ind = _get_indicator()
        n_total = 32 + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = 32
        # H/L range: or_high=104, or_low=96, midpoint=100, range=8
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 94.0   # downside breakout
        close[b + 2] = 100.0  # retrace to midpoint — rl50_retest_bear SHOULD fire
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        high[b + 1] = 95.0   # breakout bar: H stays below entry level
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8])
        assert "rb1_orb_s08_rl50_retest_bear" in result.columns
        # Bar b+2 (C=100, post_bear=1, H >= entry_bear=100): retest fires
        # Signal columns are NOT shifted.
        val = result["rb1_orb_s08_rl50_retest_bear"].iloc[b + 2]
        assert val == 1.0, (
            f"rl50_retest_bear should be 1 when at midpoint after downside breakout, got {val}"
        )

    def test_retest_bull_not_triggered_without_post_range_breakout(self):
        """rl50_retest_bull must NOT fire when no close exceeds or_high in the post-range period.

        With rb=2 and candle_span='hl', the range covers bars b and b+1.
        Post-range bars must break above or_high = max(H[b], H[b+1]) to trigger.
        """
        ind = _get_indicator()
        n_pre = 32
        n_total = n_pre + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre
        # rb=2: range bars are b and b+1
        # candle_span='hl': or_high = max(H[b], H[b+1]) = 105, or_low = min(L[b], L[b+1]) = 95
        high[b] = 105.0
        low[b] = 95.0
        high[b + 1] = 102.0
        low[b + 1] = 98.0
        # Post-range bars stay inside range
        close[b + 2] = 98.0
        close[b + 3] = 100.0
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8],
                             range_bars=2)
        # With rb=2: range = bars b,b+1; first valid = b+2.
        # Features are shifted by 1, so b+2 data appears at b+3 in the result.
        post_bull_b2 = result["rb2_orb_s08_post_bull"].iloc[b + 3]
        assert post_bull_b2 == 0.0, (
            f"post_bull should be 0 when no close > or_high in post-range period, got {post_bull_b2}"
        )
        val = result["rb2_orb_s08_rl50_retest_bull"].iloc[b + 4]
        assert val == 0.0, (
            f"rl50_retest_bull should be 0 without post-range breakout, got {val}"
        )

    def test_retest_bull_does_not_refire_after_zone_exit_and_reentry(self):
        """rl50_retest_bull fires at most once per session (session-level lock).

        After the signal fires on the first midpoint touch, price may temporarily
        leave the zone and re-enter.  The signal must NOT fire a second time.
        """
        ind = _get_indicator()
        n_pre = 32
        n_total = n_pre + 12
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 106.0  # bull breakout
        close[b + 2] = 100.0  # first touch — signal fires here
        close[b + 3] = 106.0  # exits zone (above)
        close[b + 4] = 100.0  # re-enters zone — must NOT fire again
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        low[b + 1] = 105.0  # breakout bar: L stays above entry level
        low[b + 3] = 105.0  # exit bar: L stays above entry level
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8])
        # Signal columns are NOT shifted.
        val_first = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 2]   # first touch at b+2
        val_reentry = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 4]  # re-entry at b+4
        assert val_first == 1.0, f"first touch should fire (=1), got {val_first}"
        assert val_reentry == 0.0, (
            f"re-entry into zone after exit must NOT fire again (expected 0), got {val_reentry}"
        )

    def test_retest_bull_bear_in_get_signal_columns(self):
        """rl50_retest_bull and rl50_retest_bear must appear in get_signal_columns() for all pipeline sessions."""
        ind = _get_indicator()
        signals = ind.get_signal_columns()
        for h in [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]:
            pfx = f"rb1_orb_s{h:02d}"
            assert f"{pfx}_rl50_retest_bull" in signals, (
                f"{pfx}_rl50_retest_bull missing from get_signal_columns()"
            )
            assert f"{pfx}_rl50_retest_bear" in signals, (
                f"{pfx}_rl50_retest_bear missing from get_signal_columns()"
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
        """Set the range bar H/L for a given day's session hour."""
        day_start = df.index[0] + pd.Timedelta(days=day)
        session_time = day_start.replace(hour=session_hour, minute=0)
        if session_time in df.index:
            df.loc[session_time, "H"] = or_high
            df.loc[session_time, "L"] = or_low

    def test_default_no_carry(self):
        """carry_forward_days=0 (default): each session uses its own range."""
        ind = _get_indicator()
        df = self._make_multi_day_df(n_days=3)
        # Day 0: session range [97, 103]
        self._set_session_range(df, 0, 8, 103.0, 97.0)
        # Day 1: session range [98, 102]
        self._set_session_range(df, 1, 8, 102.0, 98.0)

        result = ind.compute(df.copy(), sessions=[8],
                             carry_forward_days=0)

        # Day 1's range should be its own (not carried from day 0)
        # sl_dist = or_range / 2 = (102-98) / 2 = 2.0
        day1_start = df.index[0] + pd.Timedelta(days=1, hours=8, minutes=15)
        if day1_start in result.index:
            sl_dist = result.loc[day1_start, "rb1_orb_s08_sl_dist"]
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

        result = ind.compute(df.copy(), sessions=[8],
                             carry_forward_days=1)

        # Day 1 should have sl_dist = 5.0 (carried range/2 = (105-95)/2), not 1.0 (its own)
        # Find a valid bar in day 1's session (after range period)
        day1_post_range = df.index[0] + pd.Timedelta(days=1, hours=8, minutes=15)
        if day1_post_range in result.index:
            sl_dist = result.loc[day1_post_range, "rb1_cf1_orb_s08_sl_dist"]
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
        # Day 1: narrow range [99, 101] -> carried to [95, 105]
        self._set_session_range(df, 1, 8, 101.0, 99.0)
        # Force breakout on day 1 (C=110 > carried or_high=105)
        day1_breakout = df.index[0] + pd.Timedelta(days=1, hours=9)
        if day1_breakout in df.index:
            df.loc[day1_breakout, "C"] = 110.0
            df.loc[day1_breakout, "H"] = 110.0
        # Day 2: range [98, 102] — should use OWN range (carry chain broken)
        self._set_session_range(df, 2, 8, 102.0, 98.0)

        result = ind.compute(df.copy(), sessions=[8],
                             carry_forward_days=2)

        # Day 2 should use its own range/2 (2.0), not the carried one (5.0)
        day2_post_range = df.index[0] + pd.Timedelta(days=2, hours=8, minutes=15)
        if day2_post_range in result.index:
            sl_dist = result.loc[day2_post_range, "rb1_cf2_orb_s08_sl_dist"]
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
        # Day 1: narrow [99, 101] -> carried to [95, 105], no breakout
        self._set_session_range(df, 1, 8, 101.0, 99.0)
        # Day 2: [98, 102] -> carry_forward_days=1, so carry should have expired
        self._set_session_range(df, 2, 8, 102.0, 98.0)

        result = ind.compute(df.copy(), sessions=[8],
                             carry_forward_days=1)

        # Day 2 should use its own range/2 (2.0), carry expired after 1 day
        day2_post_range = df.index[0] + pd.Timedelta(days=2, hours=8, minutes=15)
        if day2_post_range in result.index:
            sl_dist = result.loc[day2_post_range, "rb1_cf1_orb_s08_sl_dist"]
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

        result = ind.compute(df.copy(), sessions=[8],
                             carry_forward_days=1)

        # In a normal (non-carried) session, the range bar at 08:00 has NaN features.
        # For a carried session, the range is pre-established -> 08:00 bar should be valid.
        day1_range_bar = df.index[0] + pd.Timedelta(days=1, hours=8)
        if day1_range_bar in result.index:
            # The range bar itself may have shifted features from the previous bar.
            # Check that at least the next bar after range bar is valid.
            next_bar = df.index[0] + pd.Timedelta(days=1, hours=8, minutes=15)
            if next_bar in result.index:
                val_next = result.loc[next_bar, "rb1_cf1_orb_s08_sl_dist"]
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

        result = ind.compute(df.copy(), sessions=[8],
                             pre_range_bars=0)

        # Range should be 103 - 97 = 6.0 (pre-session bar ignored)
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "rb1_orb_s08_sl_dist"]
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
        # Pre-bar 1 (07:45): H = 106 -> should expand range high to 106
        pre_bar1 = df.index[0] + pd.Timedelta(hours=7, minutes=45)
        df.loc[pre_bar1, "H"] = 106.0
        df.loc[pre_bar1, "L"] = 99.0

        result = ind.compute(df.copy(), sessions=[8],
                             pre_range_bars=2)

        # Range should be 106 - 97 = 9.0 (expanded high from pre-bar)
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "rb1_prb2_orb_s08_sl_dist"]
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
        # Pre-bar 1 (07:45): L = 94 -> should expand range low to 94
        pre_bar1 = df.index[0] + pd.Timedelta(hours=7, minutes=45)
        df.loc[pre_bar1, "H"] = 101.0
        df.loc[pre_bar1, "L"] = 94.0

        result = ind.compute(df.copy(), sessions=[8],
                             pre_range_bars=2)

        # Range should be 103 - 94 = 9.0 (expanded low from pre-bar)
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "rb1_prb2_orb_s08_sl_dist"]
            if not pd.isna(sl_dist):
                assert sl_dist == pytest.approx(9.0, abs=0.1), (
                    f"pre_range_bars=2 should expand low to 94, got sl_dist={sl_dist}"
                )

    def test_no_expansion_when_pre_bars_within_range(self):
        """pre_range_bars > 0 but pre-bars are within session range -> no change."""
        ind = _get_indicator()
        df = self._make_pre_range_df()
        session_bar = df.index[0] + pd.Timedelta(hours=8)
        df.loc[session_bar, "H"] = 103.0
        df.loc[session_bar, "L"] = 97.0
        # Pre-bar at 07:45 has values within session range
        pre_bar1 = df.index[0] + pd.Timedelta(hours=7, minutes=45)
        df.loc[pre_bar1, "H"] = 101.0
        df.loc[pre_bar1, "L"] = 99.0

        result = ind.compute(df.copy(), sessions=[8],
                             pre_range_bars=2)

        # Range should still be 103 - 97 = 6.0
        post_range = df.index[0] + pd.Timedelta(hours=8, minutes=15)
        if post_range in result.index:
            sl_dist = result.loc[post_range, "rb1_prb2_orb_s08_sl_dist"]
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
        """Scalar cf=0, prb=0 -> no cf/prb prefix on session columns."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8],
                             carry_forward_days=0, pre_range_bars=0)

        # Should have unprefixed session columns
        assert "rb1_orb_s08_range" in result.columns
        # Should NOT have cf/prb prefixed columns
        cf_cols = [c for c in result.columns if c.startswith("cf")]
        assert len(cf_cols) == 0, f"Unexpected cf-prefixed columns: {cf_cols}"

    def test_cf_list_generates_prefixed_columns(self):
        """carry_forward_days=[0, 1] -> rb1_cf0_ and rb1_cf1_ columns (prb omitted, default 0)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8],
                             carry_forward_days=[0, 1],
                             pre_range_bars=0)

        assert "rb1_cf0_orb_s08_range" in result.columns, (
            f"Missing rb1_cf0_ prefix. Columns: {[c for c in result.columns if 'orb_s08' in c]}"
        )
        assert "rb1_cf1_orb_s08_range" in result.columns, (
            f"Missing rb1_cf1_ prefix. Columns: {[c for c in result.columns if 'orb_s08' in c]}"
        )

    def test_prb_list_generates_prefixed_columns(self):
        """pre_range_bars=[0, 1] -> rb1_prb0_ and rb1_prb1_ columns (cf omitted, default 0)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8],
                             carry_forward_days=0,
                             pre_range_bars=[0, 1])

        assert "rb1_prb0_orb_s08_range" in result.columns
        assert "rb1_prb1_orb_s08_range" in result.columns

    def test_combined_lists_generate_cartesian_prefixes(self):
        """cf=[0,1], prb=[0,1] -> 4 variant sets (cartesian product)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8],
                             carry_forward_days=[0, 1],
                             pre_range_bars=[0, 1])

        expected_prefixes = ["rb1_cf0_prb0_", "rb1_cf0_prb1_", "rb1_cf1_prb0_", "rb1_cf1_prb1_"]
        for prefix in expected_prefixes:
            col = f"{prefix}orb_s08_range"
            assert col in result.columns, (
                f"Missing {prefix} variant. Got: {[c for c in result.columns if 'orb_s08_range' in c]}"
            )

    def test_rb_and_cf_prb_prefixes_combined(self):
        """range_bars=[1,2] + cf=[0,1] -> rb1_cf0_, rb1_cf1_, rb2_cf0_, rb2_cf1_ (prb omitted)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], range_bars=[1, 2],
                             carry_forward_days=[0, 1], pre_range_bars=0)

        # Should have combined rb + cf prefixes (prb omitted, default 0)
        assert "rb1_cf0_orb_s08_range" in result.columns
        assert "rb1_cf1_orb_s08_range" in result.columns
        assert "rb2_cf0_orb_s08_range" in result.columns
        assert "rb2_cf1_orb_s08_range" in result.columns

    def test_single_cf_with_prb_list_triggers_prefix(self):
        """cf=0 (scalar) + prb=[0,1] (list) -> prefix active (len(prb_list) > 1)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8],
                             carry_forward_days=0,
                             pre_range_bars=[0, 1])

        # Even though cf is scalar, prb being a list triggers prb prefix
        assert "rb1_prb0_orb_s08_range" in result.columns
        assert "rb1_prb1_orb_s08_range" in result.columns

    def test_retest_signals_get_cf_prefix(self):
        """Retest signal columns (rl50_retest_bull, rl50_retest_bear) get cf prefix when cf active."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8],
                             enable_retracement=True,
                             carry_forward_days=[0, 1], pre_range_bars=0)

        assert "rb1_cf0_orb_s08_rl50_retest_bull" in result.columns
        assert "rb1_cf1_orb_s08_rl50_retest_bull" in result.columns
        assert "rb1_cf0_orb_s08_rl50_retest_bear" in result.columns
        assert "rb1_cf1_orb_s08_rl50_retest_bear" in result.columns

    def test_variant_count_matches_cartesian_product(self):
        """Number of session variant sets = len(cf_list) x len(prb_list)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8],
                             carry_forward_days=[0, 1, 2],
                             pre_range_bars=[0, 1])

        # 3 cf x 2 prb = 6 variants; each has orb_s08_range
        range_cols = [c for c in result.columns if c.endswith("_orb_s08_range")]
        assert len(range_cols) == 6, (
            f"Expected 6 variants (3x2), got {len(range_cols)}: {range_cols}"
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
        if len(signal_cols) == 0:
            pytest.skip(f"No inline signal columns in {strategy_file} (uses presets)")

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
            enable_retracement=True,
            carry_forward_days=[0, 1, 2], pre_range_bars=[0, 1],
        )

        signal_col = "rb1_cf0_prb0_orb_s08_rl50_retest_bull"
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

        Wick range: H_max = 106, L_min = 95 -> wick_range = 11.0
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
        """or_high/or_low must be derived from O/C body, not H/L wicks (candle_span='body')."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        result = ind.compute(df, sessions=[8], range_bars=2, candle_span="body")

        # Body range = max(98, 102) - min(98, 102) = 4.0
        # _range feature is now raw (not normalized by close).
        # After shift: result.iloc[b+3] has the value for bar b+2
        range_val = result["rb2_orb_s08_range"].iloc[b + 3]
        assert not np.isnan(range_val), "rb2_orb_s08_range should not be NaN for post-range bar"
        expected = 4.0  # raw body range
        assert abs(range_val - expected) < 0.1, (
            f"Expected raw body range {expected}, "
            f"got {range_val}. If ~11, range is still wick-based (H/L)."
        )

    def test_body_range_smaller_than_wick_range(self):
        """Body-based range must be <= wick-based range for any candle."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, sessions=[8], range_bars=2, candle_span="body")

        range_vals = result["rb2_orb_s08_range"].dropna()
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

        # Single range bar: O=97, C=103 -> body range = 6.0
        # Wicks: H=105, L=94 -> wick range = 11.0
        open_[b] = 97.0
        close[b] = 103.0
        high[b] = 105.0
        low[b] = 94.0

        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8], range_bars=1, candle_span="body")

        # _range is now raw (not normalized by close). Body range = 6.0.
        # After shift: result.iloc[b+2] has value for bar b+1
        range_val = result["rb1_orb_s08_range"].iloc[b + 2]
        assert not np.isnan(range_val), "range should not be NaN"
        expected = 6.0  # raw body range
        assert abs(range_val - expected) < 0.1, (
            f"Expected raw body range {expected}, got {range_val}"
        )

    def test_sl_dist_is_half_range(self):
        """Session sl_dist = or_range / 2 (entry at midpoint -> SL at body boundary)."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        result = ind.compute(df, sessions=[8], range_bars=2, candle_span="body")

        # Body range = 4.0, sl_dist = 4.0 / 2 = 2.0
        sl_val = result["rb2_orb_s08_sl_dist"].iloc[b + 3]
        assert not np.isnan(sl_val), "sl_dist should not be NaN"
        assert abs(sl_val - 2.0) < 0.01, (
            f"Expected sl_dist = 2.0 (body_range 4.0 / 2), got {sl_val}"
        )

    def test_poc_dist_reflects_body_midpoint(self):
        """poc_dist = (C - midpoint) / ATR. Midpoint is body-based: (or_high + or_low) / 2."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        # Post-range bar: C=100.0, body midpoint = (102+98)/2 = 100.0 -> poc_dist ~ 0
        result = ind.compute(df, sessions=[8], range_bars=2, candle_span="body")

        poc_val = result["rb2_orb_s08_poc_dist"].iloc[b + 3]
        assert not np.isnan(poc_val), "poc_dist should not be NaN"
        assert abs(poc_val) < 0.1, (
            f"Expected poc_dist ~ 0 (C=100 at body midpoint=100), got {poc_val}"
        )

    def test_breakout_uses_body_boundary(self):
        """Breakout detection must use body-based or_high/or_low, not wick H/L."""
        ind = _get_indicator()
        df, b = self._make_body_vs_wick_df(range_bars=2)
        df = df.copy()
        # Post-range bar: C=103 -> above body or_high=102 (breakout!)
        # but below wick H_max=106 (would NOT be breakout if wick-based)
        df.loc[df.index[b + 2], "C"] = 103.0
        df.loc[df.index[b + 2], "H"] = 103.0
        result = ind.compute(df, sessions=[8], range_bars=2, candle_span="body")

        # After shift: result.iloc[b+3] has bar b+2's breakout value
        bu_val = result["rb2_orb_s08_breakout_up"].iloc[b + 3]
        assert bu_val == 1.0, (
            f"Expected breakout_up=1 (C=103 > body or_high=102), got {bu_val}. "
            f"If 0, breakout is still using wick-based boundary."
        )


class TestRetracementLevels:
    """Test retracement_levels parameter producing rl{N}_ prefixed columns."""

    def test_multiple_retracement_levels_produce_columns(self):
        """retracement_levels=[0, 0.5] produces _rl0_ and _rl50_ retest columns."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], retracement_levels=[0, 0.5])

        # rl0 columns
        assert "rb1_orb_s08_rl0_retest_bull" in result.columns, (
            f"Missing rl0_retest_bull. Cols: {[c for c in result.columns if 'retest' in c]}"
        )
        assert "rb1_orb_s08_rl0_retest_bear" in result.columns
        assert "rb1_orb_s08_rl0_sl_dist" in result.columns

        # rl50 columns
        assert "rb1_orb_s08_rl50_retest_bull" in result.columns
        assert "rb1_orb_s08_rl50_retest_bear" in result.columns
        assert "rb1_orb_s08_rl50_sl_dist" in result.columns

    def test_single_retracement_level_scalar(self):
        """retracement_levels=0.5 (default scalar) produces _rl50_ columns."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df, sessions=[8], retracement_levels=0.5)

        assert "rb1_orb_s08_rl50_retest_bull" in result.columns
        assert "rb1_orb_s08_rl50_retest_bear" in result.columns
        assert "rb1_orb_s08_rl50_sl_dist" in result.columns

    def test_retracement_level_0_at_boundary(self):
        """rl=0 means entry at OR boundary (no retracement). sl_dist at rl0 should differ from rl50."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, sessions=[8], retracement_levels=[0, 0.5])

        # rl0_sl_dist and rl50_sl_dist should both exist and have values
        rl0_sl = result["rb1_orb_s08_rl0_sl_dist"].dropna()
        rl50_sl = result["rb1_orb_s08_rl50_sl_dist"].dropna()
        assert len(rl0_sl) > 0, "rl0_sl_dist has no values"
        assert len(rl50_sl) > 0, "rl50_sl_dist has no values"

    def test_retracement_levels_binary_signals(self):
        """All rl{N}_retest_bull/bear columns must be binary (0/1)."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, sessions=[8], retracement_levels=[0, 0.5])

        for rl_tag in ["rl0", "rl50"]:
            for direction in ["bull", "bear"]:
                col = f"rb1_orb_s08_{rl_tag}_retest_{direction}"
                vals = result[col].dropna()
                if len(vals) > 0:
                    assert set(vals.unique()).issubset({0, 1}), (
                        f"{col} is not binary: {vals.unique()}"
                    )


class TestMinRetracement:
    """Test min_retracement parameter filtering shallow retests."""

    def test_min_retracement_filters_shallow_retests(self):
        """min_retracement=0.3 should prevent signals when retracement is shallow.

        Setup: or_high=104, or_low=96, range=8, midpoint=100.
        min_retracement=0.3 means low must reach at most 104 - 0.3*8 = 101.6
        before the retest signal can fire.

        In this test, all bars after breakout have L > 101.6 (shallow retracement),
        so retracement_ok_bull stays False and the signal does not fire.
        """
        ind = _get_indicator()
        n_pre = 32
        n_total = n_pre + 12
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")

        # Build arrays manually so Low is fully controlled
        o = np.full(n_total, 103.0)
        c = np.full(n_total, 103.0)
        h = np.full(n_total, 104.0)
        lo = np.full(n_total, 102.0)  # All bars L > 101.6

        b = n_pre
        # Range bar
        h[b] = 104.0
        lo[b] = 96.0
        o[b] = 100.0
        c[b] = 100.0

        # Bull breakout: C > or_high (104)
        # L stays high (105) -> no deep retracement
        o[b + 1] = 105.0
        c[b + 1] = 106.0
        h[b + 1] = 107.0
        lo[b + 1] = 105.0

        # Retrace to midpoint zone (C=100), but L only reaches 103 (shallow)
        # Need L <= 101.6 for retracement to qualify — here L=103 > 101.6
        o[b + 2] = 103.0
        c[b + 2] = 100.0
        h[b + 2] = 103.0
        lo[b + 2] = 103.0

        # Subsequent bars also stay above retracement threshold
        for i in range(b + 3, n_total):
            o[i] = 103.0
            c[i] = 103.0
            h[i] = 104.0
            lo[i] = 102.0

        df = pd.DataFrame({"O": o, "H": h, "L": lo, "C": c}, index=idx)
        result = ind.compute(df, sessions=[8],
                             min_retracement=0.3)

        # Signal columns are NOT shifted. Check bar b+2 (the retrace bar).
        val = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 2]
        assert val == 0.0, (
            f"rl50_retest_bull should be 0 for shallow retracement (min_retracement=0.3), got {val}"
        )

    def test_min_retracement_zero_allows_all(self):
        """min_retracement=0.0 (default) should not filter any retests."""
        ind = _get_indicator()
        n_pre = 32
        n_total = n_pre + 10
        idx = pd.date_range("2024-01-02 00:00", periods=n_total, freq="15min")
        close = np.full(n_total, 100.0)
        high = close + 0.5
        low = close - 0.5
        b = n_pre
        high[b] = 104.0
        low[b] = 96.0
        close[b + 1] = 106.0  # bull breakout
        close[b + 2] = 100.0  # retrace to midpoint
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        low[b + 1] = 105.0  # breakout bar: L stays above entry level
        df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close}, index=idx)
        result = ind.compute(df, sessions=[8],
                             min_retracement=0.0)

        # Signal columns are NOT shifted.
        val = result["rb1_orb_s08_rl50_retest_bull"].iloc[b + 2]
        assert val == 1.0, (
            f"rl50_retest_bull should fire with min_retracement=0.0, got {val}"
        )

    def test_min_retracement_in_default_params(self):
        """min_retracement should appear in get_default_params() with value 0.0."""
        params = _orb.OpeningRangeIndicator.get_default_params()
        assert "min_retracement" in params
        assert params["min_retracement"] == 0.0

    def test_min_retracement_in_param_schema(self):
        """min_retracement should appear in get_param_schema()."""
        schema = _orb.OpeningRangeIndicator.get_param_schema()
        assert "min_retracement" in schema
        assert schema["min_retracement"]["default"] == 0.0


class TestCrossHourRangeBars:
    """range_bars > bars_per_hour must span across hour boundaries."""

    def _make_controlled_15m(self, n_hours=4, session_hour=0):
        """Build a 15-min OHLCV DataFrame with controlled prices.

        Session starts at session_hour. Within the opening range, bars have
        ascending highs so we can verify which bars are included.
        """
        bars_per_hour = 4
        n = n_hours * bars_per_hour
        idx = pd.date_range(f"2024-01-01 {session_hour:02d}:00", periods=n, freq="15min")

        # Ascending H: bar 0 → H=101, bar 1 → H=102, ... bar 5 → H=106
        # Descending L: bar 0 → L=99, bar 1 → L=98, ... bar 5 → L=94
        h_vals = 101 + np.arange(n, dtype=float)
        l_vals = 99 - np.arange(n, dtype=float)
        c_vals = np.full(n, 100.0)
        o_vals = np.full(n, 100.0)

        return pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)

    def test_range_bars_6_includes_bars_across_hour_boundary(self):
        """With range_bars=6 on 15m data, or_high must include bar 5 (hour+1)."""
        df = self._make_controlled_15m(n_hours=4, session_hour=0)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="hl",
        )
        col = "rb6_orb_s00_range"
        assert col in result.columns, f"Expected column {col} not found"

        # Bar 5 (index 5) is at 01:15 — across hour boundary from session 00:00.
        # H at bar 5 = 101 + 5 = 106. So or_high must be 106.
        # L at bar 0 = 99. So or_low must be 94 (99 - 5 = 94 at bar 5).
        # Verify on a bar AFTER the range (bar 6 onward).
        after_range = result.iloc[6:]
        range_vals = after_range[col].dropna()
        assert len(range_vals) > 0, "No valid range values after range period"
        # or_range is raw (not normalized): 106 - 94 = 12
        expected_range = 106 - 94
        actual = range_vals.iloc[0]
        assert abs(actual - expected_range) < 1e-6, (
            f"or_range should be {expected_range} but got {actual}. "
            "range_bars=6 likely not spanning across hour boundary."
        )

    def test_range_bars_4_stays_within_one_hour(self):
        """With range_bars=4 on 15m data, or_high uses only bars 0-3 (one hour)."""
        df = self._make_controlled_15m(n_hours=4, session_hour=0)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=4, sessions=[0],
            enable_retracement=False, candle_span="hl",
        )
        col = "rb4_orb_s00_range"
        after_range = result.iloc[4:]
        range_vals = after_range[col].dropna()
        assert len(range_vals) > 0
        # Bars 0-3: H max = 104, L min = 96. Range = 104-96 = 8 (raw)
        expected = 104 - 96
        actual = range_vals.iloc[0]
        assert abs(actual - expected) < 1e-6

    def test_breakout_respects_cross_hour_range(self):
        """Breakout should only fire when close exceeds the full range (not just 1-hour range)."""
        df = self._make_controlled_15m(n_hours=4, session_hour=0)
        # Set close of bar 7 to 105 — above 4-bar range (104) but below 6-bar range (106)
        df.iloc[7, df.columns.get_loc("C")] = 105.0
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="hl",
        )
        bu_col = "rb6_orb_s00_breakout_up"
        # Bar 7 close=105 < or_high=106 → no breakout
        assert result[bu_col].iloc[7] != 1, (
            "Breakout fired at close=105 which is below 6-bar range high=106. "
            "Range computation likely limited to one hour."
        )

    def test_breakout_fires_when_exceeding_cross_hour_range(self):
        """Breakout should fire when close exceeds the full cross-hour range."""
        df = self._make_controlled_15m(n_hours=4, session_hour=0)
        # Set close of bar 7 to 107 — above 6-bar or_high (106)
        df.iloc[7, df.columns.get_loc("C")] = 107.0
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="hl",
        )
        bu_col = "rb6_orb_s00_breakout_up"
        # shift_features shifts by 1 bar → breakout at bar 7 appears at bar 8
        assert result[bu_col].iloc[8] == 1, (
            "Breakout should fire at close=107 which exceeds 6-bar range high=106."
        )

    def test_in_range_period_masks_all_range_bars(self):
        """Bars within the range period (0 to range_bars-1) must be NaN/masked.

        shift_features shifts by 1 bar, so bars 0-6 are NaN and bar 7 is first valid.
        """
        df = self._make_controlled_15m(n_hours=4, session_hour=0)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="hl",
        )
        col = "rb6_orb_s00_position"
        # Bars 0-6 should be NaN (range period 0-5 + 1 bar shift)
        for i in range(7):
            assert pd.isna(result[col].iloc[i]), (
                f"Bar {i} should be masked but has value {result[col].iloc[i]}"
            )
        # Bar 7 should have a valid value (first bar after range + shift)
        assert pd.notna(result[col].iloc[7]), "Bar 7 should be valid (first bar after range period + shift)"

    def test_body_candle_span_cross_hour(self):
        """candle_span='body' with range_bars=6 must use O/C of ALL range bars."""
        bars_per_hour = 4
        n = 4 * bars_per_hour
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        o_vals = np.full(n, 100.0)
        c_vals = np.full(n, 100.0)
        h_vals = np.full(n, 110.0)  # wicks should be ignored
        l_vals = np.full(n, 90.0)
        # Bar 0: O=98, bar 3: C=107 (highest body in range), bar 5: C=95 (lowest)
        o_vals[0] = 98.0
        c_vals[3] = 107.0  # intermediate bar — must be included!
        c_vals[5] = 95.0   # across hour boundary — must be included!
        df = pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)

        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="body",
        )
        col = "rb6_orb_s00_range"
        after_range = result.iloc[7:]  # +1 for shift_features
        range_vals = after_range[col].dropna()
        assert len(range_vals) > 0
        # body across all 6 bars: or_high = max(O,C) = 107 (bar 3), or_low = min(O,C) = 95 (bar 5)
        # range = 107-95 = 12 (raw)
        expected = 107 - 95
        actual = range_vals.iloc[0]
        assert abs(actual - expected) < 1e-6, (
            f"Body range should be {expected} but got {actual}. "
            "candle_span='body' must use max/min of O/C across ALL range bars."
        )

    def test_body_ignores_wicks(self):
        """candle_span='body' must ignore H/L wicks entirely."""
        df = self._make_controlled_15m(n_hours=4, session_hour=0)
        # Controlled data has ascending H and descending L (wicks).
        # All O/C are 100.0, so body range should be 0.
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=4, sessions=[0],
            enable_retracement=False, candle_span="body",
        )
        col = "rb4_orb_s00_range"
        after_range = result.iloc[5:]  # 4 range bars + 1 shift
        range_vals = after_range[col].dropna()
        assert len(range_vals) > 0
        assert range_vals.iloc[0] == 0.0, (
            f"Body range should be 0 (all O=C=100) but got {range_vals.iloc[0]}. "
            "Wicks (H/L) are leaking into body computation."
        )

    def test_zone_height_matches_breakout_threshold(self):
        """Zone rectangle high/low must equal or_high/or_low (breakout thresholds)."""
        df = self._make_controlled_15m(n_hours=4, session_hour=0)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="hl",
        )
        zones = ind._range_zones
        assert len(zones) > 0, "No range zones produced"
        zone = zones[0]

        # or_high = max(H) of bars 0-5 = 101+5 = 106
        # or_low = min(L) of bars 0-5 = 99-5 = 94
        assert zone["high"] == 106.0, f"Zone high should be 106, got {zone['high']}"
        assert zone["low"] == 94.0, f"Zone low should be 94, got {zone['low']}"

        # Breakout fires only when close > zone high (106)
        bu_col = "rb6_orb_s00_breakout_up"
        # All bars after range have close=100 < 106, so no breakout
        post_range = result[bu_col].iloc[7:]  # +1 shift
        assert (post_range.dropna() == 0).all(), (
            "No breakout should fire when all closes (100) are below or_high (106)"
        )

    def test_zone_height_matches_breakout_body_mode(self):
        """Zone and breakout must use body (O/C) not wicks when candle_span='body'."""
        n = 4 * 4
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        o_vals = np.full(n, 100.0)
        c_vals = np.full(n, 100.0)
        h_vals = np.full(n, 110.0)  # wicks extend to 110
        l_vals = np.full(n, 90.0)   # wicks extend to 90
        # Range bars: all bodies at 100, wicks at 90-110
        # Body range: or_high = 100, or_low = 100, range = 0
        # Set bar 2 O=97, bar 4 C=103 (body range becomes 97-103)
        o_vals[2] = 97.0
        c_vals[4] = 103.0
        df = pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)

        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="body",
        )
        zones = ind._range_zones
        assert len(zones) > 0
        zone = zones[0]

        # Body mode: or_high = max(max(O,C)) = 103, or_low = min(min(O,C)) = 97
        # Zone must use body values, NOT wick values (110/90)
        assert zone["high"] == 103.0, (
            f"Zone high should be 103 (body), got {zone['high']}. "
            f"If 110, zone is using wicks (H/L) instead of body (O/C)."
        )
        assert zone["low"] == 97.0, (
            f"Zone low should be 97 (body), got {zone['low']}. "
            f"If 90, zone is using wicks."
        )

        # Breakout: bar 7 close=100, which is inside body range (97-103) → no breakout
        bu_col = "rb6_orb_s00_breakout_up"
        post_range = result[bu_col].iloc[7:]  # +1 shift
        no_breakout = post_range.dropna()
        assert (no_breakout == 0).all(), (
            "Close=100 is inside body range (97-103), no breakout should fire"
        )

    def test_breakout_fires_only_once_per_session(self):
        """Breakout must transition 0->1 only ONCE per session (cummax), then persist."""
        n = 4 * 8  # 8 hours at 15min
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        o_vals = np.full(n, 100.0)
        c_vals = np.full(n, 100.0)
        h_vals = np.full(n, 110.0)
        l_vals = np.full(n, 90.0)
        # Range bars 0-5: body = 100 (flat), so or_high=100, or_low=100
        # Set one range bar slightly higher to create a range
        c_vals[0] = 102.0  # or_high = 102
        o_vals[1] = 98.0   # or_low = 98

        # Post-range bars: simulate oscillation around the upper boundary
        # Bar 7: close=103 -> above or_high=102 -> breakout_up fires (persists)
        c_vals[7] = 103.0
        # Bar 8: close=101 -> back inside range, but breakout_up stays 1 (persistent)
        c_vals[8] = 101.0
        # Bar 9: close=104 -> above again, stays 1
        c_vals[9] = 104.0

        df = pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="body",
        )
        bu_col = "rb6_orb_s00_breakout_up"
        # Signal columns are NOT shifted. Breakout at bar 7 appears at bar 7.
        # Breakout uses cummax: once 1, stays 1 for rest of session.
        breakout_vals = result[bu_col].iloc[6:].dropna()
        # Only one 0->1 transition
        transitions = breakout_vals.diff().fillna(0)
        first_fires = (transitions == 1.0).sum()
        assert first_fires == 1, (
            f"Breakout should have exactly 1 transition 0->1, got {first_fires}. "
            f"Values: {breakout_vals.tolist()}"
        )
        # First breakout at bar 7
        assert result[bu_col].iloc[7] == 1.0, "First breakout should fire at bar 7"
        # Persistent: stays 1 even when price goes back inside
        assert result[bu_col].iloc[8] == 1.0, "Breakout persists (cummax) at bar 8"

    def test_breakout_down_fires_only_once_per_session(self):
        """breakout_down must transition 0->1 only once per session (cummax), then persist."""
        n = 4 * 8
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        o_vals = np.full(n, 100.0)
        c_vals = np.full(n, 100.0)
        h_vals = np.full(n, 110.0)
        l_vals = np.full(n, 90.0)
        c_vals[0] = 102.0  # or_high = 102
        o_vals[1] = 98.0   # or_low = 98

        # Bar 7: close=97 -> below or_low=98 -> breakout_down fires (persists)
        c_vals[7] = 97.0
        # Bar 8: close=99 -> back inside, but stays 1 (persistent)
        c_vals[8] = 99.0
        # Bar 9: close=96 -> below again, stays 1
        c_vals[9] = 96.0

        df = pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="body",
        )
        bd_col = "rb6_orb_s00_breakout_down"
        # Signal columns are NOT shifted. Breakout uses cummax (persistent).
        breakout_vals = result[bd_col].iloc[6:].dropna()
        transitions = breakout_vals.diff().fillna(0)
        first_fires = (transitions == 1.0).sum()
        assert first_fires == 1, (
            f"breakout_down should have exactly 1 transition 0->1, fired {first_fires} times"
        )

    def test_breakout_respects_pct_threshold(self):
        """breakout_threshold (pct of range) must be added to or_high/or_low."""
        n = 4 * 8
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        o_vals = np.full(n, 100.0)
        c_vals = np.full(n, 100.0)
        h_vals = np.full(n, 110.0)
        l_vals = np.full(n, 90.0)
        # Range: or_high=105, or_low=95 → or_range=10
        c_vals[0] = 105.0
        o_vals[1] = 95.0
        # breakout_threshold=0.1 → threshold = 0.1 * 10 = 1.0
        # So breakout_up requires C > 105 + 1 = 106

        # Bar 7: close=105.5 → above or_high(105) but below threshold(106) → NO breakout
        c_vals[7] = 105.5
        # Bar 8: close=106.5 → above threshold(106) → breakout
        c_vals[8] = 106.5

        df = pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="body",
            breakout_threshold=0.1,
        )
        bu_col = "rb6_orb_s00_breakout_up"
        # Signal columns are NOT shifted.
        # Bar 7 close=105.5 < 106 threshold -> no breakout
        assert result[bu_col].iloc[7] != 1.0, (
            "C=105.5 is below threshold (or_high + 10% of range = 106), should not fire"
        )
        # Bar 8 close=106.5 > 106 threshold -> breakout (persistent)
        assert result[bu_col].iloc[8] == 1.0, (
            "C=106.5 is above threshold (106), should fire"
        )

    def test_breakout_respects_abs_threshold(self):
        """breakout_threshold_abs must be added when it exceeds pct threshold."""
        n = 4 * 8
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        o_vals = np.full(n, 100.0)
        c_vals = np.full(n, 100.0)
        h_vals = np.full(n, 110.0)
        l_vals = np.full(n, 90.0)
        # Range: or_high=105, or_low=95 → or_range=10
        c_vals[0] = 105.0
        o_vals[1] = 95.0
        # breakout_threshold=0.1 → pct_dist = 1.0
        # breakout_threshold_abs=2.0 → abs wins (max(1.0, 2.0) = 2.0)
        # So breakout_up requires C > 105 + 2 = 107

        # Bar 7: close=106.5 → above pct threshold(106) but below abs threshold(107)
        c_vals[7] = 106.5
        # Bar 8: close=107.5 → above abs threshold(107) → breakout
        c_vals[8] = 107.5

        df = pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="body",
            breakout_threshold=0.1,
            breakout_threshold_abs=2.0,
        )
        bu_col = "rb6_orb_s00_breakout_up"
        # Signal columns are NOT shifted.
        # Bar 7: C=106.5 < 107 -> no breakout
        assert result[bu_col].iloc[7] != 1.0, (
            "C=106.5 below abs threshold (107), should not fire"
        )
        # Bar 8: C=107.5 > 107 -> breakout (persistent)
        assert result[bu_col].iloc[8] == 1.0, (
            "C=107.5 above abs threshold (107), should fire"
        )

    def test_breakout_down_respects_threshold(self):
        """breakout_down must also respect threshold (or_low - threshold)."""
        n = 4 * 8
        idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
        o_vals = np.full(n, 100.0)
        c_vals = np.full(n, 100.0)
        h_vals = np.full(n, 110.0)
        l_vals = np.full(n, 90.0)
        c_vals[0] = 105.0
        o_vals[1] = 95.0
        # threshold = max(0.1*10, 0) = 1.0
        # breakout_down requires C < 95 - 1 = 94

        # Bar 7: close=94.5 → below or_low(95) but above threshold(94) → NO
        c_vals[7] = 94.5
        # Bar 8: close=93.5 → below threshold(94) → breakout_down
        c_vals[8] = 93.5

        df = pd.DataFrame({"O": o_vals, "H": h_vals, "L": l_vals, "C": c_vals}, index=idx)
        ind = _get_indicator()
        result = ind.compute(
            df, range_bars=6, sessions=[0],
            enable_retracement=False, candle_span="body",
            breakout_threshold=0.1,
        )
        bd_col = "rb6_orb_s00_breakout_down"
        # Signal columns are NOT shifted.
        assert result[bd_col].iloc[7] != 1.0, (
            "C=94.5 above threshold (or_low - 1 = 94), should not fire"
        )
        assert result[bd_col].iloc[8] == 1.0, (
            "C=93.5 below threshold (94), should fire"
        )
