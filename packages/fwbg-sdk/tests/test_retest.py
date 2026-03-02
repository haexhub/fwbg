"""Tests for shared retest logic (fwbg_sdk.retest)."""
import numpy as np

from fwbg_sdk.retest import apply_breakout_threshold, compute_break_state, compute_retest_signals


class TestApplyBreakoutThreshold:
    """Tests for threshold-based breakout detection."""

    def test_no_threshold_simple_comparison(self):
        close = np.array([101.0, 99.0, 100.0])
        rh = np.array([100.0, 100.0, 100.0])
        rl = np.array([98.0, 98.0, 98.0])
        rs = rh - rl
        above, below = apply_breakout_threshold(close, rh, rl, rs)
        assert above.tolist() == [True, False, False]
        assert below.tolist() == [False, False, False]

    def test_pct_threshold_filters_marginal_close(self):
        """Close 3% above range -> no breakout when threshold is 5%."""
        rh = np.array([100.0])
        rl = np.array([90.0])
        rs = rh - rl  # 10
        # 3% of range = 0.3 -> need close > 100.3
        close = np.array([100.2])
        above, below = apply_breakout_threshold(close, rh, rl, rs, breakout_threshold=0.05)
        assert not above[0]

    def test_pct_threshold_allows_strong_breakout(self):
        """Close 6% above range -> breakout when threshold is 5%."""
        rh = np.array([100.0])
        rl = np.array([90.0])
        rs = rh - rl  # 10
        # 6% of 10 = 0.6 -> need close > 100.5
        close = np.array([100.7])
        above, below = apply_breakout_threshold(close, rh, rl, rs, breakout_threshold=0.05)
        assert above[0]

    def test_abs_threshold(self):
        """Absolute threshold: close 5 points above, threshold 10 -> no breakout."""
        rh = np.array([100.0])
        rl = np.array([90.0])
        rs = rh - rl
        close = np.array([105.0])
        above, below = apply_breakout_threshold(close, rh, rl, rs, breakout_threshold_abs=10.0)
        assert not above[0]

    def test_abs_threshold_allows_breakout(self):
        close = np.array([111.0])
        rh = np.array([100.0])
        rl = np.array([90.0])
        rs = rh - rl
        above, below = apply_breakout_threshold(close, rh, rl, rs, breakout_threshold_abs=10.0)
        assert above[0]

    def test_max_of_pct_and_abs(self):
        """Effective threshold = max(pct * range, abs)."""
        rh = np.array([100.0])
        rl = np.array([90.0])
        rs = rh - rl  # 10
        # pct: 0.05 * 10 = 0.5
        # abs: 2.0
        # max = 2.0 -> need close > 102
        close = np.array([101.5])
        above, _ = apply_breakout_threshold(close, rh, rl, rs, breakout_threshold=0.05,
                                            breakout_threshold_abs=2.0)
        assert not above[0]

    def test_bear_breakout_with_threshold(self):
        rh = np.array([100.0])
        rl = np.array([90.0])
        rs = rh - rl
        close = np.array([89.0])  # 1 below range_low
        _, below = apply_breakout_threshold(close, rh, rl, rs, breakout_threshold_abs=2.0)
        assert not below[0]  # need 2 points below
        close2 = np.array([87.5])
        _, below2 = apply_breakout_threshold(close2, rh, rl, rs, breakout_threshold_abs=2.0)
        assert below2[0]


class TestComputeBreakState:
    """Tests for break state tracking with group resets."""

    def test_single_group_bull_break(self):
        above = np.array([False, True, True, False])
        below = np.array([False, False, False, False])
        nan_guard = np.array([1.0, 1.0, 1.0, 1.0])
        group_ids = np.array([0, 0, 0, 0])
        hb, lb, pb, pbe, bha, bla = compute_break_state(above, below, nan_guard, group_ids, 4)
        assert hb[1] == 1.0  # first break
        assert hb[2] == 0.0  # already broke
        assert pb[2] == 1.0  # post_bull persists
        assert bha[2]

    def test_group_reset(self):
        above = np.array([True, False, True, False])
        below = np.array([False, False, False, False])
        nan_guard = np.array([1.0, 1.0, 1.0, 1.0])
        group_ids = np.array([0, 0, 1, 1])
        hb, _, _, _, _, _ = compute_break_state(above, below, nan_guard, group_ids, 4)
        assert hb[0] == 1.0
        assert hb[2] == 1.0  # resets at group boundary

    def test_nan_guard_skips_bars(self):
        above = np.array([True, True])
        below = np.array([False, False])
        nan_guard = np.array([np.nan, 1.0])
        group_ids = np.array([0, 0])
        hb, _, _, _, _, _ = compute_break_state(above, below, nan_guard, group_ids, 2)
        assert hb[0] == 0.0  # skipped due to NaN
        assert hb[1] == 1.0

    def test_session_mask_restricts_breaks(self):
        above = np.array([True, True, False])
        below = np.array([False, False, False])
        nan_guard = np.array([1.0, 1.0, 1.0])
        group_ids = np.array([0, 0, 0])
        session_mask = np.array([False, True, True])
        hb, _, _, _, _, _ = compute_break_state(above, below, nan_guard, group_ids, 3,
                                                 session_mask=session_mask)
        assert hb[0] == 0.0  # off-session
        assert hb[1] == 1.0  # in-session


class TestComputeRetestSignals:
    """Tests for the shared retest signal computation.

    The new API uses proximity checks (low <= entry_bull for bull,
    high >= entry_bear for bear) instead of the old departure/half_band zone.
    """

    def _make_scenario(self, n, range_high, range_low, close_vals, group_ids=None,
                       broke_high=None, broke_low=None):
        """Helper to build arrays for retest signal tests."""
        close = np.array(close_vals, dtype=float)
        rh = np.full(n, range_high, dtype=float)
        rl = np.full(n, range_low, dtype=float)
        if group_ids is None:
            group_ids = np.zeros(n, dtype=int)
        else:
            group_ids = np.array(group_ids, dtype=int)
        if broke_high is None:
            broke_high = np.zeros(n, dtype=bool)
        else:
            broke_high = np.array(broke_high, dtype=bool)
        if broke_low is None:
            broke_low = np.zeros(n, dtype=bool)
        else:
            broke_low = np.array(broke_low, dtype=bool)
        return close, rh, rl, group_ids, broke_high, broke_low

    def test_bull_retest_fires_on_proximity(self):
        """Bull retest fires when broke_high, low <= entry_bull, close > range_low."""
        # range: [96, 104], entry_bull = 100
        # Bull retest: low must reach entry_bull (100) to fire.
        n = 4
        close, rh, rl, gids, bh, bl = self._make_scenario(
            n, 104.0, 96.0,
            [100.0, 105.0, 100.0, 100.0],
        )
        bh[:] = [False, True, True, True]
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)
        # low=None means close is used for proximity check
        # bar1: C=105 > 100 (not near entry) -> no
        # bar2: C=100 <= 100 -> fires (first retest)
        result = compute_retest_signals(
            close=close, high=None, low=None,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n,
        )
        assert result["retest_bull"][1] == 0.0  # breakout bar, C=105 > entry_bull=100
        assert result["retest_bull"][2] == 1.0  # C=100 <= entry_bull, fires

    def test_no_breakout_no_signal(self):
        """If broke_high is never True, no bull retest fires."""
        n = 4
        close, rh, rl, gids, bh, bl = self._make_scenario(
            n, 104.0, 96.0,
            [100.0, 100.0, 100.0, 100.0],
        )
        # bh all False
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)

        result = compute_retest_signals(
            close=close, high=None, low=None,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n,
        )
        assert result["retest_bull"].sum() == 0.0

    def test_signal_fires_once_per_group(self):
        """Only one bull retest per group, even if price re-enters zone."""
        n = 6
        close, rh, rl, gids, bh, bl = self._make_scenario(
            n, 104.0, 96.0,
            [100.0, 106.0, 100.0, 106.0, 100.0, 100.0],
        )
        bh[:] = [False, True, True, True, True, True]
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)

        result = compute_retest_signals(
            close=close, high=None, low=None,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n,
        )
        assert result["retest_bull"].sum() == 1.0
        assert result["retest_bull"][2] == 1.0  # first touch

    def test_group_boundary_resets(self):
        """State resets at group boundary -- signal can fire again in new group."""
        n = 6
        close = np.array([106.0, 100.0, 100.0, 106.0, 100.0, 100.0])
        rh = np.full(n, 104.0)
        rl = np.full(n, 96.0)
        gids = np.array([0, 0, 0, 1, 1, 1])
        bh = np.ones(n, dtype=bool)
        bl = np.zeros(n, dtype=bool)
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)

        result = compute_retest_signals(
            close=close, high=None, low=None,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n,
        )
        assert result["retest_bull"][1] == 1.0  # group 0
        assert result["retest_bull"][4] == 1.0  # group 1 (reset)

    def test_retracement_check(self):
        """With min_retracement > 0, low must reach threshold before signal fires."""
        n = 6
        # range: [96, 104], range_size=8. entry_bull=100.
        # near_entry_bull: low <= entry_bull = 100.
        # min_retracement=0.8 -> retrace_bull_threshold = 104 - 0.8*8 = 97.6.
        close = np.array([106.0, 103.0, 100.0, 103.0, 97.0, 100.0])
        low_arr = np.array([105.0, 102.0, 99.0, 102.0, 96.0, 99.0])
        high_arr = np.array([107.0, 104.0, 101.0, 104.0, 98.0, 101.0])
        rh = np.full(n, 104.0)
        rl = np.full(n, 96.0)
        gids = np.zeros(n, dtype=int)
        bh = np.ones(n, dtype=bool)
        bl = np.zeros(n, dtype=bool)
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)

        # Without retracement check:
        # bar2: L=99 <= 100 -> fires at bar2
        r1 = compute_retest_signals(
            close=close, high=high_arr, low=low_arr,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n,
        )
        assert r1["retest_bull"][2] == 1.0

        # With min_retracement=0.8: low must reach 97.6 before signal fires
        # bar2: L=99 <= 100 (near entry) but L=99 > 97.6 -> not retraced enough
        # bar4: L=96 <= 97.6 -> retraced! L=96 <= 100 -> near entry. Fires.
        r2 = compute_retest_signals(
            close=close, high=high_arr, low=low_arr,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n, min_retracement=0.8,
        )
        assert r2["retest_bull"][2] == 0.0  # not retraced enough yet
        assert r2["retest_bull"][4] == 1.0  # retracement met, fires

    def test_session_mask_blocks_signal(self):
        """Signal only fires during session bars."""
        n = 4
        close = np.array([106.0, 100.0, 100.0, 100.0])
        rh = np.full(n, 104.0)
        rl = np.full(n, 96.0)
        gids = np.zeros(n, dtype=int)
        bh = np.ones(n, dtype=bool)
        bl = np.zeros(n, dtype=bool)
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)
        session_mask = np.array([True, False, False, True])

        result = compute_retest_signals(
            close=close, high=None, low=None,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n, session_mask=session_mask,
        )
        # bar1/2 off-session -> no signal even though in zone
        assert result["retest_bull"][1] == 0.0
        assert result["retest_bull"][2] == 0.0
        # bar3 in-session -> fires
        assert result["retest_bull"][3] == 1.0

    def test_bear_retest(self):
        """Bear retest: breakout below range, return to entry level."""
        n = 4
        # entry_bear = 100. Bear retest: high >= entry_bear AND close < range_high.
        close = np.array([100.0, 94.0, 100.0, 100.0])
        high_arr = np.array([100.5, 94.5, 100.5, 100.5])
        rh = np.full(n, 104.0)
        rl = np.full(n, 96.0)
        gids = np.zeros(n, dtype=int)
        bh = np.zeros(n, dtype=bool)
        bl = np.array([False, True, True, True])
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)

        result = compute_retest_signals(
            close=close, high=high_arr, low=None,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n,
        )
        # bar1: H=94.5 < 100 -> not near entry_bear
        # bar2: H=100.5 >= 100 AND C=100 < 104 -> fires
        assert result["retest_bear"][2] == 1.0

    def test_close_above_range_high_blocks_bear(self):
        """Bear retest blocked when close >= range_high."""
        n = 3
        close = np.array([94.0, 105.0, 105.0])  # bar1 above range_high
        high_arr = np.array([94.5, 105.5, 105.5])
        rh = np.full(n, 104.0)
        rl = np.full(n, 96.0)
        gids = np.zeros(n, dtype=int)
        bh = np.zeros(n, dtype=bool)
        bl = np.ones(n, dtype=bool)
        entry_bull = np.full(n, 100.0)
        entry_bear = np.full(n, 100.0)

        result = compute_retest_signals(
            close=close, high=high_arr, low=None,
            range_high=rh, range_low=rl,
            group_ids=gids, broke_high_arr=bh, broke_low_arr=bl,
            entry_bull=entry_bull, entry_bear=entry_bear,
            n=n,
        )
        # bar1/2: C=105 > 104 (range_high) -> bear retest blocked
        assert result["retest_bear"].sum() == 0.0
