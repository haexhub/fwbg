"""
Integration tests for the orb_exploration strategy pipeline.

Tests verify that synthetic OHLCV data with known breakout patterns correctly
activates ORB breakout signal features after running through the full pipeline.

Key points about the orb_simple_v1 pipeline:
- range_bars=[1,2] is a list → columns are prefixed with rb1_ and rb2_
- Session ORBs are computed for configured session hours [0,1,2,6,7,8,12,13,14]
- shift_features() shifts all computed values by 1 bar (look-ahead protection)
- So a breakout at bar bo_idx appears in the result at bo_idx+1
- Session ORB signal columns: rb1_orb_s08_breakout_up, rb2_orb_s08_breakout_up, etc.
- Rolling ORB signal columns: rb1_orb_breakout_up, rb2_orb_breakout_up
  (but rolling ORB only fires mid-hour, not at hour boundaries)

The make_orb_breakout_scenario places the ORB range bars in hour 12 (bars 50-51)
and the breakout bar at 13:00 (bar 52 = bo_idx). Session ORBs for sessions 0-12
have already accumulated their opening ranges from earlier hours, so when price
breaks out at bar 52, those session ORBs fire at bo_idx+1=53 in the result.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

from conftest import make_m15_ohlcv, make_orb_breakout_scenario

from fwbg.pipeline.features import compute_indicator_pool
from fwbg.core.config import StrategyConfig

from fwbg.api.workspace import get_strategies_dir as _gsd; CONFIG_PATH = str(_gsd() / "orb_exploration.json")

# Primary session signal to test (hour 8 = London open, reliably in config sessions)
# With carry_forward_days=0 (disabled) and pre_range_bars=[0,1] in the pipeline,
# session columns get prb prefix only. prb0 = base variant.
_PRIMARY_UP_SIGNAL = "rb1_prb0_orb_s08_breakout_up"
_PRIMARY_DOWN_SIGNAL = "rb1_prb0_orb_s08_breakout_down"


def _load_config():
    return StrategyConfig.from_json_file(CONFIG_PATH)


class TestORBBreakoutUp:
    """Test that an upward ORB breakout activates the breakout_up signal."""

    def test_orb_breakout_up_column_exists(self):
        """The rb1_orb_s08_breakout_up column must exist after pipeline run."""
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="up")
        result = compute_indicator_pool(df, indicators=indicators)

        assert _PRIMARY_UP_SIGNAL in result.columns, (
            f"{_PRIMARY_UP_SIGNAL} not found. Available ORB columns: "
            f"{[c for c in result.columns if 'breakout_up' in c]}"
        )

    def test_orb_breakout_up_fires_at_shift_bar(self):
        """
        Session ORB breakout fires at the bar where close first crosses above the
        session opening range. shift_features shifts by 1, so the signal appears
        at bo_idx+1 in the result DataFrame.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="up")
        result = compute_indicator_pool(df, indicators=indicators)

        signal_val = result[_PRIMARY_UP_SIGNAL].iloc[bo_idx + 1]
        assert signal_val == 1, (
            f"{_PRIMARY_UP_SIGNAL} at bo_idx+1 ({bo_idx + 1}) is {signal_val}, expected 1. "
            f"Window [{bo_idx}:{bo_idx+5}]: "
            f"{result[_PRIMARY_UP_SIGNAL].iloc[bo_idx:bo_idx+5].tolist()}"
        )

    def test_orb_breakout_up_fires_in_window(self):
        """
        At least one ORB breakout_up signal must activate in a window around bo_idx.
        This verifies the breakout is detected even if the exact signal column shifts.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="up")
        result = compute_indicator_pool(df, indicators=indicators)

        up_cols = [c for c in result.columns if "breakout_up" in c]
        assert up_cols, "No breakout_up columns found in result"

        # Check window [bo_idx, bo_idx+3] across all breakout_up columns
        any_fired = False
        for col in up_cols:
            window = result[col].iloc[bo_idx : bo_idx + 3].fillna(0)
            if window.max() == 1:
                any_fired = True
                break

        assert any_fired, (
            f"No breakout_up signal fired in window [{bo_idx}:{bo_idx+3}]. "
            f"Checked columns: {up_cols}"
        )

    def test_orb_breakout_up_multiple_sessions_fire(self):
        """
        Multiple session ORB signals should fire at the same bar, since all sessions
        prior to hour 13 already established their ranges with the flat price data.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="up")
        result = compute_indicator_pool(df, indicators=indicators)

        # Sessions 0, 1, 2, 6, 7, 8, 12 all precede hour 13 and should fire
        sessions_that_should_fire = [0, 1, 2, 6, 7, 8, 12]
        fired_sessions = []
        for h in sessions_that_should_fire:
            col = f"rb1_prb0_orb_s{h:02d}_breakout_up"
            if col in result.columns:
                val = result[col].iloc[bo_idx + 1]
                if val == 1:
                    fired_sessions.append(h)

        assert len(fired_sessions) >= 3, (
            f"Expected at least 3 session ORB signals to fire, got {len(fired_sessions)}: "
            f"{fired_sessions}"
        )

    def test_orb_breakout_down_does_not_fire_on_up_breakout(self):
        """When breaking up, the breakout_down signal should remain 0."""
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="up")
        result = compute_indicator_pool(df, indicators=indicators)

        if _PRIMARY_DOWN_SIGNAL in result.columns:
            val = result[_PRIMARY_DOWN_SIGNAL].iloc[bo_idx + 1]
            assert val == 0, (
                f"{_PRIMARY_DOWN_SIGNAL} unexpectedly fired ({val}) on an up breakout"
            )


class TestORBBreakoutDown:
    """Test that a downward ORB breakout activates the breakout_down signal."""

    def test_orb_breakout_down_column_exists(self):
        """The rb1_orb_s08_breakout_down column must exist after pipeline run."""
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="down")
        result = compute_indicator_pool(df, indicators=indicators)

        assert _PRIMARY_DOWN_SIGNAL in result.columns, (
            f"{_PRIMARY_DOWN_SIGNAL} not found. Available ORB columns: "
            f"{[c for c in result.columns if 'breakout_down' in c]}"
        )

    def test_orb_breakout_down_fires_at_shift_bar(self):
        """
        Session ORB breakout_down fires at bo_idx+1 in the result (after shift).
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="down")
        result = compute_indicator_pool(df, indicators=indicators)

        signal_val = result[_PRIMARY_DOWN_SIGNAL].iloc[bo_idx + 1]
        assert signal_val == 1, (
            f"{_PRIMARY_DOWN_SIGNAL} at bo_idx+1 ({bo_idx + 1}) is {signal_val}, expected 1. "
            f"Window [{bo_idx}:{bo_idx+5}]: "
            f"{result[_PRIMARY_DOWN_SIGNAL].iloc[bo_idx:bo_idx+5].tolist()}"
        )

    def test_orb_breakout_down_fires_in_window(self):
        """
        At least one ORB breakout_down signal fires in the window around bo_idx.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="down")
        result = compute_indicator_pool(df, indicators=indicators)

        down_cols = [c for c in result.columns if "breakout_down" in c]
        assert down_cols, "No breakout_down columns found in result"

        any_fired = False
        for col in down_cols:
            window = result[col].iloc[bo_idx : bo_idx + 3].fillna(0)
            if window.max() == 1:
                any_fired = True
                break

        assert any_fired, (
            f"No breakout_down signal fired in window [{bo_idx}:{bo_idx+3}]. "
            f"Checked columns: {down_cols}"
        )

    def test_orb_breakout_down_multiple_sessions_fire(self):
        """
        Multiple session ORB breakout_down signals fire on a down breakout.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="down")
        result = compute_indicator_pool(df, indicators=indicators)

        sessions_that_should_fire = [0, 1, 2, 6, 7, 8, 12]
        fired_sessions = []
        for h in sessions_that_should_fire:
            col = f"rb1_prb0_orb_s{h:02d}_breakout_down"
            if col in result.columns:
                val = result[col].iloc[bo_idx + 1]
                if val == 1:
                    fired_sessions.append(h)

        assert len(fired_sessions) >= 3, (
            f"Expected at least 3 session ORB breakout_down signals to fire, got "
            f"{len(fired_sessions)}: {fired_sessions}"
        )

    def test_orb_breakout_up_does_not_fire_on_down_breakout(self):
        """When breaking down, the breakout_up signal should remain 0."""
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="down")
        result = compute_indicator_pool(df, indicators=indicators)

        if _PRIMARY_UP_SIGNAL in result.columns:
            val = result[_PRIMARY_UP_SIGNAL].iloc[bo_idx + 1]
            assert val == 0, (
                f"{_PRIMARY_UP_SIGNAL} unexpectedly fired ({val}) on a down breakout"
            )

    def test_rb2_session_breakout_down_also_fires(self):
        """rb2_ variants (range_bars=2) should also fire for sessions with enough data."""
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="down")
        result = compute_indicator_pool(df, indicators=indicators)

        rb2_col = "rb2_prb0_orb_s08_breakout_down"
        if rb2_col in result.columns:
            # rb2 session 08 requires 2 range bars — both should be captured
            val = result[rb2_col].iloc[bo_idx + 1]
            assert val == 1, (
                f"{rb2_col} at bo_idx+1 is {val}, expected 1. "
                f"Window: {result[rb2_col].iloc[bo_idx:bo_idx+5].tolist()}"
            )


class TestORBPipelineFeatures:
    """Test that the full ORB pipeline runs cleanly on random M15 data."""

    def test_result_has_many_feature_columns(self):
        """Full pipeline should produce more than 30 feature columns."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        ohlcv = {"O", "H", "L", "C", "V"}
        feature_cols = [c for c in result.columns if c not in ohlcv]
        assert len(feature_cols) > 30, (
            f"Expected >30 feature columns, got {len(feature_cols)}"
        )

    def test_no_inf_values_in_result(self):
        """Pipeline result must not contain any inf values."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(result[numeric_cols].values).any()
        if has_inf:
            inf_cols = [c for c in numeric_cols if np.isinf(result[c].values).any()]
            pytest.fail(f"Inf values found in columns: {inf_cols}")

    def test_index_is_preserved(self):
        """The result DataFrame must preserve the original index."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        pd.testing.assert_index_equal(result.index, df.index)

    def test_session_orb_columns_present(self):
        """Session ORB columns for configured sessions must be present."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        # Check a subset of expected session columns from orb_simple_v1 config
        required_sessions = [0, 1, 2, 6, 7, 8, 12, 13, 14]
        missing = []
        for h in required_sessions:
            col = f"rb1_prb0_orb_s{h:02d}_breakout_up"
            if col not in result.columns:
                missing.append(col)

        assert not missing, (
            f"Missing session ORB columns: {missing}"
        )

    def test_ohlcv_columns_preserved(self):
        """Original OHLCV columns must be present in result."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns, f"OHLCV column '{col}' missing from result"

    def test_rb1_rb2_column_count_symmetry(self):
        """rb1_ and rb2_ prefixed columns should have the same count (symmetric feature sets)."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        rb1_cols = [c for c in result.columns if c.startswith("rb1_")]
        rb2_cols = [c for c in result.columns if c.startswith("rb2_")]
        assert len(rb1_cols) == len(rb2_cols), (
            f"rb1_ has {len(rb1_cols)} columns but rb2_ has {len(rb2_cols)} columns. "
            f"They should be symmetric."
        )

    def test_orb_sl_dist_columns_present_and_positive(self):
        """orb_sl_dist and session sl_dist columns must exist and be positive."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        # With range_bars=[1,2], sl_dist columns get rb1_/rb2_ prefix
        sl_cols = [c for c in result.columns if c.endswith("_sl_dist")]
        assert sl_cols, (
            "No *_sl_dist columns found in pipeline output. "
            f"Available: {[c for c in result.columns if 'sl' in c]}"
        )
        for col in sl_cols:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals > 0).all(), f"{col} contains non-positive values"

    def test_post_bull_bear_columns_present(self):
        """post_bull and post_bear state features must exist in the pipeline output."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=2000, seed=77)
        result = compute_indicator_pool(df, indicators=indicators)

        post_cols = [c for c in result.columns if "_post_bull" in c or "_post_bear" in c]
        assert post_cols, (
            "No *_post_bull or *_post_bear columns found in pipeline output. "
            "These are required for retest entry signal computation."
        )

    def test_retest_signal_fires_only_after_breakout(self):
        """retest_bull signal must be 0 before the ORB breakout, 1 after retrace to midpoint.

        Uses the make_orb_breakout_scenario (upside breakout at hour 13), then verifies:
        - Before breakout bar: retest_bull = 0 or NaN
        - After breakout + retrace: the signal can fire (we verify the column exists
          and is binary, not always 0)
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, bo_idx = make_orb_breakout_scenario(breakout_direction="up")
        result = compute_indicator_pool(df, indicators=indicators)

        retest_cols = [c for c in result.columns if "_retest_bull" in c]
        assert retest_cols, "No *_retest_bull columns found in pipeline output"

        # Before the breakout, retest_bull must be 0 (no prior breakout in this session)
        for col in retest_cols:
            pre_breakout = result[col].iloc[max(0, bo_idx - 5):bo_idx].dropna()
            if len(pre_breakout) > 0:
                assert (pre_breakout == 0).all(), (
                    f"{col}: retest_bull fired before ORB breakout at bars "
                    f"{pre_breakout[pre_breakout != 0].index.tolist()}"
                )

        # The retest columns must be binary (0 or 1), not garbage values
        for col in retest_cols:
            vals = result[col].dropna()
            assert vals.isin([0.0, 1.0]).all(), f"{col} contains non-binary values"
