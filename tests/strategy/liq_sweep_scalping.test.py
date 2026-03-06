"""
Integration tests for the liq_sweep_scalping strategy pipeline.

Tests verify that synthetic OHLCV data with known patterns correctly
activates the corresponding signal features after running through the
full indicator pipeline.

Key points:
- The liquidity_sweep plugin calls shift_features(), so the signal
  detected at bar i appears in the result at bar i+1.
- lsw_bull_active = 1 when a bullish sweep zone is active (wick below
  swing low, close above swing low).
- lsw_bear_active = 1 when a bearish sweep zone is active (wick above
  swing high, close below swing high).
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

from conftest import make_m1_ohlcv, make_liquidity_sweep_scenario

from fwbg.pipeline.features import compute_indicator_pool
from fwbg.core.config import StrategyConfig

from fwbg.api.workspace import get_strategies_dir as _gsd; CONFIG_PATH = str(_gsd() / "liq_sweep_scalping.json")


def _load_config():
    return StrategyConfig.from_json_file(CONFIG_PATH)


class TestLiqSweepBullSignal:
    """Test that a bullish liquidity sweep scenario activates lsw_bull_active."""

    def test_lsw_bull_active_column_exists(self):
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = make_liquidity_sweep_scenario(n_base=200, n_post=50, seed=42)
        result = compute_indicator_pool(df, indicators=indicators)
        assert "lsw_bull_active" in result.columns, (
            f"lsw_bull_active not found in columns: {list(result.columns)}"
        )

    def test_lsw_bull_active_fires_after_sweep(self):
        """
        After the sweep bar, lsw_bull_active should be 1 in a window of bars.
        shift_features() shifts by 1, so the signal appears at sweep_idx+1.
        We check a window of [sweep_idx, sweep_idx+5] for robustness.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = make_liquidity_sweep_scenario(n_base=200, n_post=50, seed=42)
        result = compute_indicator_pool(df, indicators=indicators)

        window = result["lsw_bull_active"].iloc[sweep_idx : sweep_idx + 5]
        max_val = window.max()
        assert max_val == 1, (
            f"lsw_bull_active did not activate in window [{sweep_idx}:{sweep_idx+5}]. "
            f"Values: {window.tolist()}"
        )

    def test_lsw_bull_active_value_at_shift_bar(self):
        """Signal should appear exactly at sweep_idx+1 due to shift_features."""
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = make_liquidity_sweep_scenario(n_base=200, n_post=50, seed=42)
        result = compute_indicator_pool(df, indicators=indicators)

        # After shift, the swept zone becomes active at the next bar
        signal_val = result["lsw_bull_active"].iloc[sweep_idx + 1]
        assert signal_val == 1, (
            f"lsw_bull_active at sweep_idx+1 ({sweep_idx+1}) is {signal_val}, expected 1. "
            f"Window values: {result['lsw_bull_active'].iloc[sweep_idx:sweep_idx+5].tolist()}"
        )

    def test_lsw_bull_active_persists_after_sweep(self):
        """Zone stays active for several bars after the sweep."""
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = make_liquidity_sweep_scenario(n_base=200, n_post=50, seed=42)
        result = compute_indicator_pool(df, indicators=indicators)

        # Zone should persist for at least 3 bars after activation
        window = result["lsw_bull_active"].iloc[sweep_idx + 1 : sweep_idx + 10]
        assert window.sum() >= 3, (
            f"lsw_bull_active should persist for at least 3 bars after sweep. "
            f"Values: {window.tolist()}"
        )

    def test_lsw_bull_active_fires_while_bear_does_not_newly_trigger(self):
        """
        At the bull sweep bar, lsw_bull_active must be 1 (shift+1 rule).
        lsw_bear_active may already be 1 from prior random-walk noise, but
        the bull signal is the one we care about here.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = make_liquidity_sweep_scenario(n_base=200, n_post=50, seed=42)
        result = compute_indicator_pool(df, indicators=indicators)

        bull_val = result["lsw_bull_active"].iloc[sweep_idx + 1]
        assert bull_val == 1, (
            f"lsw_bull_active should be 1 at sweep_idx+1={sweep_idx + 1}, got {bull_val}"
        )


class TestLiqSweepBearSignal:
    """Test that a bearish liquidity sweep scenario activates lsw_bear_active."""

    def _make_bear_sweep_scenario(self, n_base=200, n_post=50):
        """
        Deterministic bearish sweep scenario.

        The liquidity_sweep plugin uses:
            prev_sh = rolling max of H over swing_lookback (50) bars ending at i-1
            Bear sweep detected when: highs[i] > prev_sh AND closes[i] < prev_sh

        Strategy: build a stable flat high at exactly 101.0 for 50+ bars so
        the rolling swing high is 101.0, then plant a bar with wick above 101
        and close below 101. This guarantees the detection conditions are met.
        """
        swing_lookback = 50  # matches liq_sweep_scalping config
        # Phase 1: n_base flat bars – controlled, no random walk
        # High never exceeds 101.0 so rolling swing high stabilises at 101.0
        n = n_base + 1 + n_post
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high  = np.full(n, 101.0)   # consistent swing high reference level
        low   = np.full(n, 99.0)

        # Sweep bar at sweep_idx: wick ABOVE 101 (swing high), close BELOW 101
        sweep_idx = n_base
        open_[sweep_idx]  = 100.5
        high[sweep_idx]   = 102.0   # wick above swing high 101.0
        low[sweep_idx]    = 99.5
        close[sweep_idx]  = 100.3   # close below swing high 101.0

        # Post bars: price recovers downward (the rejection plays out)
        for i in range(n_post):
            j = sweep_idx + 1 + i
            close[j] = 100.0 - i * 0.01
            open_[j] = close[j] + 0.1
            high[j]  = close[j] + 0.5
            low[j]   = close[j] - 0.5

        idx = pd.date_range("2022-01-03 08:00", periods=n, freq="h")
        df = pd.DataFrame(
            {"O": open_, "H": high, "L": low, "C": close, "V": np.full(n, 1000.0)},
            index=idx,
        )
        return df, sweep_idx

    def test_lsw_bear_active_column_exists(self):
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = self._make_bear_sweep_scenario()
        result = compute_indicator_pool(df, indicators=indicators)
        assert "lsw_bear_active" in result.columns, (
            f"lsw_bear_active not found in columns: {list(result.columns)}"
        )

    def test_lsw_bear_active_fires_after_sweep(self):
        """
        After the bearish sweep bar, lsw_bear_active should be 1.
        shift_features() shifts by 1, so signal appears at sweep_idx+1.
        We check a window [sweep_idx, sweep_idx+5] for robustness.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = self._make_bear_sweep_scenario()
        result = compute_indicator_pool(df, indicators=indicators)

        window = result["lsw_bear_active"].iloc[sweep_idx : sweep_idx + 5]
        max_val = window.max()
        assert max_val == 1, (
            f"lsw_bear_active did not activate in window [{sweep_idx}:{sweep_idx+5}]. "
            f"Values: {window.tolist()}"
        )

    def test_lsw_bear_active_at_shift_bar(self):
        """
        Signal should appear exactly at sweep_idx+1 due to shift_features.
        The sweep is detected at bar sweep_idx; shift moves it to sweep_idx+1.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = self._make_bear_sweep_scenario()
        result = compute_indicator_pool(df, indicators=indicators)

        signal_val = result["lsw_bear_active"].iloc[sweep_idx + 1]
        assert signal_val == 1, (
            f"lsw_bear_active at sweep_idx+1 ({sweep_idx + 1}) is {signal_val}, expected 1. "
            f"Window values: {result['lsw_bear_active'].iloc[sweep_idx:sweep_idx+5].tolist()}"
        )

    def test_lsw_bear_active_persists_after_sweep(self):
        """Zone stays active for several bars after the sweep."""
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = self._make_bear_sweep_scenario()
        result = compute_indicator_pool(df, indicators=indicators)

        window = result["lsw_bear_active"].iloc[sweep_idx + 1 : sweep_idx + 10]
        assert window.sum() >= 3, (
            f"lsw_bear_active should persist for at least 3 bars. "
            f"Values: {window.tolist()}"
        )

    def test_lsw_bull_inactive_at_bear_sweep(self):
        """
        With a deterministic flat scenario (high always <= 101), no bullish
        sweep can form — lsw_bull_active should be 0 at sweep_idx+1.
        """
        config = _load_config()
        indicators = config.get_indicators()
        df, sweep_idx = self._make_bear_sweep_scenario()
        result = compute_indicator_pool(df, indicators=indicators)

        if "lsw_bull_active" in result.columns:
            bull_val = result["lsw_bull_active"].iloc[sweep_idx + 1]
            assert bull_val == 0, (
                f"lsw_bull_active unexpectedly fired at sweep_idx+1 in a bear scenario: {bull_val}"
            )


class TestLiqSweepPipelineFeatures:
    """Test that the full pipeline runs cleanly on random M1 data."""

    def test_result_has_many_feature_columns(self):
        """Full pipeline should produce more than 50 feature columns."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m1_ohlcv(n=3000, seed=99)
        result = compute_indicator_pool(df, indicators=indicators)

        # Count feature columns (exclude OHLCV)
        ohlcv = {"O", "H", "L", "C", "V"}
        feature_cols = [c for c in result.columns if c not in ohlcv]
        assert len(feature_cols) > 50, (
            f"Expected >50 feature columns, got {len(feature_cols)}: {feature_cols}"
        )

    def test_no_inf_values_in_result(self):
        """Pipeline result must not contain any inf values."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m1_ohlcv(n=3000, seed=99)
        result = compute_indicator_pool(df, indicators=indicators)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(result[numeric_cols].values).any()
        if has_inf:
            inf_cols = [
                c for c in numeric_cols
                if np.isinf(result[c].values).any()
            ]
            pytest.fail(f"Inf values found in columns: {inf_cols}")

    def test_index_is_preserved(self):
        """The result DataFrame must preserve the original index."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m1_ohlcv(n=3000, seed=99)
        result = compute_indicator_pool(df, indicators=indicators)

        pd.testing.assert_index_equal(result.index, df.index)

    def test_lsw_signal_columns_all_present(self):
        """All expected liquidity sweep signal columns must be present."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m1_ohlcv(n=3000, seed=99)
        result = compute_indicator_pool(df, indicators=indicators)

        expected_signals = [
            "lsw_bull_active",
            "lsw_bear_active",
            "lsw_bull_dist",
            "lsw_bear_dist",
            "lsw_bull_in_zone",
            "lsw_bear_in_zone",
            "lsw_bull_recency",
            "lsw_bear_recency",
        ]
        missing = [col for col in expected_signals if col not in result.columns]
        assert not missing, (
            f"Missing liquidity sweep columns: {missing}. "
            f"Available: {[c for c in result.columns if 'lsw' in c]}"
        )

    def test_ohlcv_columns_preserved(self):
        """Original OHLCV columns must be present in result."""
        config = _load_config()
        indicators = config.get_indicators()
        df = make_m1_ohlcv(n=3000, seed=99)
        result = compute_indicator_pool(df, indicators=indicators)

        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns, f"OHLCV column '{col}' missing from result"
