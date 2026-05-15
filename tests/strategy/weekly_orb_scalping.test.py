"""
Integration tests for the weekly_orb_scalping strategy pipeline.

Verifies that M15 OHLCV data spanning multiple weeks produces correct
weekly_opening_range breakout signal features through the full indicator pipeline.

Column naming note:
  The pipeline config uses range_bars=[2,4] (a list), so the plugin prefixes
  each variant:  "wor_rb2_" + suffix  and  "wor_rb4_" + suffix.
  E.g. "wor_rb2_breakout_up", "wor_rb4_breakout_down".
  Shared statistical columns keep the plain "wor_stat_*" prefix.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

from tests.strategy.conftest import make_m15_ohlcv

from fwbg.pipeline.features import compute_indicator_pool
from fwbg.core.config import StrategyConfig

from fwbg.api.workspace import get_strategies_dir as _gsd; _CONFIG_PATH = str(_gsd() / "weekly_orb_scalping.json")

# Need at least 2 full weeks so the weekly range can form and break.
# make_m15_ohlcv starts on 2022-01-03 (Monday). 2000 bars = ~20 days.
_N_BARS = 2000


def load_config():
    return StrategyConfig.from_json_file(_CONFIG_PATH)


def _run_pipeline(n: int = _N_BARS, seed: int = 42) -> pd.DataFrame:
    config = load_config()
    indicators = config.get_indicators()
    df = make_m15_ohlcv(n=n, seed=seed)
    return compute_indicator_pool(df, indicators=indicators)


# ---------------------------------------------------------------------------
# TestWORBreakoutUp
# ---------------------------------------------------------------------------

class TestWORBreakoutUp:
    """wor_rb2_breakout_up and wor_rb4_breakout_up columns exist and fire."""

    def test_wor_rb2_breakout_up_column_exists(self):
        result = _run_pipeline()
        assert "wor_rb2_breakout_up" in result.columns, (
            f"wor_rb2_breakout_up missing. WOR columns: "
            f"{[c for c in result.columns if 'wor' in c]}"
        )

    def test_wor_rb4_breakout_up_column_exists(self):
        result = _run_pipeline()
        assert "wor_rb4_breakout_up" in result.columns

    def test_wor_rb2_breakout_up_fires_at_least_once(self):
        """Over 2000 M15 bars (~20 days) the signal should fire at least once."""
        result = _run_pipeline()
        col = result["wor_rb2_breakout_up"]
        # Skip the first ~4 bars per week which are NaN (opening range formation)
        assert col.iloc[100:].max() == 1, (
            "wor_rb2_breakout_up never fired across 2000 bars"
        )

    def test_wor_rb4_breakout_up_fires_at_least_once(self):
        result = _run_pipeline()
        col = result["wor_rb4_breakout_up"]
        assert col.iloc[100:].max() == 1, (
            "wor_rb4_breakout_up never fired across 2000 bars"
        )

    def test_wor_breakout_up_is_binary_or_nan(self):
        result = _run_pipeline()
        for col_name in ("wor_rb2_breakout_up", "wor_rb4_breakout_up"):
            col = result[col_name].dropna()
            unique_vals = set(col.unique())
            assert unique_vals.issubset({0, 1, 0.0, 1.0}), (
                f"{col_name} contains non-binary values: {unique_vals}"
            )


# ---------------------------------------------------------------------------
# TestWORBreakoutDown
# ---------------------------------------------------------------------------

class TestWORBreakoutDown:
    """wor_rb2_breakout_down and wor_rb4_breakout_down columns exist and fire."""

    def test_wor_rb2_breakout_down_column_exists(self):
        result = _run_pipeline()
        assert "wor_rb2_breakout_down" in result.columns, (
            f"wor_rb2_breakout_down missing. WOR columns: "
            f"{[c for c in result.columns if 'wor' in c]}"
        )

    def test_wor_rb4_breakout_down_column_exists(self):
        result = _run_pipeline()
        assert "wor_rb4_breakout_down" in result.columns

    def test_wor_rb2_breakout_down_fires_at_least_once(self):
        result = _run_pipeline()
        col = result["wor_rb2_breakout_down"]
        assert col.iloc[100:].max() == 1, (
            "wor_rb2_breakout_down never fired across 2000 bars"
        )

    def test_wor_rb4_breakout_down_fires_at_least_once(self):
        result = _run_pipeline()
        col = result["wor_rb4_breakout_down"]
        assert col.iloc[100:].max() == 1, (
            "wor_rb4_breakout_down never fired across 2000 bars"
        )

    def test_wor_breakout_down_is_binary_or_nan(self):
        result = _run_pipeline()
        for col_name in ("wor_rb2_breakout_down", "wor_rb4_breakout_down"):
            col = result[col_name].dropna()
            unique_vals = set(col.unique())
            assert unique_vals.issubset({0, 1, 0.0, 1.0}), (
                f"{col_name} contains non-binary values: {unique_vals}"
            )


# ---------------------------------------------------------------------------
# TestWORPipelineFeatures
# ---------------------------------------------------------------------------

class TestWORPipelineFeatures:
    """Full pipeline smoke-tests: column count, no inf, index preserved."""

    def test_feature_count_exceeds_30(self):
        result = _run_pipeline()
        ohlcv_cols = {"O", "H", "L", "C", "V"}
        feature_cols = [c for c in result.columns if c not in ohlcv_cols]
        assert len(feature_cols) > 30, (
            f"Expected >30 feature columns, got {len(feature_cols)}: {feature_cols}"
        )

    def test_no_inf_values(self):
        result = _run_pipeline()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(result[numeric_cols].values).any()
        assert not has_inf, "Result contains inf values"

    def test_index_preserved(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m15_ohlcv(n=_N_BARS)
        result = compute_indicator_pool(df, indicators=indicators)
        assert result.index.equals(df.index), "Index was not preserved through the pipeline"

    def test_wor_stat_columns_present(self):
        """Statistical features (shared, no rb prefix) should be present."""
        result = _run_pipeline()
        expected_stats = [
            "wor_stat_avg_range",
            "wor_stat_range_vs_avg",
            "wor_stat_breakout_rate",
        ]
        missing = [c for c in expected_stats if c not in result.columns]
        assert not missing, f"Missing wor_stat columns: {missing}"

    def test_wor_all_rb2_columns_present(self):
        """All rb2 feature columns should be present."""
        result = _run_pipeline()
        expected = [
            "wor_rb2_range",
            "wor_rb2_position",
            "wor_rb2_breakout_up",
            "wor_rb2_breakout_down",
            "wor_rb2_range_vs_atr",
            "wor_rb2_dist_to_high",
            "wor_rb2_dist_to_low",
        ]
        missing = [c for c in expected if c not in result.columns]
        assert not missing, f"Missing wor_rb2 columns: {missing}"

    def test_wor_all_rb4_columns_present(self):
        """All rb4 feature columns should be present."""
        result = _run_pipeline()
        expected = [
            "wor_rb4_range",
            "wor_rb4_position",
            "wor_rb4_breakout_up",
            "wor_rb4_breakout_down",
            "wor_rb4_range_vs_atr",
            "wor_rb4_dist_to_high",
            "wor_rb4_dist_to_low",
        ]
        missing = [c for c in expected if c not in result.columns]
        assert not missing, f"Missing wor_rb4 columns: {missing}"

    def test_wor_nan_only_in_opening_range_bars(self):
        """
        NaN values in wor_rb2_breakout_up should only occur in the first
        range_bars (2) of each week — not mid-week.
        """
        result = _run_pipeline(n=3000)
        col = result["wor_rb2_breakout_up"]
        # After 2 full weeks (≥ 2*5*24*4 = 960 bars), data should be largely valid
        late_slice = col.iloc[960:]
        # Some NaN may appear at the start of each new week (opening range bars),
        # but most bars should be non-NaN
        non_nan_ratio = late_slice.notna().mean()
        assert non_nan_ratio > 0.90, (
            f"Too many NaN values in wor_rb2_breakout_up after warmup: "
            f"{1 - non_nan_ratio:.1%} NaN"
        )

    def test_orb_sl_dist_columns_present_and_positive(self):
        """orb_sl_dist (session SL distance) columns must exist and be positive in WOR pipeline."""
        result = _run_pipeline(n=3000)
        sl_cols = [c for c in result.columns if c.endswith("_sl_dist")]
        assert sl_cols, (
            "No *_sl_dist columns found in weekly_orb pipeline output. "
            f"Available: {[c for c in result.columns if 'sl' in c][:10]}"
        )
        for col in sl_cols:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals > 0).all(), f"{col} contains non-positive values"

    def test_orb_retest_signal_columns_exist(self):
        """retest_bull and retest_bear signal columns must be present in WOR pipeline output."""
        result = _run_pipeline(n=3000)
        retest_cols = [c for c in result.columns if "_retest_bull" in c or "_retest_bear" in c]
        assert retest_cols, (
            "No *_retest_bull or *_retest_bear columns found in weekly_orb pipeline output"
        )
        for col in retest_cols:
            vals = result[col].dropna()
            assert vals.isin([0.0, 1.0]).all(), f"{col} contains non-binary values"
