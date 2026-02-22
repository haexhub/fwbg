"""
Integration tests for the smc_choch_fvg strategy pipeline.

Verifies that synthetic OHLCV data with known patterns produces the expected
signal activations in the full indicator pipeline.
"""
import sys
import os

import numpy as np
import pytest

# Ensure conftest helpers are importable from this directory
sys.path.insert(0, os.path.dirname(__file__))

from conftest import make_m15_ohlcv, make_fvg_scenario  # noqa: E402

from fwbg.core.config import StrategyConfig
from fwbg.pipeline.features import compute_indicator_pool

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "smc_choch_fvg.json")


@pytest.fixture(scope="module")
def strategy_config():
    return StrategyConfig.from_json_file(CONFIG_PATH)


@pytest.fixture(scope="module")
def indicators(strategy_config):
    return strategy_config.get_indicators()


# =============================================================================
# TestSMCFVGBullishFVG
# =============================================================================

class TestSMCFVGBullishFVG:
    """Verify that a planted bullish FVG pattern fires the fvg_bull_active signal."""

    @pytest.fixture(scope="class")
    def fvg_result(self, indicators):
        df, fvg_bar_idx = make_fvg_scenario(n_pre=100, seed=42)
        result = compute_indicator_pool(df, indicators=indicators)
        return result, fvg_bar_idx

    def test_fvg_bull_active_column_exists(self, fvg_result):
        result, _ = fvg_result
        assert "fvg_bull_active" in result.columns, (
            f"fvg_bull_active not found. FVG columns present: "
            f"{[c for c in result.columns if 'fvg' in c]}"
        )

    def test_fvg_bull_confirmed_column_exists(self, fvg_result):
        result, _ = fvg_result
        assert "fvg_bull_confirmed" in result.columns, (
            f"fvg_bull_confirmed not found. FVG columns: "
            f"{[c for c in result.columns if 'fvg' in c]}"
        )

    def test_fvg_bull_active_fires_near_pattern(self, fvg_result):
        """At or shortly after fvg_bar_idx the bull FVG signal should be active."""
        result, fvg_bar_idx = fvg_result
        window = result["fvg_bull_active"].iloc[fvg_bar_idx: fvg_bar_idx + 5]
        assert window.max() >= 1, (
            f"fvg_bull_active did not fire in window [{fvg_bar_idx}, {fvg_bar_idx + 5}). "
            f"Values: {window.tolist()}"
        )

    def test_fvg_bull_active_fires_in_dataset(self, fvg_result):
        """Fallback: bull FVG signal fires at least once in the entire dataset."""
        result, _ = fvg_result
        assert result["fvg_bull_active"].max() >= 1, (
            "fvg_bull_active never activated over the full dataset"
        )

    def test_fvg_bear_active_column_exists(self, fvg_result):
        result, _ = fvg_result
        assert "fvg_bear_active" in result.columns

    def test_fvg_columns_binary_where_active(self, fvg_result):
        """fvg_bull_active and fvg_bear_active should only contain 0 or 1."""
        result, _ = fvg_result
        for col in ("fvg_bull_active", "fvg_bear_active",
                    "fvg_bull_confirmed", "fvg_bear_confirmed"):
            if col in result.columns:
                vals = result[col].dropna()
                assert vals.isin([0.0, 1.0]).all(), (
                    f"{col} contains non-binary values: {vals.unique()}"
                )


# =============================================================================
# TestSMCCHOCH
# =============================================================================

class TestSMCCHOCH:
    """Verify CHOCH columns exist, are binary, and fire on random-walk data."""

    @pytest.fixture(scope="class")
    def choch_result(self, indicators):
        df = make_m15_ohlcv(n=3000, seed=42)
        return compute_indicator_pool(df, indicators=indicators)

    def test_ms_choch_bull_column_exists(self, choch_result):
        assert "ms_choch_bull" in choch_result.columns, (
            f"ms_choch_bull missing. MS columns: "
            f"{[c for c in choch_result.columns if c.startswith('ms_')]}"
        )

    def test_ms_choch_bear_column_exists(self, choch_result):
        assert "ms_choch_bear" in choch_result.columns

    def test_ms_bos_columns_exist(self, choch_result):
        assert "ms_bos_bull" in choch_result.columns
        assert "ms_bos_bear" in choch_result.columns

    def test_ms_trend_column_exists(self, choch_result):
        assert "ms_trend" in choch_result.columns

    def test_choch_bull_is_binary(self, choch_result):
        vals = choch_result["ms_choch_bull"].dropna()
        assert vals.isin([0.0, 1.0]).all(), (
            f"ms_choch_bull non-binary values: {vals.unique()}"
        )

    def test_choch_bear_is_binary(self, choch_result):
        vals = choch_result["ms_choch_bear"].dropna()
        assert vals.isin([0.0, 1.0]).all()

    def test_choch_bull_fires_occasionally(self, choch_result):
        """On 3000 bars of random-walk data, at least one CHOCH should appear."""
        assert choch_result["ms_choch_bull"].sum() >= 1, (
            "ms_choch_bull never fired on 3000-bar random walk"
        )

    def test_choch_bear_fires_occasionally(self, choch_result):
        assert choch_result["ms_choch_bear"].sum() >= 1, (
            "ms_choch_bear never fired on 3000-bar random walk"
        )

    def test_ms_trend_values_valid(self, choch_result):
        """ms_trend should be in {-1, 0, +1} only."""
        vals = choch_result["ms_trend"].dropna()
        assert vals.isin([-1.0, 0.0, 1.0]).all(), (
            f"ms_trend contains unexpected values: {vals.unique()}"
        )


# =============================================================================
# TestSMCPipelineFeatures
# =============================================================================

class TestSMCPipelineFeatures:
    """Verify the full SMC pipeline output shape and data quality."""

    @pytest.fixture(scope="class")
    def pipeline_result(self, indicators):
        df = make_m15_ohlcv(n=2000, seed=99)
        return df, compute_indicator_pool(df, indicators=indicators)

    def test_result_has_many_feature_columns(self, pipeline_result):
        _, result = pipeline_result
        ohlcv = {"O", "H", "L", "C", "V"}
        feature_cols = [c for c in result.columns if c not in ohlcv]
        assert len(feature_cols) > 60, (
            f"Expected >60 feature columns, got {len(feature_cols)}: {feature_cols}"
        )

    def test_no_inf_values(self, pipeline_result):
        _, result = pipeline_result
        numeric = result.select_dtypes(include=[np.number])
        inf_cols = [c for c in numeric.columns if np.isinf(numeric[c]).any()]
        assert not inf_cols, f"Inf values found in columns: {inf_cols}"

    def test_index_preserved(self, pipeline_result):
        df, result = pipeline_result
        assert result.index.equals(df.index), (
            "Result index does not match input index"
        )

    def test_ohlcv_columns_present(self, pipeline_result):
        _, result = pipeline_result
        for col in ("O", "H", "L", "C", "V"):
            assert col in result.columns, f"OHLCV column '{col}' missing from result"

    def test_no_all_nan_feature_columns(self, pipeline_result):
        """Most feature columns should have at least some non-NaN values.

        Distance columns (e.g. liq_eqh_dist) may be all-NaN when the
        corresponding rare pattern (equal highs / equal lows) was not
        detected in the synthetic data window, so we allow up to 10%
        of feature columns to be entirely NaN.
        """
        _, result = pipeline_result
        ohlcv = {"O", "H", "L", "C", "V"}
        feature_cols = [c for c in result.columns if c not in ohlcv]
        all_nan_cols = [c for c in feature_cols if result[c].isna().all()]
        ratio = len(all_nan_cols) / max(len(feature_cols), 1)
        assert ratio < 0.10, (
            f"{len(all_nan_cols)}/{len(feature_cols)} feature columns are entirely NaN "
            f"(>{ratio:.0%}): {all_nan_cols}"
        )
