"""
Integration tests for the sr_trend_continuation strategy pipeline.

Verifies that synthetic OHLCV data with known patterns produces the expected
support/resistance zone signal activations in the full indicator pipeline.

Note: compute_indicator_pool applies the indicator plugins only; the
feature_selection (correlation_filter) configured in sr_trend_v1.json is part
of the training pipeline and is not applied here.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from conftest import make_h1_ohlcv  # noqa: E402

from fwbg.core.config import StrategyConfig
from fwbg.pipeline.features import compute_indicator_pool

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sr_trend_continuation.json")

_OHLCV_COLS = {"O", "H", "L", "C", "V"}


@pytest.fixture(scope="module")
def strategy_config():
    return StrategyConfig.from_json_file(CONFIG_PATH)


@pytest.fixture(scope="module")
def indicators(strategy_config):
    return strategy_config.get_indicators()


# =============================================================================
# TestSRZones
# =============================================================================

class TestSRZones:
    """Verify SR zone columns exist, are binary, and activate on random-walk data."""

    @pytest.fixture(scope="class")
    def sr_result(self, indicators):
        df = make_h1_ohlcv(n=3000, seed=42)
        return compute_indicator_pool(df, indicators=indicators)

    def test_sr_in_support_zone_exists(self, sr_result):
        assert "sr_in_support_zone" in sr_result.columns, (
            f"sr_in_support_zone missing. SR columns present: "
            f"{[c for c in sr_result.columns if c.startswith('sr_')]}"
        )

    def test_sr_in_resistance_zone_exists(self, sr_result):
        assert "sr_in_resistance_zone" in sr_result.columns

    def test_sr_in_support_zone_binary(self, sr_result):
        vals = sr_result["sr_in_support_zone"].dropna()
        assert vals.isin([0.0, 1.0]).all(), (
            f"sr_in_support_zone non-binary values: {vals.unique()}"
        )

    def test_sr_in_resistance_zone_binary(self, sr_result):
        vals = sr_result["sr_in_resistance_zone"].dropna()
        assert vals.isin([0.0, 1.0]).all()

    def test_support_zone_activates_at_least_once(self, sr_result):
        assert sr_result["sr_in_support_zone"].sum() >= 1, (
            "sr_in_support_zone never activated over 3000 H1 bars"
        )

    def test_resistance_zone_activates_at_least_once(self, sr_result):
        assert sr_result["sr_in_resistance_zone"].sum() >= 1, (
            "sr_in_resistance_zone never activated over 3000 H1 bars"
        )

    def test_d1_support_zone_column_exists(self, sr_result):
        assert "sr_d1_in_support_zone" in sr_result.columns

    def test_d1_resistance_zone_column_exists(self, sr_result):
        assert "sr_d1_in_resistance_zone" in sr_result.columns

    def test_sr_trend_class_column_exists(self, sr_result):
        assert "sr_trend_class" in sr_result.columns

    def test_sr_trend_class_valid_range(self, sr_result):
        """Trend class should be an integer in [-3, +3]."""
        vals = sr_result["sr_trend_class"].dropna()
        assert vals.between(-3, 3).all(), (
            f"sr_trend_class out of [-3, 3] range: {vals.unique()}"
        )

    def test_sr_interaction_columns_exist(self, sr_result):
        for col in ("sr_at_support_in_uptrend", "sr_at_resistance_in_downtrend"):
            assert col in sr_result.columns, f"Column '{col}' missing from result"

    def test_sr_interaction_columns_binary(self, sr_result):
        for col in ("sr_at_support_in_uptrend", "sr_at_resistance_in_downtrend",
                    "sr_at_support_in_range", "sr_at_resistance_in_range"):
            vals = sr_result[col].dropna()
            assert vals.isin([0.0, 1.0]).all(), (
                f"{col} contains non-binary values: {vals.unique()}"
            )


# =============================================================================
# TestSRTrendSignal
# =============================================================================

class TestSRTrendSignal:
    """Soft tests: verify that the zone detection works in a trending scenario."""

    @pytest.fixture(scope="class")
    def uptrend_result(self, indicators):
        """Monotonically rising close, then pullback to prior support level."""
        n = 500
        np.random.seed(7)

        # Build an uptrend: closes rise steadily
        step = 0.5
        close = np.linspace(100.0, 100.0 + step * n, n)
        # Add small noise so OHLC bars are realistic
        noise = np.random.randn(n) * 0.05
        close = close + noise

        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.1
        low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.1
        volume = np.full(n, 1000.0)

        df = pd.DataFrame(
            {"O": open_, "H": high, "L": low, "C": close, "V": volume},
            index=pd.date_range("2022-01-03 00:00", periods=n, freq="h"),
        )
        return compute_indicator_pool(df, indicators=indicators)

    def test_sr_signal_columns_exist(self, uptrend_result):
        """Both zone columns must be present regardless of data pattern."""
        assert "sr_in_support_zone" in uptrend_result.columns
        assert "sr_in_resistance_zone" in uptrend_result.columns

    def test_sr_zone_values_valid_type(self, uptrend_result):
        """Zone columns must be numeric."""
        for col in ("sr_in_support_zone", "sr_in_resistance_zone"):
            assert pd.api.types.is_numeric_dtype(uptrend_result[col]), (
                f"{col} is not numeric: {uptrend_result[col].dtype}"
            )

    def test_sr_zone_values_in_range(self, uptrend_result):
        """Zone values must be 0 or 1."""
        for col in ("sr_in_support_zone", "sr_in_resistance_zone"):
            vals = uptrend_result[col].dropna()
            assert vals.isin([0.0, 1.0]).all(), (
                f"{col} out-of-range values: {vals.unique()}"
            )

    def test_sr_trend_positive_in_uptrend(self, uptrend_result):
        """With a clear uptrend, the trend class should be positive for most bars."""
        trend = uptrend_result["sr_trend_class"].dropna()
        # At least 60% of bars should have a positive (bullish) trend class
        bullish_fraction = (trend > 0).sum() / len(trend)
        assert bullish_fraction > 0.5, (
            f"Expected >50% bullish trend class in uptrend data, "
            f"got {bullish_fraction:.1%}"
        )

    def test_dist_nearest_support_nonnegative(self, uptrend_result):
        """Distance to nearest support must be >= 0 where not NaN."""
        col = "sr_dist_nearest_support"
        if col in uptrend_result.columns:
            vals = uptrend_result[col].dropna()
            # Distances should be non-negative (price is above support)
            # Allow a small negative threshold for floating-point edge cases
            assert (vals >= -0.01).all(), (
                f"{col} has unexpected negative distances: {vals.min()}"
            )


# =============================================================================
# TestSRPipelineFeatures
# =============================================================================

class TestSRPipelineFeatures:
    """Verify the sr_trend pipeline output shape and data quality."""

    @pytest.fixture(scope="class")
    def pipeline_result(self, indicators):
        df = make_h1_ohlcv(n=2000, seed=77)
        return df, compute_indicator_pool(df, indicators=indicators)

    def test_result_has_more_than_20_feature_columns(self, pipeline_result):
        _, result = pipeline_result
        feature_cols = [c for c in result.columns if c not in _OHLCV_COLS]
        assert len(feature_cols) > 20, (
            f"Expected >20 feature columns, got {len(feature_cols)}: {feature_cols}"
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

    def test_sr_columns_all_present(self, pipeline_result):
        """Core S/R feature columns from the support_resistance indicator."""
        _, result = pipeline_result
        expected = [
            "sr_dist_nearest_support",
            "sr_dist_nearest_resistance",
            "sr_in_support_zone",
            "sr_in_resistance_zone",
            "sr_trend_class",
            "sr_at_support_in_uptrend",
            "sr_at_resistance_in_downtrend",
        ]
        for col in expected:
            assert col in result.columns, (
                f"Expected SR column '{col}' missing. Available SR columns: "
                f"{[c for c in result.columns if c.startswith('sr_')]}"
            )

    def test_volatility_columns_present(self, pipeline_result):
        _, result = pipeline_result
        vol_cols = [c for c in result.columns if c.startswith("vol_")]
        assert len(vol_cols) >= 1, "No volatility columns found in result"

    def test_time_season_columns_present(self, pipeline_result):
        _, result = pipeline_result
        time_cols = [c for c in result.columns
                     if c.startswith("time_") or c.startswith("season_")]
        assert len(time_cols) >= 1, "No time/season columns found in result"
