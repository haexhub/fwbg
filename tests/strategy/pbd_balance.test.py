"""
Integration tests for the pbd_balance strategy pipeline.

Verifies that synthetic M5 OHLCV data with known patterns produces the
correct balance_zones signal features through the full indicator pipeline.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Allow importing conftest helpers directly (not as pytest conftest)
from conftest import make_m5_ohlcv

from fwbg.pipeline.features import compute_indicator_pool
from fwbg.core.config import StrategyConfig

# Path to the strategy config (resolved relative to this file)
_from fwbg.api.workspace import get_strategies_dir as _gsd; CONFIG_PATH = str(_gsd() / "pbd_balance.json")


def load_config():
    return StrategyConfig.from_json_file(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# TestPBDBalanceZoneDetection
# ---------------------------------------------------------------------------

class TestPBDBalanceZoneDetection:
    """bz_in_balance and bz_in_zone columns are produced and are binary."""

    def test_bz_in_balance_column_exists(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        assert "bz_in_balance" in result.columns, (
            f"bz_in_balance missing. Available columns: {list(result.columns)}"
        )

    def test_bz_in_zone_column_exists(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        assert "bz_in_zone" in result.columns

    def test_bz_in_balance_fires_at_least_once(self):
        """On random realistic data bz_in_balance should be 1 sometimes."""
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        col = result["bz_in_balance"].dropna()
        assert col.max() == 1.0, "bz_in_balance never fired on 3000-bar random data"

    def test_bz_in_balance_is_binary(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        col = result["bz_in_balance"].dropna()
        unique_vals = set(col.unique())
        assert unique_vals.issubset({0.0, 1.0}), (
            f"bz_in_balance contains non-binary values: {unique_vals}"
        )

    def test_bz_in_zone_is_binary(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        col = result["bz_in_zone"].dropna()
        unique_vals = set(col.unique())
        assert unique_vals.issubset({0.0, 1.0}), (
            f"bz_in_zone contains non-binary values: {unique_vals}"
        )


# ---------------------------------------------------------------------------
# TestPBDFakeBreakout
# ---------------------------------------------------------------------------

def _make_fake_bull_scenario() -> pd.DataFrame:
    """
    Craft a scenario that guarantees bz_fake_bull fires.

    The balance_zones plugin detects bz_fake_bull at bar i when:
        closes[i-1] < zone_bottom[i-2]   (prev bar closed BELOW old zone bottom)
        closes[i]   >= zone_bottom[i-1]  (current bar closed BACK ABOVE or at zone bottom)

    After shift_features shifts everything +1, the signal appears at bar i+1.

    Construction:
      - 200 warmup bars: narrow range around 100 so zone_top ≈ 100.06,
        zone_bottom ≈ 99.94  (realistic bodies)
      - bar W  : open=100, close=100 (still inside zone)
      - bar W+1: FAKE-BEAR bar — close drops to 99.50, below zone_bottom
      - bar W+2: RECOVERY bar  — close bounces back to 100 (above zone_bottom)
        → bz_fake_bull fires at bar W+2 in the raw compute array
        → appears in result at bar W+3 (after shift)
    """
    np.random.seed(0)
    n_warmup = 220
    # Warmup: tiny-body candles so balance zone is narrow
    base_price = 100.0
    closes_w = base_price + np.cumsum(np.random.randn(n_warmup) * 0.02)
    opens_w  = np.roll(closes_w, 1); opens_w[0] = closes_w[0]
    # Keep bodies tight: spread = 0.02, wicks = 0.04
    highs_w  = np.maximum(opens_w, closes_w) + 0.04
    lows_w   = np.minimum(opens_w, closes_w) - 0.04

    # Known pattern: 4 injected bars after warmup
    # bar 0: neutral inside zone
    # bar 1 (sweep): close well below zone_bottom
    # bar 2 (recovery): close back above zone_bottom → triggers bz_fake_bull
    # bar 3: trailing bar so the shift-by-1 brings the signal into the index
    extra_opens  = np.array([100.0,  100.0,  99.60, 100.0])
    extra_closes = np.array([100.0,   99.50, 100.0, 100.0])
    extra_highs  = np.array([100.04, 100.04, 100.04, 100.04])
    extra_lows   = np.array([99.96,   99.46,  99.56, 99.96])

    opens  = np.concatenate([opens_w,  extra_opens])
    closes = np.concatenate([closes_w, extra_closes])
    highs  = np.concatenate([highs_w,  extra_highs])
    lows   = np.concatenate([lows_w,   extra_lows])
    n = len(closes)
    volume = np.full(n, 1000.0)

    idx = pd.date_range("2022-01-03 00:00", periods=n, freq="5min")
    return pd.DataFrame(
        {"O": opens, "H": highs, "L": lows, "C": closes, "V": volume},
        index=idx,
    )


class TestPBDFakeBreakout:
    """bz_fake_bull and bz_fake_bear signal columns activate on crafted scenarios."""

    def test_bz_fake_bull_fires_in_scenario(self):
        """
        bz_fake_bull must fire within the 5 bars following the recovery bar.
        (bar W+2 raw detection → bar W+3 after shift, but zone evolves so we
        allow a small window: W+3 to W+7)
        """
        config = load_config()
        indicators = config.get_indicators()
        df = _make_fake_bull_scenario()
        result = compute_indicator_pool(df, indicators=indicators)

        assert "bz_fake_bull" in result.columns, "bz_fake_bull column missing"

        # The recovery bar is at raw index 222 (220 warmup + 2).
        # shift_features shifts +1, so the signal appears at result index 223.
        # With 4 trailing bars (n=224 total), index 223 is valid.
        recovery_bar = 222
        # Search from recovery_bar through the end of the DataFrame
        window = result["bz_fake_bull"].iloc[recovery_bar:]
        assert window.max() == 1.0, (
            f"bz_fake_bull did not fire in window around recovery bar.\n"
            f"Window values:\n{window.to_string()}\n"
            f"Full bz_fake_bull non-zero:\n"
            f"{result['bz_fake_bull'][result['bz_fake_bull'] == 1.0]}"
        )

    def test_bz_fake_bull_fires_on_random_data(self):
        """bz_fake_bull fires at least once across 3000 bars of random M5 data."""
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        assert "bz_fake_bull" in result.columns
        assert result["bz_fake_bull"].max() == 1.0, (
            "bz_fake_bull never fired on 3000-bar random data"
        )

    def test_bz_fake_bear_fires_on_random_data(self):
        """bz_fake_bear fires at least once across 3000 bars of random M5 data."""
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        assert "bz_fake_bear" in result.columns
        assert result["bz_fake_bear"].max() == 1.0, (
            "bz_fake_bear never fired on 3000-bar random data"
        )

    def test_bz_fake_signals_are_binary(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=3000)
        result = compute_indicator_pool(df, indicators=indicators)
        for col_name in ("bz_fake_bull", "bz_fake_bear"):
            col = result[col_name].dropna()
            unique_vals = set(col.unique())
            assert unique_vals.issubset({0.0, 1.0}), (
                f"{col_name} contains non-binary values: {unique_vals}"
            )


# ---------------------------------------------------------------------------
# TestPBDPipelineFeatures
# ---------------------------------------------------------------------------

class TestPBDPipelineFeatures:
    """Full pipeline smoke-tests: column count, no inf, index preservation."""

    def test_feature_count_exceeds_40(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=2000)
        result = compute_indicator_pool(df, indicators=indicators)
        ohlcv_cols = {"O", "H", "L", "C", "V"}
        feature_cols = [c for c in result.columns if c not in ohlcv_cols]
        assert len(feature_cols) > 40, (
            f"Expected >40 feature columns, got {len(feature_cols)}: {feature_cols}"
        )

    def test_no_inf_values(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=2000)
        result = compute_indicator_pool(df, indicators=indicators)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        has_inf = np.isinf(result[numeric_cols].values).any()
        assert not has_inf, "Result contains inf values"

    def test_index_preserved(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=2000)
        result = compute_indicator_pool(df, indicators=indicators)
        assert result.index.equals(df.index), "Index was not preserved through the pipeline"

    def test_all_balance_zone_columns_present(self):
        config = load_config()
        indicators = config.get_indicators()
        df = make_m5_ohlcv(n=2000)
        result = compute_indicator_pool(df, indicators=indicators)
        expected = [
            "bz_in_balance", "bz_in_zone", "bz_zone_width",
            "bz_zone_top_dist", "bz_zone_bottom_dist",
            "bz_breakout_bull", "bz_breakout_bear",
            "bz_fake_bear", "bz_fake_bull", "bz_balance_bars",
        ]
        missing = [c for c in expected if c not in result.columns]
        assert not missing, f"Missing balance_zones columns: {missing}"
