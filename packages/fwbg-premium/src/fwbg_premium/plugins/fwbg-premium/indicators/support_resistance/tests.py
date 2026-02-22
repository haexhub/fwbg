"""Tests for the support_resistance indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_sr = import_plugin_module("fwbg-premium", "indicators", "support_resistance")
if _sr is None:
    pytest.skip("support_resistance plugin not available", allow_module_level=True)


class TestSwingDetection:
    """Tests for _detect_swings helper function."""

    def test_detects_obvious_swing_high(self):
        """A clear peak must be detected as swing high."""
        #                          v peak at index 5
        highs = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float64)
        lows = highs - 0.5
        period = 5

        swing_highs, _ = _sr._detect_swings(highs, lows, period)

        # Peak is at index 5 (value 6), confirmed at index 5 + period = 10
        assert swing_highs[10] == 6.0

    def test_detects_obvious_swing_low(self):
        """A clear trough must be detected as swing low."""
        #                           v trough at index 5
        lows = np.array([6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6], dtype=np.float64)
        highs = lows + 0.5
        period = 5

        _, swing_lows = _sr._detect_swings(highs, lows, period)

        assert swing_lows[10] == 1.0

    def test_no_swing_in_flat_data(self):
        """Flat data should not produce swings."""
        n = 50
        highs = np.full(n, 100.0)
        lows = np.full(n, 99.0)

        swing_highs, swing_lows = _sr._detect_swings(highs, lows, period=5)

        # All values should be NaN (no clear swing)
        assert np.all(np.isnan(swing_highs)) or swing_highs[~np.isnan(swing_highs)].size <= 2

    def test_no_lookahead_bias(self):
        """Swing at index i must NOT be confirmed before index i + period."""
        highs = np.array(
            [float(i) for i in range(15)] + [float(14 - i) for i in range(15)]
        )
        lows = highs - 0.5
        period = 5

        swing_highs, _ = _sr._detect_swings(highs, lows, period)

        # Peak is at index 14 (value 14.0)
        # Must NOT appear before index 14 + period = 19
        for i in range(19):
            assert np.isnan(swing_highs[i]) or swing_highs[i] != 14.0, (
                f"Swing high leaked at index {i} (before confirmation at 19)"
            )

    def test_multiple_swings(self):
        """Multiple peaks should all be detected."""
        # Two peaks: index 5 and index 15
        vals = (
            list(range(6))
            + list(range(4, -1, -1))
            + list(range(1, 7))
            + list(range(5, -1, -1))
        )
        highs = np.array(vals, dtype=np.float64)
        lows = highs - 0.5
        period = 5

        swing_highs, _ = _sr._detect_swings(highs, lows, period)

        detected = swing_highs[~np.isnan(swing_highs)]
        assert len(detected) >= 2


class TestZoneClustering:
    """Tests for _cluster_levels."""

    def test_identical_levels_cluster(self):
        """Levels at the same price must cluster into one zone."""
        levels = np.array([100.0, 100.1, 100.05])
        atr = 1.0
        zones = _sr._cluster_levels(levels, atr, threshold=1.5)
        assert len(zones) == 1
        assert zones[0]["touches"] == 3

    def test_distant_levels_separate(self):
        """Levels far apart must be separate zones."""
        levels = np.array([100.0, 110.0, 120.0])
        atr = 1.0
        zones = _sr._cluster_levels(levels, atr, threshold=1.5)
        assert len(zones) == 3
        assert all(z["touches"] == 1 for z in zones)

    def test_mixed_clustering(self):
        """Close levels cluster, distant ones separate."""
        levels = np.array([100.0, 100.5, 100.3, 110.0, 110.2])
        atr = 1.0
        zones = _sr._cluster_levels(levels, atr, threshold=1.5)
        assert len(zones) == 2
        assert zones[0]["touches"] == 3  # 100.0, 100.3, 100.5
        assert zones[1]["touches"] == 2  # 110.0, 110.2

    def test_empty_levels(self):
        """All-NaN levels must return empty."""
        levels = np.array([np.nan, np.nan, np.nan])
        zones = _sr._cluster_levels(levels, atr=1.0, threshold=1.5)
        assert len(zones) == 0

    def test_zone_has_center(self):
        """Zone center must be the mean of clustered levels."""
        levels = np.array([100.0, 101.0])
        zones = _sr._cluster_levels(levels, atr=1.0, threshold=1.5)
        assert len(zones) == 1
        assert zones[0]["center"] == pytest.approx(100.5)


class TestFindZones:
    """Tests for _find_zones (swing detection + clustering combined)."""

    def _make_v_shape(self, n=100, peak_idx=50, peak_val=110.0, base_val=100.0):
        """Create V-shape price data with a clear peak."""
        prices = np.full(n, base_val)
        for i in range(peak_idx):
            prices[i] = base_val + (peak_val - base_val) * i / peak_idx
        for i in range(peak_idx, n):
            prices[i] = peak_val - (peak_val - base_val) * (i - peak_idx) / (n - peak_idx)
        return prices

    def test_finds_resistance_from_peak(self):
        """A peak must produce a resistance zone."""
        prices = self._make_v_shape()
        highs = prices + 0.5
        lows = prices - 0.5
        atr = np.full(len(prices), 1.0)

        zones = _sr._find_zones(
            highs, lows, atr, swing_periods=[5], lookback=200, cluster_threshold=1.5
        )

        resistance_zones = [z for z in zones if z["type"] in ("resistance", "both")]
        assert len(resistance_zones) >= 1

    def test_finds_support_from_trough(self):
        """A trough must produce a support zone."""
        n = 100
        prices = np.full(n, 110.0)
        for i in range(50):
            prices[i] = 110.0 - 10.0 * i / 50
        for i in range(50, n):
            prices[i] = 100.0 + 10.0 * (i - 50) / 50
        highs = prices + 0.5
        lows = prices - 0.5
        atr = np.full(n, 1.0)

        zones = _sr._find_zones(
            highs, lows, atr, swing_periods=[5], lookback=200, cluster_threshold=1.5
        )

        support_zones = [z for z in zones if z["type"] in ("support", "both")]
        assert len(support_zones) >= 1


class TestTrendClassification:
    """Tests for _classify_trend (Rayner-style -3..+3)."""

    def test_strong_uptrend(self):
        """Price above all MAs, MAs bullish aligned -> +3."""
        assert _sr._classify_trend(close=110, ma20=108, ma50=105, ma200=100) == 3

    def test_healthy_uptrend(self):
        """Price between MA20 and MA50, MAs bullish aligned -> +2."""
        assert _sr._classify_trend(close=106, ma20=108, ma50=105, ma200=100) == 2

    def test_weak_uptrend(self):
        """Price below MA50 but MAs still bullish aligned -> +1."""
        assert _sr._classify_trend(close=103, ma20=108, ma50=105, ma200=100) == 1

    def test_strong_downtrend(self):
        """Price below all MAs, MAs bearish aligned -> -3."""
        assert _sr._classify_trend(close=90, ma20=92, ma50=95, ma200=100) == -3

    def test_healthy_downtrend(self):
        """Price between MA20 and MA50, MAs bearish aligned -> -2."""
        assert _sr._classify_trend(close=94, ma20=92, ma50=95, ma200=100) == -2

    def test_weak_downtrend(self):
        """Price above MA50 but MAs still bearish aligned -> -1."""
        assert _sr._classify_trend(close=97, ma20=92, ma50=95, ma200=100) == -1

    def test_sideways(self):
        """MAs not aligned -> 0 (sideways/range)."""
        # MA20 > MA200 > MA50 -> not bull or bear aligned
        assert _sr._classify_trend(close=100, ma20=102, ma50=98, ma200=100) == 0


def _make_trending_df(n=500):
    """Create OHLC DataFrame with clear trends and S/R bounces."""
    np.random.seed(42)
    trend = np.linspace(100, 120, n) + np.random.randn(n) * 0.5
    oscillation = 3 * np.sin(np.linspace(0, 8 * np.pi, n))
    prices = trend + oscillation

    df = pd.DataFrame(
        {
            "O": prices + np.random.randn(n) * 0.1,
            "H": prices + np.abs(np.random.randn(n) * 0.5),
            "L": prices - np.abs(np.random.randn(n) * 0.5),
            "C": prices,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


class TestComputeTier1:
    """Tests for H1 S/R zone features."""

    def test_h1_features_present(self):
        """All H1 S/R features must be present in output."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        expected_cols = [
            "sr_dist_nearest_support",
            "sr_dist_nearest_resistance",
            "sr_support_strength",
            "sr_resistance_strength",
            "sr_in_support_zone",
            "sr_in_resistance_zone",
            "sr_nearest_is_flip_zone",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_d1_features_present(self):
        """All D1 S/R features must be present in output."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        expected_cols = [
            "sr_d1_dist_nearest_support",
            "sr_d1_dist_nearest_resistance",
            "sr_d1_support_strength",
            "sr_d1_resistance_strength",
            "sr_d1_in_support_zone",
            "sr_d1_in_resistance_zone",
            "sr_d1_nearest_is_flip_zone",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_distances_are_atr_normalized(self):
        """S/R distances must be in ATR units (typically 0-20 range)."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        dist = result["sr_dist_nearest_support"].dropna()
        assert len(dist) > 0
        assert dist.median() < 50, (
            f"Distances too large ({dist.median():.1f}), probably not ATR-normalized"
        )

    def test_in_zone_is_binary(self):
        """in_support_zone and in_resistance_zone must be 0 or 1."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_in_support_zone", "sr_in_resistance_zone"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} has non-binary values: {vals}"

    def test_strength_is_positive(self):
        """Zone strength (touches) must be >= 0."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_support_strength", "sr_resistance_strength"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals >= 0).all(), f"{col} has negative values"

    def test_shift_applied(self):
        """Features must be shifted by 1 bar (lookahead prevention)."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=300)
        result = indicator.compute(df)

        for col in ["sr_dist_nearest_support", "sr_dist_nearest_resistance"]:
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), (
                    f"{col} not shifted — first row is not NaN"
                )


class TestComputeTier2:
    """Tests for trend context features."""

    def test_trend_features_present(self):
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in [
            "sr_trend_class",
            "sr_pullback_depth",
            "sr_ma_alignment",
            "sr_price_vs_ma20",
            "sr_price_vs_ma50",
            "sr_price_vs_ma200",
        ]:
            assert col in result.columns, f"Missing: {col}"

    def test_trend_class_range(self):
        """sr_trend_class must be in [-3, +3]."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)
        vals = result["sr_trend_class"].dropna()
        assert vals.min() >= -3
        assert vals.max() <= 3

    def test_ma_alignment_range(self):
        """sr_ma_alignment must be in [-1, +1]."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)
        vals = result["sr_ma_alignment"].dropna()
        assert vals.min() >= -1.0 - 0.01
        assert vals.max() <= 1.0 + 0.01


class TestComputeTier3:
    """Tests for interaction features."""

    def test_interaction_features_present(self):
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in [
            "sr_at_support_in_uptrend",
            "sr_at_resistance_in_downtrend",
            "sr_at_support_in_range",
            "sr_at_resistance_in_range",
            "sr_range_width",
            "sr_range_position",
            "sr_breakout_up",
            "sr_breakout_down",
        ]:
            assert col in result.columns, f"Missing: {col}"

    def test_range_position_bounds(self):
        """sr_range_position must be in [0, 1] (or NaN)."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)
        vals = result["sr_range_position"].dropna()
        if len(vals) > 0:
            assert vals.min() >= -0.01
            assert vals.max() <= 1.01

    def test_breakouts_are_binary(self):
        for col in ["sr_breakout_up", "sr_breakout_down"]:
            indicator = _sr.SupportResistanceIndicator()
            df = _make_trending_df()
            result = indicator.compute(df)
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"


class TestTemporalFlip:
    """Tests for temporal S/R flip (resistance broken -> support)."""

    def test_flip_features_present(self):
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_at_flipped_support", "sr_at_flipped_resistance"]:
            assert col in result.columns, f"Missing: {col}"

    def test_flip_features_are_binary(self):
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        for col in ["sr_at_flipped_support", "sr_at_flipped_resistance"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"


class TestTrendBreak:
    """Tests for trend break signal."""

    def test_trend_break_present(self):
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        assert "sr_trend_break" in result.columns

    def test_trend_break_values(self):
        """sr_trend_break must be -1, 0, or 1."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df()
        result = indicator.compute(df)

        vals = result["sr_trend_break"].dropna().unique()
        assert set(vals).issubset({-1.0, 0.0, 1.0}), f"Unexpected values: {vals}"


class TestPluginIntegration:
    """Integration tests — plugin discovery, all features."""

    def test_plugin_discoverable(self):
        """Plugin must be found via discover_plugins."""
        from fwbg.core import discover_plugins, get_indicator

        discover_plugins()
        cls = get_indicator("support_resistance")
        assert cls is not None

    def test_all_feature_columns_present(self):
        """Every column in get_feature_columns() must appear in compute() output."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=600)
        result = indicator.compute(df)

        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Declared feature {col} missing from output"

    def test_no_extra_undeclared_features(self):
        """compute() must not add columns beyond OHLC + declared features."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=300)
        original_cols = set(df.columns)
        result = indicator.compute(df)
        new_cols = set(result.columns) - original_cols
        declared = set(indicator.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared features in output: {undeclared}"

    def test_feature_count(self):
        """Plugin must declare exactly 31 features."""
        indicator = _sr.SupportResistanceIndicator()
        assert len(indicator.get_feature_columns()) == 31

    def test_not_all_nan(self):
        """Features must not be all NaN after warmup period.

        Uses 2000 bars because D1 features need ~960+ bars warmup
        (d1_swing_periods scale by d1_bars=24).
        """
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=2000)
        result = indicator.compute(df)

        late = result.iloc[1000:]
        for col in indicator.get_feature_columns():
            vals = late[col].dropna()
            assert len(vals) > 0, f"{col} is all NaN after warmup"

    def test_no_inf_values(self):
        """No feature should contain inf values."""
        indicator = _sr.SupportResistanceIndicator()
        df = _make_trending_df(n=500)
        result = indicator.compute(df)

        for col in indicator.get_feature_columns():
            inf_count = np.isinf(result[col].dropna()).sum()
            assert inf_count == 0, f"{col} has {inf_count} inf values"
