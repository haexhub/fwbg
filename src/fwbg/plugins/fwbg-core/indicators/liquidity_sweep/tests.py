"""Tests for the liquidity_sweep indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_lsw = import_plugin_module("fwbg-core", "indicators", "liquidity_sweep")
if _lsw is None:
    pytest.skip("liquidity_sweep plugin not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_bull_sweep_df(swing_lookback=5, n_post=4):
    """DataFrame with a clear bullish sweep.

    Bars 0..swing_lookback-1: stable lows at 100 (establishing swing low)
    Bar swing_lookback:       L=98 (sweep below 100), C=104 (closes back above)
    Bars after:               L=104, C=106 (price rises, zone stays active)
    """
    sweep_i = swing_lookback
    n = swing_lookback + 1 + n_post
    lows = [100.0] * swing_lookback + [98.0] + [104.0] * n_post
    closes = [104.0] * swing_lookback + [104.0] + [106.0] * n_post
    return pd.DataFrame(
        {
            "O": [102.0] * n,
            "H": [108.0] * n,
            "L": lows,
            "C": closes,
            "V": [100.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def _make_bear_sweep_df(swing_lookback=5, n_post=4):
    """DataFrame with a clear bearish sweep.

    Bars 0..swing_lookback-1: stable highs at 110 (establishing swing high)
    Bar swing_lookback:       H=112 (sweep above 110), C=108 (closes back below)
    Bars after:               H=106, C=106 (price falls, zone stays active)
    """
    n = swing_lookback + 1 + n_post
    highs = [110.0] * swing_lookback + [112.0] + [106.0] * n_post
    closes = [108.0] * swing_lookback + [108.0] + [106.0] * n_post
    return pd.DataFrame(
        {
            "O": [108.0] * n,
            "H": highs,
            "L": [104.0] * n,
            "C": closes,
            "V": [100.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def _make_sweep_df(n=500):
    """General DataFrame with guaranteed sweep events for integration tests."""
    np.random.seed(42)
    prices = np.linspace(100, 120, n) + np.random.randn(n) * 0.2

    df = pd.DataFrame(
        {
            "O": prices + np.random.randn(n) * 0.1,
            "H": prices + np.abs(np.random.randn(n) * 0.3) + 0.5,
            "L": prices - np.abs(np.random.randn(n) * 0.3) - 0.5,
            "C": prices,
            "V": np.ones(n) * 100,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))

    # Force explicit bull sweep zones at bars 60, 120, 180, ... to ensure
    # test_not_all_nan passes (active zones within zone_lookback=50 of bar 100+)
    for sweep_bar in range(60, n - 5, 60):
        idx = df.index[sweep_bar]
        actual_swing_low = df["L"][max(0, sweep_bar - 20): sweep_bar].min()
        df.loc[idx, "L"] = actual_swing_low - 0.5   # wick below swing low
        if df.loc[idx, "C"] <= actual_swing_low:
            df.loc[idx, "C"] = actual_swing_low + 0.5
        df.loc[idx, "H"] = max(df.loc[idx, "H"], df.loc[idx, "C"] + 0.1)

    return df


# ---------------------------------------------------------------------------
# Bullish sweep detection
# ---------------------------------------------------------------------------

class TestBullishSweepDetection:
    """Tests for bullish sweep (fake-out below swing low then close back above)."""

    def test_bull_sweep_detected(self):
        """L[i] < prev_swing_low AND C[i] > prev_swing_low → lsw_bull_active=1."""
        sw = 5
        df = _make_bull_sweep_df(swing_lookback=sw)
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)

        # Sweep at bar sw; after shift it appears at iloc[sw+1]
        assert result["lsw_bull_active"].iloc[sw + 1] == 1.0, (
            f"Expected lsw_bull_active=1 at iloc[{sw+1}], "
            f"got {result['lsw_bull_active'].iloc[sw+1]}"
        )

    def test_no_bull_sweep_when_close_stays_below(self):
        """If C[i] <= swing_low (displacement), no bull sweep is recorded."""
        sw = 5
        df = _make_bull_sweep_df(swing_lookback=sw)
        # Change sweep bar: close stays below swing_low (100)
        df.iloc[sw, df.columns.get_loc("C")] = 99.0   # close below 100 → no sweep
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)
        assert result["lsw_bull_active"].fillna(0).sum() == 0.0

    def test_no_bull_sweep_when_no_wick_below(self):
        """If L[i] >= swing_low, no sweep happened."""
        sw = 5
        df = _make_bull_sweep_df(swing_lookback=sw)
        # Change sweep bar: low stays at swing_low (no wick below)
        df.iloc[sw, df.columns.get_loc("L")] = 100.5   # above swing_low=100
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)
        assert result["lsw_bull_active"].fillna(0).sum() == 0.0


# ---------------------------------------------------------------------------
# Bearish sweep detection
# ---------------------------------------------------------------------------

class TestBearishSweepDetection:
    """Tests for bearish sweep (fake-out above swing high then close back below)."""

    def test_bear_sweep_detected(self):
        """H[i] > prev_swing_high AND C[i] < prev_swing_high → lsw_bear_active=1."""
        sw = 5
        df = _make_bear_sweep_df(swing_lookback=sw)
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)

        assert result["lsw_bear_active"].iloc[sw + 1] == 1.0, (
            f"Expected lsw_bear_active=1 at iloc[{sw+1}], "
            f"got {result['lsw_bear_active'].iloc[sw+1]}"
        )

    def test_no_bear_sweep_when_close_stays_above(self):
        """If C[i] >= swing_high (displacement), no bear sweep is recorded."""
        sw = 5
        df = _make_bear_sweep_df(swing_lookback=sw)
        df.iloc[sw, df.columns.get_loc("C")] = 111.0   # close above 110 → no sweep
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)
        assert result["lsw_bear_active"].fillna(0).sum() == 0.0

    def test_no_bear_sweep_when_no_wick_above(self):
        """If H[i] <= swing_high, no sweep happened."""
        sw = 5
        df = _make_bear_sweep_df(swing_lookback=sw)
        df.iloc[sw, df.columns.get_loc("H")] = 109.5   # below swing_high=110
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)
        assert result["lsw_bear_active"].fillna(0).sum() == 0.0


# ---------------------------------------------------------------------------
# Zone persistence and invalidation
# ---------------------------------------------------------------------------

class TestSweepZones:
    """Tests for sweep zone lifecycle."""

    def test_bull_zone_persists_across_bars(self):
        """Bull sweep zone stays active for subsequent bars."""
        sw = 5
        df = _make_bull_sweep_df(swing_lookback=sw, n_post=4)
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)

        # Bars sw, sw+1, sw+2, sw+3 should all be active (after shift: sw+1 ... sw+4)
        for offset in range(1, 4):
            assert result["lsw_bull_active"].iloc[sw + offset] == 1.0, (
                f"Expected lsw_bull_active=1 at iloc[{sw+offset}]"
            )

    def test_bull_zone_invalidated_on_re_sweep(self):
        """Bull zone is removed when price wicks below the zone bottom again.

        Use a displacement scenario: L goes below zone_bottom AND C also stays
        below swing_low so no new sweep zone is created (displacement, not fake-out).
        """
        sw = 5
        df = _make_bull_sweep_df(swing_lookback=sw, n_post=4)
        # At bar sw+1: price displaces below zone_bottom (98) and closes below swing_low (100)
        # → invalidates zone and does NOT create a new bull sweep (C < prev_sl)
        re_sweep_i = sw + 1
        df.iloc[re_sweep_i, df.columns.get_loc("L")] = 97.0   # below zone_bottom=98
        df.iloc[re_sweep_i, df.columns.get_loc("C")] = 97.5   # close BELOW swing_low=100 → no new sweep
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)

        # After shift: result.iloc[re_sweep_i+1] reflects bar re_sweep_i
        assert result["lsw_bull_active"].iloc[re_sweep_i + 1] == 0.0, (
            "Zone should be invalidated after re-sweep"
        )

    def test_bear_zone_persists_across_bars(self):
        """Bear sweep zone stays active for subsequent bars."""
        sw = 5
        df = _make_bear_sweep_df(swing_lookback=sw, n_post=4)
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)

        for offset in range(1, 4):
            assert result["lsw_bear_active"].iloc[sw + offset] == 1.0, (
                f"Expected lsw_bear_active=1 at iloc[{sw+offset}]"
            )

    def test_bear_zone_invalidated_on_re_sweep(self):
        """Bear zone is removed when price wicks above zone top again.

        Use a displacement scenario: H goes above zone_top AND C also stays
        above swing_high so no new sweep zone is created (displacement, not fake-out).
        """
        sw = 5
        df = _make_bear_sweep_df(swing_lookback=sw, n_post=4)
        re_sweep_i = sw + 1
        df.iloc[re_sweep_i, df.columns.get_loc("H")] = 113.0   # above zone_top=112
        df.iloc[re_sweep_i, df.columns.get_loc("C")] = 112.5   # close ABOVE swing_high=110 → no new sweep
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)
        assert result["lsw_bear_active"].iloc[re_sweep_i + 1] == 0.0

    def test_zone_expires_after_lookback(self):
        """Zone is removed after zone_lookback bars."""
        sw = 5
        zone_lb = 3
        df = _make_bull_sweep_df(swing_lookback=sw, n_post=zone_lb + 2)
        result = _lsw.LiquiditySweepIndicator().compute(
            df, swing_lookback=sw, zone_lookback=zone_lb
        )
        sweep_i = sw
        # After shift: result.iloc[sweep_i + zone_lb + 1] reflects bar sweep_i + zone_lb
        # At bar sweep_i + zone_lb: age = zone_lb → zone should be expired (age > zone_lb)
        expired_row = sweep_i + zone_lb + 1
        assert result["lsw_bull_active"].iloc[expired_row] == 0.0, (
            f"Expected zone expired at iloc[{expired_row}]"
        )

    def test_bull_and_bear_independent(self):
        """Bull and bear sweep features are independent (bull sweep → bear_active=0)."""
        sw = 5
        df = _make_bull_sweep_df(swing_lookback=sw)
        result = _lsw.LiquiditySweepIndicator().compute(df, swing_lookback=sw)
        assert result["lsw_bear_active"].fillna(0).sum() == 0.0


# ---------------------------------------------------------------------------
# Feature properties
# ---------------------------------------------------------------------------

class TestSweepFeatureProperties:
    """Tests for feature shapes, types and values."""

    def test_all_features_present(self):
        indicator = _lsw.LiquiditySweepIndicator()
        result = indicator.compute(_make_sweep_df())
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self):
        indicator = _lsw.LiquiditySweepIndicator()
        assert len(indicator.get_feature_columns()) == 10

    def test_binary_features(self):
        indicator = _lsw.LiquiditySweepIndicator()
        result = indicator.compute(_make_sweep_df())
        for col in ["lsw_bull_active", "lsw_bear_active",
                    "lsw_bull_in_zone", "lsw_bear_in_zone"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} not binary: {vals}"

    def test_recency_in_unit_interval(self):
        """Recency features should be in [0, 1]."""
        indicator = _lsw.LiquiditySweepIndicator()
        result = indicator.compute(_make_sweep_df())
        for col in ["lsw_bull_recency", "lsw_bear_recency"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= 0.0, f"{col} has values < 0"
                assert vals.max() <= 1.0, f"{col} has values > 1"

    def test_shift_applied(self):
        """All features must be shifted by 1 bar (lookahead prevention)."""
        indicator = _lsw.LiquiditySweepIndicator()
        result = indicator.compute(_make_sweep_df(n=100))
        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted (iloc[0] not NaN)"

    def test_no_inf_values(self):
        indicator = _lsw.LiquiditySweepIndicator()
        result = indicator.compute(_make_sweep_df())
        for col in indicator.get_feature_columns():
            inf_count = np.isinf(result[col].dropna()).sum()
            assert inf_count == 0, f"{col} has {inf_count} inf values"

    def test_not_all_nan(self):
        """After warmup, each feature column must have at least some non-NaN values."""
        indicator = _lsw.LiquiditySweepIndicator()
        result = indicator.compute(_make_sweep_df(n=500))
        late = result.iloc[100:]
        for col in indicator.get_feature_columns():
            vals = late[col].dropna()
            assert len(vals) > 0, f"{col} is all NaN after warmup"

    def test_no_undeclared_features(self):
        indicator = _lsw.LiquiditySweepIndicator()
        df = _make_sweep_df(n=200)
        original_cols = set(df.columns)
        result = indicator.compute(df)
        new_cols = set(result.columns) - original_cols
        declared = set(indicator.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared features: {undeclared}"


# ---------------------------------------------------------------------------
# Plugin integration
# ---------------------------------------------------------------------------

class TestPluginIntegration:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator
        discover_plugins()
        cls = get_indicator("liquidity_sweep")
        assert cls is not None

    def test_all_declared_features_in_output(self):
        indicator = _lsw.LiquiditySweepIndicator()
        result = indicator.compute(_make_sweep_df(n=200))
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Declared {col} missing from output"
