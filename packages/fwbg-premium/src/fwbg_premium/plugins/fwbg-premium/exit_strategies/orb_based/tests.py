"""Tests for OrbExitStrategy plugin.

The orb_based strategy uses ORB-range SL and ATR-based TP:
- SL = orb_sl_dist column if present (half the ORB range), fallback to ATR * sl_mult
- TP = ATR * tp_mult (optimizer searches best multiple)
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module
from fwbg.core.context import SimulationContext

_orb_exit = import_plugin_module("fwbg-premium", "exit_strategies", "orb_based")
if _orb_exit is None:
    pytest.skip("fwbg-premium orb_based exit strategy not available", allow_module_level=True)

OrbExitStrategy = _orb_exit.OrbExitStrategy


# --- Fixtures ---

def _make_ohlc(n=100, spread=1.0, seed=42):
    """OHLC at index price ~10000 (DAX-like)."""
    rng = np.random.default_rng(seed)
    close = 10000 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    high = close + np.abs(rng.normal(0, spread * 2, n))
    low = close - np.abs(rng.normal(0, spread * 2, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame({"O": open_, "H": high, "L": low, "C": close})


@pytest.fixture
def ohlc_df():
    return _make_ohlc(n=200)


@pytest.fixture
def ctx():
    return SimulationContext(
        symbol="DAX",
        asset_class="index",
        spread=1.0,
        point=0.1,
        min_trades=10,
        max_trade_bars=100,
        exit_strategy="orb_based",
        exit_params={
            "atr_period": 14,
            "min_tp_pips": 8,
            "min_sl_pips": 5,
        }
    )


# --- Registration ---

class TestOrbExitStrategyRegistration:
    """Strategy must be discoverable via the plugin registry."""

    def test_registered_as_orb_based(self):
        """@register_exit_strategy('orb_based') must be present."""
        from fwbg.core.registry import get_exit_strategy
        cls = get_exit_strategy("orb_based")
        assert cls is OrbExitStrategy

    def test_class_name(self):
        assert OrbExitStrategy.__name__ == "OrbExitStrategy"


# --- Smoke tests ---

class TestOrbExitStrategySmoke:
    """Basic correctness: compute_targets returns valid arrays."""

    def test_compute_targets_returns_two_arrays(self, ohlc_df, ctx):
        strategy = OrbExitStrategy()
        result = strategy.compute_targets(ohlc_df, ctx, tp_mult=2.0, sl_mult=1.0)
        assert len(result) == 2
        targets_long, targets_short = result
        assert len(targets_long) == len(ohlc_df)
        assert len(targets_short) == len(ohlc_df)

    def test_targets_are_binary(self, ohlc_df, ctx):
        strategy = OrbExitStrategy()
        tl, ts = strategy.compute_targets(ohlc_df, ctx, tp_mult=2.0, sl_mult=1.0)
        assert set(np.unique(tl)).issubset({0.0, 1.0})
        assert set(np.unique(ts)).issubset({0.0, 1.0})

    def test_compute_targets_no_crash_with_orb_sl_dist(self, ohlc_df, ctx):
        """compute_targets must not crash when orb_sl_dist column is present."""
        df = ohlc_df.copy()
        df["orb_sl_dist"] = df["C"] * 0.002  # ~20 price units, realistic ORB half-range
        strategy = OrbExitStrategy()
        tl, ts = strategy.compute_targets(df, ctx, tp_mult=2.0, sl_mult=1.0)
        assert len(tl) == len(df)
        assert len(ts) == len(df)


# --- SL Source Tests ---

class TestOrbExitStrategySL:
    """SL must come from orb_sl_dist when available, fall back to ATR * sl when not."""

    def test_sl_uses_orb_sl_dist_when_present(self, ohlc_df, ctx):
        """resolve_distances must return orb_sl_dist values (not ATR * sl) for SL."""
        strategy = OrbExitStrategy()
        large_sl_dist = 500.0  # large, unmistakeable value
        df = ohlc_df.copy()
        df["orb_sl_dist"] = large_sl_dist

        _, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)

        # All SL distances must equal max(500.0, min_sl_distance)
        # min_sl_pips=5, spread=1.0 → min_sl_distance = 5.0
        # max(500.0, 5.0) = 500.0
        assert (sl_dists >= 500.0).all(), (
            f"Expected SL distances >= 500.0 from orb_sl_dist, got min={sl_dists.min():.2f}"
        )

    def test_sl_falls_back_to_atr_when_no_orb_sl_dist(self, ohlc_df, ctx):
        """When orb_sl_dist not in df, SL must fall back to ATR * sl_mult."""
        strategy = OrbExitStrategy()
        assert "orb_sl_dist" not in ohlc_df.columns
        _, sl_dists = strategy.resolve_distances(ohlc_df, tp=2.0, sl=1.0, ctx=ctx)
        # Fallback: ATR * sl_mult, should be positive and reasonable
        assert (sl_dists > 0).all(), "Fallback SL distances must be positive"
        # ATR for DAX-like data at ~10000 with spread=1.0: roughly 20-200 range
        assert sl_dists.mean() < 1000, "Fallback SL should be in reasonable range"

    def test_sl_fallback_uses_session_sl_dist_column(self, ohlc_df, ctx):
        """resolve_distances must also recognise session-specific *_sl_dist columns."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_s08_sl_dist"] = 42.0  # session-specific sl_dist column

        _, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)

        # Should detect any *_sl_dist column, including session-specific ones
        assert (sl_dists >= 42.0).all(), (
            f"Expected SL >= 42.0 from orb_s08_sl_dist, got min={sl_dists.min():.2f}"
        )

    def test_min_sl_enforced(self, ohlc_df, ctx):
        """SL must never be less than min_sl_pips * spread."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_sl_dist"] = 0.001  # near-zero orb_sl_dist
        # min_sl_pips=5, spread=1.0 → min_sl = 5.0
        _, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)
        assert (sl_dists >= 5.0).all(), f"Min SL not enforced: {sl_dists.min():.4f}"


# --- TP Tests ---

class TestOrbExitStrategyTP:
    """TP must always be ATR-based (tp_mult * ATR)."""

    def test_tp_is_atr_based(self, ohlc_df, ctx):
        """TP distances must scale with tp_mult (larger mult → larger TP)."""
        strategy = OrbExitStrategy()
        tp_lo, _ = strategy.resolve_distances(ohlc_df, tp=1.0, sl=1.0, ctx=ctx)
        tp_hi, _ = strategy.resolve_distances(ohlc_df, tp=3.0, sl=1.0, ctx=ctx)
        # tp_hi should be ~3x tp_lo (for bars where ATR determines TP)
        assert tp_hi.mean() > tp_lo.mean() * 1.5, (
            "TP at tp_mult=3 should be significantly larger than at tp_mult=1"
        )

    def test_min_tp_enforced(self, ctx):
        """TP must never be less than min_tp_pips * spread."""
        strategy = OrbExitStrategy()
        df = _make_ohlc(n=50)
        df["_atr"] = np.full(len(df), 0.001)  # near-zero ATR
        # min_tp_pips=8, spread=1.0 → min_tp = 8.0
        tp_dists, _ = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)
        assert (tp_dists >= 8.0).all(), f"Min TP not enforced: {tp_dists.min():.4f}"


# --- Grid Tests ---

class TestOrbExitStrategyGrid:
    """Grid iteration must cover tp_mult × sl multiplier combinations."""

    def test_grid_generates_tp_sl_combinations(self, ctx):
        strategy = OrbExitStrategy()
        grid_config = {"tp": [1.5, 2.0, 3.0], "sl": [1.0]}
        combos = list(strategy.iterate_grid(grid_config, ctx))
        assert len(combos) == 3

    def test_grid_combo_contains_tp_and_sl(self, ctx):
        strategy = OrbExitStrategy()
        grid_config = {"tp": [2.0], "sl": [1.0]}
        combos = list(strategy.iterate_grid(grid_config, ctx))
        assert len(combos) == 1
        c = combos[0]
        assert "tp_mult" in c
        assert "sl_mult" in c

    def test_grid_default_sl_is_one(self, ctx):
        """Default sl grid = [1.0] (exact ORB range, no buffer needed)."""
        strategy = OrbExitStrategy()
        grid_config = {"tp": [2.0]}  # no sl specified
        combos = list(strategy.iterate_grid(grid_config, ctx))
        assert all(c["sl_mult"] == 1.0 for c in combos), (
            "Default sl for orb_based should be [1.0] (exact ORB range)"
        )


# --- Default params ---

class TestOrbExitStrategyDefaults:
    def test_get_default_params(self):
        params = OrbExitStrategy.get_default_params()
        assert "tp_mult" in params
        assert "sl_mult" in params
        assert params["sl_mult"] == 1.0, "Default sl_mult must be 1.0 (exact ORB range)"

    def test_get_param_schema(self):
        schema = OrbExitStrategy.get_param_schema()
        assert "tp_mult" in schema
        assert "sl_mult" in schema
