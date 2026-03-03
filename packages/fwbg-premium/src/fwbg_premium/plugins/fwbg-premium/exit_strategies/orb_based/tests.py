"""Tests for OrbExitStrategy plugin.

The orb_based strategy uses structural-level SL and ATR-based TP:
- Explicit sl_dist_column: raw distance used as-is (no sl_mult). E.g. pdl_sl_dist.
- Auto-detect *_sl_dist: distance * sl_mult (buffer tuning). E.g. orb_sl_dist.
- No column: ATR * sl_mult fallback.
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
    open_ = np.roll(close, 1)
    open_[0] = close[0]
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
        assert cls.__name__ == "OrbExitStrategy"

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
        df["orb_sl_dist"] = df["C"] * 0.002  # ~20 price units, realistic ORB range
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

    def test_no_double_sl_mult_on_nan_fallback(self, ohlc_df, ctx):
        """NaN orb_sl_dist rows must use ATR*sl_mult, NOT ATR*sl_mult^2.

        Regression test: _get_sl_dist previously did:
            sl_vals = df[col].fillna(atr * sl_mult).values
            return sl_vals * sl_mult       # ← double multiplication for NaN rows!
        """
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        sl_mult = 2.0

        # Half the rows have orb_sl_dist, half NaN
        sl_col = np.full(len(df), np.nan)
        sl_col[::2] = 100.0  # even rows: known value
        df["orb_sl_dist"] = sl_col

        atr_v = strategy._get_atr(df, 14)
        sl_dists = strategy._get_sl_dist(df, atr_v, sl_mult)

        # Even rows: orb_sl_dist * sl_mult = 100.0 * 2.0 = 200.0
        even_mask = np.arange(len(df)) % 2 == 0
        np.testing.assert_allclose(
            sl_dists[even_mask], 200.0,
            err_msg="orb_sl_dist rows should be orb_sl_dist * sl_mult",
        )

        # Odd rows (NaN): should be ATR * sl_mult (NOT ATR * sl_mult^2)
        odd_mask = ~even_mask
        expected_fallback = atr_v[odd_mask] * sl_mult
        np.testing.assert_allclose(
            sl_dists[odd_mask], expected_fallback,
            err_msg="NaN fallback should be ATR * sl_mult, not ATR * sl_mult^2",
        )


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


# --- sl_dist_column selection ---

class TestOrbExitStrategySLDistColumn:
    """When multiple *_sl_dist columns exist (e.g. orb_sl_dist and pdl_sl_dist),
    sl_dist_column in exit_params must select the correct one."""

    def test_sl_dist_column_selects_pdl(self, ohlc_df):
        """With sl_dist_column='pdl_sl_dist', must use pdl_sl_dist (not orb_sl_dist)."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_sl_dist"] = 100.0  # wrong column
        df["hl_ses_pdl_sl_dist"] = 500.0  # correct column

        ctx = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 5,
                         "sl_dist_column": "hl_ses_pdl_sl_dist"},
        )

        _, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)
        # Must use pdl_sl_dist=500, not orb_sl_dist=100
        assert (sl_dists >= 500.0).all(), (
            f"Expected SL from pdl_sl_dist=500, got min={sl_dists.min():.2f}"
        )

    def test_explicit_sl_dist_column_ignores_sl_mult(self, ohlc_df):
        """When sl_dist_column is explicitly set, sl_mult must NOT be applied.

        The raw column value is the absolute SL distance (e.g. distance to PDL/PDH).
        """
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["hl_ses_pdl_sl_dist"] = 100.0

        atr_v = strategy._get_atr(df, 14)

        # With explicit sl_dist_column: raw=100, sl_mult=3.0 → result should be 100 (NOT 300)
        sl_dists = strategy._get_sl_dist(df, atr_v, sl_mult=3.0, sl_dist_column="hl_ses_pdl_sl_dist")
        np.testing.assert_allclose(
            sl_dists, 100.0,
            err_msg="Explicit sl_dist_column must use raw value, NOT apply sl_mult",
        )

    def test_auto_detect_sl_dist_applies_sl_mult(self, ohlc_df):
        """When no sl_dist_column is set, auto-detect applies sl_mult as buffer."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_sl_dist"] = 100.0

        atr_v = strategy._get_atr(df, 14)

        # Without explicit sl_dist_column: raw=100, sl_mult=2.0 → result should be 200
        sl_dists = strategy._get_sl_dist(df, atr_v, sl_mult=2.0)
        np.testing.assert_allclose(
            sl_dists, 200.0,
            err_msg="Auto-detect sl_dist must apply sl_mult as buffer multiplier",
        )

    def test_without_sl_dist_column_picks_first(self, ohlc_df):
        """Without sl_dist_column, auto-detect picks the first *_sl_dist column."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_sl_dist"] = 100.0

        ctx = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 5},
        )

        _, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)
        assert (sl_dists >= 100.0).all()

    def test_sl_dist_column_missing_falls_back_to_auto(self, ohlc_df):
        """If sl_dist_column specified but not in df, fall back to auto-detect."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_sl_dist"] = 200.0  # available for auto-detect

        ctx = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"atr_period": 14, "min_tp_pips": 8, "min_sl_pips": 5,
                         "sl_dist_column": "nonexistent_sl_dist"},
        )

        _, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)
        # Should fall back to orb_sl_dist
        assert (sl_dists >= 200.0).all(), (
            f"Expected fallback to orb_sl_dist=200, got min={sl_dists.min():.2f}"
        )

    def test_compute_targets_uses_sl_dist_column(self, ohlc_df):
        """compute_targets must also respect sl_dist_column from exit_params."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_sl_dist"] = 0.5   # tight SL → more wins
        df["hl_ses_pdl_sl_dist"] = 500.0  # very wide SL → almost all wins

        ctx_orb = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"atr_period": 14, "min_tp_pips": 1, "min_sl_pips": 1},
        )
        ctx_pdl = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"atr_period": 14, "min_tp_pips": 1, "min_sl_pips": 1,
                         "sl_dist_column": "hl_ses_pdl_sl_dist"},
        )

        tl_orb, _ = strategy.compute_targets(df, ctx_orb, tp_mult=2.0, sl_mult=1.0)
        tl_pdl, _ = strategy.compute_targets(df, ctx_pdl, tp_mult=2.0, sl_mult=1.0)

        # Wide SL (pdl) should produce more or equal wins than tight SL (orb)
        assert tl_pdl.sum() >= tl_orb.sum(), (
            f"Wide SL (pdl_sl_dist=500) should have >= wins than tight SL (orb_sl_dist=0.5): "
            f"pdl={tl_pdl.sum()} vs orb={tl_orb.sum()}"
        )


# --- tp_mode="range" tests ---

class TestOrbExitStrategyRangeMode:
    """tp_mode='range' uses the *_range column for TP instead of ATR."""

    def test_tp_uses_range_column(self, ohlc_df):
        """With tp_mode='range', TP must be derived from the range column, not ATR."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_range"] = 200.0  # large, unmistakeable
        df["orb_sl_dist"] = 100.0  # range/2

        ctx_range = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"tp_mode": "range", "min_tp_pips": 1, "min_sl_pips": 1},
        )
        tp_dists, _ = strategy.resolve_distances(df, tp=1.0, sl=1.0, ctx=ctx_range)
        # TP = range * tp_mult = 200 * 1.0 = 200
        np.testing.assert_allclose(tp_dists, 200.0,
            err_msg="tp_mode='range' must use range column for TP")

    def test_tp_mode_atr_ignores_range_column(self, ohlc_df):
        """Default tp_mode='atr' must ignore the range column."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_range"] = 9999.0  # should be ignored

        ctx_atr = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"min_tp_pips": 1, "min_sl_pips": 1},
        )
        tp_dists, _ = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx_atr)
        # TP should NOT be 9999 — it should be ATR-based
        assert tp_dists.mean() < 1000, (
            f"tp_mode='atr' should not use range column, got mean={tp_dists.mean():.2f}"
        )

    def test_range_mode_tp_scales_with_tp_mult(self, ohlc_df):
        """TP must scale linearly with tp_mult in range mode."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_range"] = 100.0
        df["orb_sl_dist"] = 50.0

        ctx = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"tp_mode": "range", "min_tp_pips": 1, "min_sl_pips": 1},
        )
        tp_1x, _ = strategy.resolve_distances(df, tp=1.0, sl=1.0, ctx=ctx)
        tp_2x, _ = strategy.resolve_distances(df, tp=2.0, sl=1.0, ctx=ctx)
        np.testing.assert_allclose(tp_1x, 100.0)
        np.testing.assert_allclose(tp_2x, 200.0)

    def test_range_mode_compute_targets_smoke(self, ohlc_df):
        """compute_targets must work with tp_mode='range'."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_range"] = df["C"] * 0.002
        df["orb_sl_dist"] = df["C"] * 0.001

        ctx = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"tp_mode": "range", "min_tp_pips": 1, "min_sl_pips": 1},
        )
        tl, ts = strategy.compute_targets(df, ctx, tp_mult=1.0, sl_mult=1.4)
        assert len(tl) == len(df)
        assert len(ts) == len(df)
        assert set(np.unique(tl)).issubset({0.0, 1.0})

    def test_range_column_override(self, ohlc_df):
        """range_column in exit_params must override auto-detection."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_range"] = 100.0  # auto-detect would find this
        df["custom_range"] = 500.0  # explicit override

        ctx = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={
                "tp_mode": "range",
                "range_column": "custom_range",
                "min_tp_pips": 1, "min_sl_pips": 1,
            },
        )
        tp_dists, _ = strategy.resolve_distances(df, tp=1.0, sl=1.0, ctx=ctx)
        np.testing.assert_allclose(tp_dists, 500.0,
            err_msg="range_column override must be used for TP")

    def test_range_auto_detect_finds_prefixed_column(self, ohlc_df):
        """Auto-detect must find rb1_orb_range style columns."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["rb1_orb_range"] = 300.0

        range_v = strategy._get_range(df, {"tp_mode": "range"})
        np.testing.assert_allclose(range_v, 300.0)

    def test_range_fallback_to_zeros(self, ohlc_df):
        """Without any range column, _get_range must return zeros."""
        strategy = OrbExitStrategy()
        range_v = strategy._get_range(ohlc_df, {})
        assert (range_v == 0.0).all()

    def test_range_mode_sl_unchanged(self, ohlc_df):
        """tp_mode='range' must NOT affect SL calculation."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        df["orb_range"] = 200.0
        df["orb_sl_dist"] = 100.0

        ctx_atr = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"min_tp_pips": 1, "min_sl_pips": 1},
        )
        ctx_range = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"tp_mode": "range", "min_tp_pips": 1, "min_sl_pips": 1},
        )
        _, sl_atr = strategy.resolve_distances(df, tp=2.0, sl=1.4, ctx=ctx_atr)
        _, sl_range = strategy.resolve_distances(df, tp=1.0, sl=1.4, ctx=ctx_range)
        np.testing.assert_allclose(sl_atr, sl_range,
            err_msg="tp_mode must not affect SL distances")

    def test_range_mode_sl_mult_140_gives_70pct_range(self, ohlc_df):
        """sl_mult=1.4 on orb_sl_dist (=range/2) gives SL at 70% of range from entry."""
        strategy = OrbExitStrategy()
        df = ohlc_df.copy()
        orb_range = 200.0
        df["orb_range"] = orb_range
        df["orb_sl_dist"] = orb_range / 2  # entry at midpoint

        ctx = SimulationContext(
            symbol="DAX", asset_class="index", spread=1.0, point=0.1,
            min_trades=10, max_trade_bars=100, exit_strategy="orb_based",
            exit_params={"tp_mode": "range", "min_tp_pips": 1, "min_sl_pips": 1},
        )
        _, sl_dists = strategy.resolve_distances(df, tp=1.0, sl=1.4, ctx=ctx)
        # SL = orb_sl_dist * sl_mult = 100 * 1.4 = 140 = 70% of range (200)
        np.testing.assert_allclose(sl_dists, 140.0,
            err_msg="sl_mult=1.4 on range/2 must give 70% of range as SL")


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
