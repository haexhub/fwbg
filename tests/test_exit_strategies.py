"""
Integration tests for exit strategy dispatch and comparison.

Plugin-specific tests are in each plugin's tests.py:
- src/fwbg/plugins/fwbg-core/exit_strategies/fixed/tests.py
- packages/fwbg-premium/.../exit_strategies/atr_based/tests.py
"""
import numpy as np
import pandas as pd
import pytest
import ta

from fwbg.plugins import import_plugin_module
from fwbg.core.context import SimulationContext
from fwbg.core.registry import register_exit_modifier, EXIT_MODIFIER_REGISTRY

_fixed = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
_atr = import_plugin_module("fwbg-premium", "exit_strategies", "atr_based")

if _fixed is None:
    pytest.skip("fwbg-core exit_strategies plugin not available", allow_module_level=True)
if _atr is None:
    pytest.skip("fwbg-premium exit_strategies plugin not available", allow_module_level=True)

FixedExitStrategy = _fixed.FixedExitStrategy
AtrExitStrategy = _atr.AtrExitStrategy


@pytest.fixture
def volatile_ohlc():
    """OHLC-Daten mit hoher Volatilität."""
    n = 100
    np.random.seed(42)
    base = 1.1000
    volatility = 0.002

    closes = base + np.cumsum(np.random.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "O": closes - np.random.uniform(0, volatility, n),
        "H": closes + np.abs(np.random.normal(0, volatility, n)),
        "L": closes - np.abs(np.random.normal(0, volatility, n)),
        "C": closes,
    })
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


@pytest.fixture
def forex_context():
    return SimulationContext(
        symbol="EURUSD",
        asset_class="forex",
        spread=0.0001,
        point=0.00001,
        min_trades=10,
        max_trade_bars=50,
        exit_strategy="fixed",
    )


class TestExitStrategyComparison:
    """Vergleichende Tests zwischen Fixed und ATR Exit-Strategien."""

    def test_both_strategies_same_interface(self, volatile_ohlc, forex_context):
        """Beide Strategien sollten gleiches Interface haben."""
        import ta
        volatile_ohlc["_atr"] = ta.volatility.average_true_range(
            volatile_ohlc["H"], volatile_ohlc["L"], volatile_ohlc["C"], window=14
        )

        fixed = FixedExitStrategy()
        atr = AtrExitStrategy()

        t_fixed_l, t_fixed_s = fixed.compute_targets(volatile_ohlc, forex_context, tp=30, sl=20)
        t_atr_l, t_atr_s = atr.compute_targets(volatile_ohlc, forex_context, tp_mult=2.0, sl_mult=1.5)

        # Gleiche Output-Struktur
        assert t_fixed_l.shape == t_atr_l.shape
        assert t_fixed_s.shape == t_atr_s.shape

        # Werte im gültigen Bereich
        assert np.all((t_fixed_l >= 0) & (t_fixed_l <= 1))
        assert np.all((t_atr_l >= 0) & (t_atr_l <= 1))

    def test_both_have_default_params(self):
        """Beide Strategien sollten Default-Parameter haben."""
        fixed_defaults = FixedExitStrategy.get_default_params()
        atr_defaults = AtrExitStrategy.get_default_params()

        assert "tp" in fixed_defaults
        assert "sl" in fixed_defaults

        assert "tp_mult" in atr_defaults
        assert "sl_mult" in atr_defaults


class TestAtrBasedModifierDispatch:
    """Tests that atr_based dispatches to exit modifier when ctx.exit_modifier is set."""

    @pytest.fixture
    def ohlc_with_atr(self):
        n = 200
        np.random.seed(7)
        closes = 1.1 + np.cumsum(np.random.normal(0, 0.0005, n))
        df = pd.DataFrame({
            "O": closes - 0.0001,
            "H": closes + 0.0005,
            "L": closes - 0.0005,
            "C": closes,
        })
        df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
        df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )
        return df

    def test_atr_based_dispatches_to_modifier(self, ohlc_with_atr):
        """atr_based sollte compute_targets an den modifier delegieren wenn ctx.exit_modifier gesetzt ist."""
        called_with = {}

        @register_exit_modifier("_test_mock_modifier")
        class MockModifier:
            def compute_targets(self, opens, closes, highs, lows, atr_values,
                                tp_mult, sl_mult, spread, slippage,
                                min_tp_distance, min_sl_distance,
                                max_bars, timeout_val, return_durations=False, **params):
                called_with["tp_mult"] = tp_mult
                called_with["spread"] = spread
                called_with["extra"] = params
                n = len(closes)
                return np.zeros(n), np.zeros(n)

        try:
            ctx = SimulationContext(
                symbol="EURUSD",
                asset_class="FOREX",
                spread=0.0001,
                point=0.00001,
                max_trade_bars=50,
                exit_strategy="atr_based",
                exit_modifier="_test_mock_modifier",
                exit_modifier_params={"breakeven_trigger": 0.5},
            )
            strategy = AtrExitStrategy()
            t_long, t_short = strategy.compute_targets(
                ohlc_with_atr, ctx, tp_mult=2.0, sl_mult=1.0
            )
            assert called_with.get("tp_mult") == 2.0
            assert called_with.get("spread") == pytest.approx(0.0001)
            assert called_with["extra"].get("breakeven_trigger") == 0.5
        finally:
            if "_test_mock_modifier" in EXIT_MODIFIER_REGISTRY:
                del EXIT_MODIFIER_REGISTRY["_test_mock_modifier"]

    def test_atr_based_without_modifier_unchanged(self, ohlc_with_atr):
        """atr_based ohne exit_modifier sollte wie bisher funktionieren."""
        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            max_trade_bars=50,
            exit_strategy="atr_based",
        )
        strategy = AtrExitStrategy()
        t_long, t_short = strategy.compute_targets(ohlc_with_atr, ctx, tp_mult=2.0, sl_mult=1.0)
        assert t_long.shape == (len(ohlc_with_atr),)
        assert t_short.shape == (len(ohlc_with_atr),)
        assert np.all((t_long >= 0) & (t_long <= 1))


class TestTrailingStopModifier:
    """Tests für den trailing_stop exit modifier plugin."""

    @pytest.fixture
    def ohlc_with_atr(self):
        n = 300
        np.random.seed(99)
        closes = 1.1 + np.cumsum(np.random.normal(0, 0.0005, n))
        df = pd.DataFrame({
            "O": closes - 0.0001,
            "H": closes + 0.0008,
            "L": closes - 0.0008,
            "C": closes,
        })
        df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
        df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )
        return df

    def test_trailing_stop_modifier_registered(self):
        """trailing_stop modifier sollte nach auto_discover im Registry sein."""
        from fwbg.core.registry import EXIT_MODIFIER_REGISTRY, get_exit_modifier
        from fwbg.pipeline.registry import get_registry
        registry = get_registry()
        registry.auto_discover()
        # Die Registrierung erfolgt über den @register_exit_modifier Decorator
        modifier_cls = get_exit_modifier("trailing_stop")
        assert modifier_cls is not None

    def test_trailing_stop_modifier_produces_targets(self, ohlc_with_atr):
        """trailing_stop modifier über atr_based sollte korrekte Target-Arrays produzieren."""
        from fwbg.pipeline.registry import get_registry
        registry = get_registry()
        registry.auto_discover()

        ctx = SimulationContext(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            max_trade_bars=100,
            exit_strategy="atr_based",
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
        )
        strategy = AtrExitStrategy()
        t_long, t_short = strategy.compute_targets(
            ohlc_with_atr, ctx, tp_mult=2.0, sl_mult=1.0
        )
        assert t_long.shape == (len(ohlc_with_atr),)
        assert np.all((t_long >= 0) & (t_long <= 1))
        assert np.all((t_short >= 0) & (t_short <= 1))

    def test_trailing_stop_produces_different_targets_than_base(self, ohlc_with_atr):
        """trailing_stop modifier sollte andere Targets produzieren als rein atr_based."""
        from fwbg.pipeline.registry import get_registry
        registry = get_registry()
        registry.auto_discover()

        ctx_base = SimulationContext(
            symbol="EURUSD", asset_class="FOREX", spread=0.0001, point=0.00001,
            max_trade_bars=100, exit_strategy="atr_based",
        )
        ctx_trailing = SimulationContext(
            symbol="EURUSD", asset_class="FOREX", spread=0.0001, point=0.00001,
            max_trade_bars=100, exit_strategy="atr_based",
            exit_modifier="trailing_stop",
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
        )
        strategy = AtrExitStrategy()
        t_long_base, _ = strategy.compute_targets(ohlc_with_atr, ctx_base, tp_mult=2.0, sl_mult=1.0)
        t_long_trail, _ = strategy.compute_targets(ohlc_with_atr, ctx_trailing, tp_mult=2.0, sl_mult=1.0)
        # Trailing stop changes outcomes — arrays should differ
        assert not np.array_equal(t_long_base, t_long_trail)

    def test_trail_tp_atr_mult_changes_targets(self, ohlc_with_atr):
        """trail_tp_atr_mult > 0 sollte andere Targets als trail_tp_atr_mult=0 produzieren."""
        from fwbg.pipeline.registry import get_registry
        registry = get_registry()
        registry.auto_discover()
        from fwbg.core.registry import get_exit_modifier
        modifier_cls = get_exit_modifier("trailing_stop")
        modifier = modifier_cls()

        opens = ohlc_with_atr["O"].values
        closes = ohlc_with_atr["C"].values
        highs = ohlc_with_atr["H"].values
        lows = ohlc_with_atr["L"].values
        atr_values = ohlc_with_atr["_atr"].values

        # tp_mult=0.5 → TP within single-bar range → many TP hits
        # breakeven_trigger=0.0 → trailing TP starts immediately
        # trail_atr_mult=0.0 → no trailing SL, only trailing TP matters
        shared = dict(
            tp_mult=0.5, sl_mult=3.0, spread=0.0, slippage=0.0,
            min_tp_distance=0.0, min_sl_distance=0.0,
            max_bars=150, timeout_val=0,
            breakeven_trigger=0.0, trail_atr_mult=0.0,
        )
        t_long_no_trail_tp, _ = modifier.compute_targets(
            opens, closes, highs, lows, atr_values,
            **shared, trail_tp_atr_mult=0.0,
        )
        t_long_trail_tp, _ = modifier.compute_targets(
            opens, closes, highs, lows, atr_values,
            **shared, trail_tp_atr_mult=1.0,
        )
        # Trailing TP ratchets the TP upward, making TP hits harder → different outcomes
        assert not np.array_equal(t_long_no_trail_tp, t_long_trail_tp)
