"""
Integration tests for exit strategy dispatch and comparison.

Plugin-specific tests are in each plugin's tests.py:
- src/fwbg/plugins/fwbg-core/exit_strategies/fixed/tests.py
- packages/fwbg-premium/.../exit_strategies/atr_based/tests.py
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module
from fwbg.core.context import SimulationContext

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
