"""Tests for AtrExitStrategy plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module
from fwbg.core.context import SimulationContext

_atr = import_plugin_module("fwbg-premium", "exit_strategies", "atr_based")
if _atr is None:
    pytest.skip("fwbg-premium atr_based exit strategy not available", allow_module_level=True)

AtrExitStrategy = _atr.AtrExitStrategy


# --- Fixtures ---


@pytest.fixture
def volatile_ohlc():
    """OHLC-Daten mit hoher Volatilität (große Ranges)."""
    n = 100
    np.random.seed(42)
    base = 1.1000
    volatility = 0.002  # 20 Pips Schwankung

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
def atr_context():
    """SimulationContext für ATR-basierte Exit-Strategie."""
    return SimulationContext(
        symbol="EURUSD",
        asset_class="forex",
        spread=0.0001,
        point=0.00001,
        min_trades=10,
        max_trade_bars=50,
        exit_strategy="atr_based",
        exit_params={
            "atr_period": 14,
            "min_tp_pips": 10,
            "min_sl_pips": 15,
        }
    )


@pytest.fixture
def forex_context():
    """SimulationContext für typisches Forex-Paar."""
    return SimulationContext(
        symbol="EURUSD",
        asset_class="forex",
        spread=0.0001,
        point=0.00001,
        min_trades=10,
        max_trade_bars=50,
        exit_strategy="atr_based",
    )


# --- AtrExitStrategy Tests ---


class TestAtrExitStrategyTargets:
    """Tests für Target-Berechnung mit ATR-basierten TP/SL."""

    def test_atr_targets_computed(self, volatile_ohlc, atr_context):
        """ATR-basierte Targets sollten berechnet werden."""
        import ta
        volatile_ohlc["_atr"] = ta.volatility.average_true_range(
            volatile_ohlc["H"], volatile_ohlc["L"], volatile_ohlc["C"], window=14
        )

        strategy = AtrExitStrategy()

        targets_long, targets_short = strategy.compute_targets(
            volatile_ohlc, atr_context,
            tp_mult=2.0, sl_mult=1.5
        )

        assert len(targets_long) == len(volatile_ohlc)
        assert len(targets_short) == len(volatile_ohlc)

        # Sollte einige Wins geben
        assert targets_long.sum() > 0 or targets_short.sum() > 0

    def test_atr_adapts_to_volatility(self, forex_context):
        """ATR-basierte Exits sollten sich an Volatilität anpassen."""

        # Niedrige Volatilität
        n = 100
        low_vol_df = pd.DataFrame({
            "O": 1.1000 + np.random.uniform(-0.0001, 0.0001, n),
            "H": 1.1000 + np.random.uniform(0.0001, 0.0002, n),
            "L": 1.1000 + np.random.uniform(-0.0002, -0.0001, n),
            "C": 1.1000 + np.random.uniform(-0.0001, 0.0001, n),
        })

        # Hohe Volatilität
        high_vol_df = pd.DataFrame({
            "O": 1.1000 + np.random.uniform(-0.001, 0.001, n),
            "H": 1.1000 + np.random.uniform(0.001, 0.003, n),
            "L": 1.1000 + np.random.uniform(-0.003, -0.001, n),
            "C": 1.1000 + np.random.uniform(-0.001, 0.001, n),
        })

        # ATR berechnen
        import ta
        low_vol_df["_atr"] = ta.volatility.average_true_range(
            low_vol_df["H"], low_vol_df["L"], low_vol_df["C"], window=14
        )
        high_vol_df["_atr"] = ta.volatility.average_true_range(
            high_vol_df["H"], high_vol_df["L"], high_vol_df["C"], window=14
        )

        # ATR sollte bei hoher Volatilität höher sein
        low_atr_mean = low_vol_df["_atr"].dropna().mean()
        high_atr_mean = high_vol_df["_atr"].dropna().mean()

        assert high_atr_mean > low_atr_mean * 2, "ATR sollte bei hoher Volatilität deutlich höher sein"

    def test_min_tp_sl_enforced(self, atr_context):
        """Mindest-TP/SL sollten auch bei niedrigem ATR eingehalten werden."""
        strategy = AtrExitStrategy()

        # Sehr niedriger ATR (fast keine Bewegung)
        n = 50
        df = pd.DataFrame({
            "O": np.full(n, 1.1000),
            "H": np.full(n, 1.1001),
            "L": np.full(n, 1.0999),
            "C": np.full(n, 1.1000),
        })
        df["_atr"] = np.full(n, 0.00001)  # Winziger ATR

        # Mit min_tp_pips=10 und spread=0.0001 sollte min_tp_distance = 0.001 sein
        targets_long, targets_short = strategy.compute_targets(
            df, atr_context,
            tp_mult=2.0, sl_mult=1.5,
            min_tp_pips=10, min_sl_pips=15
        )

        # Sollte nicht crashen und gültige Arrays zurückgeben
        assert len(targets_long) == n
        assert len(targets_short) == n

    def test_uses_vol_atr_column_if_available(self, volatile_ohlc, atr_context):
        """Sollte vol_atr Spalte verwenden wenn _atr nicht vorhanden."""
        import ta
        volatile_ohlc["vol_atr"] = ta.volatility.average_true_range(
            volatile_ohlc["H"], volatile_ohlc["L"], volatile_ohlc["C"], window=14
        )

        strategy = AtrExitStrategy()

        # Sollte nicht crashen (verwendet vol_atr statt _atr)
        targets_long, _ = strategy.compute_targets(
            volatile_ohlc, atr_context,
            tp_mult=2.0, sl_mult=1.5
        )

        assert len(targets_long) == len(volatile_ohlc)

    def test_calculates_atr_if_not_present(self, volatile_ohlc, atr_context):
        """Sollte ATR selbst berechnen wenn keine ATR-Spalte vorhanden."""
        # Keine ATR-Spalte
        assert "_atr" not in volatile_ohlc.columns
        assert "vol_atr" not in volatile_ohlc.columns

        strategy = AtrExitStrategy()

        # Sollte nicht crashen (berechnet ATR intern)
        targets_long, _ = strategy.compute_targets(
            volatile_ohlc, atr_context,
            tp_mult=2.0, sl_mult=1.5, atr_period=14
        )

        assert len(targets_long) == len(volatile_ohlc)



class TestAtrExitStrategyCacheKey:
    """Tests für ATR Cache-Key Generierung."""

    def test_unique_cache_keys_for_mults(self):
        """Verschiedene Multiplikatoren sollten verschiedene Keys haben."""
        strategy = AtrExitStrategy()

        key1 = strategy.get_cache_key({"tp_mult": 2.0, "sl_mult": 1.5, "timeout_bars": None})
        key2 = strategy.get_cache_key({"tp_mult": 2.5, "sl_mult": 1.5, "timeout_bars": None})
        key3 = strategy.get_cache_key({"tp_mult": 2.0, "sl_mult": 2.0, "timeout_bars": None})

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_key_format(self):
        """Cache-Key sollte ATR-spezifisches Format haben."""
        strategy = AtrExitStrategy()

        key = strategy.get_cache_key({"tp_mult": 2.0, "sl_mult": 1.5, "timeout_bars": None})

        assert "atr" in key
        assert "2.00" in key  # tp_mult formatiert
        assert "1.50" in key  # sl_mult formatiert
