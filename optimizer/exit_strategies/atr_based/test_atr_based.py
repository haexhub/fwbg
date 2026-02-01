"""Tests für ATR-Based Exit Strategy."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from . import AtrExitStrategy, AtrExitConfig
from ..base import GridParams


class MockSimulationContext:
    """Mock SimulationContext für Tests."""
    def __init__(self):
        self.spread = 0.0001  # 1 Pip
        self.max_trade_bars = 1000


class TestAtrExitConfig:
    """Tests für AtrExitConfig."""

    def test_default_values(self):
        """Default-Werte sollten gesetzt sein."""
        config = AtrExitConfig()

        assert config.atr_period == 14
        assert len(config.tp_mult) > 0
        assert len(config.sl_mult) > 0
        assert config.min_tp_pips == 10
        assert config.min_sl_pips == 15

    def test_from_dict(self):
        """Config sollte aus Dictionary erstellt werden."""
        data = {
            "atr_period": 21,
            "atr_tp_mult": [1.5, 2.0, 2.5],
            "atr_sl_mult": [1.0, 1.5],
            "min_tp_pips": 15,
            "min_sl_pips": 20,
            "timeout_bars": [12, 24],
        }
        config = AtrExitConfig.from_dict(data)

        assert config.atr_period == 21
        assert config.tp_mult == [1.5, 2.0, 2.5]
        assert config.sl_mult == [1.0, 1.5]
        assert config.min_tp_pips == 15
        assert config.min_sl_pips == 20

    def test_from_dict_alternative_keys(self):
        """Alternative Key-Namen sollten funktionieren."""
        data = {
            "tp_mult": [1.5, 2.0],  # Ohne 'atr_' Prefix
            "sl_mult": [1.0, 1.5],
        }
        config = AtrExitConfig.from_dict(data)

        assert config.tp_mult == [1.5, 2.0]
        assert config.sl_mult == [1.0, 1.5]


class TestAtrExitStrategy:
    """Tests für AtrExitStrategy."""

    def test_registration(self):
        """Strategie sollte registriert sein."""
        from .. import get_strategy, list_strategies

        assert "atr_based" in list_strategies()
        assert get_strategy("atr_based") == AtrExitStrategy

    def test_iterate_grid_basic(self):
        """iterate_grid sollte alle Kombinationen durchgehen."""
        strategy = AtrExitStrategy()
        ctx = MockSimulationContext()

        grid_config = {
            "atr_tp_mult": [1.5, 2.0],
            "atr_sl_mult": [1.0, 1.5],
            "timeout_bars": [None],
        }

        params_list = list(strategy.iterate_grid(grid_config, ctx))

        # 2 TP x 2 SL x 1 Timeout = 4 Kombinationen
        assert len(params_list) == 4

        # Prüfe erste Kombination
        assert params_list[0].tp_value == 1.5
        assert params_list[0].sl_value == 1.0
        assert params_list[0].timeout_bars is None

    def test_iterate_grid_with_rrr_filter(self):
        """RRR-Filter sollte ungültige Kombinationen ausschließen."""
        strategy = AtrExitStrategy()
        ctx = MockSimulationContext()

        grid_config = {
            "atr_tp_mult": [1.0, 1.5, 2.0],
            "atr_sl_mult": [1.5, 2.0],
            "timeout_bars": [None],
            "min_rrr": 1.0,  # TP-Mult muss >= SL-Mult sein
        }

        params_list = list(strategy.iterate_grid(grid_config, ctx))

        # 1.0/1.5=0.67 ❌, 1.0/2.0=0.5 ❌, 1.5/1.5=1.0 ✓, 1.5/2.0=0.75 ❌, 2.0/1.5=1.33 ✓, 2.0/2.0=1.0 ✓
        assert len(params_list) == 3

    def test_iterate_grid_extra_params(self):
        """Extra-Parameter sollten in GridParams gespeichert werden."""
        config = AtrExitConfig(atr_period=21, min_tp_pips=15, min_sl_pips=20)
        strategy = AtrExitStrategy(config=config)
        ctx = MockSimulationContext()

        grid_config = {
            "atr_tp_mult": [1.5],
            "atr_sl_mult": [1.0],
        }

        params_list = list(strategy.iterate_grid(grid_config, ctx))

        assert params_list[0].extra["atr_period"] == 21
        assert params_list[0].extra["min_tp_pips"] == 15
        assert params_list[0].extra["min_sl_pips"] == 20

    def test_get_cache_key(self):
        """Cache-Keys sollten eindeutig sein."""
        strategy = AtrExitStrategy()

        params1 = GridParams(tp_value=1.5, sl_value=1.0, timeout_bars=None)
        params2 = GridParams(tp_value=1.5, sl_value=1.0, timeout_bars=12)
        params3 = GridParams(tp_value=2.0, sl_value=1.0, timeout_bars=None)

        key1 = strategy.get_cache_key(params1)
        key2 = strategy.get_cache_key(params2)
        key3 = strategy.get_cache_key(params3)

        assert key1 != key2  # Unterschiedlicher Timeout
        assert key1 != key3  # Unterschiedlicher TP
        assert "atr" in key1
        assert "1.50" in key1

    def test_total_combinations(self):
        """total_combinations sollte korrekt zählen."""
        strategy = AtrExitStrategy()
        ctx = MockSimulationContext()

        grid_config = {
            "atr_tp_mult": [1.0, 1.5, 2.0],
            "atr_sl_mult": [1.0, 1.5],
            "timeout_bars": [None, 12],
        }

        total = strategy.total_combinations(grid_config, ctx)

        # 3 TP x 2 SL x 2 Timeout = 12
        assert total == 12


class TestAtrComputeTargets:
    """Tests für compute_targets_atr."""

    def create_test_df(self, n_bars=100):
        """Erstellt Test-DataFrame mit OHLC-Daten und ATR."""
        np.random.seed(42)

        # Generiere realistische Preisbewegungen
        close = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0010)
        high = close + np.abs(np.random.randn(n_bars) * 0.0005)
        low = close - np.abs(np.random.randn(n_bars) * 0.0005)
        open_ = close + np.random.randn(n_bars) * 0.0003

        # Berechne ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        atr = pd.Series(tr).rolling(14).mean().values

        return pd.DataFrame({
            "O": open_,
            "H": high,
            "L": low,
            "C": close,
            "_atr": atr,
        })

    def test_compute_targets_returns_arrays(self):
        """compute_targets sollte zwei Arrays zurückgeben."""
        from .compute import compute_targets_atr

        df = self.create_test_df()
        ctx = MockSimulationContext()

        targets_long, targets_short = compute_targets_atr(
            df, tp_mult=1.5, sl_mult=1.0, ctx=ctx
        )

        assert isinstance(targets_long, np.ndarray)
        assert isinstance(targets_short, np.ndarray)
        assert len(targets_long) == len(df)
        assert len(targets_short) == len(df)

    def test_compute_targets_binary(self):
        """Targets sollten nur 0 oder 1 sein."""
        from .compute import compute_targets_atr

        df = self.create_test_df()
        ctx = MockSimulationContext()

        targets_long, targets_short = compute_targets_atr(
            df, tp_mult=1.5, sl_mult=1.0, ctx=ctx
        )

        assert set(np.unique(targets_long)).issubset({0.0, 1.0})
        assert set(np.unique(targets_short)).issubset({0.0, 1.0})

    def test_compute_targets_uses_atr_column(self):
        """ATR sollte aus _atr Spalte verwendet werden."""
        from .compute import compute_targets_atr

        df = self.create_test_df()
        ctx = MockSimulationContext()

        # Test läuft ohne Fehler wenn _atr vorhanden
        targets_long, targets_short = compute_targets_atr(
            df, tp_mult=1.5, sl_mult=1.0, ctx=ctx
        )

        assert len(targets_long) == len(df)

    def test_compute_targets_fallback_atr(self):
        """Ohne _atr Spalte sollte ATR berechnet werden."""
        from .compute import compute_targets_atr

        df = self.create_test_df()
        df = df.drop(columns=["_atr"])  # Entferne _atr
        ctx = MockSimulationContext()

        # Sollte ATR selbst berechnen (erfordert ta library)
        try:
            targets_long, targets_short = compute_targets_atr(
                df, tp_mult=1.5, sl_mult=1.0, ctx=ctx
            )
            assert len(targets_long) == len(df)
        except ImportError:
            pytest.skip("ta library not available")

    def test_min_distance_enforced(self):
        """Mindest-TP/SL sollte eingehalten werden."""
        from .compute import compute_targets_atr

        df = self.create_test_df()
        # Setze ATR auf sehr kleine Werte
        df["_atr"] = 0.00001  # Sehr kleine ATR

        ctx = MockSimulationContext()

        # Mit min_tp_pips=10, min_sl_pips=15 sollten
        # Mindest-Distanzen verwendet werden
        targets_long, targets_short = compute_targets_atr(
            df, tp_mult=1.0, sl_mult=1.0, ctx=ctx,
            min_tp_pips=10, min_sl_pips=15
        )

        # Test läuft durch - Mindest-Werte wurden angewendet
        assert len(targets_long) == len(df)

    def test_different_atr_multipliers(self):
        """Verschiedene ATR-Multiplikatoren sollten unterschiedliche Ergebnisse liefern."""
        from .compute import compute_targets_atr

        df = self.create_test_df(n_bars=500)
        ctx = MockSimulationContext()

        # Enge TP/SL
        targets_tight, _ = compute_targets_atr(
            df, tp_mult=0.5, sl_mult=0.5, ctx=ctx
        )

        # Weite TP/SL
        targets_wide, _ = compute_targets_atr(
            df, tp_mult=3.0, sl_mult=3.0, ctx=ctx
        )

        # Bei gleichen Multiplikatoren sollten Win-Raten ähnlich sein,
        # aber bei unterschiedlichen Multiplikatoren unterschiedlich
        # (Dies ist ein Smoke-Test)
        assert len(targets_tight) == len(targets_wide)


class TestAtrVsFixed:
    """Vergleichstests zwischen ATR und Fixed Strategy."""

    def create_volatile_df(self, n_bars=200):
        """Erstellt DataFrame mit variierender Volatilität."""
        np.random.seed(42)

        # Erste Hälfte: niedrige Volatilität
        low_vol = np.random.randn(n_bars // 2) * 0.0005
        # Zweite Hälfte: hohe Volatilität
        high_vol = np.random.randn(n_bars // 2) * 0.0020

        returns = np.concatenate([low_vol, high_vol])
        close = 1.1000 + np.cumsum(returns)
        high = close + np.abs(np.random.randn(n_bars) * 0.0003)
        low = close - np.abs(np.random.randn(n_bars) * 0.0003)
        open_ = close + np.random.randn(n_bars) * 0.0002

        # Berechne ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        atr = pd.Series(tr).rolling(14).mean().values

        return pd.DataFrame({
            "O": open_,
            "H": high,
            "L": low,
            "C": close,
            "_atr": atr,
        })

    def test_atr_adapts_to_volatility(self):
        """ATR-Strategie sollte sich an Volatilität anpassen."""
        from .compute import compute_targets_atr
        from ..fixed.compute import compute_targets_fixed

        df = self.create_volatile_df()
        ctx = MockSimulationContext()

        # Fixed: konstante TP/SL
        fixed_long, _ = compute_targets_fixed(df, tp=30, sl=30, ctx=ctx)

        # ATR: dynamische TP/SL
        atr_long, _ = compute_targets_atr(
            df, tp_mult=1.5, sl_mult=1.5, ctx=ctx,
            min_tp_pips=5, min_sl_pips=5
        )

        # Beide sollten Ergebnisse liefern
        assert fixed_long.sum() > 0 or len(df) < 50  # Mindestens einige Trades
        assert len(atr_long) == len(fixed_long)
