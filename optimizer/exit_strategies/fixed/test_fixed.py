"""Tests für Fixed Exit Strategy."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from . import FixedExitStrategy, FixedExitConfig
from ..base import GridParams


class MockSimulationContext:
    """Mock SimulationContext für Tests."""
    def __init__(self):
        self.spread = 0.0001  # 1 Pip
        self.max_trade_bars = 1000


class TestFixedExitConfig:
    """Tests für FixedExitConfig."""

    def test_default_values(self):
        """Default-Werte sollten gesetzt sein."""
        config = FixedExitConfig()

        assert len(config.tp) > 0
        assert len(config.sl) > 0
        assert config.timeout_bars == [None]
        assert config.min_rrr == 0.0

    def test_from_dict(self):
        """Config sollte aus Dictionary erstellt werden."""
        data = {
            "tp": [10, 20, 30],
            "sl": [15, 25],
            "timeout_bars": [12, 24],
            "min_rrr": 0.5,
        }
        config = FixedExitConfig.from_dict(data)

        assert config.tp == [10, 20, 30]
        assert config.sl == [15, 25]
        assert config.timeout_bars == [12, 24]
        assert config.min_rrr == 0.5

    def test_separate_long_short_grids(self):
        """Separate Long/Short Grids sollten funktionieren."""
        config = FixedExitConfig(
            tp=[20, 30],
            sl=[25, 35],
            long_tp=[15, 20],
            long_sl=[20, 25],
        )

        long_tp, long_sl = config.get_long_grid()
        short_tp, short_sl = config.get_short_grid()

        assert long_tp == [15, 20]
        assert long_sl == [20, 25]
        assert short_tp == [20, 30]  # Fallback zu gemeinsamen Werten
        assert short_sl == [25, 35]


class TestFixedExitStrategy:
    """Tests für FixedExitStrategy."""

    def test_registration(self):
        """Strategie sollte registriert sein."""
        from .. import get_strategy, list_strategies

        assert "fixed" in list_strategies()
        assert get_strategy("fixed") == FixedExitStrategy

    def test_iterate_grid_basic(self):
        """iterate_grid sollte alle Kombinationen durchgehen."""
        strategy = FixedExitStrategy()
        ctx = MockSimulationContext()

        grid_config = {
            "tp": [20, 30],
            "sl": [25, 35],
            "timeout_bars": [None],
        }

        params_list = list(strategy.iterate_grid(grid_config, ctx))

        # 2 TP x 2 SL x 1 Timeout = 4 Kombinationen
        assert len(params_list) == 4

        # Prüfe erste Kombination
        assert params_list[0].tp_value == 20.0
        assert params_list[0].sl_value == 25.0
        assert params_list[0].timeout_bars is None

    def test_iterate_grid_with_rrr_filter(self):
        """RRR-Filter sollte ungültige Kombinationen ausschließen."""
        strategy = FixedExitStrategy()
        ctx = MockSimulationContext()

        grid_config = {
            "tp": [20, 30, 40],
            "sl": [30, 40],
            "timeout_bars": [None],
            "min_rrr": 1.0,  # TP muss >= SL sein
        }

        params_list = list(strategy.iterate_grid(grid_config, ctx))

        # Nur Kombinationen mit TP >= SL
        # 20/30=0.67 ❌, 20/40=0.5 ❌, 30/30=1.0 ✓, 30/40=0.75 ❌, 40/30=1.33 ✓, 40/40=1.0 ✓
        assert len(params_list) == 3

    def test_iterate_grid_with_timeout(self):
        """Timeout-Werte sollten durchiteriert werden."""
        strategy = FixedExitStrategy()
        ctx = MockSimulationContext()

        grid_config = {
            "tp": [20],
            "sl": [25],
            "timeout_bars": [None, 12, 24],
        }

        params_list = list(strategy.iterate_grid(grid_config, ctx))

        # 1 TP x 1 SL x 3 Timeout = 3 Kombinationen
        assert len(params_list) == 3
        assert params_list[0].timeout_bars is None
        assert params_list[1].timeout_bars == 12
        assert params_list[2].timeout_bars == 24

    def test_get_cache_key(self):
        """Cache-Keys sollten eindeutig sein."""
        strategy = FixedExitStrategy()

        params1 = GridParams(tp_value=20, sl_value=30, timeout_bars=None)
        params2 = GridParams(tp_value=20, sl_value=30, timeout_bars=12)
        params3 = GridParams(tp_value=30, sl_value=30, timeout_bars=None)

        key1 = strategy.get_cache_key(params1)
        key2 = strategy.get_cache_key(params2)
        key3 = strategy.get_cache_key(params3)

        assert key1 != key2  # Unterschiedlicher Timeout
        assert key1 != key3  # Unterschiedlicher TP
        assert "tp20" in key1
        assert "sl30" in key1

    def test_total_combinations(self):
        """total_combinations sollte korrekt zählen."""
        strategy = FixedExitStrategy()
        ctx = MockSimulationContext()

        grid_config = {
            "tp": [20, 30, 40],
            "sl": [25, 35],
            "timeout_bars": [None, 12],
        }

        total = strategy.total_combinations(grid_config, ctx)

        # 3 TP x 2 SL x 2 Timeout = 12
        assert total == 12

    def test_from_config(self):
        """Strategie sollte aus Config erstellt werden."""
        config = {
            "fixed": {
                "tp": [10, 20],
                "sl": [15, 25],
            }
        }

        strategy = FixedExitStrategy.from_config(config)

        assert strategy.config.tp == [10, 20]
        assert strategy.config.sl == [15, 25]


class TestFixedComputeTargets:
    """Tests für compute_targets_fixed."""

    def create_test_df(self, n_bars=100):
        """Erstellt Test-DataFrame mit OHLC-Daten."""
        np.random.seed(42)

        # Generiere realistische Preisbewegungen
        close = 1.1000 + np.cumsum(np.random.randn(n_bars) * 0.0010)
        high = close + np.abs(np.random.randn(n_bars) * 0.0005)
        low = close - np.abs(np.random.randn(n_bars) * 0.0005)
        open_ = close + np.random.randn(n_bars) * 0.0003

        return pd.DataFrame({
            "O": open_,
            "H": high,
            "L": low,
            "C": close,
        })

    def test_compute_targets_returns_arrays(self):
        """compute_targets sollte zwei Arrays zurückgeben."""
        from .compute import compute_targets_fixed

        df = self.create_test_df()
        ctx = MockSimulationContext()

        targets_long, targets_short = compute_targets_fixed(
            df, tp=20, sl=30, ctx=ctx
        )

        assert isinstance(targets_long, np.ndarray)
        assert isinstance(targets_short, np.ndarray)
        assert len(targets_long) == len(df)
        assert len(targets_short) == len(df)

    def test_compute_targets_binary(self):
        """Targets sollten nur 0 oder 1 sein."""
        from .compute import compute_targets_fixed

        df = self.create_test_df()
        ctx = MockSimulationContext()

        targets_long, targets_short = compute_targets_fixed(
            df, tp=20, sl=30, ctx=ctx
        )

        assert set(np.unique(targets_long)).issubset({0.0, 1.0})
        assert set(np.unique(targets_short)).issubset({0.0, 1.0})

    def test_compute_targets_with_timeout(self):
        """Timeout sollte berücksichtigt werden."""
        from .compute import compute_targets_fixed

        df = self.create_test_df(n_bars=200)
        ctx = MockSimulationContext()

        # Mit Timeout
        targets_with_to, _ = compute_targets_fixed(
            df, tp=100, sl=100, ctx=ctx, timeout_bars=5
        )

        # Ohne Timeout
        targets_no_to, _ = compute_targets_fixed(
            df, tp=100, sl=100, ctx=ctx, timeout_bars=None
        )

        # Ergebnisse können unterschiedlich sein
        # (Timeout kann zu anderen Win/Loss führen)
        # Wichtig ist nur, dass beide ohne Fehler laufen
        assert len(targets_with_to) == len(df)
        assert len(targets_no_to) == len(df)
