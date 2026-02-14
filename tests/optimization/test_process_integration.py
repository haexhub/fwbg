"""
Integration Tests für process.py - Optimizer Pipeline.

Testet die komplette Optimierungs-Pipeline inkl.:
- Import-Pfade (keine relativen Imports zu nicht-existierenden Modulen)
- SimulationContext Attribute
- Exit-Strategy Dispatch
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from fwbg.core.config import StrategyConfig, GridConfig


def create_test_df(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Erstellt einen Test-DataFrame mit OHLC-Daten und Features."""
    np.random.seed(seed)

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_rows)]

    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.1)
    high = close + np.abs(np.random.randn(n_rows) * 0.05)
    low = close - np.abs(np.random.randn(n_rows) * 0.05)
    open_price = close + np.random.randn(n_rows) * 0.02

    df = pd.DataFrame({
        "O": open_price,
        "H": high,
        "L": low,
        "C": close,
        "_atr": np.abs(np.random.randn(n_rows) * 0.1) + 0.05,
        "_regime_ok": np.ones(n_rows, dtype=bool),
    }, index=pd.DatetimeIndex(dates))

    # Feature-Spalten hinzufügen (für verschiedene Gruppen)
    df["trend_rsi_14"] = np.random.rand(n_rows) * 100
    df["trend_adx_14"] = np.random.rand(n_rows) * 50
    df["trend_ema_21"] = close * (1 + np.random.randn(n_rows) * 0.01)
    df["mom_stoch_14"] = np.random.rand(n_rows) * 100
    df["vol_atr_14"] = np.abs(np.random.randn(n_rows) * 0.05) + 0.02

    return df


class TestImportPaths:
    """Testet dass alle Import-Pfade korrekt sind."""

    def test_process_symbol_imports(self):
        """Test: process_symbol hat korrekte Imports."""
        # Import sollte nicht fehlschlagen
        from fwbg.optimization.process import process_symbol

    def test_grid_search_imports(self):
        """Test: run_grid_search kann importiert werden."""
        from fwbg.optimization.grid_search import run_grid_search

    def test_optimization_imports(self):
        """Test: optimization modules have correct imports."""
        from fwbg.optimization.targets import (
            compute_targets_cached,
            slice_targets_for_fold,
        )
        from fwbg.optimization.nested_cv import run_inner_cv


class TestSimulationContextAttributes:
    """Testet dass SimulationContext alle benötigten Attribute hat."""

    def test_context_has_exit_strategy(self):
        """Test: SimulationContext hat exit_strategy (nicht exit_strategy_mode)."""
        from fwbg.core.context import SimulationContext

        # Prüfe dass das Attribut existiert
        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0002,
            point=0.0001,
        )

        assert hasattr(ctx, "exit_strategy")
        assert ctx.exit_strategy == "atr_based"  # Default

    def test_context_has_exit_params(self):
        """Test: SimulationContext hat exit_params."""
        from fwbg.core.context import SimulationContext

        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0002,
            point=0.0001,
            exit_params={"atr_period": 14},
        )

        assert hasattr(ctx, "exit_params")
        assert ctx.exit_params.get("atr_period") == 14

    def test_context_create_from_strategy(self):
        """Test: SimulationContext.create() setzt exit_strategy korrekt."""
        from fwbg.core.context import SimulationContext
        from fwbg.data.assets import AssetConfig

        # Minimale Strategy
        strategy = StrategyConfig(
            exit_strategy="atr_based",
            exit_params={"atr_period": 21, "min_tp_pips": 15},
        )

        # Asset-Config Mock
        asset = AssetConfig(
            symbol="EURUSD",
            asset_class="FOREX",
            spread=0.0002,
            point=0.0001,
            currencies=["EUR", "USD"],
        )

        ctx = SimulationContext.create(asset, strategy)

        assert ctx.exit_strategy == "atr_based"
        assert ctx.exit_params.get("atr_period") == 21


class TestExitStrategyDispatch:
    """Testet die Exit-Strategy Dispatch-Logik in nested_cv."""

    def test_atr_based_strategy_dispatch(self):
        """Test: ATR-basierte Strategie wird korrekt aufgerufen."""
        from fwbg.core import get_exit_strategy as get_strategy
        from fwbg.core.context import SimulationContext

        # Strategie laden
        strategy_cls = get_strategy("atr_based")
        strategy = strategy_cls()

        # Mock-Context
        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0002,
            point=0.0001,
            exit_params={"atr_period": 14},
        )

        # Test-DataFrame
        df = create_test_df(500)

        # compute_targets sollte nicht crashen
        targets_long, targets_short = strategy.compute_targets(
            df, ctx,
            tp_mult=2.0,
            sl_mult=1.5,
            atr_period=14,
            min_tp_pips=10,
            min_sl_pips=15,
        )

        assert len(targets_long) == len(df)
        assert len(targets_short) == len(df)
        assert set(np.unique(targets_long)).issubset({0.0, 1.0})

    def test_compute_targets_cached_with_atr(self):
        """Test: compute_targets_cached funktioniert mit ATR-Strategie."""
        from fwbg.optimization.targets import compute_targets_cached
        from fwbg.core.context import SimulationContext

        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0002,
            point=0.0001,
            exit_params={"atr_period": 14, "min_tp_pips": 10, "min_sl_pips": 15},
        )

        df = create_test_df(500)

        # Sollte nicht crashen (war der ursprüngliche Bug)
        targets_l, targets_s = compute_targets_cached(
            df, tp=2.0, sl=1.5, ctx=ctx,
            exit_strategy_mode="atr_based"
        )

        assert len(targets_l) == len(df)
        assert len(targets_s) == len(df)

    def test_fixed_strategy_dispatch(self):
        """Test: Fixed-Strategie funktioniert weiterhin."""
        from fwbg.optimization.targets import compute_targets_cached
        from fwbg.core.context import SimulationContext

        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0002,
            point=0.0001,
        )

        df = create_test_df(500)

        targets_l, targets_s = compute_targets_cached(
            df, tp=20, sl=10, ctx=ctx,
            exit_strategy_mode="fixed"
        )

        assert len(targets_l) == len(df)


class TestStrategyIndicators:
    """Tests für Strategy-Indikatoren."""

    def test_strategy_get_indicators(self):
        """Test: Strategy.get_indicators() returns pipeline indicators."""
        strategy = StrategyConfig(
            pipeline={
                "indicators": [
                    {"name": "trend", "params": {}},
                    {"name": "momentum", "params": {}},
                    {"name": "volatility", "params": {}},
                ]
            }
        )

        indicators = strategy.get_indicators()

        assert len(indicators) == 3
        assert indicators[0]["name"] == "trend"
        assert indicators[1]["name"] == "momentum"
        assert indicators[2]["name"] == "volatility"


class TestStrategyConfigIntegration:
    """Integration-Tests für StrategyConfig."""

    def test_load_exploration_strategy(self):
        """Test: exploration.json kann geladen werden."""
        import os

        strategy_path = "strategies/exploration.json"
        if os.path.exists(strategy_path):
            strategy = StrategyConfig.from_json_file(strategy_path)

            assert strategy.name == "Exploration"
            assert strategy.exit_strategy == "fixed"
            assert "FOREX" in strategy.grids

    def test_grid_config_parsing(self):
        """Test: Grid-Config wird korrekt geparsed."""
        data = {
            "grids": {
                "FOREX": {
                    "tp": [1.0, 1.5, 2.0],
                    "sl": [1.0, 1.5],
                    "ct": [0.5, 0.55, 0.60],
                    "timeout_bars": [None, 24, 48]
                }
            }
        }

        strategy = StrategyConfig.from_dict(data)
        grid = strategy.get_grid_for_class("FOREX")

        assert grid.tp == [1.0, 1.5, 2.0]
        assert grid.sl == [1.0, 1.5]
        assert len(grid.timeout_bars) == 3
        assert None in grid.timeout_bars


class TestRunGridSearch:
    """Tests für run_grid_search Funktion."""

    def test_function_exists_and_imports(self):
        """Test: run_grid_search kann importiert werden."""
        from fwbg.optimization.grid_search import run_grid_search

    def test_grid_search_with_few_features(self):
        """Test: Grid-Search mit 1 Feature funktioniert."""
        from fwbg.optimization.grid_search import run_grid_search
        from fwbg.core.config import GridConfig
        import pandas as pd
        import numpy as np

        # Mock-Context mit minimalen Attributen
        ctx = Mock()
        ctx.symbol = "TEST"
        ctx.grid_combinations_per_run = Mock(return_value=1)
        ctx.total_grid_combinations = Mock(return_value=1)
        ctx.min_rrr = 0
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}

        grid = GridConfig(tp=[1.0], sl=[1.0], ct=[0.5])

        # Nur 1 Feature
        full_pool = ["trend_rsi_14"]

        # Dummy inner_df
        inner_df = pd.DataFrame({
            "trend_rsi_14": np.random.randn(100)
        })

        candidates, grid_results = run_grid_search(
            full_pool=full_pool,
            inner_folds=[],
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=inner_df,
        )

        # Sollte leer zurückkehren (keine inner_folds)
        assert candidates == []
        assert grid_results == []
