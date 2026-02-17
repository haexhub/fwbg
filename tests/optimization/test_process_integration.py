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
        "_regime": np.full(n_rows, 7, dtype=np.int8),
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

    def test_process_fold_imports(self):
        """Test: process_fold module can be imported."""
        from fwbg.optimization.process_fold import precompute_indicators, process_single_fold

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
        assert ctx.exit_strategy == "fixed"  # Default

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
        from fwbg.core import get_exit_strategy
        from fwbg.core.context import SimulationContext

        # Strategie laden
        exit_strategy_class = get_exit_strategy("atr_based")
        exit_strategy = exit_strategy_class()

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
        targets_long, targets_short = exit_strategy.compute_targets(
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
        grid = strategy.get_grid("EURUSD", "FOREX")

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


class TestPrecomputedMergeNoDuplicates:
    """Precomputed raw indicator merge must not create duplicate columns.

    Bug: When data_loading adds macro_* columns to df, and those same columns
    pass through compute_indicator_pool into precomputed_raw_df, pd.concat
    creates duplicates. Then train_df[col] returns a DataFrame instead of a
    Series, causing 'truth value of Series is ambiguous' in the inf check.
    """

    def test_no_duplicate_columns_after_merge(self):
        """Precomputed features must exclude columns already in base df."""
        n = 500
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "O": 100 + np.cumsum(np.random.randn(n) * 0.1),
            "H": 101 + np.cumsum(np.random.randn(n) * 0.1),
            "L": 99 + np.cumsum(np.random.randn(n) * 0.1),
            "C": 100 + np.cumsum(np.random.randn(n) * 0.1),
            "V": np.random.randint(100, 10000, n).astype(float),
            # Simulate macro columns added by data_loading
            "macro_vix": np.random.rand(n) * 30,
            "macro_tnx": np.random.rand(n) * 5,
        }, index=dates)

        from fwbg.pipeline import compute_indicator_pool

        # compute_indicator_pool passes through existing columns (incl. macro_*)
        precomputed = compute_indicator_pool(
            df, indicators=[{"name": "trend", "params": {"adx_periods": [14]}}]
        )

        # Filter: only NEW features, not columns already in df
        base_cols = set(df.columns) | {"O", "H", "L", "C", "V"}
        raw_feature_cols = [c for c in precomputed.columns
                           if c not in base_cols and not c.startswith("_")]
        precomputed_raw_df = precomputed[raw_feature_cols]

        # macro columns must NOT be in precomputed_raw_df
        assert "macro_vix" not in precomputed_raw_df.columns
        assert "macro_tnx" not in precomputed_raw_df.columns

        # Merge like process.py does
        train_df = df.iloc[:400].copy()
        merged = pd.concat([train_df, precomputed_raw_df.reindex(train_df.index)], axis=1)

        # No duplicate columns
        dupes = merged.columns[merged.columns.duplicated()].tolist()
        assert dupes == [], f"Duplicate columns after merge: {dupes}"

        # The inf check that was crashing must work on every column
        for col in merged.columns:
            if col in ["O", "H", "L", "C", "V"]:
                continue
            has_inf = np.isinf(merged[col]).any()
            # Must be a scalar bool, not a Series
            assert isinstance(has_inf, (bool, np.bool_)), \
                f"has_inf for '{col}' is {type(has_inf)}, not bool — duplicate column?"


class TestStrategyFilePluginResolution:
    """All strategy JSON files must reference only resolvable plugin names.

    Short names must resolve to a fully qualified name via the plugin
    registry. Unresolvable names (no ':' after normalization) would crash
    at runtime in PluginRegistry.get().
    """

    @staticmethod
    def _collect_plugin_names(strategy):
        """Extract all plugin names from a strategy's pipeline config."""
        names = []
        for ind in strategy.get_indicators():
            names.append(("indicator", ind["name"]))
        for pp in strategy.get_preprocessing():
            names.append(("preprocessing", pp["name"]))
        for fs in strategy.get_feature_selection():
            names.append(("feature_selection", fs["name"]))
        return names

    def test_all_strategy_plugins_resolve(self):
        """Every plugin name in strategies/*.json must resolve to a fully qualified name."""
        import glob
        from fwbg.pipeline.features import normalize_plugin_name

        strategy_files = sorted(glob.glob("strategies/*.json"))
        assert strategy_files, "No strategy files found"

        errors = []
        for path in strategy_files:
            strategy = StrategyConfig.from_json_file(path)
            for kind, name in self._collect_plugin_names(strategy):
                fq_name = normalize_plugin_name(name)
                if ":" not in fq_name:
                    errors.append(f"{path}: {kind} '{name}' did not resolve (got '{fq_name}')")

        assert not errors, (
            "Unresolvable plugin names in strategy files:\n"
            + "\n".join(errors)
        )


class TestAllNanColumnProtection:
    """All-NaN feature columns must not destroy entire fold via dropna.

    Regression test: mtf_y1_ema200d_dist was ALL NaN on 4000-row test
    subsets (warmup > test size). The old code did dropna() before
    removing bad columns, which killed ALL rows.
    """

    def test_all_nan_column_removed_before_dropna(self):
        """An all-NaN feature column must not cause dropna to kill all rows."""
        from fwbg.pipeline.features import get_feature_columns

        n = 500
        df = pd.DataFrame({
            "O": np.random.randn(n) + 100,
            "H": np.random.randn(n) + 101,
            "L": np.random.randn(n) + 99,
            "C": np.random.randn(n) + 100,
            "trend_rsi_14": np.random.rand(n) * 100,
            "trend_adx_14": np.random.rand(n) * 50,
            "vol_atr_14": np.abs(np.random.randn(n)) + 0.02,
            # This column is ALL NaN (simulates indicator with warmup > data size)
            "mtf_y1_ema200d_dist": np.full(n, np.nan),
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        # Reproduce the fixed logic from process_fold.py
        full_pool = get_feature_columns(df)
        assert "mtf_y1_ema200d_dist" in full_pool

        clean_pool = []
        drop_cols = []
        for col in full_pool:
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio >= 0.1:
                drop_cols.append(col)
            else:
                clean_pool.append(col)

        df_clean = df.drop(columns=drop_cols, errors="ignore")
        df_clean = df_clean.dropna()

        # The all-NaN column must be excluded
        assert "mtf_y1_ema200d_dist" in drop_cols
        assert "mtf_y1_ema200d_dist" not in clean_pool

        # Rows must survive
        assert len(df_clean) == n, (
            f"dropna killed {n - len(df_clean)} rows — all-NaN column was not removed first"
        )

    def test_test_set_all_nan_column_also_caught(self):
        """A column valid in train but all-NaN in test must be excluded."""
        from fwbg.pipeline.features import get_feature_columns

        n = 500
        base = {
            "O": np.random.randn(n) + 100,
            "H": np.random.randn(n) + 101,
            "L": np.random.randn(n) + 99,
            "C": np.random.randn(n) + 100,
            "trend_rsi_14": np.random.rand(n) * 100,
        }
        idx = pd.date_range("2024-01-01", periods=n, freq="h")

        # Train has valid data for the column
        train_df = pd.DataFrame({
            **base,
            "slow_indicator": np.random.rand(n),
        }, index=idx)

        # Test has all NaN (warmup > test size)
        test_df = pd.DataFrame({
            **base,
            "slow_indicator": np.full(n, np.nan),
        }, index=idx)

        # Reproduce fixed process_fold logic
        full_pool = get_feature_columns(train_df)
        clean_pool = []
        drop_cols = []
        for col in full_pool:
            nan_ratio = train_df[col].isna().sum() / len(train_df)
            if nan_ratio >= 0.1:
                drop_cols.append(col)
            else:
                clean_pool.append(col)

        # Check test set for all-NaN columns
        for col in list(clean_pool):
            if col in test_df.columns and test_df[col].isna().all():
                clean_pool.remove(col)
                drop_cols.append(col)

        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        test_df = test_df.drop(columns=drop_cols, errors="ignore")
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        assert "slow_indicator" in drop_cols
        assert len(train_df) == n
        assert len(test_df) == n
