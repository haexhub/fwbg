"""Tests für Grid-Search Progress-Callbacks.

Diese Tests verifizieren, dass progress_callback für jede Grid-Kombination aufgerufen wird.
"""
import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np


class TestProcessSymbolProgressReporting:
    """Tests für report_progress Aufrufe in process_symbol."""

    def test_process_single_fold_passes_progress_callback(self):
        """process_single_fold sollte progress_callback an run_grid_search übergeben."""
        from fwbg.optimization import process_fold
        import inspect

        source = inspect.getsource(process_fold.process_single_fold)

        # Prüfe ob run_grid_search mit progress_callback aufgerufen wird
        assert "run_grid_search" in source, "process_single_fold sollte run_grid_search aufrufen"
        assert "progress_callback" in source, "process_single_fold sollte progress_callback verwenden"


class TestGridProgressCallback:
    """Tests für run_grid_search progress_callback Aufrufe."""

    def test_progress_callback_called_for_each_combo(self):
        """progress_callback sollte für jede abgeschlossene Grid-Kombination aufgerufen werden."""
        from fwbg.optimization.grid_search import run_grid_search

        # Setup: Mock-Objekte für die notwendigen Parameter
        progress_callback = MagicMock()

        # Minimaler ctx mit 2 TP, 2 SL = 4 Kombinationen
        ctx = MagicMock()
        ctx.grid_combinations_per_run.return_value = 4
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0  # Kein RRR-Filter
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.model_type = "xgboost"
        ctx.model_arch = "long_short_separate"
        ctx.trade_directions = ["long", "short"]
        ctx.model_hyperparams = {"n_estimators": 10, "max_depth": 2}

        # Minimaler grid mit 2x2 = 4 Kombinationen
        grid = MagicMock()
        grid.tp = [10, 20]
        grid.sl = [20, 30]
        grid.ct = [0.6]
        grid.timeout_bars = None

        # Dummy inner_df mit genug Daten
        np.random.seed(42)
        inner_df = pd.DataFrame({
            "trend_adx_14": np.random.randn(1000),
            "trend_ema_21": np.random.randn(1000),
            "trend_slope_21": np.random.randn(1000),
            "momentum_rsi_14": np.random.randn(1000),
        })

        # Dummy inner_folds
        inner_folds = [
            MagicMock(train_idx=list(range(500)), val_idx=list(range(500, 700))),
            MagicMock(train_idx=list(range(200, 700)), val_idx=list(range(700, 900))),
        ]

        # full_pool mit Feature-Namen
        full_pool = ["trend_adx_14", "trend_ema_21", "trend_slope_21", "momentum_rsi_14"]

        # Aufruf der Funktion
        with patch("fwbg.optimization.grid_search.select_features") as mock_select:
            # Feature-Selection überspringen - Features direkt zurückgeben
            mock_select.return_value = (full_pool, full_pool)

            with patch("fwbg.optimization.grid_search._process_single_grid_combo") as mock_combo:
                # Jede Combo liefert ein Ergebnis
                mock_combo.return_value = (
                    {"params": (10, 20, 0.6), "inner_val_pnl": 100},  # candidate
                    {"tp": 10, "sl": 20, "pnl": 100},  # grid_result
                )

                candidates, grid_results = run_grid_search(
                    full_pool=full_pool,
                    inner_folds=inner_folds,
                    grid=grid,
                    ctx=ctx,
                    regime_config={},
                    sym="EURUSD",
                    progress_callback=progress_callback,
                    inner_df=inner_df,
                )

        # ASSERTION: progress_callback sollte für JEDE der 4 Kombinationen aufgerufen werden
        assert progress_callback.call_count == 4, (
            f"progress_callback sollte 4x aufgerufen werden (für 4 Grid-Kombinationen), "
            f"wurde aber {progress_callback.call_count}x aufgerufen"
        )

        # Die Aufrufe sollten den Fortschritt korrekt reporten
        # Erwartet: (1, 4), (2, 4), (3, 4), (4, 4) - jeweils (position, total)
        calls = [call.args for call in progress_callback.call_args_list]
        for i, (pos, total) in enumerate(calls):
            assert total == 4, f"grid_total sollte 4 sein, ist aber {total}"
            assert pos >= 1 and pos <= 4, f"grid_pos sollte zwischen 1-4 sein, ist aber {pos}"


    def test_progress_callback_called_on_early_exit_no_features(self):
        """progress_callback sollte auch bei Early-Exit (keine Features) aufgerufen werden."""
        from fwbg.optimization.grid_search import run_grid_search

        progress_callback = MagicMock()

        # ctx mit 4 Grid-Kombinationen
        ctx = MagicMock()
        ctx.grid_combinations_per_run.return_value = 4
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0

        grid = MagicMock()
        grid.tp = [10, 20]
        grid.sl = [20, 30]
        grid.ct = [0.6]
        grid.timeout_bars = None

        # Leerer inner_df - wird zu Early-Exit führen (keine Features)
        inner_df = pd.DataFrame({
            "some_col": [1, 2, 3],
        })

        inner_folds = [MagicMock(train_idx=[0, 1], val_idx=[2])]
        full_pool = []  # Keine Features - Early-Exit

        candidates, grid_results = run_grid_search(
            full_pool=full_pool,
            inner_folds=inner_folds,
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="EURUSD",
            progress_callback=progress_callback,
            inner_df=inner_df,
        )

        # ASSERTION: Auch bei Early-Exit sollten ALLE Grid-Kombinationen reportet werden
        # Damit die Progress-Anzeige korrekt weiterläuft
        assert progress_callback.call_count == 4, (
            f"progress_callback sollte 4x aufgerufen werden auch bei Early-Exit, "
            f"wurde aber {progress_callback.call_count}x aufgerufen"
        )


    def test_progress_callback_called_on_boruta_no_selection(self):
        """progress_callback sollte aufgerufen werden wenn Boruta keine Features selektiert."""
        from fwbg.optimization.grid_search import run_grid_search

        progress_callback = MagicMock()

        ctx = MagicMock()
        ctx.grid_combinations_per_run.return_value = 4
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0

        grid = MagicMock()
        grid.tp = [10, 20]
        grid.sl = [20, 30]
        grid.ct = [0.6]
        grid.timeout_bars = None

        # inner_df mit genug Features
        inner_df = pd.DataFrame({
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100),
            "feat3": np.random.randn(100),
            "feat4": np.random.randn(100),
        })

        inner_folds = [MagicMock(train_idx=list(range(80)), val_idx=list(range(80, 100)))]
        full_pool = ["feat1", "feat2", "feat3", "feat4"]

        with patch("fwbg.optimization.grid_search.select_features") as mock_select:
            # Boruta findet keine Features
            mock_select.return_value = ([], [])

            candidates, grid_results = run_grid_search(
                full_pool=full_pool,
                inner_folds=inner_folds,
                grid=grid,
                ctx=ctx,
                regime_config={},
                sym="EURUSD",
                progress_callback=progress_callback,
                inner_df=inner_df,
            )

        # ASSERTION: Progress sollte trotzdem reportet werden
        assert progress_callback.call_count == 4, (
            f"progress_callback sollte 4x aufgerufen werden auch wenn Boruta keine Features findet, "
            f"wurde aber {progress_callback.call_count}x aufgerufen"
        )


class TestProgressCallbackSequence:
    """Tests dass progress_callback mit korrekter Sequenz aufgerufen wird."""

    def test_progress_callback_increments_sequentially(self):
        """progress_callback sollte sequentiell von 1 bis total inkrementieren."""
        from fwbg.optimization.grid_search import run_grid_search
        from unittest.mock import MagicMock, patch

        progress_callback = MagicMock()

        # ctx mit 4 Grid-Kombinationen (2 TP x 2 SL)
        ctx = MagicMock()
        ctx.grid_combinations_per_run.return_value = 4
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0  # Kein RRR-Filter - alle Combos werden verarbeitet
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.model_type = "xgboost"
        ctx.model_arch = "long_short_separate"
        ctx.trade_directions = ["long", "short"]
        ctx.model_hyperparams = {"n_estimators": 10, "max_depth": 2}

        grid = MagicMock()
        grid.tp = [10, 20]
        grid.sl = [20, 30]
        grid.ct = [0.6]
        grid.timeout_bars = None

        np.random.seed(42)
        inner_df = pd.DataFrame({
            "feat1": np.random.randn(1000),
            "feat2": np.random.randn(1000),
        })

        inner_folds = [
            MagicMock(train_idx=list(range(500)), val_idx=list(range(500, 700))),
        ]
        full_pool = ["feat1", "feat2"]

        with patch("fwbg.optimization.grid_search.select_features") as mock_select:
            mock_select.return_value = (full_pool, full_pool)

            with patch("fwbg.optimization.grid_search._process_single_grid_combo") as mock_combo:
                mock_combo.return_value = (
                    {"params": (10, 20, 0.6), "inner_val_pnl": 100},
                    {"tp": 10, "sl": 20, "pnl": 100},
                )

                run_grid_search(
                    full_pool=full_pool,
                    inner_folds=inner_folds,
                    grid=grid,
                    ctx=ctx,
                    regime_config={},
                    sym="TEST",
                    progress_callback=progress_callback,
                    inner_df=inner_df,
                )

        # Extrahiere alle grid_pos Werte aus den Aufrufen
        grid_positions = [call.args[0] for call in progress_callback.call_args_list]

        # Sollte sequentiell 1, 2, 3, 4 sein
        assert grid_positions == [1, 2, 3, 4], (
            f"progress_callback sollte sequentiell 1-4 aufgerufen werden, "
            f"wurde aber mit {grid_positions} aufgerufen"
        )

    def test_progress_callback_with_rrr_skipped_combos(self):
        """progress_callback sollte auch bei RRR-übersprungenen Combos korrekt zählen."""
        from fwbg.optimization.grid_search import run_grid_search
        from unittest.mock import MagicMock, patch

        progress_callback = MagicMock()

        # ctx mit min_rrr Filter - einige Combos werden übersprungen
        ctx = MagicMock()
        ctx.grid_combinations_per_run.return_value = 4  # 2 TP x 2 SL = 4
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0.5  # RRR-Filter: TP/SL muss >= 0.5 sein
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.model_type = "xgboost"
        ctx.model_arch = "long_short_separate"
        ctx.trade_directions = ["long", "short"]
        ctx.model_hyperparams = {"n_estimators": 10, "max_depth": 2}

        grid = MagicMock()
        # TP=10, SL=30 -> RRR=0.33 < 0.5 -> SKIP
        # TP=10, SL=20 -> RRR=0.5 >= 0.5 -> OK
        # TP=20, SL=30 -> RRR=0.67 >= 0.5 -> OK
        # TP=20, SL=20 -> RRR=1.0 >= 0.5 -> OK
        grid.tp = [10, 20]
        grid.sl = [20, 30]
        grid.ct = [0.6]
        grid.timeout_bars = None

        np.random.seed(42)
        inner_df = pd.DataFrame({
            "feat1": np.random.randn(1000),
            "feat2": np.random.randn(1000),
        })

        inner_folds = [
            MagicMock(train_idx=list(range(500)), val_idx=list(range(500, 700))),
        ]
        full_pool = ["feat1", "feat2"]

        with patch("fwbg.optimization.grid_search.select_features") as mock_select:
            mock_select.return_value = (full_pool, full_pool)

            with patch("fwbg.optimization.grid_search._process_single_grid_combo") as mock_combo:
                mock_combo.return_value = (
                    {"params": (10, 20, 0.6), "inner_val_pnl": 100},
                    {"tp": 10, "sl": 20, "pnl": 100},
                )

                run_grid_search(
                    full_pool=full_pool,
                    inner_folds=inner_folds,
                    grid=grid,
                    ctx=ctx,
                    regime_config={},
                    sym="TEST",
                    progress_callback=progress_callback,
                    inner_df=inner_df,
                )

        # Callback sollte trotzdem 4x aufgerufen werden (total combos = 4)
        assert progress_callback.call_count == 4, (
            f"progress_callback sollte 4x aufgerufen werden, "
            f"wurde aber {progress_callback.call_count}x aufgerufen"
        )

        # Die grid_pos Werte sollten sequentiell 1-4 sein
        grid_positions = [call.args[0] for call in progress_callback.call_args_list]
        assert grid_positions == [1, 2, 3, 4], (
            f"progress_callback sollte sequentiell 1-4 aufgerufen werden (auch bei RRR-Skips), "
            f"wurde aber mit {grid_positions} aufgerufen"
        )


class TestGridResultsInProcessSymbolOutput:
    """Tests dass grid_results im process_symbol Ergebnis enthalten sind."""

    def test_process_symbol_returns_grid_results(self):
        """process_symbol sollte grid_results im Ergebnis-Dictionary enthalten."""
        from fwbg.optimization import process
        import re
        import inspect

        # Hole den Quellcode
        source = inspect.getsource(process.process_symbol)

        # Prüfe ob grid_results zum result dictionary hinzugefügt wird
        # Suche nach "grid_results" als Key in einem return-Statement oder result-Dict
        grid_results_pattern = r'"grid_results":\s*\w+'

        matches = re.findall(grid_results_pattern, source)
        assert len(matches) >= 1, (
            "process_symbol sollte 'grid_results' im result Dictionary enthalten.\n"
            "Erwartet: \"grid_results\": <variable>\n"
            "Dies ist notwendig, damit grid_details Dateien gespeichert werden können."
        )
