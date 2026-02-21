"""Tests für Grid-Search Progress-Callbacks.

Diese Tests verifizieren, dass progress_callback für jede Grid-Kombination aufgerufen wird.
"""
import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np


class TestGridProgressCallback:
    """Tests für run_grid_search progress_callback Aufrufe."""

    def test_progress_callback_called_for_each_combo(self):
        """progress_callback sollte für jede abgeschlossene Grid-Kombination aufgerufen werden."""
        from fwbg.optimization.grid_search import run_grid_search

        # Setup: Mock-Objekte für die notwendigen Parameter
        progress_callback = MagicMock()

        # Minimaler ctx mit 2 TP, 2 SL = 4 Kombinationen
        ctx = MagicMock()
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0
        ctx.early_pruning_enabled = False  # Kein RRR-Filter
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.model_type = "xgboost"
        ctx.model_arch = "long_short_separate"
        ctx.trade_directions = ["long", "short"]
        ctx.model_hyperparams = {"n_estimators": 10, "max_depth": 2}
        ctx.grid_exit_modifier_params = [None]

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
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0
        ctx.early_pruning_enabled = False
        ctx.grid_exit_modifier_params = [None]

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
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0
        ctx.early_pruning_enabled = False
        ctx.grid_exit_modifier_params = [None]

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
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0
        ctx.early_pruning_enabled = False  # Kein RRR-Filter - alle Combos werden verarbeitet
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.model_type = "xgboost"
        ctx.model_arch = "long_short_separate"
        ctx.trade_directions = ["long", "short"]
        ctx.model_hyperparams = {"n_estimators": 10, "max_depth": 2}
        ctx.grid_exit_modifier_params = [None]

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
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0.5  # RRR-Filter: TP/SL muss >= 0.5 sein
        ctx.early_pruning_enabled = False
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.model_type = "xgboost"
        ctx.model_arch = "long_short_separate"
        ctx.trade_directions = ["long", "short"]
        ctx.model_hyperparams = {"n_estimators": 10, "max_depth": 2}
        ctx.grid_exit_modifier_params = [None]

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


class TestSuccessiveHalving:
    """Tests for successive halving grid search pruning."""

    def _make_ctx(self, keep_ratio=0.5, min_survivors=2):
        ctx = MagicMock()
        ctx.early_pruning_enabled = True
        ctx.early_pruning_keep_ratio = keep_ratio
        ctx.early_pruning_min_survivors = min_survivors
        ctx.sample_weights = False
        ctx.separate_long_short = False
        return ctx

    def _make_combos(self, n):
        """Create n dummy combo tuples with tp=10+i, sl=20."""
        combos = []
        for i in range(n):
            combos.append((
                10 + i, 20, None, i,
                ["feat1"], [], MagicMock(), {},
                0, n, None,
                ["feat1"], ["feat1"],
            ))
        return combos

    def _mock_eval(self, fold_idx, train_df, val_df, features, tp, sl, ctx,
                   timeout_bars=None, **kwargs):
        """Mock fold evaluation returning PnL proportional to tp."""
        return {
            "success": True,
            "pnl": float(tp),
            "best_ct": 0.5,
            "trades_by_ct": {},
            "selected_features_long": ["feat1"],
            "selected_features_short": ["feat1"],
        }

    def test_pruning_reduces_evaluations(self):
        """Later folds should evaluate fewer combos due to pruning."""
        from fwbg.optimization.grid_search import _run_with_successive_halving

        n_combos = 10
        combos = self._make_combos(n_combos)
        inner_folds = [(MagicMock(), MagicMock()) for _ in range(3)]
        ctx = self._make_ctx(keep_ratio=0.5, min_survivors=2)

        eval_calls = []

        def tracking_eval(fold_idx, train_df, val_df, features, tp, sl, ctx,
                          timeout_bars=None, **kwargs):
            eval_calls.append(fold_idx)
            return self._mock_eval(fold_idx, train_df, val_df, features, tp, sl,
                                   ctx, timeout_bars, **kwargs)

        with patch("fwbg.optimization.grid_search._evaluate_single_fold", side_effect=tracking_eval), \
             patch("fwbg.optimization.grid_search._compute_cached_targets", return_value={}):
            candidates, grid_results, _ = _run_with_successive_halving(
                combos, inner_folds, ctx,
                ["feat1"], {}, None,
                ["feat1"], ["feat1"],
                "TEST", None, 0, n_combos,
            )

        fold_counts = {}
        for f in eval_calls:
            fold_counts[f] = fold_counts.get(f, 0) + 1

        # Fold 0: all 10
        assert fold_counts[0] == 10
        # Fold 1: 5 survivors (50% of 10)
        assert fold_counts[1] == 5
        # Fold 2: 2 survivors (max(2, int(5*0.5))=2)
        assert fold_counts[2] == 2
        # Only final survivors produce candidates
        assert len(candidates) == 2

    def test_min_survivors_prevents_aggressive_pruning(self):
        """min_survivors should prevent pruning below the threshold."""
        from fwbg.optimization.grid_search import _run_with_successive_halving

        n_combos = 4
        combos = self._make_combos(n_combos)
        inner_folds = [(MagicMock(), MagicMock()) for _ in range(3)]
        # keep_ratio=0.1 would want 0 combos, but min_survivors=3 overrides
        ctx = self._make_ctx(keep_ratio=0.1, min_survivors=3)

        eval_calls = []

        def tracking_eval(fold_idx, *args, **kwargs):
            eval_calls.append(fold_idx)
            return self._mock_eval(fold_idx, *args, **kwargs)

        with patch("fwbg.optimization.grid_search._evaluate_single_fold", side_effect=tracking_eval), \
             patch("fwbg.optimization.grid_search._compute_cached_targets", return_value={}):
            _run_with_successive_halving(
                combos, inner_folds, ctx,
                ["feat1"], {}, None,
                ["feat1"], ["feat1"],
                "TEST", None, 0, n_combos,
            )

        fold_counts = {}
        for f in eval_calls:
            fold_counts[f] = fold_counts.get(f, 0) + 1

        assert fold_counts[0] == 4
        assert fold_counts[1] == 3  # min_survivors=3 overrides keep_ratio
        assert fold_counts[2] == 3  # min_survivors still respected

    def test_pruning_skipped_single_fold(self):
        """With 1 inner fold, pruning should not activate."""
        from fwbg.optimization.grid_search import run_grid_search

        ctx = MagicMock()
        ctx.total_grid_combinations.return_value = 4
        ctx.min_rrr = 0
        ctx.early_pruning_enabled = False
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.early_pruning_enabled = True
        ctx.early_pruning_keep_ratio = 0.5
        ctx.early_pruning_min_survivors = 2
        ctx.grid_exit_modifier_params = [None]

        grid = MagicMock()
        grid.tp = [10, 20]
        grid.sl = [20, 30]
        grid.timeout_bars = None

        inner_df = pd.DataFrame({"feat1": np.random.randn(100)})
        inner_folds = [(MagicMock(), MagicMock())]  # Single fold!

        with patch("fwbg.optimization.grid_search.select_features", return_value=(["feat1"], ["feat1"])), \
             patch("fwbg.optimization.grid_search._run_with_successive_halving") as mock_sh, \
             patch("fwbg.optimization.grid_search._process_tp_sl_combo_wrapper",
                   return_value=({"params": (10, 20, 0.6), "inner_val_pnl": 1.0}, {}, 0)):
            run_grid_search(["feat1"], inner_folds, grid, ctx, {}, "TEST", inner_df=inner_df)
            mock_sh.assert_not_called()

    def test_pruning_skipped_few_combos(self):
        """With fewer combos than min_survivors, pruning should not activate."""
        from fwbg.optimization.grid_search import run_grid_search

        ctx = MagicMock()
        ctx.total_grid_combinations.return_value = 2
        ctx.min_rrr = 0
        ctx.early_pruning_enabled = False
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}
        ctx.early_pruning_enabled = True
        ctx.early_pruning_keep_ratio = 0.5
        ctx.early_pruning_min_survivors = 10  # More than available combos
        ctx.grid_exit_modifier_params = [None]

        grid = MagicMock()
        grid.tp = [10]
        grid.sl = [20, 30]
        grid.timeout_bars = None

        inner_df = pd.DataFrame({"feat1": np.random.randn(100)})
        inner_folds = [(MagicMock(), MagicMock()), (MagicMock(), MagicMock())]

        with patch("fwbg.optimization.grid_search.select_features", return_value=(["feat1"], ["feat1"])), \
             patch("fwbg.optimization.grid_search._run_with_successive_halving") as mock_sh, \
             patch("fwbg.optimization.grid_search._process_tp_sl_combo_wrapper",
                   return_value=({"params": (10, 20, 0.6), "inner_val_pnl": 1.0}, {}, 0)):
            run_grid_search(["feat1"], inner_folds, grid, ctx, {}, "TEST", inner_df=inner_df)
            mock_sh.assert_not_called()

    def test_progress_reports_all_combos(self):
        """Progress should be reported for all combos (pruned + survivors)."""
        from fwbg.optimization.grid_search import _run_with_successive_halving

        n_combos = 10
        combos = self._make_combos(n_combos)
        inner_folds = [(MagicMock(), MagicMock()) for _ in range(3)]
        ctx = self._make_ctx(keep_ratio=0.5, min_survivors=2)
        progress_cb = MagicMock()

        with patch("fwbg.optimization.grid_search._evaluate_single_fold", side_effect=self._mock_eval), \
             patch("fwbg.optimization.grid_search._compute_cached_targets", return_value={}):
            _, _, progress_reported = _run_with_successive_halving(
                combos, inner_folds, ctx,
                ["feat1"], {}, None,
                ["feat1"], ["feat1"],
                "TEST", progress_cb, 0, n_combos,
            )

        # All combos accounted for: pruned after fold 0 (5) + after fold 1 (3) + survivors (2)
        assert progress_reported == n_combos
        assert progress_cb.call_count == n_combos

    def test_failed_folds_get_low_ranking(self):
        """Combos with failed folds should rank below successful ones."""
        from fwbg.optimization.grid_search import _run_with_successive_halving

        n_combos = 6
        combos = self._make_combos(n_combos)
        inner_folds = [(MagicMock(), MagicMock()) for _ in range(2)]
        ctx = self._make_ctx(keep_ratio=0.5, min_survivors=1)

        def mixed_eval(fold_idx, train_df, val_df, features, tp, sl, ctx,
                       timeout_bars=None, **kwargs):
            # Combos 0-2 (tp=10,11,12) fail, combos 3-5 (tp=13,14,15) succeed
            if tp < 13:
                return {"success": False}
            return {
                "success": True, "pnl": float(tp), "best_ct": 0.5,
                "trades_by_ct": {},
                "selected_features_long": ["feat1"],
                "selected_features_short": ["feat1"],
            }

        with patch("fwbg.optimization.grid_search._evaluate_single_fold", side_effect=mixed_eval), \
             patch("fwbg.optimization.grid_search._compute_cached_targets", return_value={}):
            candidates, _, _ = _run_with_successive_halving(
                combos, inner_folds, ctx,
                ["feat1"], {}, None,
                ["feat1"], ["feat1"],
                "TEST", None, 0, n_combos,
            )

        # Only successful combos should produce candidates
        # After fold 0: 6 combos, top 3 kept. Failed combos have -inf PnL → pruned.
        # After fold 1: last fold, 3 survivors aggregated.
        assert len(candidates) == 3
        # All candidates should have tp >= 13
        for c in candidates:
            assert c["params"][0] >= 13


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


class TestProbabilityCalibration:
    """Tests for probability calibration replacing CT grid search.

    Uses real objects (SimulationContext, XGBoost, CalibratedClassifierCV) instead
    of mocks to verify actual behavior end-to-end.
    """

    @staticmethod
    def _make_ctx(probability_calibration=False, calibration_method="isotonic",
                  separate_long_short=False):
        """Create a real SimulationContext for calibration tests."""
        from fwbg.core.context import SimulationContext
        return SimulationContext(
            symbol="TESTUSD",
            asset_class="FOREX",
            spread=0.0002,
            point=0.0001,
            min_trades=10,
            grid_ct=[0.50, 0.55, 0.60],
            grid_tp=[10, 20],
            grid_sl=[20, 30],
            long_enabled=True,
            short_enabled=True,
            separate_long_short=separate_long_short,
            model_hyperparameters={"n_estimators": 10, "max_depth": 2, "random_state": 42},
            probability_calibration=probability_calibration,
            calibration_method=calibration_method,
        )

    @staticmethod
    def _make_ohlc_df(n=500, seed=42):
        """Create a realistic OHLC DataFrame with _atr and _regime columns."""
        rng = np.random.default_rng(seed)
        # Random walk price
        price = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
        atr = np.abs(rng.normal(0.001, 0.0003, n))
        df = pd.DataFrame({
            "O": price,
            "H": price + np.abs(rng.normal(0.0005, 0.0002, n)),
            "L": price - np.abs(rng.normal(0.0005, 0.0002, n)),
            "C": price + rng.normal(0, 0.0003, n),
            "_atr": atr,
            "_regime": np.full(n, 7, dtype=np.int8),
            "feat1": rng.standard_normal(n),
            "feat2": rng.standard_normal(n),
            "feat3": rng.standard_normal(n),
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))
        return df

    def test_config_roundtrip(self):
        """ValidationConfig parses probability_calibration from dict and defaults correctly."""
        from fwbg.core.config import ValidationConfig

        # Explicit values
        cfg = ValidationConfig.from_dict({
            "probability_calibration": True,
            "calibration_method": "sigmoid",
        })
        assert cfg.probability_calibration is True
        assert cfg.calibration_method == "sigmoid"

        # Defaults
        cfg_default = ValidationConfig.from_dict({})
        assert cfg_default.probability_calibration is False
        assert cfg_default.calibration_method == "isotonic"

    def test_context_wiring_from_strategy(self):
        """SimulationContext.create() wires probability_calibration from StrategyConfig."""
        from fwbg.core.config import StrategyConfig

        strategy = StrategyConfig.from_dict({
            "validation": {
                "probability_calibration": True,
                "calibration_method": "sigmoid",
            },
            "grids": {"FOREX": {"tp": [10], "sl": [20], "ct": [0.5]}},
        })

        from fwbg.data.assets import AssetConfig
        asset = AssetConfig(
            symbol="TESTUSD", asset_class="FOREX",
            spread=0.0002, point=0.0001, currencies=["USD"],
        )

        from fwbg.core.context import SimulationContext
        ctx = SimulationContext.create(asset, strategy)
        assert ctx.probability_calibration is True
        assert ctx.calibration_method == "sigmoid"

    def test_train_model_calibrated_returns_model_with_calibration(self):
        """train_model with calibration returns a BaseModel with calibrated predictions."""
        from fwbg.optimization.nested_cv import train_model
        from fwbg_sdk.models import BaseModel

        ctx = self._make_ctx(probability_calibration=True)
        df = self._make_ohlc_df(300)
        targets = np.random.default_rng(42).integers(0, 2, len(df)).astype(float)

        model = train_model(df, targets, ["feat1", "feat2"], min_trades=10, ctx=ctx)

        assert isinstance(model, BaseModel)
        assert model._calibrated_model is not None
        probs = model.predict_probability_calibrated(df[["feat1", "feat2"]])
        assert probs.shape == (len(df), 2)
        # Calibrated probs should sum to 1 per row
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_train_model_uncalibrated_returns_model(self):
        """train_model without calibration returns a BaseModel without calibration."""
        from fwbg.optimization.nested_cv import train_model
        from fwbg_sdk.models import BaseModel

        ctx = self._make_ctx(probability_calibration=False)
        df = self._make_ohlc_df(300)
        targets = np.random.default_rng(42).integers(0, 2, len(df)).astype(float)

        model = train_model(df, targets, ["feat1", "feat2"], min_trades=10, ctx=ctx)

        assert isinstance(model, BaseModel)
        assert model._calibrated_model is None

    def test_evaluate_on_validation_calibrated_uses_ev_threshold(self):
        """With calibration, evaluate_on_validation uses EV-optimal CT instead of grid."""
        from fwbg.optimization.nested_cv import train_model
        from fwbg.optimization.targets import evaluate_on_validation

        ctx = self._make_ctx(probability_calibration=True)
        df = self._make_ohlc_df(500)
        rng = np.random.default_rng(42)
        targets_long = (rng.random(len(df)) > 0.6).astype(float)
        targets_short = (rng.random(len(df)) > 0.6).astype(float)

        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        features = ["feat1", "feat2", "feat3"]

        mod_long = train_model(train_df, targets_long[:350], features, 10, ctx)
        mod_short = train_model(train_df, targets_short[:350], features, 10, ctx)

        tp, sl = 20, 30
        expected_ct = sl / (tp + sl)  # 0.60

        best_ct, best_pnl, trades_by_ct = evaluate_on_validation(
            val_df, mod_long, mod_short, features, features, tp, sl, ctx,
        )

        # CT must be the EV-optimal threshold, not from grid
        assert best_ct == pytest.approx(expected_ct)
        # Only one CT evaluated (no grid search)
        assert len(trades_by_ct) == 1
        assert expected_ct in trades_by_ct

    def test_evaluate_on_validation_uncalibrated_uses_grid(self):
        """Without calibration, evaluate_on_validation searches the CT grid."""
        from fwbg.optimization.nested_cv import train_model
        from fwbg.optimization.targets import evaluate_on_validation

        ctx = self._make_ctx(probability_calibration=False)
        df = self._make_ohlc_df(500)
        rng = np.random.default_rng(42)
        targets_long = (rng.random(len(df)) > 0.6).astype(float)
        targets_short = (rng.random(len(df)) > 0.6).astype(float)

        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        features = ["feat1", "feat2", "feat3"]

        mod_long = train_model(train_df, targets_long[:350], features, 10, ctx)
        mod_short = train_model(train_df, targets_short[:350], features, 10, ctx)

        tp, sl = 20, 30

        best_ct, best_pnl, trades_by_ct = evaluate_on_validation(
            val_df, mod_long, mod_short, features, features, tp, sl, ctx,
        )

        # Grid has 3 CT values [0.50, 0.55, 0.60], all should be evaluated
        assert len(trades_by_ct) == 3
        # best_ct must be from the grid
        if best_ct is not None:
            assert best_ct in ctx.grid_ct

    def test_calibrated_separate_long_short_returns_ct_tuple(self):
        """With calibration + separate_long_short, CT should be (ct_ev, ct_ev) tuple."""
        from fwbg.optimization.nested_cv import train_model
        from fwbg.optimization.targets import evaluate_on_validation

        ctx = self._make_ctx(probability_calibration=True, separate_long_short=True)
        df = self._make_ohlc_df(500)
        rng = np.random.default_rng(42)
        targets_long = (rng.random(len(df)) > 0.6).astype(float)
        targets_short = (rng.random(len(df)) > 0.6).astype(float)

        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        features = ["feat1", "feat2", "feat3"]

        mod_long = train_model(train_df, targets_long[:350], features, 10, ctx)
        mod_short = train_model(train_df, targets_short[:350], features, 10, ctx)

        tp, sl = 15, 30
        expected_ct = sl / (tp + sl)  # 30/45 ≈ 0.667

        best_ct, _, _ = evaluate_on_validation(
            val_df, mod_long, mod_short, features, features, tp, sl, ctx,
        )

        assert isinstance(best_ct, tuple)
        assert best_ct[0] == pytest.approx(expected_ct)
        assert best_ct[1] == pytest.approx(expected_ct)

    def test_ev_threshold_varies_with_tp_sl(self):
        """Different TP/SL ratios should produce different EV thresholds."""
        from fwbg.optimization.nested_cv import train_model
        from fwbg.optimization.targets import evaluate_on_validation

        ctx = self._make_ctx(probability_calibration=True)
        df = self._make_ohlc_df(500)
        rng = np.random.default_rng(42)
        targets = (rng.random(len(df)) > 0.6).astype(float)

        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        features = ["feat1", "feat2", "feat3"]

        mod = train_model(train_df, targets[:350], features, 10, ctx)

        # TP=10, SL=40 → ct = 40/50 = 0.80
        ct1, _, _ = evaluate_on_validation(val_df, mod, mod, features, features, 10, 40, ctx)
        assert ct1 == pytest.approx(0.80)

        # TP=30, SL=20 → ct = 20/50 = 0.40
        ct2, _, _ = evaluate_on_validation(val_df, mod, mod, features, features, 30, 20, ctx)
        assert ct2 == pytest.approx(0.40)

        # TP=20, SL=20 → ct = 0.50
        ct3, _, _ = evaluate_on_validation(val_df, mod, mod, features, features, 20, 20, ctx)
        assert ct3 == pytest.approx(0.50)
