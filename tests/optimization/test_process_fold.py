"""Integration tests for process_single_fold.

These tests actually CALL process_single_fold to catch issues like
NameErrors, wrong variable ordering, and broken progress reporting.
Mock only external dependencies (XGBoost, Feature Selection),
use real config objects for everything else.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

from fwbg.core.config import GridConfig, RegimeFilterGridConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization.robust_validation import WalkForwardFold


def _make_ohlc_df(n=800, seed=42):
    """Create a realistic OHLC DataFrame with features."""
    rng = np.random.default_rng(seed)
    price = 1.1 + np.cumsum(rng.normal(0, 0.0005, n))
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
        "feat4": rng.standard_normal(n),
        "feat5": rng.standard_normal(n),
        "feat6": rng.standard_normal(n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="h"))
    return df


def _make_fold(df, fold_id=0, train_ratio=0.7):
    """Create a WalkForwardFold from a DataFrame."""
    split = int(len(df) * train_ratio)
    return WalkForwardFold(
        fold_id=fold_id,
        train_start=0,
        train_end=split,
        test_start=split,
        test_end=len(df),
        train_df=df.iloc[:split].copy(),
        test_df=df.iloc[split:].copy(),
    )


def _make_ctx(**overrides):
    """Create a minimal SimulationContext for testing."""
    defaults = dict(
        symbol="TESTUSD",
        asset_class="FOREX",
        spread=0.0002,
        point=0.0001,
        min_trades=5,
        grid_tp=[10, 20],
        grid_sl=[20, 30],
        grid_ct=[0.5, 0.55],
        long_enabled=True,
        short_enabled=True,
        n_inner_folds=2,
        embargo_bars=0,
        sample_weights=False,
        early_pruning_enabled=False,
        exit_strategy="fixed",
        exit_params={},
        model_hyperparameters={"n_estimators": 10, "max_depth": 2, "random_state": 42},
    )
    defaults.update(overrides)
    return SimulationContext(**defaults)


def _make_grid(regime_conditions=None):
    """Create a GridConfig with optional regime filter conditions."""
    regime_grid = RegimeFilterGridConfig(
        condition_grids=regime_conditions or []
    )
    return GridConfig(
        tp=[10, 20],
        sl=[20, 30],
        ct=[0.5, 0.55],
        timeout_bars=[None],
        regime_filter_grid=regime_grid,
    )


class TestProcessSingleFoldIntegration:
    """Integration tests that actually call process_single_fold."""

    def _run_fold(self, fold_idx=0, grid=None, ctx=None, regime_conditions=None):
        """Run process_single_fold with mocked XGB/feature selection internals."""
        from fwbg.optimization.process_fold import process_single_fold

        df = _make_ohlc_df()
        fold = _make_fold(df, fold_id=fold_idx)
        grid = grid or _make_grid(regime_conditions=regime_conditions)
        ctx = ctx or _make_ctx()
        features = ["feat1", "feat2", "feat3", "feat4", "feat5", "feat6"]

        # Mock select_features to return immediately
        mock_select = patch(
            "fwbg.optimization.process_fold.select_features",
            return_value=(features, features),
        )
        # Mock run_grid_search to return dummy candidates
        def fake_grid_search(full_pool, inner_folds, grid, ctx, regime_config,
                             sym, progress_callback=None, inner_df=None,
                             preselected_features_long=None,
                             preselected_features_short=None):
            # Simulate progress
            n = ctx.total_grid_combinations()
            if progress_callback:
                for i in range(1, n + 1):
                    progress_callback(i, n)
            return [
                {
                    "inner_val_pnl": 50.0,
                    "params": (10, 20, 0.5),
                    "timeout_bars": None,
                    "feats": features,
                    "rrr": 0.5,
                    "selected_features_long": features,
                    "selected_features_short": features,
                    "fold_stability": 0.8,
                    "regime_filter": regime_config,
                    "score": 50.0,
                }
            ], [{"tp_mult": 10, "sl_mult": 20, "inner_val_pnl": 50.0}]

        mock_grid = patch(
            "fwbg.optimization.process_fold.run_grid_search",
            side_effect=fake_grid_search,
        )
        # Mock evaluate_on_holdout
        mock_holdout = patch(
            "fwbg.optimization.process_fold.evaluate_on_holdout",
            return_value={
                "pnl": 30.0,
                "win_rate": 0.55,
                "n_trades": 20,
                "trades": [],
            },
        )
        # Mock plateau functions (just pass through)
        mock_plateau = patch(
            "fwbg.optimization.process_fold.calculate_param_plateau_score",
            side_effect=lambda candidates, *args, **kwargs: candidates,
        )
        mock_select_plateau = patch(
            "fwbg.optimization.process_fold.select_best_plateau_candidate",
            side_effect=lambda candidates, *args, **kwargs: candidates[0] if candidates else None,
        )
        # Mock nested_cv_split
        train_df = fold.train_df
        split = len(train_df) // 2
        mock_cv = patch(
            "fwbg.optimization.process_fold.nested_cv_split",
            return_value={
                "inner_folds": [
                    (train_df.iloc[:split].copy(), train_df.iloc[split:].copy()),
                ],
            },
        )
        # Suppress progress queue
        mock_phase = patch("fwbg.optimization.process_fold.report_phase")
        mock_meta = patch("fwbg.optimization.process_fold.report_meta")
        mock_progress = patch("fwbg.optimization.process_fold.report_progress")

        with mock_select, mock_grid as m_grid, mock_holdout, mock_plateau, \
             mock_select_plateau, mock_cv, mock_phase as m_phase, \
             mock_meta as m_meta, mock_progress as m_progress:
            result, grid_results = process_single_fold(
                fold=fold,
                fold_idx=fold_idx,
                n_folds=8,
                fold_indicators=[],
                precomputed_raw_df=None,
                preprocessing_configs=None,
                grid=grid,
                ctx=ctx,
                sym="TESTUSD",
                total_indicators=6,
            )

        return result, grid_results, m_grid, m_phase, m_meta, m_progress

    def test_fold_idx_zero_no_crash(self):
        """process_single_fold with fold_idx=0 must not crash (NameError regression)."""
        result, grid_results, _, _, _, _ = self._run_fold(fold_idx=0)
        assert result is not None
        assert result["fold_id"] == 0
        assert result["test_pnl"] == 30.0

    def test_fold_idx_nonzero(self):
        """process_single_fold with fold_idx > 0 works correctly."""
        result, _, _, _, _, _ = self._run_fold(fold_idx=3)
        assert result is not None
        assert result["fold_id"] == 3

    def test_report_meta_called_only_first_fold(self):
        """report_meta should only be called for fold_idx=0."""
        _, _, _, _, m_meta, _ = self._run_fold(fold_idx=0)
        assert m_meta.call_count >= 1

        _, _, _, _, m_meta2, _ = self._run_fold(fold_idx=1)
        assert m_meta2.call_count == 0

    def test_report_meta_includes_regime_combos(self):
        """report_meta should include regime_combos count."""
        _, _, _, _, m_meta, _ = self._run_fold(
            fold_idx=0,
            regime_conditions=[
                {"column": "feat1", "operator": ">=", "values": [None, 0.5],
                 "directions": 6, "else_directions": 0},
            ],
        )
        # Find the call with regime_combos
        meta_kwargs = m_meta.call_args.kwargs if m_meta.call_args.kwargs else {}
        if not meta_kwargs:
            # positional → keyword in some mock versions
            _, meta_kwargs = m_meta.call_args
        assert "regime_combos" in meta_kwargs
        assert meta_kwargs["regime_combos"] == 2  # [null, 0.5] = 2 combos

    def test_regime_combos_all_get_grid_search(self):
        """With 2 regime combos, run_grid_search should be called 2x."""
        _, _, m_grid, _, _, _ = self._run_fold(
            regime_conditions=[
                {"column": "feat1", "operator": ">=", "values": [None, 0.5],
                 "directions": 6, "else_directions": 0},
            ],
        )
        assert m_grid.call_count == 2

    def test_preselected_features_passed_to_grid_search(self):
        """Feature selection results should be passed as preselected_features."""
        _, _, m_grid, _, _, _ = self._run_fold()

        # Check first call to run_grid_search
        _, kwargs = m_grid.call_args
        assert "preselected_features_long" in kwargs
        assert "preselected_features_short" in kwargs
        assert kwargs["preselected_features_long"] is not None
        assert kwargs["preselected_features_short"] is not None

    def test_grid_progress_monotonic_across_regime_combos(self):
        """Grid progress should be monotonically increasing across regime combos."""
        _, _, _, _, _, m_progress = self._run_fold(
            regime_conditions=[
                {"column": "feat1", "operator": ">=", "values": [None, 0.5],
                 "directions": 6, "else_directions": 0},
            ],
        )

        if m_progress.call_count > 0:
            grid_positions = [
                c.kwargs.get("grid_pos", c.args[4] if len(c.args) > 4 else 0)
                for c in m_progress.call_args_list
            ]
            # Positions must be monotonically non-decreasing
            for i in range(1, len(grid_positions)):
                assert grid_positions[i] >= grid_positions[i - 1], (
                    f"Grid progress jumped backwards at index {i}: "
                    f"{grid_positions[i-1]} -> {grid_positions[i]}"
                )

    def test_grid_total_includes_regime_multiplier(self):
        """grid_total in progress reports should include regime combo count."""
        ctx = _make_ctx()
        base_combos = ctx.total_grid_combinations()  # 2 TP * 2 SL = 4

        _, _, _, _, _, m_progress = self._run_fold(
            ctx=ctx,
            regime_conditions=[
                {"column": "feat1", "operator": ">=", "values": [None, 0.5],
                 "directions": 6, "else_directions": 0},
            ],
        )

        if m_progress.call_count > 0:
            grid_totals = [
                c.kwargs.get("grid_total", c.args[5] if len(c.args) > 5 else 0)
                for c in m_progress.call_args_list
            ]
            expected_total = base_combos * 2  # 2 regime combos
            for total in grid_totals:
                assert total == expected_total, (
                    f"grid_total should be {expected_total} (base={base_combos} × 2 regime), "
                    f"got {total}"
                )

    def test_no_features_selected_returns_none(self):
        """When feature selection finds nothing, fold should be skipped."""
        from fwbg.optimization.process_fold import process_single_fold

        df = _make_ohlc_df()
        fold = _make_fold(df, fold_id=0)
        grid = _make_grid()
        ctx = _make_ctx()

        with patch("fwbg.optimization.process_fold.select_features", return_value=(None, None)), \
             patch("fwbg.optimization.process_fold.nested_cv_split", return_value={"inner_folds": []}), \
             patch("fwbg.optimization.process_fold.report_phase"), \
             patch("fwbg.optimization.process_fold.report_meta"), \
             patch("fwbg.optimization.process_fold.report_progress"):
            result, grid_results = process_single_fold(
                fold=fold, fold_idx=0, n_folds=8,
                fold_indicators=[], precomputed_raw_df=None,
                preprocessing_configs=None,
                grid=grid, ctx=ctx, sym="TEST", total_indicators=6,
            )

        assert result is None
        assert grid_results == []

    def test_phase_messages_include_regime_info(self):
        """With multiple regime combos, phase messages should show R{n}/{total}."""
        _, _, _, m_phase, _, _ = self._run_fold(
            regime_conditions=[
                {"column": "feat1", "operator": ">=", "values": [None, 0.5],
                 "directions": 6, "else_directions": 0},
            ],
        )

        phase_texts = [c.args[1] for c in m_phase.call_args_list if len(c.args) > 1]
        regime_phases = [t for t in phase_texts if "R1/" in t or "R2/" in t]
        assert len(regime_phases) >= 2, (
            f"Expected regime info in phase messages, got: {phase_texts}"
        )
