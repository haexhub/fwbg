"""Tests für _build_combo_tuples und Combo-Zählung.

Stellt sicher, dass:
- Die Anzahl gebauter Combos exakt ctx.total_grid_combinations() entspricht
- Das exit_modifier_params_grid die Combo-Anzahl korrekt multipliziert
- Die Modifier-Params pro Combo-Ctx korrekt gesetzt werden
- Early-Exit in run_grid_search genau total_grid_combinations() Callbacks sendet
"""
import pytest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from fwbg.core.context import SimulationContext


def _make_ctx(modifier_params_grid=None, timeout_bars=None):
    """Erstellt einen minimalen SimulationContext für Combo-Tests."""
    if modifier_params_grid is None:
        modifier_params_grid = [None]
    return SimulationContext(
        symbol="TEST",
        asset_class="FOREX",
        spread=0.0001,
        point=0.0001,
        grid_tp=[10.0, 20.0],
        grid_sl=[20.0, 30.0],
        grid_ct=[0.5],
        grid_timeout_bars=timeout_bars,
        grid_exit_modifier_params=modifier_params_grid,
        exit_modifier_params={},
    )


def _make_grid():
    """Minimales grid-Mock mit tp=[10, 20], sl=[20, 30]."""
    grid = MagicMock()
    grid.tp = [10.0, 20.0]
    grid.sl = [20.0, 30.0]
    return grid


class TestBuildComboTuples:
    """Unit-Tests für _build_combo_tuples."""

    def test_combo_count_matches_total_grid_combinations_single_modifier(self):
        """Ohne Modifier-Grid: Combo-Anzahl = tp×sl×timeout = 2×2×2 = 8."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = _make_ctx(modifier_params_grid=[None], timeout_bars=[32, 96])
        grid = _make_grid()
        timeout_values = [32, 96]

        combos, skipped = _build_combo_tuples(
            grid, ctx, timeout_values, ["feat1"], [], {}, 8, None, None, None, "TEST"
        )

        assert skipped == 0
        assert len(combos) == 8
        assert len(combos) == ctx.total_grid_combinations()

    def test_combo_count_doubles_with_two_modifier_params(self):
        """Mit 2 Modifier-Params: Combo-Anzahl verdoppelt sich."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx_single = _make_ctx(modifier_params_grid=[None], timeout_bars=[32, 96])
        ctx_double = _make_ctx(
            modifier_params_grid=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
            timeout_bars=[32, 96],
        )
        grid = _make_grid()
        timeout_values = [32, 96]

        combos_single, _ = _build_combo_tuples(
            grid, ctx_single, timeout_values, ["feat1"], [], {}, 8, None, None, None, "TEST"
        )
        combos_double, _ = _build_combo_tuples(
            grid, ctx_double, timeout_values, ["feat1"], [], {}, 16, None, None, None, "TEST"
        )

        assert len(combos_double) == 2 * len(combos_single)

    def test_combo_count_with_modifier_matches_total_grid_combinations(self):
        """Combo-Anzahl mit Modifier-Grid = ctx.total_grid_combinations()."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = _make_ctx(
            modifier_params_grid=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
            timeout_bars=[32, 96],
        )
        grid = _make_grid()
        timeout_values = [32, 96]

        combos, skipped = _build_combo_tuples(
            grid, ctx, timeout_values, ["feat1"], [], {},
            ctx.total_grid_combinations(), None, None, None, "TEST"
        )

        assert skipped == 0
        assert len(combos) == ctx.total_grid_combinations()  # 2×2×2×2 = 16

    def test_modifier_params_applied_to_combo_ctx(self):
        """Jeder Combo-Ctx trägt die richtigen exit_modifier_params."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        modifier_no_trail = {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0}
        modifier_trail = {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5}
        ctx = _make_ctx(modifier_params_grid=[modifier_no_trail, modifier_trail])
        grid = _make_grid()
        timeout_values = [32]  # 1 timeout für übersichtliche Prüfung

        combos, _ = _build_combo_tuples(
            grid, ctx, timeout_values, ["feat1"], [], {}, 8, None, None, None, "TEST"
        )

        # 2 tp × 2 sl × 1 timeout × 2 modifier = 8 Combos
        assert len(combos) == 8

        combo_ctxs = [combo[6] for combo in combos]
        params_in_first_half = {str(sorted(c.exit_modifier_params.items())) for c in combo_ctxs[:4]}
        params_in_second_half = {str(sorted(c.exit_modifier_params.items())) for c in combo_ctxs[4:]}

        assert len(params_in_first_half) == 1, "Erste Hälfte: alle Combos haben gleiche Modifier-Params"
        assert len(params_in_second_half) == 1, "Zweite Hälfte: alle Combos haben gleiche Modifier-Params"
        assert params_in_first_half != params_in_second_half, "Erste und zweite Hälfte müssen unterschiedliche Params haben"

    def test_no_modifier_combo_ctx_uses_default_exit_modifier_params(self):
        """Ohne Modifier-Grid (None-Eintrag): Combo-Ctx entspricht dem Original-Ctx."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = _make_ctx(modifier_params_grid=[None])
        ctx_with_default = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0001,
            point=0.0001,
            grid_tp=[10.0, 20.0],
            grid_sl=[20.0, 30.0],
            grid_ct=[0.5],
            exit_modifier_params={"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            grid_exit_modifier_params=[None],
        )
        grid = _make_grid()
        timeout_values = [32]

        combos, _ = _build_combo_tuples(
            grid, ctx_with_default, timeout_values, ["feat1"], [], {}, 4, None, None, None, "TEST"
        )

        # Alle Combos müssen den Default-exit_modifier_params aus ctx tragen
        for combo in combos:
            combo_ctx = combo[6]
            assert combo_ctx.exit_modifier_params == {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5}

    def test_rrr_filter_skips_correct_count_with_modifier_grid(self):
        """RRR-Filter überspringt korrekte Anzahl auch mit Modifier-Grid."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0001,
            point=0.0001,
            grid_tp=[10.0, 20.0],
            grid_sl=[20.0, 30.0],
            grid_ct=[0.5],
            min_rrr=0.5,  # tp=10, sl=30 → RRR=0.33 → SKIP
            grid_exit_modifier_params=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
            exit_modifier_params={},
        )
        grid = _make_grid()
        timeout_values = [32, 96]  # 2 timeouts

        # tp=10, sl=30 → RRR=0.33 < 0.5 → SKIP; fails for BOTH modifier iterations
        # Expected: skipped = 2 modifier × 2 timeout = 4
        combos, skipped = _build_combo_tuples(
            grid, ctx, timeout_values, ["feat1"], [], {}, 12, None, None, None, "TEST"
        )

        assert skipped == 4  # 1 tp/sl pair × 2 timeout × 2 modifiers
        assert len(combos) == 12  # (4-1) valid pairs × 2 timeout × 2 modifiers


class TestEarlyExitProgressCallbacks:
    """Tests für Early-Exit Progress-Callback-Anzahl in run_grid_search."""

    def test_no_features_reports_total_grid_combinations_callbacks(self):
        """Bei leeren Features: genau total_grid_combinations() Callbacks."""
        from fwbg.optimization.grid_search import run_grid_search

        progress_calls = []

        def progress_cb(pos, total):
            progress_calls.append((pos, total))

        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0001,
            point=0.0001,
            grid_tp=[10.0, 20.0],
            grid_sl=[20.0, 30.0],
            grid_ct=[0.5],
            grid_exit_modifier_params=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
            exit_modifier_params={},
        )
        # total_grid_combinations() = 2×2×1×2 = 8 (no timeout, 2 modifiers)

        # inner_df mit nur inf-Werten → alle Features werden herausgefiltert
        inner_df = pd.DataFrame({"feat1": [np.inf] * 50, "feat2": [np.inf] * 50})

        grid = MagicMock()
        grid.tp = [10.0, 20.0]
        grid.sl = [20.0, 30.0]
        grid.timeout_bars = None
        grid.ct = [0.5]

        gs_cands, gs_grid = run_grid_search(
            full_pool=["feat1", "feat2"],
            inner_folds=[],
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            progress_callback=progress_cb,
            inner_df=inner_df,
        )

        assert gs_cands == []
        assert gs_grid == []
        # Muss genau total_grid_combinations() Callbacks senden (nicht grid_combinations_per_run())
        expected = ctx.total_grid_combinations()
        assert len(progress_calls) == expected, (
            f"Expected {expected} callbacks (=total_grid_combinations()), "
            f"got {len(progress_calls)}"
        )

    def test_no_selected_features_reports_total_grid_combinations_callbacks(self):
        """Bei fehlgeschlagener Feature-Selection: genau total_grid_combinations() Callbacks."""
        from fwbg.optimization.grid_search import run_grid_search

        progress_calls = []

        def progress_cb(pos, total):
            progress_calls.append((pos, total))

        ctx = SimulationContext(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0001,
            point=0.0001,
            grid_tp=[10.0, 20.0],
            grid_sl=[20.0, 30.0],
            grid_ct=[0.5],
            grid_exit_modifier_params=[
                {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
                {"breakeven_trigger": 0.5, "trail_atr_mult": 0.5},
            ],
            exit_modifier_params={},
        )

        inner_df = pd.DataFrame({"feat1": np.random.randn(50)})

        grid = MagicMock()
        grid.tp = [10.0, 20.0]
        grid.sl = [20.0, 30.0]
        grid.timeout_bars = None
        grid.ct = [0.5]

        # run_grid_search mit preselected_features=[](leer) → sofortiger Early-Exit nach Feature-Selection
        gs_cands, gs_grid = run_grid_search(
            full_pool=["feat1"],
            inner_folds=[],
            grid=grid,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            progress_callback=progress_cb,
            inner_df=inner_df,
            preselected_features_long=[],
            preselected_features_short=[],
        )

        assert gs_cands == []
        assert gs_grid == []
        expected = ctx.total_grid_combinations()
        assert len(progress_calls) == expected, (
            f"Expected {expected} callbacks (=total_grid_combinations()), "
            f"got {len(progress_calls)}"
        )
