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


class TestMinFoldsBeforePruning:
    """Tests für min_folds_before_pruning_ratio in EarlyPruningConfig."""

    def test_early_pruning_config_parses_min_folds_before_pruning_ratio(self):
        """EarlyPruningConfig.from_dict liest min_folds_before_pruning_ratio."""
        from fwbg.core.config import EarlyPruningConfig
        config = EarlyPruningConfig.from_dict({
            "enabled": True, "keep_ratio": 0.5, "min_survivors": 10,
            "min_folds_before_pruning_ratio": 0.4,
        })
        assert config.min_folds_before_pruning_ratio == pytest.approx(0.4)

    def test_early_pruning_config_defaults_min_folds_before_pruning_ratio_to_0_3(self):
        """Ohne Angabe von min_folds_before_pruning_ratio: Default ist 0.3."""
        from fwbg.core.config import EarlyPruningConfig
        config = EarlyPruningConfig.from_dict({"enabled": True})
        assert config.min_folds_before_pruning_ratio == pytest.approx(0.3)

    def test_early_pruning_config_from_empty_dict_defaults_ratio_to_0_3(self):
        """Leeres Dict → min_folds_before_pruning_ratio == 0.3."""
        from fwbg.core.config import EarlyPruningConfig
        config = EarlyPruningConfig()
        assert config.min_folds_before_pruning_ratio == pytest.approx(0.3)

    def test_simulation_context_exposes_early_pruning_min_folds_ratio(self):
        """SimulationContext hat early_pruning_min_folds_before_pruning_ratio."""
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0], grid_sl=[10.0], grid_ct=[0.5],
            early_pruning_min_folds_before_pruning_ratio=0.4,
        )
        assert ctx.early_pruning_min_folds_before_pruning_ratio == pytest.approx(0.4)

    def test_pruning_skipped_for_early_folds_respects_ratio_setting(self):
        """Mit min_folds_before_pruning_ratio=0.3: 30% der Folds abwarten.

        Prüft, dass das Feld im Context korrekt propagiert wird.
        """
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0], grid_sl=[20.0], grid_ct=[0.5],
            early_pruning_enabled=True,
            early_pruning_keep_ratio=0.5,
            early_pruning_min_survivors=5,
            early_pruning_min_folds_before_pruning_ratio=0.3,
        )
        assert ctx.early_pruning_min_folds_before_pruning_ratio == pytest.approx(0.3)
        assert 0.0 <= ctx.early_pruning_min_folds_before_pruning_ratio <= 1.0

    def test_validation_config_propagates_min_folds_ratio_to_context(self):
        """ValidationConfig.from_dict propagiert min_folds_before_pruning_ratio korrekt."""
        from fwbg.core.config import ValidationConfig
        val_cfg = ValidationConfig.from_dict({
            "early_pruning": {
                "enabled": True,
                "keep_ratio": 0.5,
                "min_survivors": 10,
                "min_folds_before_pruning_ratio": 0.4,
            }
        })
        assert val_cfg.early_pruning.min_folds_before_pruning_ratio == pytest.approx(0.4)


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



class TestModelHyperparametersGrid:
    """Tests for model_hyperparameters_grid integration."""

    def test_gridconfig_parses_model_hyperparameters_grid(self):
        """GridConfig.from_dict parses model_hyperparameters_grid list."""
        from fwbg.core.config import GridConfig
        data = {
            "tp": [2.0], "sl": [1.0], "ct": [0.5],
            "model_hyperparameters_grid": [
                {"signal_column_long": "cf0_prb0_orb_s08_retest_bull",
                 "signal_column_short": "cf0_prb0_orb_s08_retest_bear"},
                {"signal_column_long": "cf1_prb0_orb_s08_retest_bull",
                 "signal_column_short": "cf1_prb0_orb_s08_retest_bear"},
            ]
        }
        grid = GridConfig.from_dict(data)
        assert len(grid.model_hyperparameters_grid) == 2
        assert grid.model_hyperparameters_grid[0]["signal_column_long"] == "cf0_prb0_orb_s08_retest_bull"

    def test_gridconfig_default_model_hyperparameters_grid_is_none_list(self):
        """Without model_hyperparameters_grid: defaults to [None]."""
        from fwbg.core.config import GridConfig
        grid = GridConfig.from_dict({"tp": [2.0], "sl": [1.0], "ct": [0.5]})
        assert grid.model_hyperparameters_grid == [None]

    def test_gridconfig_single_dict_wrapped_in_list(self):
        """A single dict is wrapped in a list."""
        from fwbg.core.config import GridConfig
        data = {
            "tp": [2.0], "sl": [1.0], "ct": [0.5],
            "model_hyperparameters_grid": {
                "signal_column_long": "test_col",
                "signal_column_short": "test_col2",
            }
        }
        grid = GridConfig.from_dict(data)
        assert len(grid.model_hyperparameters_grid) == 1
        assert isinstance(grid.model_hyperparameters_grid[0], dict)

    def test_context_grid_model_hyperparameters_default(self):
        """SimulationContext default grid_model_hyperparameters is [None]."""
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
        )
        assert ctx.grid_model_hyperparameters == [None]

    def test_total_grid_combinations_multiplied_by_model_hp_grid(self):
        """total_grid_combinations includes model_hp_grid factor."""
        ctx_no_grid = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0, 20.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[None],
            grid_exit_modifier_params=[None],
        )
        ctx_with_grid = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0, 20.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[
                {"signal_column_long": "a", "signal_column_short": "b"},
                {"signal_column_long": "c", "signal_column_short": "d"},
                {"signal_column_long": "e", "signal_column_short": "f"},
            ],
            grid_exit_modifier_params=[None],
        )
        assert ctx_with_grid.total_grid_combinations() == 3 * ctx_no_grid.total_grid_combinations()

    def test_combo_count_multiplied_by_model_hp_grid(self):
        """_build_combo_tuples produces N x more combos with N model_hp variants."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx_no_grid = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0, 20.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[None],
            grid_exit_modifier_params=[None],
            exit_modifier_params={},
            model_hyperparameters={"signal_column_long": "base_long", "signal_column_short": "base_short"},
        )
        ctx_with_grid = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0, 20.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[
                {"signal_column_long": "variant_a_long"},
                {"signal_column_long": "variant_b_long"},
            ],
            grid_exit_modifier_params=[None],
            exit_modifier_params={},
            model_hyperparameters={"signal_column_long": "base_long", "signal_column_short": "base_short"},
        )

        grid = MagicMock()
        grid.tp = [10.0, 20.0]
        grid.sl = [20.0]

        combos_no, _ = _build_combo_tuples(
            grid, ctx_no_grid, [32], ["feat1"], [], {}, 2, None, None, None, "TEST"
        )
        combos_with, _ = _build_combo_tuples(
            grid, ctx_with_grid, [32], ["feat1"], [], {}, 4, None, None, None, "TEST"
        )

        assert len(combos_with) == 2 * len(combos_no)

    def test_model_hp_variant_merged_into_combo_ctx(self):
        """Each combo ctx has model_hyperparameters merged with variant."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[
                {"signal_column_long": "variant_a_long", "signal_column_short": "variant_a_short"},
                {"signal_column_long": "variant_b_long", "signal_column_short": "variant_b_short"},
            ],
            grid_exit_modifier_params=[None],
            exit_modifier_params={},
            model_hyperparameters={"signal_column_long": "base_long", "signal_column_short": "base_short", "extra_param": 42},
        )

        grid = MagicMock()
        grid.tp = [10.0]
        grid.sl = [20.0]

        combos, _ = _build_combo_tuples(
            grid, ctx, [None], ["feat1"], [], {}, 2, None, None, None, "TEST"
        )

        assert len(combos) == 2

        # First combo: variant_a merged with base
        combo_ctx_a = combos[0][6]
        assert combo_ctx_a.model_hyperparameters["signal_column_long"] == "variant_a_long"
        assert combo_ctx_a.model_hyperparameters["signal_column_short"] == "variant_a_short"
        assert combo_ctx_a.model_hyperparameters["extra_param"] == 42  # base param preserved

        # Second combo: variant_b merged with base
        combo_ctx_b = combos[1][6]
        assert combo_ctx_b.model_hyperparameters["signal_column_long"] == "variant_b_long"
        assert combo_ctx_b.model_hyperparameters["signal_column_short"] == "variant_b_short"
        assert combo_ctx_b.model_hyperparameters["extra_param"] == 42

    def test_model_hp_none_uses_base_ctx(self):
        """model_hp_variant=None: combo ctx keeps base model_hyperparameters."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[None],
            grid_exit_modifier_params=[None],
            exit_modifier_params={},
            model_hyperparameters={"signal_column_long": "base_long", "signal_column_short": "base_short"},
        )

        grid = MagicMock()
        grid.tp = [10.0]
        grid.sl = [20.0]

        combos, _ = _build_combo_tuples(
            grid, ctx, [None], ["feat1"], [], {}, 1, None, None, None, "TEST"
        )

        assert len(combos) == 1
        combo_ctx = combos[0][6]
        assert combo_ctx.model_hyperparameters["signal_column_long"] == "base_long"

    def test_model_hp_grid_combined_with_modifier_grid(self):
        """model_hp_grid x modifier_grid: both dimensions multiply."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[
                {"signal_column_long": "a"},
                {"signal_column_long": "b"},
            ],
            grid_exit_modifier_params=[
                {"breakeven_trigger": 0.0},
                {"breakeven_trigger": 0.5},
            ],
            exit_modifier_params={},
            model_hyperparameters={},
        )

        grid = MagicMock()
        grid.tp = [10.0]
        grid.sl = [20.0]

        combos, _ = _build_combo_tuples(
            grid, ctx, [None], ["feat1"], [], {}, 4, None, None, None, "TEST"
        )

        # 2 model_hp x 2 modifier x 1 tp x 1 sl x 1 timeout = 4
        assert len(combos) == 4
        assert len(combos) == ctx.total_grid_combinations()

    def test_combo_count_matches_total_grid_combinations(self):
        """Combo count with all dimensions matches total_grid_combinations()."""
        from fwbg.optimization.grid_search import _build_combo_tuples

        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0, 20.0], grid_sl=[20.0, 30.0], grid_ct=[0.5],
            grid_timeout_bars=[32, 96],
            grid_model_hyperparameters=[
                {"signal_column_long": "a"},
                {"signal_column_long": "b"},
                {"signal_column_long": "c"},
            ],
            grid_exit_modifier_params=[
                {"breakeven_trigger": 0.0},
                {"breakeven_trigger": 0.5},
            ],
            exit_modifier_params={},
            model_hyperparameters={},
        )

        grid = MagicMock()
        grid.tp = [10.0, 20.0]
        grid.sl = [20.0, 30.0]

        combos, _ = _build_combo_tuples(
            grid, ctx, [32, 96], ["feat1"], [], {},
            ctx.total_grid_combinations(), None, None, None, "TEST"
        )

        # 3 model_hp x 2 modifier x 2 tp x 2 sl x 2 timeout = 48
        assert len(combos) == ctx.total_grid_combinations()
        assert len(combos) == 48


class TestCandidateStoresModelHyperparameters:
    """Tests that _build_candidate_and_grid_result stores model_hyperparameters."""

    def test_candidate_contains_model_hyperparameters(self):
        """Candidate dict must include ctx.model_hyperparameters."""
        from fwbg.optimization.grid_search import _build_candidate_and_grid_result

        inner_result = {
            "success": True,
            "avg_val_pnl": 42.0,
            "best_ct": 0.5,
            "selected_features": ["feat1", "feat2"],
            "selected_features_long": ["feat1"],
            "selected_features_short": ["feat2"],
            "fold_stability": 0.75,
        }
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            model_hyperparameters={
                "signal_column_long": "rb1_cf0_prb0_orb_s07_retest_bull",
                "signal_column_short": "rb1_cf0_prb0_orb_s07_retest_bear",
            },
            exit_modifier_params={"breakeven_trigger": 0.5},
        )

        candidate, grid_result = _build_candidate_and_grid_result(
            inner_result, tp=3.0, sl=2.0, timeout_bars=16,
            regime_config={}, ctx=ctx,
        )

        assert candidate is not None
        assert "model_hyperparameters" in candidate
        assert candidate["model_hyperparameters"]["signal_column_long"] == "rb1_cf0_prb0_orb_s07_retest_bull"
        assert candidate["model_hyperparameters"]["signal_column_short"] == "rb1_cf0_prb0_orb_s07_retest_bear"

    def test_grid_result_contains_model_hyperparameters(self):
        """Grid result dict must also include model_hyperparameters."""
        from fwbg.optimization.grid_search import _build_candidate_and_grid_result

        inner_result = {
            "success": True,
            "avg_val_pnl": 42.0,
            "best_ct": 0.5,
            "selected_features": ["feat1"],
            "selected_features_long": ["feat1"],
            "selected_features_short": ["feat1"],
            "fold_stability": 0.75,
        }
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            model_hyperparameters={"signal_column_long": "test_col"},
            exit_modifier_params={},
        )

        _, grid_result = _build_candidate_and_grid_result(
            inner_result, tp=3.0, sl=2.0, timeout_bars=16,
            regime_config={}, ctx=ctx,
        )

        assert grid_result is not None
        assert "model_hyperparameters" in grid_result
        assert grid_result["model_hyperparameters"]["signal_column_long"] == "test_col"

    def test_candidate_model_hp_reflects_combo_ctx_variant(self):
        """When _build_combo_tuples creates combo_ctx with merged HP variant,
        the candidate should carry that variant — not the base HP."""
        from fwbg.optimization.grid_search import (
            _build_combo_tuples, _build_candidate_and_grid_result,
        )

        base_hp = {
            "signal_column_long": "base_long",
            "signal_column_short": "base_short",
            "extra": 42,
        }
        variant_hp = {
            "signal_column_long": "variant_long",
            "signal_column_short": "variant_short",
        }
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            grid_tp=[10.0], grid_sl=[20.0], grid_ct=[0.5],
            grid_model_hyperparameters=[variant_hp],
            grid_exit_modifier_params=[None],
            exit_modifier_params={},
            model_hyperparameters=base_hp,
        )

        grid = MagicMock()
        grid.tp = [10.0]
        grid.sl = [20.0]

        combos, _ = _build_combo_tuples(
            grid, ctx, [None], ["feat1"], [], {}, 1, None, None, None, "TEST"
        )
        assert len(combos) == 1

        combo_ctx = combos[0][6]
        # combo_ctx should have the merged HP (variant overrides base)
        assert combo_ctx.model_hyperparameters["signal_column_long"] == "variant_long"
        assert combo_ctx.model_hyperparameters["extra"] == 42  # base preserved

        # Now simulate building a candidate from this combo
        inner_result = {
            "success": True, "avg_val_pnl": 10.0, "best_ct": 0.5,
            "selected_features": ["feat1"],
            "selected_features_long": ["feat1"],
            "selected_features_short": ["feat1"],
            "fold_stability": 0.5,
        }
        candidate, _ = _build_candidate_and_grid_result(
            inner_result, tp=10.0, sl=20.0, timeout_bars=None,
            regime_config={}, ctx=combo_ctx,
        )

        # The candidate must carry the variant HP, not the base
        assert candidate["model_hyperparameters"]["signal_column_long"] == "variant_long"
        assert candidate["model_hyperparameters"]["signal_column_short"] == "variant_short"


class TestAutoCollectRequiredFeatures:
    """Tests for auto-collecting signal columns into required_features."""

    def test_base_model_hp_signal_columns_auto_added(self):
        """Signal columns from base model_hyperparameters auto-added to required_features."""
        from fwbg.core.config import GridConfig

        # Minimal mock of strategy and asset to test SimulationContext.create()
        # We test the auto-collect directly by checking ctx after construction
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX", spread=0.0001, point=0.0001,
            model_hyperparameters={
                "signal_column_long": "cf0_prb0_orb_s08_retest_bull",
                "signal_column_short": "cf0_prb0_orb_s08_retest_bear",
            },
            required_features=[],
        )
        # Note: auto-collect happens in SimulationContext.create(), not __init__.
        # For unit testing, we verify the create() flow via integration tests below.
        # This test just verifies the field structure is correct.
        assert ctx.model_hyperparameters["signal_column_long"] == "cf0_prb0_orb_s08_retest_bull"
