"""Test fold k is replayed with the indicator variant that is best on folds 0..k."""

from types import SimpleNamespace

import pytest

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization import causal_reporting, causal_variants, indicator_grid, process
from fwbg.optimization.causal_variants import (
    VariantRecord,
    prefix_variant_choice,
    replay_prefix_variants,
    replay_view,
    variant_score,
)
from fwbg.optimization.process import process_symbol
from fwbg.optimization.signal_fold import prepare_signal_fold_data
from tests.optimization import symbol_harness
from tests.optimization.symbol_harness import STRATEGY, SYMBOL_FILE, WINNING

FOLDS = [SimpleNamespace(fold_id=i) for i in range(4)]


def _results(inner_pnls):
    """Fold results with these inner validation PnLs; None = failed fold."""
    config = {"tp": 2.0, "sl": 1.0, "ct": 0.5}
    return [
        {"fold_id": i, "inner_val_pnl": pnl, "best_config": config}
        for i, pnl in enumerate(inner_pnls)
        if pnl is not None
    ]


def test_variant_score_without_successful_folds():
    assert variant_score([]) == (0, float("-inf"))


def test_variant_score_counts_folds_and_averages_inner_pnl():
    assert variant_score(_results([1.0, None, 2.0])) == (2, 1.5)


def test_prefix_choice_switches_where_the_overall_best_overtakes():
    early = _results([10.0, 10.0, 0.0, 0.0])
    late = _results([1.0, 1.0, 30.0, 30.0])
    assert prefix_variant_choice(FOLDS, [early, late]) == [0, 0, 1, 1]


def test_prefix_choice_ties_go_to_the_lower_index():
    same = _results([5.0, 5.0, 5.0, 5.0])
    assert prefix_variant_choice(FOLDS, [same, list(same)]) == [0, 0, 0, 0]
    assert prefix_variant_choice(FOLDS, [_results([None] * 4)] * 2) == [0, 0, 0, 0]


def test_prefix_choice_prefers_more_successful_folds_to_a_better_mean():
    steady = _results([1.0, 1.0, 1.0, 1.0])
    gappy = _results([None, 50.0, 50.0, 50.0])
    assert prefix_variant_choice(FOLDS, [gappy, steady]) == [1, 1, 1, 1]


def test_prefix_choice_ignores_fold_result_order():
    early = _results([10.0, 10.0, 0.0, 0.1])
    late = _results([1.0, 0.3, 30.0, 30.0])
    expected = prefix_variant_choice(FOLDS, [early, late])
    assert prefix_variant_choice(FOLDS, [early[::-1], late[::-1]]) == expected


# ---------------------------------------------------------------------------
# The replay through process_symbol
# ---------------------------------------------------------------------------

# Variant 0 ("late", frozen: best over all folds) is best only from fold 2 on;
# variant 1 ("early") is best on folds 0-1. TP = base + fold_id tells them apart.
BASE_TP = {"late": 5.0, "early": 1.0}


def _inner_pnl(label, fold_id):
    if label == "late":
        return 1.0 if fold_id < 2 else 30.0
    return 10.0 if fold_id < 2 else 0.0


def _patch_two_variants(monkeypatch):
    """Two fake indicator variants; records the replay's precomputes and simulations."""
    contexts, precomputed, replays = {}, [], []
    strategies = {"late": STRATEGY, "early": StrategyConfig.from_dict(symbol_harness.METADATA)}

    def variant_fold(label, fold):
        result = symbol_harness.fold_result(fold)
        tp = BASE_TP[label] + fold.fold_id
        return {
            **result,
            "inner_val_pnl": _inner_pnl(label, fold.fold_id),
            "best_config": {**result["best_config"], "tp": tp, "rrr": tp},
        }

    def single_variant(idx, label, strategy, df, wf_folds, asset, sym, n_variants):
        results = [variant_fold(label, fold) for fold in wf_folds]
        contexts[label] = SimulationContext.create(asset, strategy)
        return {
            "score": variant_score(results),
            "label": label,
            "strategy": strategy,
            "ctx": contexts[label],
            "all_fold_results": results,
            "accumulated_grid_results": [],
            "fold_indicators": f"indicators-{label}",
            "precomputed_raw_df": f"raw-{label}",
            "preprocessing_configs": [],
        }

    def precompute(df, strategy, sym):
        precomputed.append(strategy)
        return "indicators-recomputed", "raw-recomputed", 0

    def simulate(
        folds, candidates, fold_indicators, raw_df, configs, ctx, sym, prepare_data_fn=None
    ):
        replays.append(
            {
                "fold_ids": [fold.fold_id for fold in folds],
                "tps": [candidate["params"][0] for candidate in candidates],
                "ctx": ctx,
                "fold_indicators": fold_indicators,
            }
        )
        return [
            {**symbol_harness.unified_fold(fold, WINNING), "fold_id": fold.fold_id}
            for fold in reversed(folds)
        ]

    monkeypatch.setattr(process, "load_data_aligned", lambda path: symbol_harness.hourly_frame())
    monkeypatch.setattr(
        indicator_grid, "expand_indicator_grid", lambda strategy: list(strategies.items())
    )
    monkeypatch.setattr(process, "_process_single_variant", single_variant)
    monkeypatch.setattr(causal_variants, "precompute_indicators", precompute)
    monkeypatch.setattr(causal_variants, "run_unified_simulation", simulate)
    monkeypatch.setattr(
        process, "run_unified_simulation", lambda *a, **k: pytest.fail("single path")
    )
    return strategies, contexts, precomputed, replays


def test_each_fold_is_replayed_once_with_the_variant_best_up_to_it(monkeypatch, capsys):
    strategies, contexts, precomputed, replays = _patch_two_variants(monkeypatch)
    reported = []
    compute = causal_reporting.compute_causal_reporting

    def spy(fold_results, *args):
        reported.append([r["fold_id"] for r in fold_results])
        return compute(fold_results, *args)

    monkeypatch.setattr(causal_reporting, "compute_causal_reporting", spy)
    result = process_symbol(SYMBOL_FILE, STRATEGY)

    n_folds = len(reported[0])
    assert n_folds > 2
    assert reported == [list(range(n_folds))]
    late, early = replays
    assert late["fold_ids"] == list(range(2, n_folds))
    assert early["fold_ids"] == [0, 1]
    assert late["ctx"] is contexts["late"] and early["ctx"] is contexts["early"]
    # Each fold's candidate is its variant's prefix merge: median TP of base + 0..k.
    assert late["tps"] == [BASE_TP["late"] + k / 2 for k in range(2, n_folds)]
    assert early["tps"] == [BASE_TP["early"] + k / 2 for k in (0, 1)]
    # Only the non-frozen variant recomputes its indicators.
    assert late["fold_indicators"] == "indicators-late"
    assert early["fold_indicators"] == "indicators-recomputed"
    assert precomputed == [strategies["early"]]
    # The frozen setting stays the frozen variant's full merge.
    assert result["indicator_variant"] == "late"
    assert result["unified_candidate"]["params"][0] == BASE_TP["late"] + (n_folds - 1) / 2
    mapping = " ".join(f"{k + 1}:{'early' if k < 2 else 'late'}" for k in range(n_folds))
    assert f"Replay variants per fold: {mapping}" in capsys.readouterr().err


def _record_simulations(monkeypatch):
    """Replace the fold simulation with a recorder of (folds, candidates, data, prepare)."""
    calls = []

    def simulate(folds, candidates, indicators, *args, prepare_data_fn=None):
        calls.append((folds, candidates, indicators, prepare_data_fn))
        return [{"fold_id": fold.fold_id} for fold in folds]

    monkeypatch.setattr(causal_variants, "run_unified_simulation", simulate)
    return calls


def test_a_variant_without_a_setting_on_its_folds_is_not_recomputed(monkeypatch):
    calls = _record_simulations(monkeypatch)
    monkeypatch.setattr(causal_variants, "precompute_indicators", pytest.fail)
    ctx = SimpleNamespace(model_type="xgboost")
    records = [
        VariantRecord("first", None, ctx, [], _results([None, 1.0, 1.0, 1.0])),
        VariantRecord("frozen", None, ctx, [], _results([None, 2.0, 2.0, 2.0])),
    ]
    results = replay_prefix_variants(FOLDS, records, 1, "frozen-indicators", None, None, "TEST")

    assert [r["fold_id"] for r in results] == [0, 1, 2, 3]
    (first_folds, first_candidates, first_data, _), (frozen_folds, _, frozen_data, _) = calls
    assert [fold.fold_id for fold in first_folds] == [0] and first_candidates == [None]
    assert first_data is None
    assert [fold.fold_id for fold in frozen_folds] == [1, 2, 3]
    assert frozen_data == "frozen-indicators"


def test_a_signal_variant_prepares_signal_fold_data(monkeypatch):
    calls = _record_simulations(monkeypatch)
    monkeypatch.setattr(causal_variants, "precompute_indicators", lambda *args: (None, None, 0))
    records = [
        VariantRecord(
            "signal",
            None,
            SimpleNamespace(model_type="signal"),
            [],
            _results([9.0, 0.0, 0.0, -20.0]),
        ),
        VariantRecord("ml", None, SimpleNamespace(model_type="xgboost"), [], _results([1.0] * 4)),
    ]
    replay_prefix_variants(FOLDS, records, 1, None, None, None, "TEST")

    assert [prepare for *_, prepare in calls] == [prepare_signal_fold_data, None]


def test_the_last_fold_must_replay_the_frozen_variant(monkeypatch):
    _record_simulations(monkeypatch)
    ctx = SimpleNamespace(model_type="xgboost")
    records = [
        VariantRecord("worse", None, ctx, [], _results([1.0] * 4)),
        VariantRecord("best", None, ctx, [], _results([2.0] * 4)),
    ]
    with pytest.raises(ValueError, match="frozen"):
        replay_prefix_variants(FOLDS, records, 0, None, None, None, "TEST")


def test_replay_view_keeps_only_what_the_replay_reads():
    result = {
        **_results([1.0])[0],
        "selected_features_long": ["f1"],
        "test_trades_trace": [{"pnl_raw": 1.0}],
        "test_trades_detail": [{"entry_time": "t"}],
        "test_pnl": 3.0,
    }
    assert replay_view([result]) == [
        {
            "fold_id": 0,
            "inner_val_pnl": 1.0,
            "best_config": result["best_config"],
            "selected_features_long": ["f1"],
        }
    ]
