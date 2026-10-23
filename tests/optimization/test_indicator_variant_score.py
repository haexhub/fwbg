"""Indicator-grid variants are ranked on inner validation PnL, never on test PnL."""

from types import SimpleNamespace

from fwbg.optimization import process

FOLDS = [SimpleNamespace(fold_id=0), SimpleNamespace(fold_id=1)]


def _variant(label, folds):
    """A variant whose folds report (inner_val_pnl, test_pnl); None = failed fold."""
    results = [
        (
            None
            if pnls is None
            else {
                "fold_id": i,
                "inner_val_pnl": pnls[0],
                "test_pnl": pnls[1],
                "best_config": {"tp": 2.0, "sl": 1.0, "ct": 0.5},
                "test_trades_trace": [{"pnl_raw": pnls[1]}],
                "test_trades_detail": [{"pnl_raw": pnls[1]}],
            }
        )
        for i, pnls in enumerate(folds)
    ]
    strategy = SimpleNamespace(
        resources=SimpleNamespace(max_parallel_folds=1),
        get_preprocessing=lambda: [],
        fold_results=results,
    )
    return label, strategy


def _rank(monkeypatch, variants):
    monkeypatch.setattr(
        process.SimulationContext,
        "create",
        lambda asset, strategy: SimpleNamespace(
            total_grid_combinations=lambda: 1, fold_results=strategy.fold_results
        ),
    )
    monkeypatch.setattr(process, "precompute_indicators", lambda df, strategy, sym: ([], None, 0))
    monkeypatch.setattr(
        process,
        "process_single_fold",
        lambda fold, fold_idx, *args: (args[4].fold_results[fold_idx], []),
    )
    return process._run_indicator_variants(variants, None, FOLDS, None, "TEST")


def test_variants_rank_by_inner_validation_pnl(monkeypatch):
    best = _rank(
        monkeypatch,
        [
            _variant("A", [(10.0, 1.0), (10.0, 1.0)]),
            _variant("B", [(1.0, 10.0), (1.0, 10.0)]),
        ],
    )
    assert best["label"] == "A"
    assert best["score"] == (2, 10.0)


def test_failed_variant_still_ranks_last(monkeypatch):
    best = _rank(
        monkeypatch,
        [
            _variant("failed", [None, None]),
            _variant("negative", [(-5.0, 3.0), (-1.0, 3.0)]),
        ],
    )
    assert best["label"] == "negative"
    assert best["score"] == (2, -3.0)


def test_parallel_fold_results_are_ordered_by_fold_id(monkeypatch):
    """Folds finishing in reverse order leave the same fold list and score."""
    label, strategy = _variant("A", [(1.0, 0.0), (2.0, 0.0)])
    strategy.resources.max_parallel_folds = 2
    monkeypatch.setattr(process, "as_completed", lambda futures: reversed(list(futures)))
    best = _rank(monkeypatch, [(label, strategy)])
    assert [r["fold_id"] for r in best["all_fold_results"]] == [0, 1]
    assert best["score"] == (2, 1.5)


def test_a_grid_keeps_a_light_record_of_every_variant(monkeypatch):
    """The replay needs every variant's folds and context, never their indicator data."""
    best = _rank(
        monkeypatch,
        [
            _variant("A", [(1.0, 0.0), (1.0, 0.0)]),
            _variant("B", [(5.0, 0.0), None]),
            _variant("C", [(2.0, 0.0), (2.0, 0.0)]),
        ],
    )
    assert best["label"] == "C"
    assert best["frozen_index"] == 2
    assert [record.label for record in best["variants"]] == ["A", "B", "C"]
    assert [len(record.fold_results) for record in best["variants"]] == [2, 1, 2]
    assert best["variants"][2].ctx is best["ctx"]
    assert not any(hasattr(record, "fold_indicators") for record in best["variants"])
    # Test trades are not needed by the replay; the frozen variant keeps its own in full.
    kept = [set(r) for record in best["variants"] for r in record.fold_results]
    assert not any({"test_trades_trace", "test_trades_detail"} & keys for keys in kept)
    assert all("test_trades_trace" in r for r in best["all_fold_results"])


def test_a_single_variant_keeps_no_records(monkeypatch):
    best = _rank(monkeypatch, [_variant("base", [(1.0, 0.0), (1.0, 0.0)])])
    assert "variants" not in best and "frozen_index" not in best
