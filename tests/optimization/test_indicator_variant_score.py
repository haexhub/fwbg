"""Indicator-grid variants are ranked on inner validation PnL, never on test PnL."""

from types import SimpleNamespace

from fwbg.optimization import process

FOLDS = [SimpleNamespace(fold_id=0), SimpleNamespace(fold_id=1)]


def _variant(label, folds):
    """A variant whose folds report (inner_val_pnl, test_pnl); None = failed fold."""
    results = [
        None if pnls is None else {"inner_val_pnl": pnls[0], "test_pnl": pnls[1]} for pnls in folds
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
