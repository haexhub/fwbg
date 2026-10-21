import json

import pytest

from fwbg.research.ledger import TrialStatus
from fwbg.research.legacy import LegacyRunError, run_outcome
from tests.research.factories import make_run


def test_run_outcome_counts_evaluated_symbols_only(tmp_path):
    """A macro asset is not evaluated; the run is ok through its evaluated symbol."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"EURUSD": {"unified": {"sharpe": 0.5}}, "DXY": {"status": "macro_asset"}},
    )
    status, metrics = run_outcome(run_dir)
    assert status is TrialStatus.OK
    assert set(metrics) == {"EURUSD"}
    assert metrics["EURUSD"]["sharpe"] == 0.5


def test_run_outcome_refuses_an_unknown_status_by_default(tmp_path):
    """In strict mode an unclassified status stays a legacy error."""
    run_dir = make_run(tmp_path, "run-1", symbols={"GOLD": {"status": "mystery"}})
    with pytest.raises(LegacyRunError, match="run-1.*GOLD"):
        run_outcome(run_dir)


def test_lenient_run_outcome_counts_an_unknown_status_as_evaluated(tmp_path):
    """Live runs never drop a trial over an unknown symbol status."""
    run_dir = make_run(tmp_path, "run-1", symbols={"GOLD": {"status": "mystery"}})
    status, metrics = run_outcome(run_dir, lenient=True)
    assert status is TrialStatus.OK
    assert set(metrics) == {"GOLD"}


@pytest.mark.parametrize("filename", ["config.json", "unified_metrics.json"])
def test_lenient_run_outcome_counts_an_unreadable_symbol_as_evaluated(tmp_path, filename):
    """A killed run may leave a truncated symbol file; it was looked at, so it counts."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"GOLD": {"unified": {"sharpe": 0.5}}, "DAX": {"unified": {"sharpe": 0.1}}},
    )
    (run_dir / "grid_details" / "DAX" / filename).write_text('{"status": "o', encoding="utf-8")
    with pytest.raises(LegacyRunError, match="run-1/grid_details/DAX"):
        run_outcome(run_dir)
    status, metrics = run_outcome(run_dir, lenient=True)
    assert status is TrialStatus.OK
    assert metrics["DAX"] == {}
    assert metrics["GOLD"]["sharpe"] == 0.5


def test_lenient_run_outcome_ignores_an_unreadable_elite_list(tmp_path):
    """A truncated results.json only loses the elite values, not the trial."""
    run_dir = make_run(tmp_path, "run-1", symbols={"GOLD": {"unified": {"sharpe": 0.5}}})
    (run_dir / "results.json").write_text('{"elite_results": [', encoding="utf-8")
    with pytest.raises(LegacyRunError, match="results.json"):
        run_outcome(run_dir)
    status, metrics = run_outcome(run_dir, lenient=True)
    assert status is TrialStatus.OK
    assert metrics["GOLD"]["sharpe"] == 0.5


def test_run_outcome_without_symbol_results_is_failed(tmp_path):
    """A run directory without grid details has no evaluated symbol."""
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    assert run_outcome(run_dir) == (TrialStatus.FAILED, {})


def test_lenient_run_outcome_keeps_the_return_statistics(tmp_path):
    """Live runs record the return statistics; the sample size stays an integer."""
    unified = {
        "sharpe": 1.2,
        "sr_per_trade": 0.11,
        "skew": -0.4,
        "kurtosis": 4.5,
        "n_returns": 230,
        "test_period_years": 2.5,
    }
    run_dir = make_run(tmp_path, "run-1", symbols={"GOLD": {"unified": unified}})
    status, metrics = run_outcome(run_dir, lenient=True)
    assert status is TrialStatus.OK
    assert {k: metrics["GOLD"][k] for k in unified} == unified
    assert type(metrics["GOLD"]["n_returns"]) is int


def _holdout_symbol(status, evaluated):
    return {
        "status": status,
        "result": {"holdout": {"cutoff": "2024-01-01", "evaluated": evaluated}},
    }


@pytest.mark.parametrize("status", ["no_edge", "not_significant", "no_successful_folds", "error"])
def test_a_holdout_never_looked_at_does_not_count(tmp_path, status):
    """A holdout symbol counts only when its holdout was looked at, whatever its status."""
    run_dir = make_run(tmp_path, "run-1", symbols={"GOLD": _holdout_symbol(status, False)})
    assert run_outcome(run_dir) == (TrialStatus.FAILED, {})


@pytest.mark.parametrize("status", ["ok", "not_significant", "no_unified_trades", "error"])
def test_a_holdout_looked_at_counts(tmp_path, status):
    """An error after the holdout trades existed still looked at the data."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"GOLD": _holdout_symbol(status, True), "DAX": _holdout_symbol("no_edge", False)},
    )
    status_, metrics = run_outcome(run_dir)
    assert status_ is TrialStatus.OK
    assert set(metrics) == {"GOLD"}


@pytest.mark.parametrize("holdout", [None, "x", {"cutoff": "2024-01-01"}, {"evaluated": "yes"}])
def test_an_unreadable_holdout_block_counts_only_leniently(tmp_path, holdout):
    """Like an unknown status: strict refuses, live runs count the symbol."""
    run_dir = make_run(
        tmp_path, "run-1", symbols={"GOLD": {"status": "no_edge", "result": {"holdout": holdout}}}
    )
    with pytest.raises(LegacyRunError, match="run-1.*GOLD.*holdout"):
        run_outcome(run_dir)
    status, metrics = run_outcome(run_dir, lenient=True)
    assert status is TrialStatus.OK
    assert set(metrics) == {"GOLD"}


def test_run_outcome_carries_the_selection_protocol(tmp_path):
    """Marked runs record their protocol per symbol; runs before the marker record None."""
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={
            "GOLD": {
                "unified": {"sharpe": 0.5},
                "fold": {"walk_forward": {"selection_protocol": 2}},
            },
            "DAX": {"unified": {"sharpe": 0.4}, "fold": {"walk_forward": {"fold_stability": 0.5}}},
        },
    )
    _, metrics = run_outcome(run_dir)
    assert metrics["GOLD"]["selection_protocol"] == 2
    assert metrics["DAX"]["selection_protocol"] is None


def test_a_holdout_look_carries_its_developments_selection_protocol(tmp_path):
    """A holdout look has no fold_results.json; development.json keeps the marker."""
    run_dir = make_run(tmp_path, "run-1", symbols={"GOLD": _holdout_symbol("ok", True)})
    development = {"status": "ok", "walk_forward": {"selection_protocol": 2}}
    (run_dir / "grid_details" / "GOLD" / "development.json").write_text(
        json.dumps(development), encoding="utf-8"
    )
    _, metrics = run_outcome(run_dir)
    assert metrics["GOLD"]["selection_protocol"] == 2
