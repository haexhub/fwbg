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
