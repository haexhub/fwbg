import json
import subprocess

import pytest

from fwbg.research.card import CardError
from fwbg.research.cli import main
from fwbg.research.ledger import LEDGER_FILENAME, LedgerError, TrialKind, append_trial, read_trials
from fwbg.research.legacy import LegacyRunError
from fwbg.research.pending import STAGING, write_pending
from fwbg.research.reconcile import ReconcileResult, reconcile, repair, unindexed_runs
from fwbg.research.repo import ResearchRepoError
from tests.research.factories import dead_pid, make_pending, make_run, make_trial

NOTHING = ReconcileResult(recovered=(), committed=False, unindexed=())


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Point every workspace directory into *tmp_path*."""
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("FWBG_STRATEGIES_DIR", str(tmp_path / "strategies" / "configs"))
    (tmp_path / "results").mkdir()


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results"


def commits(register) -> int:
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=register,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(count.stdout)


def dirty(register) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=register, capture_output=True, text=True, check=True
    ).stdout


def test_clean_register_changes_nothing_twice(gate_register, results):
    """A clean register: nothing recovered, nothing committed, the same result again."""
    before = commits(gate_register)
    assert reconcile(gate_register, results) == NOTHING
    assert reconcile(gate_register, results) == NOTHING
    assert commits(gate_register) == before


def test_uncommitted_trial_is_committed_once(gate_register, results):
    """A ledger append left uncommitted is committed, and only once."""
    append_trial(gate_register / LEDGER_FILENAME, make_trial(run_id="r-1"))
    before = commits(gate_register)

    assert reconcile(gate_register, results) == ReconcileResult((), True, ())
    assert commits(gate_register) == before + 1
    assert dirty(gate_register) == ""
    assert reconcile(gate_register, results) == NOTHING
    assert commits(gate_register) == before + 1


def test_uncommitted_reports_are_committed(gate_register, results):
    """Reports are part of the register commit."""
    (gate_register / "reports").mkdir(exist_ok=True)
    (gate_register / "reports" / "S-0007.md").write_text("report\n", encoding="utf-8")

    assert reconcile(gate_register, results).committed
    assert dirty(gate_register) == ""


def test_stale_marker_is_recovered_and_committed(gate_register, results):
    """A pending run whose process is gone is recorded and committed."""
    run_dir = make_run(results, "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))

    result = reconcile(gate_register, results)

    trials = read_trials(gate_register / LEDGER_FILENAME)
    assert result == ReconcileResult((str(trials[0].trial_id),), True, ())
    assert not (gate_register / STAGING / "run-1.json").exists()
    assert dirty(gate_register) == ""
    assert reconcile(gate_register, results) == NOTHING


def test_live_marker_is_left_alone(gate_register, results):
    """A running gated run is neither recorded nor reported."""
    run_dir = make_run(results, "run-1")
    write_pending(gate_register, make_pending(run_dir))

    assert reconcile(gate_register, results) == NOTHING
    assert (gate_register / STAGING / "run-1.json").exists()


def truncate_result(run_dir, symbol="GOLD"):
    """Cut a symbol's result file short, as a run still writing it leaves it."""
    (run_dir / "grid_details" / symbol / "config.json").write_text('{"sta', encoding="utf-8")


def test_running_gated_run_with_partial_results_is_skipped_unread(gate_register, results):
    """A run with a live marker is not loaded, so its half-written files do not fail the scan."""
    run_dir = make_run(results, "run-1")
    truncate_result(run_dir)
    write_pending(gate_register, make_pending(run_dir))

    assert reconcile(gate_register, results) == NOTHING


def test_run_in_progress_with_partial_results_is_skipped_unread(gate_register, results):
    """A run whose progress.json says it is running is not loaded either."""
    run_dir = make_run(results, "r-running")
    truncate_result(run_dir)
    (run_dir / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")

    assert unindexed_runs(gate_register, results) == ()


def test_repair_and_report_are_separate_steps(gate_register, results):
    """The repair commits on its own; a broken run fails only the report."""
    run_dir = make_run(results, "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))
    truncate_result(make_run(results, "r-broken"))

    recovered, committed = repair(gate_register)
    assert [trial.run_id for trial in recovered] == ["run-1"]
    assert committed
    assert dirty(gate_register) == ""
    with pytest.raises(LegacyRunError):
        unindexed_runs(gate_register, results)


def test_runs_missing_from_the_ledger_are_reported(gate_register, results):
    """Untagged runs not in the ledger are reported; recorded and running ones are not."""
    make_run(results, "r-missing")
    make_run(results, "r-iteration")
    make_run(results, "r-legacy")
    running = make_run(results, "r-running")
    (running / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    append_trial(gate_register / LEDGER_FILENAME, make_trial(run_id="r-iteration"))
    append_trial(
        gate_register / LEDGER_FILENAME, make_trial(run_id="r-legacy", kind=TrialKind.LEGACY)
    )
    reconcile(gate_register, results)

    assert reconcile(gate_register, results) == ReconcileResult((), False, ("r-missing",))


def test_broken_ledger_raises_and_commits_nothing(gate_register, results):
    """An unparsable ledger line stops reconcile before anything is recovered or committed."""
    run_dir = make_run(results, "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))
    with open(gate_register / LEDGER_FILENAME, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    before = commits(gate_register)

    with pytest.raises(LedgerError):
        reconcile(gate_register, results)
    assert commits(gate_register) == before
    assert (gate_register / STAGING / "run-1.json").exists()


def test_broken_card_raises_and_commits_nothing(gate_register, results):
    """A card that does not parse stops reconcile before anything is committed."""
    append_trial(gate_register / LEDGER_FILENAME, make_trial(run_id="r-1"))
    (gate_register / "strategies" / "S-0008-broken.md").write_text("no front matter\n")
    before = commits(gate_register)

    with pytest.raises(CardError):
        reconcile(gate_register, results)
    assert commits(gate_register) == before


def test_failed_commit_raises_and_keeps_the_data(gate_register, results, monkeypatch):
    """A commit that fails raises; the recovered trial stays in the ledger."""
    run_dir = make_run(results, "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))

    def fail(*args, **kwargs):
        raise ResearchRepoError("git failed")

    monkeypatch.setattr("fwbg.research.reconcile.commit_register", fail)

    with pytest.raises(ResearchRepoError):
        reconcile(gate_register, results)
    assert [trial.run_id for trial in read_trials(gate_register / LEDGER_FILENAME)] == ["run-1"]


def test_cli_reports_recovered_committed_and_unindexed(gate_register, results, capsys):
    """The CLI prints each part of the result and the backfill hint."""
    run_dir = make_run(results, "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))
    make_run(results, "r-missing")

    make_run(results, "r-other")

    assert main(["reconcile"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "recovered T-000001",
        "committed",
        "2 runs not in the ledger:",
        "  r-missing",
        "  r-other",
        "map them with: python -m fwbg.cli research backfill scan --out mapping.yaml",
    ]

    assert main(["reconcile", "--results", str(results)]) == 0
    out = capsys.readouterr().out
    assert "nothing to commit" in out
    assert "recovered" not in out


def test_cli_clean_register_reports_nothing_unindexed(gate_register, capsys):
    """Without gaps, the CLI says so and gives no hint."""
    assert main(["reconcile"]) == 0
    out = capsys.readouterr().out
    assert "nothing to commit" in out
    assert "backfill" not in out


def test_cli_reports_errors_without_traceback(gate_register, capsys):
    """A broken ledger exits 1 with a one-line error."""
    with open(gate_register / LEDGER_FILENAME, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    assert main(["reconcile"]) == 1
    assert capsys.readouterr().err.startswith("Error: ")


def test_cli_reports_the_repair_before_a_failing_scan(gate_register, results, capsys):
    """A run that cannot be read exits 1, after the repair has been reported."""
    run_dir = make_run(results, "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))
    truncate_result(make_run(results, "r-broken"))

    assert main(["reconcile"]) == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["recovered T-000001", "committed"]
    assert captured.err.startswith("Error: r-broken")
    assert dirty(gate_register) == ""


def test_cli_warns_about_runs_it_did_not_check(gate_register, results, capsys):
    """Runs that may still be running are named on stderr, one per line."""
    for run_id in ("r-a", "r-b"):
        run_dir = make_run(results, run_id)
        (run_dir / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")

    assert main(["reconcile"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["nothing to commit"]
    assert captured.err.splitlines() == [
        "Warning: not checked, may still be running: r-a",
        "Warning: not checked, may still be running: r-b",
    ]


def test_cli_without_a_register_points_to_init(tmp_path, capsys):
    """No register: exit 1 with the init hint, and no lock file is left behind."""
    assert main(["reconcile"]) == 1
    assert "research init" in capsys.readouterr().err
    assert not (tmp_path / "research").exists()
