import json
import subprocess

import pytest

from fwbg.research.gate import DenialCode
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    TrialKind,
    TrialStatus,
    append_trial,
    read_trials,
)
from fwbg.research.legacy import LegacyRunError
from fwbg.research.pending import STAGING, read_pending
from fwbg.research.repo import ResearchRepoError
from fwbg.research.runs import (
    GATE_ENV,
    GateChecked,
    GateDenied,
    ResearchRun,
    gate_enabled,
    prepare_metadata,
    run_gated,
)
from tests.research.factories import dead_pid, make_run, make_trial

METADATA = {"name": "x", "timeframe": "HOUR_1", "pipeline": {"indicators": []}, "model": {}}


@pytest.fixture
def results(tmp_path):
    return tmp_path / "results"


def git_status(register) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=register, capture_output=True, text=True, check=True
    ).stdout


def rewrite_marker_pid(register, run_id, pid) -> None:
    path = register / STAGING / f"{run_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["pid"] = pid
    path.write_text(json.dumps(record), encoding="utf-8")


def denial_codes(denied) -> list[DenialCode]:
    return [denial.code for denial in denied.value.denials]


def test_before_run_returns_the_clamped_metadata_and_marks_the_run(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "baseline")
    assert run.before_run(METADATA, ["eurusd"])["end_date"] == "2025-04-30"
    assert [p.run_id for p in read_pending(gate_register)] == ["run-1"]
    assert "end_date" not in METADATA


def test_holdout_run_is_on_record_as_a_holdout(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "look", holdout=True)
    metadata = run.before_run(METADATA, ["eurusd"])
    assert metadata["holdout_cutoff"] == "2025-04-30"
    assert "end_date" not in metadata
    [pending] = read_pending(gate_register)
    assert pending.trial.kind is TrialKind.HOLDOUT
    assert pending.config["strategy"]["holdout_cutoff"] == "2025-04-30"


def holdout_look(register, results, run_id, metadata=METADATA, **kwargs):
    run = ResearchRun(register, run_id, results / run_id, "S-0007", "look", holdout=True, **kwargs)
    return run.before_run(metadata, ["EURUSD"])


def test_pending_holdout_look_counts_as_a_look(gate_register, results):
    holdout_look(gate_register, results, "run-1")
    other = {**METADATA, "model": {"depth": 3}}
    with pytest.raises(GateDenied) as denied:
        holdout_look(gate_register, results, "run-2", other)
    assert denial_codes(denied) == [DenialCode.HOLDOUT_REPEAT]
    assert "run-1" in denied.value.denials[0].message
    holdout_look(gate_register, results, "run-2", other, reason="second look on purpose")
    assert [p.run_id for p in read_pending(gate_register)] == ["run-1", "run-2"]


def test_pending_iteration_does_not_count_as_a_look(gate_register, results):
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a").before_run(
        METADATA, ["EURUSD"]
    )
    holdout_look(gate_register, results, "run-2")


def test_refusal_writes_nothing(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", None)
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.MISSING_NOTE]
    assert read_pending(gate_register) == [] and not results.exists()


def test_unreadable_register_fails_closed(tmp_path):
    run = ResearchRun(tmp_path / "missing", "r", tmp_path / "r", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denied.value.denials[0].code is DenialCode.REGISTER_UNREADABLE


def test_unreadable_ledger_fails_closed(gate_register, results):
    (gate_register / LEDGER_FILENAME).write_text("not json\n", encoding="utf-8")
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.REGISTER_UNREADABLE]
    assert read_pending(gate_register) == []


@pytest.mark.parametrize("research_id", ["S-7", "s-0007", "../S-0007"])
def test_malformed_research_id_is_card_not_found(gate_register, results, research_id):
    run = ResearchRun(gate_register, "run-1", results / "run-1", research_id, "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denied.value.denials[0].code is DenialCode.CARD_NOT_FOUND


def test_check_only_stops_without_a_marker(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x", check_only=True)
    with pytest.raises(GateChecked) as checked:
        run.before_run(METADATA, ["eurusd", "GOLD"])
    assert checked.value.symbols == ("EURUSD", "GOLD")
    assert checked.value.decision.effective_metadata["end_date"] == "2025-04-30"
    assert read_pending(gate_register) == [] and git_status(gate_register) == ""


def test_run_gated_exits_0_after_a_check(gate_register, results, capsys):
    research = ResearchRun(
        gate_register, "run-1", results / "run-1", "S-0007", "x", check_only=True
    )
    with pytest.raises(SystemExit) as exited:
        run_gated(research, lambda before_run: before_run(METADATA, ["EURUSD"]))
    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert "allowed (iteration)" in out and "open .. 2025-04-30" in out


def test_run_gated_exits_2_on_a_refusal_and_records_nothing(gate_register, results, capsys):
    research = ResearchRun(gate_register, "run-1", results / "run-1", None, "x")
    with pytest.raises(SystemExit) as exited:
        run_gated(research, lambda before_run: before_run(METADATA, ["EURUSD"]))
    assert exited.value.code == 2
    assert "[missing_research_id]" in capsys.readouterr().err
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_used_run_id_is_refused(gate_register, results):
    make_run(results, "run-1")
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denied.value.denials[0].code is DenialCode.RUN_ID_IN_USE


@pytest.mark.parametrize("run_id", ["a/b", "..", ".", ""])
def test_invalid_run_id_is_refused(gate_register, results, run_id):
    run = ResearchRun(gate_register, run_id, results / "run-1", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.INVALID_RUN_ID]
    assert "invalid run id" in denied.value.denials[0].message
    assert read_pending(gate_register) == []


def test_run_id_in_the_ledger_is_refused(gate_register, results):
    append_trial(gate_register / LEDGER_FILENAME, make_trial(run_id="run-1"))
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.RUN_ID_IN_USE]


def test_run_id_of_a_running_run_is_refused(gate_register, results):
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a").before_run(
        METADATA, ["EURUSD"]
    )
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "b")
    with pytest.raises(GateDenied) as denied:
        run.before_run({**METADATA, "model": {"x": 1}}, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.RUN_ID_IN_USE]


def test_run_id_of_an_unrecordable_stale_marker_is_refused(gate_register, results, monkeypatch):
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a").before_run(
        METADATA, ["EURUSD"]
    )
    rewrite_marker_pid(gate_register, "run-1", dead_pid())

    def unrecordable(*_):
        raise LegacyRunError("run-1: broken")

    monkeypatch.setattr("fwbg.research.pending.finish", unrecordable)
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "b")
    with pytest.raises(GateDenied) as denied:
        run.before_run({**METADATA, "model": {"x": 1}}, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.RUN_ID_IN_USE]


def test_bare_run_directory_is_allowed(gate_register, results):
    (results / "run-1").mkdir(parents=True)  # the API pre-creates it for logs
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    run.before_run(METADATA, ["EURUSD"])
    assert [p.run_id for p in read_pending(gate_register)] == ["run-1"]


def test_record_appends_tags_and_commits(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "baseline")
    effective = run.before_run(METADATA, ["EURUSD"])
    make_run(results, "run-1", symbols={"EURUSD": {"unified": {"sharpe": 0.4, "trades": 120}}})
    trial = run.record()
    assert trial is not None
    assert trial.trial_id == "T-000001" and trial.status is TrialStatus.OK
    assert trial.metrics["EURUSD"]["sharpe"] == 0.4 and trial.code["commit"] == "abc123"
    assert trial.window == {"start": None, "end": "2025-04-30"}
    assert json.loads((results / "run-1" / "config.json").read_text())["research"] == {
        "id": "S-0007",
        "trial": "T-000001",
        "fingerprint": trial.fingerprint,
        "kind": "iteration",
        "note": "baseline",
        "reason": None,
    }
    assert read_pending(gate_register) == [] and git_status(gate_register) == ""
    again = ResearchRun(gate_register, "run-2", results / "run-2", "S-0007", "again")
    with pytest.raises(GateDenied) as denied:
        again.before_run(effective, ["EURUSD"])
    assert denied.value.denials[0].code is DenialCode.DUPLICATE_FINGERPRINT


def test_record_without_a_run_directory_is_a_failed_trial(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "baseline")
    run.before_run(METADATA, ["EURUSD"])
    trial = run.record()
    assert trial is not None and trial.status is TrialStatus.FAILED
    assert git_status(gate_register) == ""


def test_record_before_the_gate_records_nothing(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "baseline")
    assert run.record() is None
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_identical_run_in_progress_is_a_duplicate(gate_register, results):
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a").before_run(
        METADATA, ["EURUSD"]
    )
    with pytest.raises(GateDenied) as denied:
        ResearchRun(gate_register, "run-2", results / "run-2", "S-0007", "b").before_run(
            METADATA, ["EURUSD"]
        )
    assert denied.value.denials[0].code is DenialCode.DUPLICATE_FINGERPRINT
    assert "running now" in denied.value.denials[0].message


def test_identical_run_in_progress_runs_again_with_a_reason(gate_register, results):
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a").before_run(
        METADATA, ["EURUSD"]
    )
    ResearchRun(
        gate_register, "run-2", results / "run-2", "S-0007", "b", reason="seed check"
    ).before_run(METADATA, ["EURUSD"])
    assert [p.run_id for p in read_pending(gate_register)] == ["run-1", "run-2"]


def test_killed_run_is_recorded_by_the_next_gated_run(gate_register, results):
    first = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a")
    first.before_run(METADATA, ["EURUSD"])
    make_run(results, "run-1", symbols={"EURUSD": {"unified": {"sharpe": 0.4}}})
    rewrite_marker_pid(gate_register, "run-1", dead_pid())  # simulates os._exit
    ResearchRun(gate_register, "run-2", results / "run-2", "S-0007", "b").before_run(
        {**METADATA, "model": {"x": 1}}, ["EURUSD"]
    )
    trials = read_trials(gate_register / LEDGER_FILENAME)
    assert [(t.run_id, t.status) for t in trials] == [("run-1", TrialStatus.OK)]
    assert [p.run_id for p in read_pending(gate_register)] == ["run-2"]
    assert git_status(gate_register) == ""


def test_crash_after_evaluation_keeps_the_outcome(gate_register, results):
    research = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a")

    def crashing_run(before_run):
        before_run(METADATA, ["EURUSD"])
        make_run(results, "run-1", symbols={"EURUSD": {"unified": {"sharpe": 0.4}}})
        raise RuntimeError("report generation failed")

    with pytest.raises(RuntimeError, match="report generation failed"):
        run_gated(research, crashing_run)
    trials = read_trials(gate_register / LEDGER_FILENAME)
    assert [(t.run_id, t.status) for t in trials] == [("run-1", TrialStatus.OK)]
    assert read_pending(gate_register) == []


def test_record_failure_only_warns_and_keeps_the_marker(
    gate_register, results, monkeypatch, capsys
):
    run = ResearchRun(gate_register, "run-4", results / "run-4", "S-0007", "baseline")
    run.before_run(METADATA, ["EURUSD"])

    def disk_full(*_):
        raise OSError("disk full")

    monkeypatch.setattr("fwbg.research.pending.append_trial", disk_full)
    assert run.record() is None
    assert "run-4 is not in the ledger yet" in capsys.readouterr().err
    assert [p.run_id for p in read_pending(gate_register)] == ["run-4"]


def test_prepare_metadata_resolves_and_canonicalises(tmp_path):
    pipelines = tmp_path / "strategies" / "pipelines"
    pipelines.mkdir(parents=True)
    preset = {"indicators": [{"name": "cusum_events"}]}
    (pipelines / "cusum.json").write_text(json.dumps(preset), encoding="utf-8")
    metadata = {
        "timeframe": "DAY",
        "pipeline": "cusum",
        "_strategy_dir": str(tmp_path / "strategies" / "configs"),
        "_cli_data_path": "data/raw",
    }
    prepared = prepare_metadata(metadata, "HOUR_1")
    assert prepared["pipeline"] == preset and prepared["timeframe"] == "DAY_1"
    assert prepared["_cli_data_path"].startswith("/")
    assert prepared["_cli_data_path"].endswith("data/raw")
    assert metadata["pipeline"] == "cusum"
    assert prepare_metadata({"name": "x"}, "1h")["timeframe"] == "HOUR_1"


def test_prepare_metadata_refuses_an_unknown_timeframe():
    with pytest.raises(ValueError, match="unknown timeframe"):
        prepare_metadata({"timeframe": "FORTNIGHT"}, "HOUR_1")


def test_gate_is_on_unless_switched_off(monkeypatch):
    monkeypatch.delenv(GATE_ENV, raising=False)
    assert gate_enabled()
    monkeypatch.setenv(GATE_ENV, "on")
    assert gate_enabled()
    monkeypatch.setenv(GATE_ENV, " OFF ")
    assert not gate_enabled()


def test_gate_off_is_ignored_outside_the_test_runner(monkeypatch):
    monkeypatch.setenv(GATE_ENV, "off")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert gate_enabled()


def test_failed_marker_write_refuses_and_records_nothing(gate_register, results, monkeypatch):
    def broken(*_):
        raise OSError("read-only file system")

    monkeypatch.setattr("fwbg.research.runs.write_pending", broken)
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.REGISTER_UNREADABLE]
    assert run.record() is None
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_mutating_the_returned_metadata_does_not_break_recording(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    effective = run.before_run(METADATA, ["EURUSD"])
    effective["pipeline"]["indicators"].append({"name": "added_by_the_run"})
    trial = run.record()
    assert trial is not None and trial.trial_id == "T-000001"
    assert git_status(gate_register) == ""


def test_failed_recovery_commit_only_warns(gate_register, results, monkeypatch, capsys):
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a").before_run(
        METADATA, ["EURUSD"]
    )
    rewrite_marker_pid(gate_register, "run-1", dead_pid())

    def broken(*_):
        raise ResearchRepoError("git commit failed: index.lock exists")

    monkeypatch.setattr("fwbg.research.runs.commit_register", broken)
    ResearchRun(gate_register, "run-2", results / "run-2", "S-0007", "b").before_run(
        {**METADATA, "model": {"x": 1}}, ["EURUSD"]
    )
    assert "not committed yet" in capsys.readouterr().err
    assert [t.run_id for t in read_trials(gate_register / LEDGER_FILENAME)] == ["run-1"]
    assert [p.run_id for p in read_pending(gate_register)] == ["run-2"]


def test_unreadable_marker_tells_the_operator_where_to_look(gate_register, results):
    (gate_register / STAGING).mkdir(exist_ok=True)
    (gate_register / STAGING / "run-0.json").write_text("{", encoding="utf-8")
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(METADATA, ["EURUSD"])
    assert denial_codes(denied) == [DenialCode.REGISTER_UNREADABLE]
    assert str(gate_register / STAGING) in denied.value.denials[0].message


def test_run_that_bypasses_the_gate_is_reported(gate_register, results, capsys):
    research = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    assert run_gated(research, lambda before_run: "done") == "done"
    assert "never passed the research gate" in capsys.readouterr().err
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_record_twice_records_once(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    run.before_run(METADATA, ["EURUSD"])
    make_run(results, "run-1", symbols={"EURUSD": {"unified": {"sharpe": 0.4}}})
    first, second = run.record(), run.record()
    assert first is not None and second is not None
    assert first.trial_id == second.trial_id == "T-000001"
    assert len(read_trials(gate_register / LEDGER_FILENAME)) == 1
    assert git_status(gate_register) == ""
