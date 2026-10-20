import json
import logging
import os
import socket
import subprocess
from dataclasses import replace

import pytest

from fwbg.research import pending as pending_module
from fwbg.research.fingerprint import digest
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    LedgerError,
    Trial,
    TrialStatus,
    append_trial,
    read_trials,
)
from fwbg.research.legacy import LegacyRunError
from fwbg.research.pending import (
    STAGING,
    PendingError,
    PendingRun,
    finish,
    live_runs,
    read_pending,
    recover_stale,
    write_pending,
)
from fwbg.research.repo import init_research_repo
from tests.research.factories import make_run, make_trial

CONFIG = {"name": "orb-test", "timeframe": "MINUTE_15"}


@pytest.fixture
def register(tmp_path, git_env):
    """An initialised research register."""
    research = tmp_path / "research"
    init_research_repo(research)
    return research


def dead_pid() -> int:
    """The pid of a process that has already ended and been reaped."""
    with subprocess.Popen(["true"]) as process:
        process.wait()
    return process.pid


def make_pending(
    run_dir, *, run_id="run-1", pid=None, strategy="S-0007", ts="2026-10-19T10:00:00"
) -> PendingRun:
    """A pending run whose trial matches CONFIG's fingerprint and has no outcome yet."""
    trial = make_trial(
        strategy=strategy,
        run_id=run_id,
        ts=ts,
        status=TrialStatus.FAILED,
        fingerprint=digest(CONFIG),
        code={},
        metrics={},
    )
    return PendingRun(
        run_id=run_id,
        run_dir=run_dir,
        pid=os.getpid() if pid is None else pid,
        trial=trial,
        config=CONFIG,
    )


def ledger(register) -> list[Trial]:
    return read_trials(register / LEDGER_FILENAME)


def marker(register, run_id="run-1"):
    return register / STAGING / f"{run_id}.json"


def test_written_marker_reads_back_unchanged(register, tmp_path):
    run = make_pending(tmp_path / "results" / "run-1")
    write_pending(register, run)
    assert read_pending(register) == [run]


def test_a_run_id_is_pending_only_once(register, tmp_path):
    run = make_pending(tmp_path / "results" / "run-1")
    write_pending(register, run)
    with pytest.raises(PendingError, match="run-1"):
        write_pending(register, run)


@pytest.mark.parametrize("run_id", ["", ".", "..", "a/b"])
def test_invalid_run_ids_are_refused(register, tmp_path, run_id):
    with pytest.raises(PendingError, match="invalid run id"):
        write_pending(register, make_pending(tmp_path / "x", run_id=run_id))


def test_unreadable_marker_is_a_pending_error(register):
    (register / STAGING).mkdir()
    (register / STAGING / "run-1.json").write_text("{", encoding="utf-8")
    with pytest.raises(PendingError, match="run-1.json"):
        read_pending(register)


def test_marker_with_an_invalid_trial_is_a_pending_error(register, tmp_path):
    write_pending(register, make_pending(tmp_path / "results" / "run-1"))
    record = json.loads(marker(register).read_text(encoding="utf-8"))
    record["trial"]["strategy"] = "nonsense"
    marker(register).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(PendingError, match="strategy"):
        read_pending(register)


def rewrite_marker(register, changes):
    record = json.loads(marker(register).read_text(encoding="utf-8"))
    record.update(changes)
    marker(register).write_text(json.dumps(record), encoding="utf-8")


def test_marker_records_this_host(register, tmp_path):
    write_pending(register, make_pending(tmp_path / "results" / "run-1"))
    record = json.loads(marker(register).read_text(encoding="utf-8"))
    assert record["host"] == socket.gethostname()
    assert read_pending(register)[0].host == socket.gethostname()


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "run-2"},
        {"pid": "123"},
        {"pid": 0},
        {"pid": -1},
        {"pid": True},
        {"host": ""},
        {"host": 7},
    ],
)
def test_marker_with_invalid_fields_is_a_pending_error(register, tmp_path, changes):
    write_pending(register, make_pending(tmp_path / "results" / "run-1"))
    rewrite_marker(register, changes)
    with pytest.raises(PendingError, match="run-1.json"):
        read_pending(register)


def test_marker_from_another_host_counts_as_alive(register, tmp_path):
    write_pending(register, make_pending(tmp_path / "results" / "run-1", pid=dead_pid()))
    rewrite_marker(register, {"host": f"other-{socket.gethostname()}"})
    assert recover_stale(register) == []
    assert marker(register).exists()
    assert [run.run_id for run in live_runs(register, "S-0007")] == ["run-1"]


def test_finish_records_the_outcome_and_drops_the_marker(register, tmp_path):
    run_dir = make_run(
        tmp_path / "results", "run-1", symbols={"GOLD": {"unified": {"sharpe": 0.5}}}
    )
    run = make_pending(run_dir)
    write_pending(register, run)
    stored = finish(register, run)
    assert stored.trial_id == "T-000001"
    assert stored.status is TrialStatus.OK
    assert stored.metrics["GOLD"]["sharpe"] == 0.5
    assert stored.code == {"commit": "abc123", "branch": "main", "dirty": True}
    assert ledger(register) == [stored]
    config_file = register / "configs" / f"{digest(CONFIG).removeprefix('sha256:')}.json"
    assert json.loads(config_file.read_text(encoding="utf-8")) == CONFIG
    assert not marker(register).exists()


def test_finish_without_a_run_directory_records_a_failed_trial(register, tmp_path):
    run = make_pending(tmp_path / "results" / "never-created")
    write_pending(register, run)
    stored = finish(register, run)
    assert stored.status is TrialStatus.FAILED
    assert stored.metrics == {}
    assert stored.code == {}
    assert not stored.counted
    assert not marker(register).exists()


def test_finish_of_an_already_recorded_run_only_drops_the_marker(register, tmp_path):
    run = make_pending(tmp_path / "results" / "run-1")
    recorded = append_trial(register / LEDGER_FILENAME, run.trial)
    write_pending(register, run)
    assert finish(register, run) == recorded
    assert ledger(register) == [recorded]
    assert not marker(register).exists()


def test_finish_of_a_reused_run_id_records_a_new_trial(register, tmp_path):
    earlier = append_trial(
        register / LEDGER_FILENAME,
        make_trial(run_id="run-1", status=TrialStatus.FAILED, ts="2026-10-19T10:00:00"),
    )
    run = make_pending(tmp_path / "results" / "run-1", ts="2026-10-20T09:00:00")
    write_pending(register, run)
    stored = finish(register, run)
    assert stored.trial_id == "T-000002"
    assert ledger(register) == [earlier, stored]
    assert not marker(register).exists()


def test_finish_without_a_marker_still_records(register, tmp_path):
    stored = finish(register, make_pending(tmp_path / "results" / "run-1"))
    assert ledger(register) == [stored]


def test_recover_stale_records_dead_runs_and_keeps_live_ones(register, tmp_path):
    results = tmp_path / "results"
    write_pending(register, make_pending(results / "run-dead", run_id="run-dead", pid=dead_pid()))
    live = make_pending(results / "run-live", run_id="run-live")
    write_pending(register, live)
    recovered = recover_stale(register)
    assert [trial.run_id for trial in recovered] == ["run-dead"]
    assert ledger(register) == recovered
    assert read_pending(register) == [live]
    assert [run.run_id for run in live_runs(register, "S-0007")] == ["run-live"]
    assert [run.run_id for run in live_runs(register)] == ["run-live"]
    assert live_runs(register, "S-0008") == []


def test_one_broken_stale_marker_does_not_block_the_others(register, tmp_path, monkeypatch, caplog):
    results = tmp_path / "results"
    for run_id in ("run-a", "run-b"):
        write_pending(register, make_pending(results / run_id, run_id=run_id, pid=dead_pid()))
    real_finish = pending_module.finish

    def finish_failing_for_run_a(research_dir, run):
        if run.run_id == "run-a":
            raise LegacyRunError("run-a: broken")
        return real_finish(research_dir, run)

    monkeypatch.setattr(pending_module, "finish", finish_failing_for_run_a)
    with caplog.at_level(logging.WARNING, logger=pending_module.__name__):
        recovered = recover_stale(register)
    assert [trial.run_id for trial in recovered] == ["run-b"]
    assert ledger(register) == recovered
    assert marker(register, "run-a").exists()
    assert not marker(register, "run-b").exists()
    assert "run-a" in caplog.text and "broken" in caplog.text


def test_marker_survives_a_failing_append_for_later_recovery(register, tmp_path, monkeypatch):
    run = replace(make_pending(tmp_path / "results" / "run-1"), pid=dead_pid())
    write_pending(register, run)

    def broken_append(ledger_path, trial):
        raise LedgerError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(pending_module, "append_trial", broken_append)
        with pytest.raises(LedgerError):
            finish(register, run)
    assert marker(register).exists()
    assert ledger(register) == []
    assert [trial.run_id for trial in recover_stale(register)] == ["run-1"]
    assert not marker(register).exists()


def test_unnumbered_trial_json_requires_a_null_trial_id():
    line = make_trial().to_json()
    assert Trial.from_json(line, numbered=False).trial_id is None
    with pytest.raises(LedgerError, match="trial"):
        Trial.from_json(line)
    numbered = replace(make_trial(), trial_id="T-000001").to_json()
    with pytest.raises(LedgerError, match="trial"):
        Trial.from_json(numbered, numbered=False)


def test_recovered_run_is_tagged_with_its_trial(register, tmp_path):
    run_dir = make_run(tmp_path / "results", "run-1")
    write_pending(register, make_pending(run_dir, pid=dead_pid()))
    [trial] = recover_stale(register)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config["research"] == {
        "id": "S-0007",
        "trial": trial.trial_id,
        "fingerprint": trial.fingerprint,
        "kind": "iteration",
        "note": "baseline",
        "reason": None,
    }
    assert config["run_id"] == "run-1"


def test_failing_tag_of_a_recovered_run_only_logs(register, tmp_path, caplog):
    run_dir = make_run(tmp_path / "results", "run-1")
    (run_dir / "config.json").write_text("[]", encoding="utf-8")
    write_pending(register, make_pending(run_dir, pid=dead_pid()))
    with caplog.at_level(logging.WARNING, logger=pending_module.__name__):
        [trial] = recover_stale(register)
    assert ledger(register) == [trial]
    assert not marker(register).exists()
    assert "could not tag run run-1" in caplog.text
