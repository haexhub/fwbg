import json
import subprocess

import pytest

from fwbg.research.card import Status, replace_card
from fwbg.research.fingerprint import canonical_config, fingerprint
from fwbg.research.gate import DenialCode
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, TrialStatus, read_trials
from fwbg.research.pending import PendingRun, read_pending, write_pending
from fwbg.research.repo import REGISTER_PATHS, ResearchRepoError, commit_register
from fwbg.research.runs import GateDenied
from fwbg.research.study import register_trial
from tests.research.factories import dead_pid, make_card, make_pending, make_trial

CONFIG = {
    "name": "x",
    "timeframe": "HOUR_1",
    "pipeline": {"indicators": []},
    "model": {},
    "start_date": "2024-01-01",
    "end_date": "2025-04-01",
}
METRICS = {"EURUSD": {"sharpe": 0.4, "trades": 120}}


def git(register, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=register, capture_output=True, text=True, check=True
    ).stdout


def denial_codes(denied) -> list[DenialCode]:
    return [denial.code for denial in denied.value.denials]


def record(register, config=CONFIG, note="baseline", **kwargs):
    return register_trial(
        "S-0007", config, ["eurusd"], METRICS, note, research_dir=register, **kwargs
    )


def test_records_an_iteration_trial_with_its_config_and_commits(gate_register):
    trial = record(gate_register, run_id="study-1")
    assert trial.trial_id == "T-000001" and trial.kind is TrialKind.ITERATION
    assert trial.status is TrialStatus.OK and trial.counted
    assert trial.metrics == METRICS and trial.run_id == "study-1"
    assert trial.window == {"start": "2024-01-01", "end": "2025-04-01"}
    assert read_trials(gate_register / LEDGER_FILENAME) == [trial]
    stored = gate_register / "configs" / f"{trial.fingerprint.removeprefix('sha256:')}.json"
    assert json.loads(stored.read_text(encoding="utf-8"))["symbols"] == ["EURUSD"]
    assert git(gate_register, "status", "--porcelain") == ""
    assert "trial T-000001: S-0007 study" in git(gate_register, "log", "-1", "--pretty=%s")


def test_missing_end_date_is_refused_and_writes_nothing(gate_register):
    config = {key: value for key, value in CONFIG.items() if key != "end_date"}
    with pytest.raises(GateDenied) as denied:
        record(gate_register, config)
    assert denial_codes(denied) == [DenialCode.INVALID_WINDOW]
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_end_after_the_cutoff_is_refused(gate_register):
    with pytest.raises(GateDenied) as denied:
        record(gate_register, {**CONFIG, "end_date": "2025-06-01"})
    assert denial_codes(denied) == [DenialCode.AFTER_CUTOFF]
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_parked_card_is_refused(gate_register):
    replace_card(gate_register, make_card(status=Status.PARKED))
    commit_register(gate_register, REGISTER_PATHS, "park")
    with pytest.raises(GateDenied) as denied:
        record(gate_register)
    assert denial_codes(denied) == [DenialCode.NOT_ACTIVE]


def test_missing_note_is_refused(gate_register):
    with pytest.raises(GateDenied) as denied:
        record(gate_register, note=" ")
    assert denial_codes(denied) == [DenialCode.MISSING_NOTE]


def test_same_config_twice_needs_a_reason(gate_register):
    first = record(gate_register)
    with pytest.raises(GateDenied) as denied:
        record(gate_register, note="again")
    assert denial_codes(denied) == [DenialCode.DUPLICATE_FINGERPRINT]
    again = record(gate_register, note="again", reason="new data vendor")
    assert again.fingerprint == first.fingerprint and again.reason == "new data vendor"
    assert len(read_trials(gate_register / LEDGER_FILENAME)) == 2


def test_identical_running_config_needs_a_reason(gate_register, tmp_path):
    running = make_trial(fingerprint=fingerprint(CONFIG, ("EURUSD",)))
    write_pending(
        gate_register,
        PendingRun(run_id="run-1", run_dir=tmp_path / "run-1", pid=1, trial=running, config={}),
    )
    with pytest.raises(GateDenied) as denied:
        record(gate_register)
    assert denial_codes(denied) == [DenialCode.DUPLICATE_FINGERPRINT]
    assert record(gate_register, reason="rerun while the CLI run is going").counted


def test_failed_variant_is_recorded_but_not_counted(gate_register):
    trial = record(gate_register, failed=True)
    assert trial.status is TrialStatus.FAILED and not trial.counted


def test_unreadable_register_is_refused(tmp_path):
    with pytest.raises(GateDenied) as denied:
        register_trial(
            "S-0007", CONFIG, ["EURUSD"], METRICS, "x", research_dir=tmp_path / "missing"
        )
    assert denial_codes(denied) == [DenialCode.REGISTER_UNREADABLE]


def test_defaults_to_the_workspace_register(gate_register, monkeypatch):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(gate_register))
    trial = register_trial("S-0007", CONFIG, ["EURUSD"], METRICS, "baseline")
    assert read_trials(gate_register / LEDGER_FILENAME) == [trial]


@pytest.mark.parametrize("end_date", [None, 20250401, ""])
def test_end_date_must_be_a_string(gate_register, end_date):
    with pytest.raises(GateDenied) as denied:
        record(gate_register, {**CONFIG, "end_date": end_date})
    assert denial_codes(denied) == [DenialCode.INVALID_WINDOW]
    assert "YYYY-MM-DD string" in str(denied.value)


def test_malformed_end_date_is_refused(gate_register):
    with pytest.raises(GateDenied) as denied:
        record(gate_register, {**CONFIG, "end_date": "2025/04/01"})
    assert denial_codes(denied) == [DenialCode.INVALID_WINDOW]
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_start_date_is_optional(gate_register):
    config = {key: value for key, value in CONFIG.items() if key != "start_date"}
    assert record(gate_register, config).window == {"start": None, "end": "2025-04-01"}


def test_timeframe_spelling_does_not_change_the_fingerprint(gate_register):
    first = record(gate_register, {**CONFIG, "timeframe": "1h"})
    assert first.fingerprint == fingerprint(CONFIG, ("EURUSD",))


def test_unknown_timeframe_is_refused(gate_register):
    with pytest.raises(GateDenied) as denied:
        record(gate_register, {**CONFIG, "timeframe": "fortnightly"})
    assert denial_codes(denied) == [DenialCode.INVALID_CONFIG]


def test_symbols_as_one_string_are_refused(gate_register):
    with pytest.raises(GateDenied) as denied:
        register_trial("S-0007", CONFIG, "EURUSD", METRICS, "x", research_dir=gate_register)
    assert denial_codes(denied) == [DenialCode.INVALID_CONFIG]


def test_nan_metric_is_refused_and_writes_nothing(gate_register):
    with pytest.raises(GateDenied) as denied:
        register_trial(
            "S-0007",
            CONFIG,
            ["EURUSD"],
            {"EURUSD": {"sharpe": float("nan")}},
            "x",
            research_dir=gate_register,
        )
    assert denial_codes(denied) == [DenialCode.INVALID_CONFIG]
    assert read_trials(gate_register / LEDGER_FILENAME) == []
    assert not any((gate_register / "configs").iterdir())


def test_metric_symbols_are_upper_cased(gate_register):
    trial = register_trial(
        "S-0007",
        CONFIG,
        ["eurusd"],
        {"eurusd": {"sharpe": 0.4}},
        "x",
        research_dir=gate_register,
    )
    assert trial.metrics == {"EURUSD": {"sharpe": 0.4}}


def test_metric_symbols_equal_after_upper_casing_are_refused(gate_register):
    with pytest.raises(GateDenied) as denied:
        register_trial(
            "S-0007",
            CONFIG,
            ["EURUSD"],
            {"eurusd": {"sharpe": 0.4}, "EURUSD": {"sharpe": 0.9}},
            "x",
            research_dir=gate_register,
        )
    assert denial_codes(denied) == [DenialCode.INVALID_CONFIG]
    assert "EURUSD" in str(denied.value)
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_stale_markers_are_recorded_first(gate_register, tmp_path):
    stale = make_trial(fingerprint=fingerprint(CONFIG, ("EURUSD",)))
    write_pending(
        gate_register,
        PendingRun(
            run_id="run-1",
            run_dir=tmp_path / "run-1",
            pid=dead_pid(),
            trial=stale,
            config=canonical_config(CONFIG, ("EURUSD",)),
        ),
    )
    trial = record(gate_register, {**CONFIG, "model": {"x": 1}})
    ledger = read_trials(gate_register / LEDGER_FILENAME)
    assert [t.run_id for t in ledger] == ["run-1", None] and ledger[1] == trial
    assert read_pending(gate_register) == []


def test_failed_commit_only_warns(gate_register, monkeypatch, capsys):
    def broken(*_):
        raise ResearchRepoError("git commit failed: index.lock exists")

    monkeypatch.setattr("fwbg.research.study.commit_register", broken)
    trial = record(gate_register)
    assert read_trials(gate_register / LEDGER_FILENAME) == [trial]
    assert "not committed yet" in capsys.readouterr().err


@pytest.mark.parametrize("run_id", ["a/b", "..", "", "study:S-0007:config-4"])
def test_invalid_or_reserved_run_id_is_refused(gate_register, run_id):
    with pytest.raises(GateDenied) as denied:
        record(gate_register, run_id=run_id)
    assert denial_codes(denied) == [DenialCode.INVALID_RUN_ID]
    assert read_trials(gate_register / LEDGER_FILENAME) == []


def test_run_id_already_in_the_ledger_is_refused(gate_register):
    record(gate_register, run_id="study-1")
    with pytest.raises(GateDenied) as denied:
        record(gate_register, {**CONFIG, "model": {"x": 1}}, run_id="study-1")
    assert denial_codes(denied) == [DenialCode.RUN_ID_IN_USE]
    assert len(read_trials(gate_register / LEDGER_FILENAME)) == 1


def test_pending_run_id_is_refused(gate_register, tmp_path):
    write_pending(gate_register, make_pending(tmp_path / "run-1"))
    with pytest.raises(GateDenied) as denied:
        record(gate_register, run_id="run-1")
    assert denial_codes(denied) == [DenialCode.RUN_ID_IN_USE]
    assert read_trials(gate_register / LEDGER_FILENAME) == []
