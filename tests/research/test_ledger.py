import multiprocessing

import pytest

from fwbg.research.ledger import LedgerError, TrialKind, TrialStatus, append_trial, read_trials
from tests.research.factories import append_many, make_trial


def test_missing_ledger_has_no_trials(tmp_path):
    assert read_trials(tmp_path / "trials.jsonl") == []


def test_append_assigns_sequential_ids_and_round_trips(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    first = append_trial(ledger, make_trial(metrics={"GOLD": {"sharpe": 0.14, "trades": 37}}))
    second = append_trial(ledger, make_trial(run_id="run-2"))
    assert (first.trial_id, second.trial_id) == ("T-000001", "T-000002")
    assert read_trials(ledger) == [first, second]


def test_append_refuses_trials_that_already_have_an_id(tmp_path):
    with pytest.raises(ValueError, match="assigns trial ids"):
        append_trial(tmp_path / "trials.jsonl", make_trial(trial_id="T-000009"))


def test_malformed_line_is_reported_with_its_number(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    append_trial(ledger, make_trial())
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(LedgerError, match="line 2"):
        read_trials(ledger)


def test_malformed_holdout_cutoff_is_a_ledger_error(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    line = make_trial(trial_id="T-000001").to_json().replace("2025-04-30", "someday", 1)
    ledger.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="line 1"):
        read_trials(ledger)


def test_append_repairs_missing_trailing_newline(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    append_trial(ledger, make_trial())
    ledger.write_text(ledger.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    append_trial(ledger, make_trial(run_id="run-2"))
    assert [t.trial_id for t in read_trials(ledger)] == ["T-000001", "T-000002"]


@pytest.mark.parametrize(
    ("kind", "status", "counted"),
    [
        (TrialKind.ITERATION, TrialStatus.OK, True),
        (TrialKind.HOLDOUT, TrialStatus.OK, True),
        (TrialKind.LEGACY, TrialStatus.OK, True),
        (TrialKind.ITERATION, TrialStatus.FAILED, False),
        (TrialKind.SMOKE, TrialStatus.OK, False),
    ],
)
def test_counted_trials(kind, status, counted):
    assert make_trial(kind=kind, status=status).counted is counted


def test_concurrent_appends_lose_no_lines(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=append_many, args=(str(ledger), worker, 25)) for worker in range(4)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join(timeout=60)
    assert [process.exitcode for process in workers] == [0, 0, 0, 0]
    trials = read_trials(ledger)
    assert len(trials) == 100
    assert len({t.trial_id for t in trials}) == 100
    assert len({t.run_id for t in trials}) == 100
