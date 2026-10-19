import json
import multiprocessing
from datetime import datetime

import pytest

from fwbg.research.ledger import LedgerError, TrialKind, TrialStatus, append_trial, read_trials
from tests.research.factories import append_many, make_trial


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.touch()
    return path


def _line(**changes):
    record = json.loads(make_trial(trial_id="T-000001").to_json())
    record.update(changes)
    return json.dumps(record)


def test_missing_ledger_cannot_be_read(tmp_path):
    with pytest.raises(LedgerError, match="does not exist"):
        read_trials(tmp_path / "trials.jsonl")


def test_missing_ledger_cannot_be_appended_to(tmp_path):
    path = tmp_path / "trials.jsonl"
    with pytest.raises(LedgerError, match="does not exist"):
        append_trial(path, make_trial())
    assert not path.exists()


def test_append_assigns_sequential_ids_and_round_trips(ledger):
    first = append_trial(ledger, make_trial(metrics={"GOLD": {"sharpe": 0.14, "trades": 37}}))
    second = append_trial(ledger, make_trial(run_id="run-2"))
    assert (first.trial_id, second.trial_id) == ("T-000001", "T-000002")
    assert read_trials(ledger) == [first, second]


def test_failed_trial_with_reason_round_trips(ledger):
    stored = append_trial(
        ledger, make_trial(status=TrialStatus.FAILED, reason="backtest crashed", metrics={})
    )
    assert read_trials(ledger) == [stored]


def test_next_id_follows_the_highest_existing_id(ledger):
    ledger.write_text(
        _line(trial="T-000001") + "\n" + _line(trial="T-000005") + "\n", encoding="utf-8"
    )
    assert append_trial(ledger, make_trial()).trial_id == "T-000006"


def test_append_refuses_trials_that_already_have_an_id(ledger):
    with pytest.raises(ValueError, match="assigns trial ids"):
        append_trial(ledger, make_trial(trial_id="T-000009"))


@pytest.mark.parametrize("separator", [" ", " ", "\x85"], ids=["LS", "PS", "NEL"])
def test_notes_with_unicode_line_breaks_round_trip_as_ascii(ledger, separator):
    stored = append_trial(ledger, make_trial(note=f"first{separator}second"))
    assert read_trials(ledger) == [stored]
    assert ledger.read_bytes().isascii()


def test_append_refuses_a_trial_that_would_not_read_back(ledger):
    append_trial(ledger, make_trial())
    ledger.write_bytes(ledger.read_bytes().rstrip(b"\n"))
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="holdout_cutoff"):
        append_trial(ledger, make_trial(holdout_cutoff=datetime(2025, 4, 30, 12, 0)))
    assert ledger.read_bytes() == before


def test_append_refuses_nan_metrics(ledger):
    with pytest.raises(ValueError, match="JSON"):
        append_trial(ledger, make_trial(metrics={"GOLD": {"sharpe": float("nan")}}))
    assert ledger.read_bytes() == b""


def test_non_utf8_ledger_is_a_ledger_error(ledger):
    append_trial(ledger, make_trial())
    offset = ledger.stat().st_size
    with ledger.open("ab") as handle:
        handle.write(b"\xff\n")
    with pytest.raises(LedgerError, match=rf"UTF-8.*byte {offset}\b"):
        read_trials(ledger)
    with pytest.raises(LedgerError, match="UTF-8"):
        append_trial(ledger, make_trial(run_id="run-2"))


def test_malformed_line_is_reported_with_its_number(ledger):
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


@pytest.mark.parametrize(
    "line",
    [
        pytest.param("[1]", id="not-an-object"),
        pytest.param("{}", id="missing-keys"),
        pytest.param(_line(trial="T-000002", extra=1), id="unknown-key"),
        pytest.param(_line(trial="T-000002", strategy=7), id="strategy-not-an-id"),
        pytest.param(_line(trial="X-1"), id="bad-trial-id"),
        pytest.param(_line(trial="T-000002", run_id=3), id="run-id-not-text"),
        pytest.param(_line(trial="T-000002", kind="exotic"), id="unknown-kind"),
        pytest.param(_line(trial="T-000002", ts=5), id="ts-not-text"),
        pytest.param(_line(trial="T-000002", fingerprint="sha256:abc"), id="short-fingerprint"),
        pytest.param(_line(trial="T-000002", fingerprint_version="1"), id="version-as-text"),
        pytest.param(_line(trial="T-000002", holdout_cutoff=20250430), id="cutoff-not-text"),
        pytest.param(_line(trial="T-000002", code=[]), id="code-not-an-object"),
        pytest.param(_line(trial="T-000002", window="all"), id="window-not-an-object"),
        pytest.param(
            _line(trial="T-000002", window={"start": None, "end": None, "step": "1d"}),
            id="window-extra-key",
        ),
        pytest.param(_line(trial="T-000002", metrics={"GOLD": "x"}), id="metrics-not-nested"),
        pytest.param(
            _line(trial="T-000002", metrics={"GOLD": {"sharpe": float("nan")}}), id="nan-metric"
        ),
        pytest.param(
            _line(trial="T-000002", metrics={"GOLD": {"sharpe": 0.5}}).replace("0.5", "1e400"),
            id="overflowing-metric",
        ),
        pytest.param(
            _line(trial="T-000002", code={"commit": "abc123", "speed": float("nan")}),
            id="nan-literal-anywhere",
        ),
    ],
)
def test_malformed_records_are_ledger_errors(ledger, line):
    ledger.write_text(_line() + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="line 2"):
        read_trials(ledger)


def test_duplicate_trial_ids_are_a_ledger_error(ledger):
    ledger.write_text(_line() + "\n" + _line(run_id="run-2") + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="line 2.*duplicate"):
        read_trials(ledger)


def test_torn_last_line_blocks_appends(ledger):
    append_trial(ledger, make_trial())
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(_line(trial="T-000002")[:40])
    before = ledger.read_bytes()
    with pytest.raises(LedgerError, match="interrupted"):
        read_trials(ledger)
    with pytest.raises(LedgerError, match="interrupted"):
        append_trial(ledger, make_trial(run_id="run-2"))
    assert ledger.read_bytes() == before


def test_append_repairs_missing_trailing_newline(ledger):
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


def test_concurrent_appends_lose_no_lines(ledger):
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=append_many, args=(str(ledger), worker, 25)) for worker in range(4)
    ]
    try:
        for process in workers:
            process.start()
        for process in workers:
            process.join(timeout=60)
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join()
    assert [process.exitcode for process in workers] == [0, 0, 0, 0]
    trials = read_trials(ledger)
    assert len(trials) == 100
    assert len({t.trial_id for t in trials}) == 100
    assert len({t.run_id for t in trials}) == 100
