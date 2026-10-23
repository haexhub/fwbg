"""``--check-json``: the gate check as one machine-readable stdout line."""

import json

import pytest

from fwbg.research.gate import Decision, Denial, DenialCode
from fwbg.research.ledger import TrialKind
from fwbg.research.runs import (
    CHECK_PREFIX,
    GateChecked,
    GateDenied,
    ResearchRun,
    check_result,
    parse_check_output,
    run_gated,
)

DECISION = Decision(
    denials=(),
    kind=TrialKind.ITERATION,
    effective_metadata={"start_date": "2024-01-01", "end_date": "2025-04-30"},
    fingerprint="abc123",
)
DENIALS = (
    Denial(DenialCode.DUPLICATE_FINGERPRINT, "identical configuration"),
    Denial(DenialCode.MISSING_NOTE, "every run needs a note"),
)


def test_check_result_for_an_allowed_decision():
    assert check_result(decision=DECISION, symbols=("EURUSD", "GOLD")) == {
        "allowed": True,
        "kind": "iteration",
        "fingerprint": "abc123",
        "symbols": ["EURUSD", "GOLD"],
        "window": {"start": "2024-01-01", "end": "2025-04-30"},
        "holdout_cutoff": None,
        "denials": [],
    }


def test_check_result_names_the_cutoff_of_a_holdout():
    holdout = Decision(
        denials=(),
        kind=TrialKind.HOLDOUT,
        effective_metadata={"start_date": "2024-01-01", "holdout_cutoff": "2025-04-30"},
        fingerprint="abc123",
    )
    payload = check_result(decision=holdout, symbols=("EURUSD",))
    assert payload["kind"] == "holdout" and payload["holdout_cutoff"] == "2025-04-30"
    assert payload["window"] == {"start": "2024-01-01", "end": None}
    assert parse_check_output(CHECK_PREFIX + json.dumps(payload), 0) == payload


@pytest.mark.parametrize("cutoff", [20250430, ["2025-04-30"]])
def test_parse_check_output_rejects_a_malformed_cutoff(cutoff):
    payload = {**check_result(decision=DECISION, symbols=("EURUSD",)), "holdout_cutoff": cutoff}
    assert parse_check_output(CHECK_PREFIX + json.dumps(payload), 0) is None


def test_check_result_lists_every_denial_with_its_status():
    assert check_result(DENIALS) == {
        "allowed": False,
        "denials": [
            {"code": "duplicate_fingerprint", "status": 409, "message": "identical configuration"},
            {"code": "missing_note", "status": 422, "message": "every run needs a note"},
        ],
    }


def test_parse_check_output_finds_the_last_payload_among_noise():
    allowed = check_result(decision=DECISION, symbols=("EURUSD",))
    stdout = "\n".join(
        [
            "Research gate: allowed (iteration)",
            CHECK_PREFIX + json.dumps(check_result(DENIALS)),
            "progress 50%",
            CHECK_PREFIX + json.dumps(allowed),
            "done",
        ]
    )
    assert parse_check_output(stdout, 0) == allowed


@pytest.mark.parametrize(
    "stdout",
    ["", "Research gate: allowed (iteration)\n", CHECK_PREFIX + "{broken", CHECK_PREFIX + "[1]"],
)
def test_parse_check_output_without_a_payload(stdout):
    assert parse_check_output(stdout, 0) is None


@pytest.mark.parametrize(("allowed", "returncode"), [(True, 1), (True, 2), (False, 0), (False, 1)])
def test_parse_check_output_rejects_a_payload_that_contradicts_the_exit_code(allowed, returncode):
    payload = (
        check_result(decision=DECISION, symbols=("EURUSD",)) if allowed else check_result(DENIALS)
    )
    stdout = CHECK_PREFIX + json.dumps(payload)
    assert parse_check_output(stdout, returncode) is None


@pytest.mark.parametrize(("allowed", "returncode"), [(True, 0), (False, 2)])
def test_parse_check_output_accepts_a_payload_that_matches_the_exit_code(allowed, returncode):
    payload = (
        check_result(decision=DECISION, symbols=("EURUSD",)) if allowed else check_result(DENIALS)
    )
    stdout = CHECK_PREFIX + json.dumps(payload)
    assert parse_check_output(stdout, returncode) == payload


def test_parse_check_output_rejects_an_incomplete_allowed_payload():
    stdout = CHECK_PREFIX + json.dumps({"allowed": True, "denials": []})
    assert parse_check_output(stdout, 0) is None


def research_run(tmp_path) -> ResearchRun:
    return ResearchRun(tmp_path / "research", "run-1", tmp_path / "run-1", "S-0007", "x")


def test_run_gated_prints_a_refusal_as_json(tmp_path, capsys):
    def run(before_run):
        raise GateDenied(DENIALS)

    with pytest.raises(SystemExit) as exited:
        run_gated(research_run(tmp_path), run, as_json=True)
    assert exited.value.code == 2
    captured = capsys.readouterr()
    assert "[missing_note]" in captured.err
    assert parse_check_output(captured.out, 2) == check_result(DENIALS)


def test_run_gated_prints_an_allowed_check_as_json(tmp_path, capsys):
    def run(before_run):
        raise GateChecked(DECISION, ("EURUSD",))

    with pytest.raises(SystemExit) as exited:
        run_gated(research_run(tmp_path), run, as_json=True)
    assert exited.value.code == 0
    out = capsys.readouterr().out
    assert "Research gate: allowed (iteration)" in out
    assert parse_check_output(out, 0) == check_result(decision=DECISION, symbols=("EURUSD",))


def test_run_gated_prints_no_json_by_default(tmp_path, capsys):
    def run(before_run):
        raise GateChecked(DECISION, ("EURUSD",))

    with pytest.raises(SystemExit):
        run_gated(research_run(tmp_path), run)
    assert parse_check_output(capsys.readouterr().out, 0) is None
