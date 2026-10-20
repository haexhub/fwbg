import subprocess
from dataclasses import replace
from datetime import date

import pytest

from fwbg.research.card import CardError, Status, criteria_hash, load_card, replace_card
from fwbg.research.commands import (
    CommandError,
    amend_criteria,
    months_before,
    new_card,
    reopen,
    write_verdict,
)
from fwbg.research.gate import RunRequest, check
from fwbg.research.ledger import LEDGER_FILENAME, append_trial, read_trials
from fwbg.research.pending import write_pending
from fwbg.research.repo import REGISTER_PATHS, commit_register
from tests.research.factories import make_run, make_trial
from tests.research.test_pending import dead_pid, make_pending

TODAY = date(2026, 10, 20)
TEMPLATE = {
    "min_sharpe": 1.0,
    "min_trades": 100.0,
    "max_drawdown": 0.40,
    "min_fold_stability": 0.6,
    "max_p_value": 0.05,
}


def git_status(register) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=register,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def card(register, card_id="S-0007"):
    found = load_card(register, card_id)
    assert found is not None
    return found


def set_card(register, **changes) -> None:
    replace_card(register, replace(card(register), **changes))
    commit_register(register, REGISTER_PATHS, "test: edit card")


def ledger(register):
    return read_trials(register / LEDGER_FILENAME)


# --- new ---------------------------------------------------------------------


def test_new_card_registers_the_next_active_card(gate_register):
    path = new_card(
        gate_register,
        "Opening range breakout",
        ["orb", " session "],
        TODAY,
        criteria={"min_sharpe": 0.8},
        hypothesis="Breakouts persist.",
    )
    assert path == gate_register / "strategies" / "S-0008-opening-range-breakout.md"
    created = card(gate_register, "S-0008")
    assert created.status is Status.ACTIVE
    assert created.tags == ("orb", "session")
    assert created.created == TODAY
    assert created.holdout_cutoff == date(2025, 4, 20)
    assert created.holdout_contaminated is False
    assert dict(created.success_criteria) == {**TEMPLATE, "min_sharpe": 0.8}
    assert created.registered_criteria_hash == criteria_hash(created.success_criteria)
    assert created.criteria_amendments == ()
    headings = [line for line in created.body.splitlines() if line.startswith("## ")]
    assert headings == ["## Hypothesis", "## Mechanism", "## History", "## Verdict"]
    assert "Breakouts persist." in created.body
    assert git_status(gate_register) == ""


def test_new_card_takes_cutoff_and_slug(gate_register):
    path = new_card(gate_register, "Ä ö", [], TODAY, cutoff=date(2025, 1, 31), slug="umlauts")
    assert path.name == "S-0008-umlauts.md"
    assert card(gate_register, "S-0008").holdout_cutoff == date(2025, 1, 31)


def test_new_card_refuses_a_future_cutoff(gate_register):
    with pytest.raises(CommandError, match="future"):
        new_card(gate_register, "x", [], TODAY, cutoff=date(2026, 10, 21))


def test_new_card_refuses_an_unknown_criterion(gate_register):
    with pytest.raises(CardError, match="min_sharp"):
        new_card(gate_register, "x", [], TODAY, criteria={"min_sharp": 1.0})


def test_new_card_refuses_a_multi_line_title(gate_register):
    with pytest.raises(CommandError, match="title"):
        new_card(gate_register, "a\nb", [], TODAY)


def test_new_card_refuses_a_title_without_ascii_words_and_no_slug(gate_register):
    with pytest.raises(CardError, match="empty slug"):
        new_card(gate_register, "ää", [], TODAY)
    assert git_status(gate_register) == ""
    assert sorted(path.name for path in (gate_register / "strategies").iterdir()) == [
        "S-0007-cusum-event-continuation.md"
    ]


def test_new_card_needs_an_initialised_register(strategies_dir):
    with pytest.raises(CommandError, match="not initialised"):
        new_card(strategies_dir, "x", [], TODAY)


def test_months_before_clamps_to_the_end_of_a_shorter_month():
    assert months_before(date(2026, 8, 31), 18) == date(2025, 2, 28)
    assert months_before(date(2026, 10, 20), 18) == date(2025, 4, 20)
    assert months_before(date(2026, 1, 15), 1) == date(2025, 12, 15)


# --- amend -------------------------------------------------------------------


def test_amend_records_the_change_and_the_gate_accepts_it(gate_register):
    append_trial(gate_register / LEDGER_FILENAME, make_trial())
    updated = amend_criteria(gate_register, "S-0007", {"min_sharpe": 0.5}, "too strict", TODAY)
    stored = card(gate_register)
    assert stored == updated
    assert dict(stored.success_criteria) == {"min_sharpe": 0.5, "min_trades": 100.0}
    (amendment,) = stored.criteria_amendments
    assert (amendment.date, amendment.reason) == (TODAY, "too strict")
    assert amendment.criteria_hash == stored.criteria_hash
    metadata = {"name": "cusum", "timeframe": "DAY_1", "pipeline": {"indicators": []}}
    request = RunRequest("S-0007", metadata, ("GOLD",), "after the amendment")
    assert check(stored, request, ledger(gate_register)).allowed
    assert git_status(gate_register) == ""


def test_amend_refuses_unchanged_criteria(gate_register):
    with pytest.raises(CommandError, match="unchanged"):
        amend_criteria(gate_register, "S-0007", {"min_sharpe": 1.0}, "same", TODAY)


def test_amend_refuses_an_unknown_criterion(gate_register):
    with pytest.raises(CardError, match="min_sharp"):
        amend_criteria(gate_register, "S-0007", {"min_sharp": 0.5}, "typo", TODAY)


def test_amend_refuses_a_multi_line_reason(gate_register):
    with pytest.raises(CommandError, match="reason"):
        amend_criteria(gate_register, "S-0007", {"min_sharpe": 0.5}, "a\nb", TODAY)


def test_amend_refuses_hand_edited_criteria(gate_register):
    set_card(gate_register, success_criteria={"min_sharpe": 0.1, "min_trades": 100.0})
    with pytest.raises(CommandError, match="edited by hand"):
        amend_criteria(gate_register, "S-0007", {"min_sharpe": 0.5}, "loosen", TODAY)


def test_amend_refuses_an_unknown_card(gate_register):
    with pytest.raises(CommandError, match="no research card S-0099"):
        amend_criteria(gate_register, "S-0099", {"min_sharpe": 0.5}, "x", TODAY)


# --- verdict -----------------------------------------------------------------


def test_verdict_closes_the_card_with_its_statistics(gate_register):
    ledger_path = gate_register / LEDGER_FILENAME
    append_trial(ledger_path, make_trial(metrics={"GOLD": {"sharpe": 0.8, "trades": 120}}))
    append_trial(ledger_path, make_trial(strategy="S-0001", metrics={"X": {}, "Y": {}}))
    updated = write_verdict(
        gate_register, "S-0007", Status.REJECTED, "no edge", "new data source", TODAY
    )
    stored = card(gate_register)
    assert stored == updated
    assert stored.status is Status.REJECTED
    assert stored.body == (
        "## Verdict\n\n"
        "### 2026-10-20: rejected\n\n"
        "Reason: no edge\n"
        "Reopen if: new data source\n"
        "Trials: 1 counted on S-0007 (1 symbol evaluations); "
        "2 counted in the register (3 symbol evaluations).\n"
        "Criteria versions used: 1.\n"
        "Holdout: unused.\n"
        "Best development result: T-000001 GOLD: "
        "min_sharpe fail (0.8 < 1); min_trades pass (120 >= 100).\n"
        "Deflated Sharpe ratio: not computed yet (register step 5).\n"
    )
    assert git_status(gate_register) == ""


def test_verdict_without_results_says_so(gate_register):
    write_verdict(gate_register, "S-0007", Status.PARKED, "paused", "time", TODAY)
    assert "Best development result: none with a Sharpe ratio." in card(gate_register).body


def test_verdict_refuses_a_card_that_is_not_active(gate_register):
    set_card(gate_register, status=Status.PARKED)
    with pytest.raises(CommandError, match="parked"):
        write_verdict(gate_register, "S-0007", Status.REJECTED, "x", "y", TODAY)


def test_verdict_refuses_the_status_active(gate_register):
    with pytest.raises(CommandError, match="not active"):
        write_verdict(gate_register, "S-0007", Status.ACTIVE, "x", "y", TODAY)


def test_verdict_waits_for_live_runs_of_the_card(gate_register, tmp_path):
    write_pending(gate_register, make_pending(tmp_path / "results" / "run-1"))
    with pytest.raises(CommandError, match="run-1"):
        write_verdict(gate_register, "S-0007", Status.REJECTED, "x", "y", TODAY)
    assert card(gate_register).status is Status.ACTIVE


def test_verdict_records_a_stale_run_of_the_card_first(gate_register, tmp_path):
    run_dir = make_run(tmp_path / "results", "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))
    write_verdict(gate_register, "S-0007", Status.REJECTED, "no edge", "never", TODAY)
    assert [trial.run_id for trial in ledger(gate_register)] == ["run-1"]
    assert "Trials: 1 counted on S-0007" in card(gate_register).body
    assert git_status(gate_register) == ""


def test_refused_verdict_still_commits_recovered_runs(gate_register, tmp_path):
    set_card(gate_register, status=Status.PARKED)
    run_dir = make_run(tmp_path / "results", "run-1")
    write_pending(gate_register, make_pending(run_dir, pid=dead_pid()))
    with pytest.raises(CommandError):
        write_verdict(gate_register, "S-0007", Status.REJECTED, "x", "y", TODAY)
    assert len(ledger(gate_register)) == 1
    assert git_status(gate_register) == ""


# --- reopen ------------------------------------------------------------------


def test_reopen_records_why_under_history_before_the_verdict(gate_register):
    set_card(gate_register, status=Status.PARKED, body="## Hypothesis\n\nx\n\n## Verdict\n\nOld.\n")
    updated = reopen(gate_register, "S-0007", "new data", TODAY)
    stored = card(gate_register)
    assert stored == updated
    assert stored.status is Status.ACTIVE
    assert stored.body == (
        "## Hypothesis\n\nx\n\n"
        "## History\n\n- 2026-10-20: reopened from parked: new data\n\n"
        "## Verdict\n\nOld.\n"
    )
    assert git_status(gate_register) == ""


def test_reopen_appends_to_an_existing_history(gate_register):
    set_card(gate_register, status=Status.REJECTED, body="## History\n\n- earlier\n")
    reopen(gate_register, "S-0007", "second look", TODAY)
    assert card(gate_register).body == (
        "## History\n\n- earlier\n\n- 2026-10-20: reopened from rejected: second look\n"
    )


@pytest.mark.parametrize("status", [Status.ACTIVE, Status.CANDIDATE])
def test_reopen_refuses_active_and_candidate_cards(gate_register, status):
    set_card(gate_register, status=status)
    with pytest.raises(CommandError, match=status.value):
        reopen(gate_register, "S-0007", "x", TODAY)
