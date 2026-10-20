import subprocess
from dataclasses import replace
from datetime import date

import pytest

from fwbg.research import commands, pending
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
from fwbg.research.ledger import TrialKind
from fwbg.research.repo import REGISTER_PATHS, ResearchRepoError, commit_register
from tests.research.factories import make_card, make_run, make_trial
from tests.research.test_pending import dead_pid, make_pending

TODAY = date(2026, 10, 20)
REFUSED_AMEND = (
    r"^the criteria of S-0007 do not match its recorded registration and amendments; "
    r"restore them by hand \(see the register's git history\) before amending$"
)
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


def short(hash_: str) -> str:
    return hash_.removeprefix("sha256:")[:12]


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
    assert created.body.endswith("## History\n\n## Verdict\n")
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
    with pytest.raises(CommandError, match=REFUSED_AMEND):
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
        f"Criteria versions used: 1 ({short(make_card().criteria_hash)}).\n"
        "Holdout: unused.\n"
        "Best development result: T-000001 GOLD, checked against the current criteria "
        "(registration): "
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


# --- review fixes ------------------------------------------------------------


@pytest.mark.parametrize("text", ["x\n## Verdict\n\nfake", "# Title", "a\n#\tb"])
def test_new_card_refuses_headings_in_section_text(gate_register, text):
    with pytest.raises(CommandError, match="heading"):
        new_card(gate_register, "x", [], TODAY, hypothesis=text)
    with pytest.raises(CommandError, match="heading"):
        new_card(gate_register, "x", [], TODAY, mechanism=text)


def test_new_card_keeps_subheadings_in_section_text(gate_register):
    new_card(gate_register, "x", [], TODAY, hypothesis="### Detail\n\ntext")
    assert "### Detail" in card(gate_register, "S-0008").body


def test_amend_refuses_a_hand_edit_with_a_recomputed_registration(gate_register):
    append_trial(gate_register / LEDGER_FILENAME, make_trial())
    edited = {"min_sharpe": 0.1, "min_trades": 100.0}
    set_card(gate_register, success_criteria=edited, registered_criteria_hash=criteria_hash(edited))
    with pytest.raises(CommandError, match=REFUSED_AMEND):
        amend_criteria(gate_register, "S-0007", {"min_sharpe": 0.5}, "loosen", TODAY)
    assert card(gate_register).criteria_amendments == ()


def test_amend_refuses_a_card_that_is_not_active(gate_register):
    set_card(gate_register, status=Status.PARKED)
    with pytest.raises(CommandError, match="reopen it first"):
        amend_criteria(gate_register, "S-0007", {"min_sharpe": 0.5}, "loosen", TODAY)


def test_second_amendment_builds_on_the_first(gate_register):
    amend_criteria(gate_register, "S-0007", {"min_sharpe": 0.5}, "first", TODAY)
    updated = amend_criteria(gate_register, "S-0007", {"min_trades": 50.0}, "second", TODAY)
    assert dict(updated.success_criteria) == {"min_sharpe": 0.5, "min_trades": 50.0}
    assert [item.reason for item in updated.criteria_amendments] == ["first", "second"]
    assert updated.criteria_amendments[-1].criteria_hash == updated.criteria_hash
    assert card(gate_register) == updated


def test_verdict_refuses_while_a_stale_run_cannot_be_recorded(gate_register, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pending, "finish", broken)
    write_pending(gate_register, make_pending(tmp_path / "results" / "run-1", pid=dead_pid()))
    with pytest.raises(CommandError, match=r"could not be recorded \(run-1\); see the warning"):
        write_verdict(gate_register, "S-0007", Status.REJECTED, "x", "y", TODAY)
    assert card(gate_register).status is Status.ACTIVE


def test_verdict_names_pending_runs_of_other_cards(gate_register, tmp_path):
    write_pending(
        gate_register, make_pending(tmp_path / "results" / "o", run_id="o", strategy="S-0001")
    )
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    assert "register N excludes 1 pending runs" in card(gate_register).body


def test_verdict_without_pending_runs_does_not_mention_them(gate_register):
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    assert "pending" not in card(gate_register).body


def test_verdict_reports_holdout_looks_and_excludes_them_from_the_best_result(gate_register):
    ledger_path = gate_register / LEDGER_FILENAME
    append_trial(ledger_path, make_trial(metrics={"GOLD": {"sharpe": 0.8}}))
    append_trial(ledger_path, make_trial(kind=TrialKind.HOLDOUT, metrics={"GOLD": {"sharpe": 2.0}}))
    write_verdict(gate_register, "S-0007", Status.REJECTED, "x", "y", TODAY)
    body = card(gate_register).body
    assert "Holdout: used by T-000002." in body
    assert "Best development result: T-000001 GOLD" in body


def test_verdict_lists_every_criteria_version(gate_register):
    ledger_path = gate_register / LEDGER_FILENAME
    old = make_card().criteria_hash
    append_trial(ledger_path, make_trial(metrics={"GOLD": {"sharpe": 0.4}}))
    amended = amend_criteria(gate_register, "S-0007", {"min_sharpe": 0.5}, "loosen", TODAY)
    append_trial(
        ledger_path,
        make_trial(criteria_hash=amended.criteria_hash, metrics={"GOLD": {"sharpe": 0.6}}),
    )
    write_verdict(gate_register, "S-0007", Status.CANDIDATE, "x", "y", TODAY)
    body = card(gate_register).body
    assert f"Criteria versions used: 2 ({short(old)}, {short(amended.criteria_hash)})." in body
    assert (
        "Best development result: T-000002 GOLD, checked against the current criteria "
        "(amendment 1): min_sharpe pass (0.6 >= 0.5)"
    ) in body


def _parked(register):
    set_card(register, status=Status.PARKED)


@pytest.mark.parametrize(
    ("prepare", "command", "card_id"),
    [
        (None, lambda r: new_card(r, "x", [], TODAY), "S-0008"),
        (None, lambda r: amend_criteria(r, "S-0007", {"min_sharpe": 0.5}, "x", TODAY), "S-0007"),
        (None, lambda r: write_verdict(r, "S-0007", Status.PARKED, "x", "y", TODAY), "S-0007"),
        (_parked, lambda r: reopen(r, "S-0007", "x", TODAY), "S-0007"),
    ],
    ids=["new", "amend", "verdict", "reopen"],
)
def test_a_failed_commit_is_reported_and_leaves_the_change_on_disk(
    gate_register, monkeypatch, prepare, command, card_id
):
    if prepare is not None:
        prepare(gate_register)

    def broken(*args, **kwargs):
        raise ResearchRepoError("git commit failed: boom")

    monkeypatch.setattr(commands, "commit_register", broken)
    with pytest.raises(CommandError, match=f"{card_id} was updated on disk but not committed"):
        command(gate_register)
    assert card(gate_register, card_id) is not None
    assert git_status(gate_register) != ""
