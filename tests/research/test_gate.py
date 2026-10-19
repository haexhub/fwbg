from datetime import date

import pytest

from fwbg.research.card import CriteriaAmendment, Status, criteria_hash
from fwbg.research.fingerprint import fingerprint
from fwbg.research.gate import DenialCode, RunRequest, check
from fwbg.research.ledger import TrialKind, TrialStatus
from tests.research.factories import make_card, make_trial

METADATA = {"name": "cusum", "timeframe": "DAY_1", "pipeline": {"indicators": []}}
SYMBOLS = ("GOLD",)
OTHER_FINGERPRINT = "sha256:" + "1" * 64


def request(**overrides) -> RunRequest:
    values = {
        "research_id": "S-0007",
        "strategy_metadata": METADATA,
        "symbols": SYMBOLS,
        "note": "baseline",
    }
    values.update(overrides)
    return RunRequest(**values)


def codes(decision) -> set[DenialCode]:
    return {denial.code for denial in decision.denials}


def test_allows_first_iteration_and_clamps_end_date_to_cutoff():
    decision = check(make_card(), request(), [])
    assert decision.allowed
    assert decision.kind is TrialKind.ITERATION
    assert decision.effective_metadata["end_date"] == "2025-04-30"
    assert decision.fingerprint == fingerprint(decision.effective_metadata, SYMBOLS)
    assert "end_date" not in METADATA


def test_missing_research_id_is_a_request_error():
    decision = check(None, request(research_id=None), [])
    assert codes(decision) == {DenialCode.MISSING_RESEARCH_ID}
    assert decision.denials[0].code.http_status == 422
    assert decision.fingerprint is None


def test_unknown_card_is_a_request_error():
    assert codes(check(None, request(), [])) == {DenialCode.CARD_NOT_FOUND}


@pytest.mark.parametrize("note", [None, "", "   "])
def test_note_is_required(note):
    assert codes(check(make_card(), request(note=note), [])) == {DenialCode.MISSING_NOTE}


@pytest.mark.parametrize("status", [Status.PARKED, Status.REJECTED, Status.CANDIDATE])
def test_only_active_cards_accept_runs(status):
    decision = check(make_card(status=status), request(), [])
    assert codes(decision) == {DenialCode.NOT_ACTIVE}
    assert decision.denials[0].code.http_status == 409


def test_collects_all_denials():
    decision = check(make_card(status=Status.PARKED), request(note=""), [])
    assert codes(decision) == {DenialCode.NOT_ACTIVE, DenialCode.MISSING_NOTE}


def test_end_date_before_cutoff_is_kept():
    metadata = {**METADATA, "end_date": "2024-12-31"}
    decision = check(make_card(), request(strategy_metadata=metadata), [])
    assert decision.allowed
    assert decision.effective_metadata["end_date"] == "2024-12-31"


def test_end_date_after_cutoff_is_refused():
    metadata = {**METADATA, "end_date": "2025-05-01"}
    assert codes(check(make_card(), request(strategy_metadata=metadata), [])) == {
        DenialCode.AFTER_CUTOFF
    }


def test_malformed_end_date_is_a_request_error():
    metadata = {**METADATA, "end_date": "soon"}
    decision = check(make_card(), request(strategy_metadata=metadata), [])
    assert codes(decision) == {DenialCode.INVALID_WINDOW}
    assert decision.denials[0].code.http_status == 422


def test_first_holdout_look_is_free_and_unclamped():
    decision = check(make_card(), request(holdout=True), [])
    assert decision.allowed
    assert decision.kind is TrialKind.HOLDOUT
    assert "end_date" not in decision.effective_metadata


def test_second_holdout_look_needs_a_reason():
    used = [make_trial(trial_id="T-000003", kind=TrialKind.HOLDOUT, fingerprint=OTHER_FINGERPRINT)]
    assert codes(check(make_card(), request(holdout=True), used)) == {DenialCode.HOLDOUT_REPEAT}
    assert check(make_card(), request(holdout=True, reason="exit bug fixed"), used).allowed


def test_duplicate_fingerprint_is_refused_with_pointer_to_earlier_trial():
    first = check(make_card(), request(), [])
    earlier = [make_trial(trial_id="T-000001", fingerprint=first.fingerprint)]
    decision = check(make_card(), request(), earlier)
    assert codes(decision) == {DenialCode.DUPLICATE_FINGERPRINT}
    assert "T-000001" in decision.denials[0].message


def test_duplicate_under_another_research_id_is_refused_too():
    first = check(make_card(), request(), [])
    earlier = [make_trial(strategy="S-0001", trial_id="T-000001", fingerprint=first.fingerprint)]
    assert codes(check(make_card(), request(), earlier)) == {DenialCode.DUPLICATE_FINGERPRINT}


def test_reason_allows_rerunning_an_identical_config():
    first = check(make_card(), request(), [])
    earlier = [make_trial(trial_id="T-000001", fingerprint=first.fingerprint)]
    assert check(make_card(), request(reason="simulator bugfix"), earlier).allowed


def test_failed_trial_does_not_block_a_rerun():
    first = check(make_card(), request(), [])
    earlier = [
        make_trial(trial_id="T-000001", fingerprint=first.fingerprint, status=TrialStatus.FAILED)
    ]
    assert check(make_card(), request(), earlier).allowed


def test_changed_criteria_without_amendment_are_refused():
    changed = make_card(success_criteria={"min_sharpe": 0.5, "min_trades": 100.0})
    trials = [make_trial(trial_id="T-000001")]
    assert codes(check(changed, request(), trials)) == {DenialCode.CRITERIA_CHANGED}


def test_amended_criteria_are_accepted():
    new_criteria = {"min_sharpe": 0.5, "min_trades": 100.0}
    amendment = CriteriaAmendment(date(2026, 10, 20), "too strict", criteria_hash(new_criteria))
    amended = make_card(success_criteria=new_criteria, criteria_amendments=(amendment,))
    assert check(amended, request(), [make_trial(trial_id="T-000001")]).allowed


def test_first_trial_uses_the_registered_criteria_hash():
    changed = make_card(registered_criteria_hash="sha256:other")
    assert codes(check(changed, request(), [])) == {DenialCode.CRITERIA_CHANGED}


def test_amended_criteria_are_accepted_before_the_first_trial():
    new_criteria = {"min_sharpe": 0.5, "min_trades": 100.0}
    amendment = CriteriaAmendment(date(2026, 10, 20), "too strict", criteria_hash(new_criteria))
    amended = make_card(
        success_criteria=new_criteria,
        registered_criteria_hash="sha256:old",
        criteria_amendments=(amendment,),
    )
    assert check(amended, request(), []).allowed


def test_moving_the_cutoff_later_after_trials_is_refused():
    moved = make_card(holdout_cutoff=date(2025, 6, 30))
    assert codes(check(moved, request(), [make_trial(trial_id="T-000001")])) == {
        DenialCode.CUTOFF_MOVED
    }


def test_moving_the_cutoff_earlier_is_allowed():
    moved = make_card(holdout_cutoff=date(2025, 1, 31))
    assert check(moved, request(), [make_trial(trial_id="T-000001")]).allowed


def test_note_must_be_one_line():
    assert codes(check(make_card(), request(note="baseline\nextra"), [])) == {
        DenialCode.MISSING_NOTE
    }


def test_blank_reason_does_not_bypass_duplicate_protection():
    first = check(make_card(), request(), [])
    earlier = [make_trial(trial_id="T-000001", fingerprint=first.fingerprint)]
    assert codes(check(make_card(), request(reason=" "), earlier)) == {
        DenialCode.DUPLICATE_FINGERPRINT
    }
