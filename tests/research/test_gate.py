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


def test_moving_the_cutoff_earlier_after_trials_is_refused():
    moved = make_card(holdout_cutoff=date(2025, 1, 31))
    assert codes(check(moved, request(), [make_trial(trial_id="T-000001")])) == {
        DenialCode.CUTOFF_MOVED
    }


def test_unchanged_cutoff_after_trials_is_allowed():
    trials = [
        make_trial(
            trial_id="T-000001",
            kind=TrialKind.HOLDOUT,
            status=TrialStatus.FAILED,
            fingerprint=OTHER_FINGERPRINT,
        ),
        make_trial(trial_id="T-000002"),
    ]
    assert check(make_card(), request(), trials).allowed


def test_every_own_trial_pins_the_cutoff():
    trials = [
        make_trial(
            trial_id="T-000001",
            holdout_cutoff=date(2025, 6, 30),
            kind=TrialKind.HOLDOUT,
            status=TrialStatus.FAILED,
            fingerprint=OTHER_FINGERPRINT,
        ),
        make_trial(trial_id="T-000002"),
    ]
    assert codes(check(make_card(), request(), trials)) == {DenialCode.CUTOFF_MOVED}


def test_a_later_own_trial_also_pins_the_cutoff():
    trials = [
        make_trial(trial_id="T-000001"),
        make_trial(
            trial_id="T-000002",
            holdout_cutoff=date(2025, 1, 31),
            kind=TrialKind.HOLDOUT,
            status=TrialStatus.FAILED,
            fingerprint=OTHER_FINGERPRINT,
        ),
    ]
    assert codes(check(make_card(), request(), trials)) == {DenialCode.CUTOFF_MOVED}


def test_another_cards_cutoff_does_not_affect_this_card():
    trials = [
        make_trial(trial_id="T-000001"),
        make_trial(strategy="S-0001", trial_id="T-000002", holdout_cutoff=date(2024, 12, 31)),
    ]
    assert check(make_card(), request(), trials).allowed


@pytest.mark.parametrize("note", ["baseline\nextra", "a\rb", "a\u2028b", "baseline\n"])
def test_note_must_be_one_line(note):
    assert codes(check(make_card(), request(note=note), [])) == {DenialCode.MISSING_NOTE}


def test_blank_reason_does_not_bypass_duplicate_protection():
    first = check(make_card(), request(), [])
    earlier = [make_trial(trial_id="T-000001", fingerprint=first.fingerprint)]
    assert codes(check(make_card(), request(reason=" "), earlier)) == {
        DenialCode.DUPLICATE_FINGERPRINT
    }


HTTP_STATUS = {
    "missing_research_id": 422,
    "card_not_found": 422,
    "missing_note": 422,
    "invalid_window": 422,
    "invalid_config": 422,
    "not_active": 409,
    "criteria_changed": 409,
    "cutoff_moved": 409,
    "after_cutoff": 409,
    "holdout_repeat": 409,
    "holdout_contaminated": 409,
    "duplicate_fingerprint": 409,
}


@pytest.mark.parametrize("code", list(DenialCode))
def test_http_status_of_every_denial_code(code):
    assert code.http_status == HTTP_STATUS[code.value]


def criteria(min_sharpe: float) -> dict[str, float]:
    return {"min_sharpe": min_sharpe, "min_trades": 100.0}


def amendment(min_sharpe: float) -> CriteriaAmendment:
    return CriteriaAmendment(
        date(2026, 10, 20), "recalibrated", criteria_hash(criteria(min_sharpe))
    )


def test_criteria_are_anchored_to_the_last_own_trial():
    current = criteria(0.8)
    card = make_card(success_criteria=current, registered_criteria_hash=criteria_hash(current))
    a, b = criteria_hash(criteria(1.0)), criteria_hash(current)
    a_then_b = [
        make_trial(trial_id="T-000001", criteria_hash=a),
        make_trial(trial_id="T-000002", criteria_hash=b),
    ]
    b_then_a = [
        make_trial(trial_id="T-000001", criteria_hash=b),
        make_trial(trial_id="T-000002", criteria_hash=a),
    ]
    assert check(card, request(), a_then_b).allowed
    assert codes(check(card, request(), b_then_a)) == {DenialCode.CRITERIA_CHANGED}


def test_an_earlier_amendment_does_not_cover_the_current_criteria():
    card = make_card(
        success_criteria=criteria(0.6), criteria_amendments=(amendment(0.6), amendment(0.5))
    )
    assert codes(check(card, request(), [make_trial(trial_id="T-000001")])) == {
        DenialCode.CRITERIA_CHANGED
    }


def test_the_latest_amendment_covers_the_current_criteria():
    card = make_card(
        success_criteria=criteria(0.5), criteria_amendments=(amendment(0.6), amendment(0.5))
    )
    assert check(card, request(), [make_trial(trial_id="T-000001")]).allowed


def test_hand_edited_registration_hash_is_refused():
    edited = criteria(0.8)
    card = make_card(success_criteria=edited, registered_criteria_hash=criteria_hash(edited))
    assert codes(check(card, request(), [make_trial(trial_id="T-000001")])) == {
        DenialCode.CRITERIA_CHANGED
    }


def test_another_cards_criteria_do_not_affect_this_card():
    trials = [
        make_trial(trial_id="T-000001"),
        make_trial(
            strategy="S-0001", trial_id="T-000002", criteria_hash=criteria_hash(criteria(0.5))
        ),
    ]
    assert check(make_card(), request(), trials).allowed


def test_failed_holdout_look_still_counts():
    used = [
        make_trial(
            trial_id="T-000003",
            kind=TrialKind.HOLDOUT,
            status=TrialStatus.FAILED,
            fingerprint=OTHER_FINGERPRINT,
        )
    ]
    assert codes(check(make_card(), request(holdout=True), used)) == {DenialCode.HOLDOUT_REPEAT}


def test_another_cards_holdout_look_does_not_count():
    other = make_trial(
        strategy="S-0001",
        trial_id="T-000001",
        kind=TrialKind.HOLDOUT,
        fingerprint=OTHER_FINGERPRINT,
    )
    assert check(make_card(), request(holdout=True), [other]).allowed


def test_iteration_after_a_holdout_look_needs_no_reason():
    used = [make_trial(trial_id="T-000001", kind=TrialKind.HOLDOUT, fingerprint=OTHER_FINGERPRINT)]
    assert check(make_card(), request(), used).allowed


def test_holdout_may_end_after_the_cutoff():
    metadata = {**METADATA, "end_date": "2025-05-01"}
    decision = check(make_card(), request(strategy_metadata=metadata, holdout=True), [])
    assert decision.allowed
    assert decision.effective_metadata["end_date"] == "2025-05-01"


@pytest.mark.parametrize("end", ["2025-04-30", "2025-04-29"])
def test_holdout_must_end_after_the_cutoff(end):
    metadata = {**METADATA, "end_date": end}
    assert codes(check(make_card(), request(strategy_metadata=metadata, holdout=True), [])) == {
        DenialCode.INVALID_WINDOW
    }


@pytest.mark.parametrize("reason", [None, "exit bug fixed"])
def test_contaminated_holdout_is_refused_even_with_a_reason(reason):
    card = make_card(holdout_contaminated=True)
    decision = check(card, request(holdout=True, reason=reason), [])
    assert codes(decision) == {DenialCode.HOLDOUT_CONTAMINATED}


def test_contaminated_holdout_is_reported_once_after_an_earlier_look():
    card = make_card(holdout_contaminated=True)
    used = [make_trial(trial_id="T-000001", kind=TrialKind.HOLDOUT, fingerprint=OTHER_FINGERPRINT)]
    decision = check(card, request(holdout=True), used)
    assert codes(decision) == {DenialCode.HOLDOUT_CONTAMINATED}


def test_contaminated_card_still_accepts_iterations():
    assert check(make_card(holdout_contaminated=True), request(), []).allowed


def test_iteration_may_end_on_the_cutoff():
    metadata = {**METADATA, "end_date": "2025-04-30"}
    assert check(make_card(), request(strategy_metadata=metadata), []).allowed


@pytest.mark.parametrize(
    "end", ["2025-04-30T23:59:59", "20250430", "2025-W18-3", "2025-02-30", 20250430]
)
def test_end_date_must_be_a_plain_iso_date(end):
    metadata = {**METADATA, "end_date": end}
    assert codes(check(make_card(), request(strategy_metadata=metadata), [])) == {
        DenialCode.INVALID_WINDOW
    }


def test_malformed_start_date_is_refused():
    metadata = {**METADATA, "start_date": "2024-01-01T00:00:00"}
    assert codes(check(make_card(), request(strategy_metadata=metadata), [])) == {
        DenialCode.INVALID_WINDOW
    }


def test_start_date_before_end_date_is_kept():
    metadata = {**METADATA, "start_date": "2024-01-01", "end_date": "2024-12-31"}
    decision = check(make_card(), request(strategy_metadata=metadata), [])
    assert decision.allowed
    assert decision.effective_metadata["start_date"] == "2024-01-01"


@pytest.mark.parametrize("start", ["2024-12-31", "2025-01-01"])
def test_start_date_must_precede_end_date(start):
    metadata = {**METADATA, "start_date": start, "end_date": "2024-12-31"}
    assert codes(check(make_card(), request(strategy_metadata=metadata), [])) == {
        DenialCode.INVALID_WINDOW
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {**METADATA, "pipeline": "orb_simple_v1"},
        {**METADATA, "params": {"threshold": float("nan")}},
        {**METADATA, "params": {"levels": {1, 2}}},
    ],
    ids=["unresolved-preset", "nan", "non-json-value"],
)
def test_unhashable_config_is_a_request_error(metadata):
    decision = check(make_card(), request(strategy_metadata=metadata), [])
    assert codes(decision) == {DenialCode.INVALID_CONFIG}
    assert decision.denials[0].code.http_status == 422
    assert decision.fingerprint is None


def test_run_without_symbols_is_a_request_error():
    decision = check(make_card(), request(symbols=()), [])
    assert codes(decision) == {DenialCode.INVALID_CONFIG}
    assert decision.denials[0].code.http_status == 422
    assert "at least one symbol" in decision.denials[0].message
    assert decision.fingerprint is None


def test_card_must_match_the_requested_research_id():
    decision = check(make_card(id="S-0008"), request(), [])
    assert codes(decision) == {DenialCode.CARD_NOT_FOUND}
    assert "S-0007" in decision.denials[0].message
    assert "S-0008" in decision.denials[0].message


@pytest.mark.parametrize("start", ["2025-04-30", "2025-05-01"])
def test_open_ended_iteration_must_start_before_the_cutoff(start):
    metadata = {**METADATA, "start_date": start}
    assert codes(check(make_card(), request(strategy_metadata=metadata), [])) == {
        DenialCode.INVALID_WINDOW
    }
