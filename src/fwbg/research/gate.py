"""Pre-run gate of the research register.

``check`` is pure: the caller loads the card and the ledger, calls ``check``,
and starts the run with ``Decision.effective_metadata`` only if
``Decision.allowed``. The CLI run path and ``POST /api/runs/start`` share it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from fwbg.research.card import ResearchCard, Status
from fwbg.research.fingerprint import fingerprint
from fwbg.research.ledger import Trial, TrialKind, TrialStatus


class DenialCode(str, Enum):
    MISSING_RESEARCH_ID = "missing_research_id"
    CARD_NOT_FOUND = "card_not_found"
    MISSING_NOTE = "missing_note"
    INVALID_WINDOW = "invalid_window"
    NOT_ACTIVE = "not_active"
    CRITERIA_CHANGED = "criteria_changed"
    CUTOFF_MOVED = "cutoff_moved"
    AFTER_CUTOFF = "after_cutoff"
    HOLDOUT_REPEAT = "holdout_repeat"
    DUPLICATE_FINGERPRINT = "duplicate_fingerprint"

    @property
    def http_status(self) -> int:
        """422 for an incomplete or malformed request, 409 for a conflict with the register."""
        return 422 if self in _REQUEST_ERRORS else 409


_REQUEST_ERRORS = frozenset(
    {
        DenialCode.MISSING_RESEARCH_ID,
        DenialCode.CARD_NOT_FOUND,
        DenialCode.MISSING_NOTE,
        DenialCode.INVALID_WINDOW,
    }
)


@dataclass(frozen=True)
class Denial:
    code: DenialCode
    message: str


@dataclass(frozen=True)
class RunRequest:
    """A run after overrides, preset resolution and canonical asset resolution.

    ``strategy_metadata`` must already be passed through
    ``fwbg.core.presets.resolve_presets`` (the fingerprint refuses unresolved
    preset names). ``symbols`` contains the uppercase, sorted concrete symbols
    produced by ``AssetRegistry``. CLI/API wiring must resolve asset classes
    before constructing this request; unresolved names never reach the gate.
    """

    research_id: str | None
    strategy_metadata: Mapping[str, Any]
    symbols: tuple[str, ...]
    note: str | None
    reason: str | None = None
    holdout: bool = False


@dataclass(frozen=True)
class Decision:
    denials: tuple[Denial, ...]
    kind: TrialKind
    effective_metadata: Mapping[str, Any]
    fingerprint: str | None

    @property
    def allowed(self) -> bool:
        return not self.denials


def check(card: ResearchCard | None, request: RunRequest, trials: Sequence[Trial]) -> Decision:
    """Decide whether *request* may run, given its card and the whole ledger."""
    kind = TrialKind.HOLDOUT if request.holdout else TrialKind.ITERATION
    metadata = dict(request.strategy_metadata)
    if not request.research_id:
        return _refuse(
            DenialCode.MISSING_RESEARCH_ID, "every run needs a research id (S-NNNN)", kind, metadata
        )
    if card is None:
        return _refuse(
            DenialCode.CARD_NOT_FOUND, f"no research card {request.research_id}", kind, metadata
        )

    own_trials = [trial for trial in trials if trial.strategy == card.id]
    denials: list[Denial] = []
    if not _valid_note(request.note):
        denials.append(
            Denial(
                DenialCode.MISSING_NOTE,
                "every run needs a non-empty one-line note: what changed and why",
            )
        )
    if card.status is not Status.ACTIVE:
        denials.append(
            Denial(
                DenialCode.NOT_ACTIVE,
                f"{card.id} is {card.status.value}; reopen it with a reason before running",
            )
        )
    denials.extend(_criteria_denials(card, own_trials))
    denials.extend(_cutoff_denials(card, own_trials))
    window_denial = _apply_window(card, metadata, request.holdout)
    if window_denial is not None:
        denials.append(window_denial)
    has_reason = _has_text(request.reason)
    if request.holdout and not has_reason:
        denials.extend(_holdout_denials(card, own_trials))
    run_fingerprint = fingerprint(metadata, request.symbols)
    if not has_reason:
        denials.extend(_duplicate_denials(run_fingerprint, trials))
    return Decision(tuple(denials), kind, metadata, run_fingerprint)


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _valid_note(note: str | None) -> bool:
    return note is not None and _has_text(note) and "\n" not in note and "\r" not in note


def _refuse(code: DenialCode, message: str, kind: TrialKind, metadata: dict[str, Any]) -> Decision:
    return Decision((Denial(code, message),), kind, metadata, None)


def _criteria_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    current = card.criteria_hash
    if not own_trials:
        if current == card.registered_criteria_hash:
            return []
        if any(amendment.criteria_hash == current for amendment in card.criteria_amendments):
            return []
        return [
            Denial(
                DenialCode.CRITERIA_CHANGED,
                f"success criteria of {card.id} differ from the registration baseline "
                "and no amendment records the change",
            )
        ]
    last = own_trials[-1]
    if current == last.criteria_hash:
        return []
    if any(amendment.criteria_hash == current for amendment in card.criteria_amendments):
        return []
    return [
        Denial(
            DenialCode.CRITERIA_CHANGED,
            f"success criteria of {card.id} differ from those of {last.trial_id} "
            "and no amendment records the change",
        )
    ]


def _cutoff_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    """The holdout may shrink after trials ran, never grow into data they could not see."""
    if not own_trials:
        return []
    earliest = min(trial.holdout_cutoff for trial in own_trials)
    if card.holdout_cutoff <= earliest:
        return []
    return [
        Denial(
            DenialCode.CUTOFF_MOVED,
            f"holdout cutoff of {card.id} moved to {card.holdout_cutoff} after trials ran "
            f"with {earliest}; it may only move earlier",
        )
    ]


def _apply_window(card: ResearchCard, metadata: dict[str, Any], holdout: bool) -> Denial | None:
    """Clamp an open-ended iteration to the holdout cutoff; refuse windows past it."""
    end = metadata.get("end_date")
    if end is None:
        if not holdout:
            metadata["end_date"] = card.holdout_cutoff.isoformat()
        return None
    try:
        end_day = date.fromisoformat(str(end)[:10])
    except ValueError:
        return Denial(DenialCode.INVALID_WINDOW, f"end_date {end!r} is not an ISO date")
    if end_day > card.holdout_cutoff and not holdout:
        return Denial(
            DenialCode.AFTER_CUTOFF,
            f"end_date {end_day} lies after the holdout cutoff {card.holdout_cutoff} "
            f"of {card.id}; only a holdout run may see that data",
        )
    return None


def _holdout_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    used = [
        str(trial.trial_id)
        for trial in own_trials
        if trial.kind is TrialKind.HOLDOUT and trial.status is TrialStatus.OK
    ]
    if not used:
        return []
    return [
        Denial(
            DenialCode.HOLDOUT_REPEAT,
            f"the holdout of {card.id} was already used ({', '.join(used)}); "
            "a further look needs a reason",
        )
    ]


def _duplicate_denials(run_fingerprint: str, trials: Sequence[Trial]) -> list[Denial]:
    earlier = [
        trial
        for trial in trials
        if trial.fingerprint == run_fingerprint and trial.status is TrialStatus.OK
    ]
    if not earlier:
        return []
    listed = ", ".join(
        f"{trial.trial_id} ({trial.strategy}, run {trial.run_id}, {trial.ts[:10]})"
        for trial in earlier
    )
    return [
        Denial(
            DenialCode.DUPLICATE_FINGERPRINT,
            f"identical configuration already ran: {listed}; pass a reason to run it again",
        )
    ]
