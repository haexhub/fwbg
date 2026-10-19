"""Pre-run gate of the research register.

``check`` is pure: the caller loads the card and the ledger, calls ``check``,
and starts the run with ``Decision.effective_metadata`` only if
``Decision.allowed``. The CLI run path and ``POST /api/runs/start`` share it.
"""

from __future__ import annotations

import re
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
    INVALID_CONFIG = "invalid_config"
    NOT_ACTIVE = "not_active"
    CRITERIA_CHANGED = "criteria_changed"
    CUTOFF_MOVED = "cutoff_moved"
    AFTER_CUTOFF = "after_cutoff"
    HOLDOUT_REPEAT = "holdout_repeat"
    HOLDOUT_CONTAMINATED = "holdout_contaminated"
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
        DenialCode.INVALID_CONFIG,
    }
)

# A plain calendar date; ``date.fromisoformat`` alone also accepts "20250430" and week dates.
_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


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
    """Decide whether *request* may run, given its card and the whole ledger.

    *trials* must be in ledger order (append order); ``Trial.ts`` is not used.
    ``end_date`` is exclusive: iterations see ``[.., cutoff)``, a holdout run
    ``[cutoff, ..)``.
    """
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
    if card.id != request.research_id:
        return _refuse(
            DenialCode.CARD_NOT_FOUND,
            f"research card {card.id} was loaded for research id {request.research_id}",
            kind,
            metadata,
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
    if request.holdout and card.holdout_contaminated:
        denials.append(
            Denial(
                DenialCode.HOLDOUT_CONTAMINATED,
                f"{card.id} has no clean historical holdout; "
                "confirm with forward data or paper trading",
            )
        )
    if request.holdout and not has_reason:
        denials.extend(_holdout_denials(card, own_trials))
    try:
        run_fingerprint: str | None = fingerprint(metadata, request.symbols)
    except (ValueError, TypeError) as exc:
        # ValueError: unresolved preset names, NaN; TypeError: values JSON cannot encode.
        denials.append(Denial(DenialCode.INVALID_CONFIG, f"configuration cannot be hashed: {exc}"))
        run_fingerprint = None
    if run_fingerprint is not None and not has_reason:
        denials.extend(_duplicate_denials(run_fingerprint, trials))
    return Decision(tuple(denials), kind, metadata, run_fingerprint)


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _valid_note(note: str | None) -> bool:
    # One line: no line break of any kind (\r, \v, \f, NEL, U+2028, ...), trailing ones included.
    return note is not None and _has_text(note) and note.splitlines() == [note]


def _refuse(code: DenialCode, message: str, kind: TrialKind, metadata: dict[str, Any]) -> Decision:
    return Decision((Denial(code, message),), kind, metadata, None)


def _criteria_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    current = card.criteria_hash
    if card.criteria_amendments:
        # ponytail: a hand-written amendment entry passes this check; the research
        # repo's git history is the audit trail.
        expected, source = card.criteria_amendments[-1].criteria_hash, "the latest amendment"
    else:
        expected, source = card.registered_criteria_hash, "the registration"
    if current != expected:
        return [
            Denial(
                DenialCode.CRITERIA_CHANGED,
                f"success criteria of {card.id} differ from {source}; "
                "record the change as a new criteria amendment",
            )
        ]
    # Guards a hand-edited registration hash.
    if not card.criteria_amendments and own_trials and own_trials[-1].criteria_hash != current:
        return [
            Denial(
                DenialCode.CRITERIA_CHANGED,
                f"success criteria of {card.id} differ from those of {own_trials[-1].trial_id} "
                "and no amendment records the change",
            )
        ]
    return []


def _cutoff_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    """The holdout cutoff is frozen once trials ran.

    Moving it earlier would put data the trials already saw into the holdout;
    moving it later would tune on holdout data, possibly after a holdout look.
    """
    moved = next(
        (trial for trial in own_trials if trial.holdout_cutoff != card.holdout_cutoff), None
    )
    if moved is None:
        return []
    return [
        Denial(
            DenialCode.CUTOFF_MOVED,
            f"holdout cutoff of {card.id} changed from {moved.holdout_cutoff} "
            f"to {card.holdout_cutoff}; it is frozen once trials ran",
        )
    ]


def _iso_day(value: object) -> date | None:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _apply_window(card: ResearchCard, metadata: dict[str, Any], holdout: bool) -> Denial | None:
    """Check the run window against the split; clamp an open-ended iteration to the cutoff."""
    cutoff = card.holdout_cutoff
    raw_start, raw_end = metadata.get("start_date"), metadata.get("end_date")
    start, end = _iso_day(raw_start), _iso_day(raw_end)
    for key, raw, day in (("start_date", raw_start, start), ("end_date", raw_end, end)):
        if raw is not None and day is None:
            return Denial(DenialCode.INVALID_WINDOW, f"{key} {raw!r} is not a YYYY-MM-DD date")
    if end is None and not holdout:
        end = cutoff
        metadata["end_date"] = cutoff.isoformat()
    if end is None:
        return None
    if not holdout and end > cutoff:
        return Denial(
            DenialCode.AFTER_CUTOFF,
            f"end_date {end} lies after the holdout cutoff {cutoff} "
            f"of {card.id}; only a holdout run may see that data",
        )
    if holdout and end <= cutoff:
        return Denial(
            DenialCode.INVALID_WINDOW,
            f"a holdout run must include data after the cutoff {cutoff}; end_date is {end}",
        )
    if start is not None and start >= end:
        return Denial(
            DenialCode.INVALID_WINDOW, f"start_date {start} must lie before end_date {end}"
        )
    return None


def _holdout_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    # Any look counts, failed ones too: a failed run may already have shown results.
    used = [str(trial.trial_id) for trial in own_trials if trial.kind is TrialKind.HOLDOUT]
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
    # ponytail: a FINGERPRINT_VERSION bump stops matching older trials; upgrade path:
    # re-fingerprint the stored configs/<fp>.json files on a bump.
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
