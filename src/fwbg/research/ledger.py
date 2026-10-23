"""Append-only trial ledger (``trials.jsonl``) of the research register."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn, TypeVar

from fwbg.research.card import CARD_ID_RE
from fwbg.research.fingerprint import DIGEST_RE

LEDGER_FILENAME = "trials.jsonl"
TRIAL_ID_PREFIX = "T-"
# Canonical ids only (as _next_id writes them): six digits, more only without padding.
TRIAL_ID_RE = re.compile(r"T-(?:[0-9]{6}|[1-9][0-9]{6,})")
_FIELDS = frozenset(
    {
        "trial",
        "strategy",
        "run_id",
        "kind",
        "status",
        "ts",
        "fingerprint",
        "fingerprint_version",
        "criteria_hash",
        "holdout_cutoff",
        "code",
        "window",
        "note",
        "reason",
        "metrics",
    }
)
_WINDOW_FIELDS = frozenset({"start", "end"})

Metrics = Mapping[str, Mapping[str, float | int | None]]

_E = TypeVar("_E", bound=Enum)


class LedgerError(Exception):
    """The ledger is missing, unreadable or contains a line that is not a valid trial."""


class TrialKind(str, Enum):
    ITERATION = "iteration"
    HOLDOUT = "holdout"
    LEGACY = "legacy"
    SMOKE = "smoke"


class TrialStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True)
class Trial:
    """One evaluation; ``trial_id`` is assigned by :func:`append_trial`."""

    strategy: str
    run_id: str | None
    kind: TrialKind
    status: TrialStatus
    ts: str
    fingerprint: str
    fingerprint_version: int
    criteria_hash: str
    holdout_cutoff: date
    code: Mapping[str, Any]
    window: Mapping[str, str | None]
    note: str | None
    reason: str | None = None
    metrics: Metrics = field(default_factory=dict)
    trial_id: str | None = None

    @property
    def counted(self) -> bool:
        """Whether this trial counts towards N in multiple-testing corrections."""
        return self.status is TrialStatus.OK and self.kind is not TrialKind.SMOKE

    def to_json(self) -> str:
        record = {
            "trial": self.trial_id,
            "strategy": self.strategy,
            "run_id": self.run_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "ts": self.ts,
            "fingerprint": self.fingerprint,
            "fingerprint_version": self.fingerprint_version,
            "criteria_hash": self.criteria_hash,
            "holdout_cutoff": self.holdout_cutoff.isoformat(),
            "code": dict(self.code),
            "window": dict(self.window),
            "note": self.note,
            "reason": self.reason,
            "metrics": {symbol: dict(values) for symbol, values in self.metrics.items()},
        }
        # ASCII only: raw U+2028/U+2029/NEL would be line breaks to other readers.
        return json.dumps(record, allow_nan=False)

    @classmethod
    def from_json(cls, line: str, *, numbered: bool = True) -> Trial:
        """Parse one stored ledger line; raise :class:`LedgerError` on any defect.

        With *numbered* false the line is a trial before its append (a pending
        run): its ``"trial"`` must be null and ``trial_id`` stays ``None``.
        """
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            raise LedgerError(f"not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise LedgerError(f"expected a JSON object, got {type(record).__name__}")
        unknown = sorted(record.keys() - _FIELDS)
        if unknown:
            raise LedgerError(f"unknown keys: {', '.join(unknown)}")
        missing = sorted(_FIELDS - record.keys())
        if missing:
            raise LedgerError(f"missing keys: {', '.join(missing)}")
        if not numbered and record["trial"] is not None:
            raise LedgerError(f"trial: expected null before the append, got {record['trial']!r}")
        return cls(
            trial_id=_trial_id(record["trial"]) if numbered else None,
            strategy=_strategy(record["strategy"]),
            run_id=_optional_text(record["run_id"], "run_id"),
            kind=_choice(TrialKind, record["kind"], "kind"),
            status=_choice(TrialStatus, record["status"], "status"),
            ts=_text(record["ts"], "ts"),
            fingerprint=_digest(record["fingerprint"], "fingerprint"),
            fingerprint_version=_integer(record["fingerprint_version"], "fingerprint_version"),
            criteria_hash=_digest(record["criteria_hash"], "criteria_hash"),
            holdout_cutoff=_iso_date(record["holdout_cutoff"], "holdout_cutoff"),
            code=_object(record["code"], "code"),
            window=_window(record["window"]),
            note=_optional_text(record["note"], "note"),
            reason=_optional_text(record["reason"], "reason"),
            metrics=_metrics(record["metrics"]),
        )


def read_trials(ledger_path: Path) -> list[Trial]:
    """All trials in ledger order; a missing ledger is an error, not an empty history."""
    try:
        with ledger_path.open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            data = handle.read()
    except FileNotFoundError as exc:
        raise _missing_ledger(ledger_path) from exc
    return _parse(_decode(data))


def append_trial(ledger_path: Path, trial: Trial) -> Trial:
    """Assign the next trial id and append *trial*; safe across processes.

    The ledger must already exist. Callers that also commit the research repo
    must serialise with a separate lock file (e.g. ``.register.lock``), never by
    locking ``trials.jsonl`` themselves: a second ``flock`` on this file from
    the same process deadlocks. Metrics must not contain NaN; convert NaN to
    ``None`` before building the :class:`Trial`.

    Raises:
        ValueError: If *trial* already has an id or cannot be stored as one
            ledger line that reads back unchanged; nothing is written.
        LedgerError: If the ledger is missing or unreadable.
    """
    if trial.trial_id is not None:
        raise ValueError("append_trial assigns trial ids; got a trial that already has one")
    try:
        descriptor = os.open(ledger_path, os.O_RDWR | os.O_APPEND)
    except FileNotFoundError as exc:
        raise _missing_ledger(ledger_path) from exc
    # ponytail: re-reads the whole ledger under one exclusive lock to find the
    # next id — O(n) per append, fine for thousands of trials; switch to a
    # counter file if the ledger grows far beyond that. The first append does
    # not fsync the directory either; add that if a crash right after
    # initialising the register must not lose the ledger file.
    with os.fdopen(descriptor, "r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        text = _decode(handle.read())
        stored = replace(trial, trial_id=_next_id(_parse(text)))
        line = readable_line(stored)
        repair = "\n" if text and not text.endswith("\n") else ""
        handle.write((repair + line + "\n").encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    return stored


def readable_line(trial: Trial) -> str:
    """Serialise *trial*, refusing anything the reader would not read back unchanged.

    :func:`append_trial` writes only lines that pass this check.

    Raises:
        ValueError: *trial* would not be stored as one line that reads back unchanged
            (a trial without an id never does).
    """
    try:
        line = trial.to_json()
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"trial cannot be serialised: {exc}") from exc
    if "\n" in line:
        raise ValueError("trial would be stored as more than one line")
    try:
        read_back = Trial.from_json(line)
    except LedgerError as exc:
        raise ValueError(f"trial would be stored as an unreadable line: {exc}") from exc
    if read_back != trial:
        raise ValueError("trial would not read back unchanged from the ledger")
    return line


def _missing_ledger(ledger_path: Path) -> LedgerError:
    """Build an error explaining how to initialise or restore a missing ledger."""
    return LedgerError(
        f"ledger {ledger_path} does not exist; run 'fwbg research init' for a new register, "
        f"or restore {LEDGER_FILENAME} from the register's git history"
    )


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LedgerError(
            f"{LEDGER_FILENAME} is not valid UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc


def _parse(text: str) -> list[Trial]:
    # Split on "\n" only: str.splitlines() also breaks on U+2028, U+2029 and NEL.
    lines = text.split("\n")
    trials: list[Trial] = []
    seen: set[str | None] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            trial = Trial.from_json(line)
        except LedgerError as exc:
            message = f"{LEDGER_FILENAME} line {number}: {exc}"
            if number == len(lines):
                message += (
                    "; the line has no trailing newline, so the last append was probably"
                    " interrupted: repair the line before any further append"
                )
            raise LedgerError(message) from exc
        if trial.trial_id in seen:
            raise LedgerError(
                f"{LEDGER_FILENAME} line {number}: duplicate trial id {trial.trial_id}"
            )
        seen.add(trial.trial_id)
        trials.append(trial)
    return trials


def _next_id(trials: list[Trial]) -> str:
    numbers = [int(t.trial_id.removeprefix(TRIAL_ID_PREFIX)) for t in trials if t.trial_id]
    return f"{TRIAL_ID_PREFIX}{max(numbers, default=0) + 1:06d}"


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"non-finite number {name} is not allowed")


def _trial_id(value: object) -> str:
    if not isinstance(value, str) or not TRIAL_ID_RE.fullmatch(value):
        raise LedgerError(f"trial: invalid trial id {value!r} (expected T-NNNNNN)")
    return value


def _strategy(value: object) -> str:
    if not isinstance(value, str) or not CARD_ID_RE.fullmatch(value):
        raise LedgerError(f"strategy: invalid research id {value!r} (expected S-NNNN)")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"{field_name}: expected non-empty text, got {value!r}")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise LedgerError(f"{field_name}: expected text or null, got {value!r}")


def _choice(enum: type[_E], value: object, field_name: str) -> _E:
    try:
        return enum(value)
    except ValueError as exc:
        valid = ", ".join(str(member.value) for member in enum)
        raise LedgerError(
            f"{field_name}: invalid value {value!r} (expected one of {valid})"
        ) from exc


def _digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
        raise LedgerError(f"{field_name}: expected sha256:<64 hex digits>, got {value!r}")
    return value


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError(f"{field_name}: expected an integer, got {value!r}")
    return value


def _iso_date(value: object, field_name: str) -> date:
    if not isinstance(value, str):
        raise LedgerError(f"{field_name}: expected an ISO date, got {value!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise LedgerError(f"{field_name}: not an ISO date: {value!r}") from exc


def _object(value: object, field_name: str) -> dict[str, Any]:
    # JSON object keys are always strings.
    if not isinstance(value, dict):
        raise LedgerError(f"{field_name}: expected an object, got {value!r}")
    return value


def _window(value: object) -> dict[str, str | None]:
    window = _object(value, "window")
    if window.keys() != _WINDOW_FIELDS:
        raise LedgerError(f"window: expected exactly start and end, got {sorted(window)}")
    return {key: _optional_text(bound, f"window.{key}") for key, bound in window.items()}


def _metrics(value: object) -> dict[str, dict[str, float | int | None]]:
    symbols = _object(value, "metrics")
    return {
        symbol: _metric_values(values, f"metrics.{symbol}") for symbol, values in symbols.items()
    }


def _metric_values(value: object, field_name: str) -> dict[str, float | int | None]:
    values = _object(value, field_name)
    for name, number in values.items():
        if number is None:
            continue
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise LedgerError(f"{field_name}.{name}: expected a number or null, got {number!r}")
        # Ints are always finite; a float literal such as 1e400 parses to inf.
        if isinstance(number, float) and not math.isfinite(number):
            raise LedgerError(f"{field_name}.{name}: expected a finite number, got {number!r}")
    return values
