"""Append-only trial ledger (``trials.jsonl``) of the research register."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

LEDGER_FILENAME = "trials.jsonl"
TRIAL_ID_PREFIX = "T-"

Metrics = Mapping[str, Mapping[str, float | int | None]]


class LedgerError(Exception):
    """The ledger contains a line that is not a valid trial."""


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
        return json.dumps(record, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, line: str) -> Trial:
        record = json.loads(line)
        return cls(
            trial_id=record["trial"],
            strategy=record["strategy"],
            run_id=record["run_id"],
            kind=TrialKind(record["kind"]),
            status=TrialStatus(record["status"]),
            ts=record["ts"],
            fingerprint=record["fingerprint"],
            fingerprint_version=record["fingerprint_version"],
            criteria_hash=record["criteria_hash"],
            holdout_cutoff=date.fromisoformat(record["holdout_cutoff"]),
            code=record["code"],
            window=record["window"],
            note=record["note"],
            reason=record.get("reason"),
            metrics=record.get("metrics", {}),
        )


def read_trials(ledger_path: Path) -> list[Trial]:
    """All trials in ledger order; a ledger that does not exist yet has none."""
    if not ledger_path.exists():
        return []
    with ledger_path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return _parse(handle.read())


def append_trial(ledger_path: Path, trial: Trial) -> Trial:
    """Assign the next trial id and append *trial*; safe across processes."""
    if trial.trial_id is not None:
        raise ValueError("append_trial assigns trial ids; got a trial that already has one")
    # ponytail: re-reads the whole ledger under one exclusive lock to find the
    # next id — O(n) per append, fine for thousands of trials; switch to a
    # counter file if the ledger grows far beyond that.
    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read()
        stored = replace(trial, trial_id=_next_id(_parse(text)))
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.write(stored.to_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return stored


def _parse(text: str) -> list[Trial]:
    trials = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            trials.append(Trial.from_json(line))
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(f"{LEDGER_FILENAME} line {number}: {exc!r}") from exc
    return trials


def _next_id(trials: list[Trial]) -> str:
    numbers = [int(t.trial_id.removeprefix(TRIAL_ID_PREFIX)) for t in trials if t.trial_id]
    return f"{TRIAL_ID_PREFIX}{max(numbers, default=0) + 1:06d}"
