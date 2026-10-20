"""Pending trials: a gated run is on record from the moment it starts.

The gate writes ``.staging/<run_id>.json`` (ignored by the register's git)
before the run starts. :func:`finish` turns it into a ledger line. A run
that never records itself — Ctrl-C ends the process through ``os._exit``,
SIGKILL, the OOM killer — leaves its marker behind, and :func:`recover_stale`
records it once its process is gone.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fwbg.research.ledger import (
    LEDGER_FILENAME,
    LedgerError,
    Metrics,
    Trial,
    TrialStatus,
    append_trial,
    read_trials,
)
from fwbg.research.legacy import run_code, run_outcome
from fwbg.research.repo import store_config

STAGING = ".staging"


class PendingError(Exception):
    """A pending marker cannot be written or read."""


@dataclass(frozen=True)
class PendingRun:
    """A gated run whose trial is complete except for its outcome."""

    run_id: str
    # Stored so recovery finds the directory even if RESULTS_BASE_PATH differs later.
    run_dir: Path
    pid: int
    trial: Trial
    config: Mapping[str, Any]


def write_pending(research_dir: Path, pending: PendingRun) -> None:
    """Write the marker atomically; a run id that already has one is refused."""
    path = _path(research_dir, pending.run_id)
    if path.exists():
        raise PendingError(f"run {pending.run_id} is already pending")
    path.parent.mkdir(exist_ok=True)
    record = {
        "run_id": pending.run_id,
        "run_dir": str(pending.run_dir),
        "pid": pending.pid,
        "trial": json.loads(pending.trial.to_json()),
        "config": pending.config,
    }
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_pending(research_dir: Path) -> list[PendingRun]:
    """All markers, oldest run id first."""
    runs = []
    for path in sorted((research_dir / STAGING).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            trial = Trial.from_json(json.dumps(record["trial"]), numbered=False)
            runs.append(
                PendingRun(
                    record["run_id"],
                    Path(record["run_dir"]),
                    int(record["pid"]),
                    trial,
                    record["config"],
                )
            )
        except (OSError, ValueError, KeyError, TypeError, LedgerError) as exc:
            raise PendingError(f"{path.name}: {exc}") from exc
    return runs


def finish(research_dir: Path, pending: PendingRun) -> Trial:
    """Append the run's trial with its outcome and drop the marker; call under the lock."""
    marker = _path(research_dir, pending.run_id)
    # A crash between append and unlink must not record the run twice.
    recorded = next(
        (
            trial
            for trial in read_trials(research_dir / LEDGER_FILENAME)
            if trial.run_id == pending.run_id and trial.strategy == pending.trial.strategy
        ),
        None,
    )
    if recorded is not None:
        marker.unlink()
        return recorded
    status = TrialStatus.FAILED
    metrics: Metrics = {}
    code: Mapping[str, Any] = {}
    if pending.run_dir.is_dir():
        status, metrics = run_outcome(pending.run_dir, count_unknown=True)
        code = _recorded_code(pending.run_dir)
    trial = replace(pending.trial, status=status, metrics=metrics, code=code)
    # Config first: the ledger must never name a fingerprint without its config.
    store_config(research_dir, trial.fingerprint, pending.config)
    stored = append_trial(research_dir / LEDGER_FILENAME, trial)
    marker.unlink()
    return stored


def recover_stale(research_dir: Path) -> list[Trial]:
    """Record every pending run whose process is gone; call under the lock."""
    return [finish(research_dir, run) for run in read_pending(research_dir) if not _alive(run.pid)]


def live_runs(research_dir: Path, card_id: str | None = None) -> list[PendingRun]:
    """Pending runs whose process still runs, optionally of one card."""
    return [
        run
        for run in read_pending(research_dir)
        if _alive(run.pid) and card_id in (None, run.trial.strategy)
    ]


def _alive(pid: int) -> bool:
    # ponytail: a reused pid keeps a stale marker looking alive until that
    # process ends; store the process start time if that ever bites.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recorded_code(run_dir: Path) -> dict[str, Any]:
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return run_code(config) if isinstance(config, dict) else {}


def _path(research_dir: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id in (".", ".."):
        raise PendingError(f"invalid run id {run_id!r}")
    return research_dir / STAGING / f"{run_id}.json"
