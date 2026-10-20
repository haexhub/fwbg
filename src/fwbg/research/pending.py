"""Pending trials: a gated run is on record from the moment it starts.

The gate writes ``.staging/<run_id>.json`` (ignored by the register's git)
before the run starts. :func:`finish` turns it into a ledger line. A run
that never records itself — Ctrl-C ends the process through ``os._exit``,
SIGKILL, the OOM killer — leaves its marker behind, and :func:`recover_stale`
records it once its process is gone.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
from fwbg.research.legacy import LegacyRunError, run_code, run_outcome
from fwbg.research.repo import ResearchRepoError, store_config

STAGING = ".staging"

log = logging.getLogger(__name__)


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
    # Another container shares the register volume but not the PID namespace.
    host: str = field(default_factory=socket.gethostname)


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
        "host": pending.host,
        "trial": json.loads(pending.trial.to_json()),
        "config": pending.config,
    }
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_pending(research_dir: Path) -> list[PendingRun]:
    """All markers, in run id order."""
    runs = []
    for path in sorted((research_dir / STAGING).glob("*.json")):
        try:
            runs.append(_parse_marker(path))
        except (OSError, ValueError, KeyError, TypeError, LedgerError) as exc:
            raise PendingError(f"{path.name}: {exc}") from exc
    return runs


def _parse_marker(path: Path) -> PendingRun:
    record = json.loads(path.read_text(encoding="utf-8"))
    run_id, pid, host = record["run_id"], record["pid"], record["host"]
    if run_id != path.stem:
        raise ValueError(f"run id {run_id!r} does not match the file name")
    if type(pid) is not int or pid <= 0:
        raise ValueError(f"invalid pid {pid!r}")
    if not isinstance(host, str) or not host:
        raise ValueError(f"invalid host {host!r}")
    return PendingRun(
        run_id=run_id,
        run_dir=Path(record["run_dir"]),
        pid=pid,
        trial=Trial.from_json(json.dumps(record["trial"]), numbered=False),
        config=record["config"],
        host=host,
    )


def finish(research_dir: Path, pending: PendingRun) -> Trial:
    """Append the run's trial with its outcome and drop the marker; call under the lock."""
    marker = _path(research_dir, pending.run_id)
    # A crash between append and unlink must not record the run twice; a reused
    # run id with an earlier trial in the ledger is a new run.
    recorded = next(
        (
            trial
            for trial in read_trials(research_dir / LEDGER_FILENAME)
            if trial.run_id == pending.run_id
            and trial.strategy == pending.trial.strategy
            and trial.ts == pending.trial.ts
            and trial.fingerprint == pending.trial.fingerprint
        ),
        None,
    )
    if recorded is not None:
        marker.unlink(missing_ok=True)
        return recorded
    status = TrialStatus.FAILED
    metrics: Metrics = {}
    code: Mapping[str, Any] = {}
    if pending.run_dir.is_dir():
        status, metrics = run_outcome(pending.run_dir, lenient=True)
        code = _recorded_code(pending.run_dir)
    trial = replace(pending.trial, status=status, metrics=metrics, code=code)
    # Config first: the ledger must never name a fingerprint without its config.
    store_config(research_dir, trial.fingerprint, pending.config)
    stored = append_trial(research_dir / LEDGER_FILENAME, trial)
    marker.unlink(missing_ok=True)
    return stored


def recover_stale(research_dir: Path) -> list[Trial]:
    """Record every pending run whose process is gone; call under the lock.

    A marker that cannot be recorded stays for a later attempt and does not
    keep the others from being recorded. Returns the recorded trials.
    """
    recorded = []
    for run in read_pending(research_dir):
        if _alive(run):
            continue
        try:
            recorded.append(finish(research_dir, run))
        except (OSError, ValueError, LedgerError, LegacyRunError, ResearchRepoError) as exc:
            log.warning("pending run %s stays unrecorded: %s", run.run_id, exc)
    return recorded


def live_runs(research_dir: Path, card_id: str | None = None) -> list[PendingRun]:
    """Pending runs whose process still runs, optionally of one card."""
    return [
        run
        for run in read_pending(research_dir)
        if _alive(run) and card_id in (None, run.trial.strategy)
    ]


def _alive(run: PendingRun) -> bool:
    # ponytail: a reused pid keeps a stale marker looking alive until that
    # process ends, and a marker from another host (container) always looks
    # alive, since its pid cannot be checked here; store the process start time
    # and record such markers from their own host (or by hand) if that ever bites.
    if run.host != socket.gethostname():
        return True
    try:
        os.kill(run.pid, 0)
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
