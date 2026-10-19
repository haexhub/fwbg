"""Builders for research-register test data."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fwbg.research.card import ResearchCard, Status, criteria_hash
from fwbg.research.ledger import Trial, TrialKind, TrialStatus, append_trial


def make_card(**overrides: Any) -> ResearchCard:
    values: dict[str, Any] = {
        "id": "S-0007",
        "title": "CUSUM event continuation",
        "status": Status.ACTIVE,
        "tags": ("event",),
        "created": date(2026, 10, 19),
        "holdout_cutoff": date(2025, 4, 30),
        "holdout_contaminated": False,
        "success_criteria": {"min_sharpe": 1.0, "min_trades": 100.0},
        "registered_criteria_hash": criteria_hash({"min_sharpe": 1.0, "min_trades": 100.0}),
        "criteria_amendments": (),
        "body": "",
    }
    values.update(overrides)
    return ResearchCard(**values)


def make_trial(**overrides: Any) -> Trial:
    values: dict[str, Any] = {
        "strategy": "S-0007",
        "run_id": "run-1",
        "kind": TrialKind.ITERATION,
        "status": TrialStatus.OK,
        "ts": "2026-10-19T10:00:00",
        "fingerprint": "sha256:" + "0" * 64,
        "fingerprint_version": 1,
        "criteria_hash": make_card().criteria_hash,
        "holdout_cutoff": date(2025, 4, 30),
        "code": {"commit": "abc123", "dirty": False, "patch": None},
        "window": {"start": None, "end": "2025-04-30"},
        "note": "baseline",
    }
    values.update(overrides)
    return Trial(**values)


def append_many(ledger: str, worker: int, count: int) -> None:
    """Process target for concurrency tests: append *count* trials."""
    for index in range(count):
        append_trial(Path(ledger), make_trial(run_id=f"w{worker}-{index}"))
