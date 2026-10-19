"""Builders for research-register test data."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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


def make_run(
    results_dir: Path,
    run_id: str,
    *,
    name: str = "orb-test",
    timestamp: str = "2026-03-05T10:00:00",
    timeframe: str | None = "MINUTE_15",
    symbols: Mapping[str, Mapping[str, Any] | None] | None = None,
    strategy_extra: Mapping[str, Any] | None = None,
    elite: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write a minimal legacy run directory; ``symbols`` maps symbol -> unified metrics."""
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True)
    strategy: dict[str, Any] = {"name": name, "pipeline": {"indicators": []}, "model": {}}
    if timeframe is not None:
        strategy["timeframe"] = timeframe
    strategy.update(strategy_extra or {})
    (run_dir / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")
    config = {
        "run_id": run_id,
        "timestamp": timestamp,
        "timeframe": timeframe or "HOUR_1",
        "git": {"commit": "abc123", "branch": "main", "dirty": True, "last_commit_message": "x"},
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for symbol, metrics in (symbols if symbols is not None else {"GOLD": {"sharpe": 0.5}}).items():
        folder = run_dir / "grid_details" / symbol
        folder.mkdir(parents=True)
        if metrics is not None:
            (folder / "unified_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if elite:
        (run_dir / "results.json").write_text(
            json.dumps({"elite_results": list(elite)}), encoding="utf-8"
        )
    return run_dir
