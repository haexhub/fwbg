"""Read legacy run directories (``test_results/<run_id>``) for the backfill."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from fwbg_sdk.enums import Timeframe

from fwbg.core.presets import resolve_presets
from fwbg.research.fingerprint import canonical_config, digest
from fwbg.research.ledger import Metrics, TrialKind, TrialStatus

METRIC_KEYS = (
    "sharpe",
    "trades",
    "max_drawdown",
    "win_rate",
    "annual_return",
    "calmar",
    "profit_factor",
    "rrr",
)
_SMOKE_NAME_RE = re.compile(r"m[0-9]+_smoke")
_GIT_KEYS = ("commit", "branch", "dirty", "patch")


class LegacyRunError(Exception):
    """A run directory cannot be read as a legacy run."""


@dataclass(frozen=True)
class LegacyRun:
    """One stored run, reduced to what a legacy trial records."""

    run_id: str
    ts: str
    run_date: date
    name: str
    kind: TrialKind
    status: TrialStatus
    symbols: tuple[str, ...]
    metrics: Metrics
    config: Mapping[str, Any] | None
    fingerprint: str
    unresolved: str | None
    code: Mapping[str, Any]
    window: Mapping[str, str | None]


def scan_runs(results_dir: Path, preset_base: Path) -> list[LegacyRun]:
    """Read every run directory under *results_dir* that has a ``strategy.json``.

    *preset_base* is the workspace strategies directory; it stands in for a
    run's recorded ``_strategy_dir`` when that location no longer holds the
    presets the run referenced.
    """
    return [
        load_run(run_dir, preset_base)
        for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir())
        if (run_dir / "strategy.json").exists()
    ]


def load_run(run_dir: Path, preset_base: Path) -> LegacyRun:
    """Read one run directory; raise :class:`LegacyRunError` if its files are unusable."""
    strategy = _read_object(run_dir / "strategy.json")
    config = _read_object(run_dir / "config.json")
    ts = config.get("timestamp")
    if not isinstance(ts, str) or len(ts) < 10:
        raise LegacyRunError(f"{run_dir.name}: config.json has no timestamp")
    try:
        run_date = date.fromisoformat(ts[:10])
    except ValueError as exc:
        raise LegacyRunError(f"{run_dir.name}: bad timestamp {ts!r}") from exc
    name = strategy.get("name")
    name = name if isinstance(name, str) and name.strip() else run_dir.name
    symbols = _symbols(run_dir)
    metrics = _metrics(run_dir, symbols)
    canonical, unresolved = _canonical(strategy, config, symbols, preset_base)
    # An unresolvable run still counts as a trial, under a stand-in that never
    # matches a real configuration.
    fingerprint = (
        digest(canonical)
        if canonical is not None
        else digest({"unresolvable_legacy_run": run_dir.name})
    )
    return LegacyRun(
        run_id=run_dir.name,
        ts=ts,
        run_date=run_date,
        name=name,
        kind=TrialKind.SMOKE if _SMOKE_NAME_RE.match(name) else TrialKind.LEGACY,
        status=TrialStatus.OK if metrics else TrialStatus.FAILED,
        symbols=symbols,
        metrics=metrics,
        config=canonical,
        fingerprint=fingerprint,
        unresolved=unresolved,
        code=_code(config),
        window=_window(strategy),
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LegacyRunError(f"{path.parent.name}/{path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise LegacyRunError(f"{path.parent.name}/{path.name}: expected a JSON object")
    return data


def _symbols(run_dir: Path) -> tuple[str, ...]:
    details = run_dir / "grid_details"
    if not details.is_dir():
        return ()
    return tuple(sorted(p.name.upper() for p in details.iterdir() if p.is_dir()))


def _metrics(run_dir: Path, symbols: tuple[str, ...]) -> dict[str, dict[str, float | int | None]]:
    """Per-symbol metrics for every evaluated symbol (not only the reported elite)."""
    elite = _elite_by_symbol(run_dir / "results.json")
    metrics: dict[str, dict[str, float | int | None]] = {}
    for symbol in symbols:
        path = run_dir / "grid_details" / symbol / "unified_metrics.json"
        if not path.exists():
            continue
        values = {
            key: _number(raw) for key, raw in _read_object(path).items() if key in METRIC_KEYS
        }
        extra = elite.get(symbol, {})
        values["p_value"] = _number(extra.get("monte_carlo", {}).get("p_value"))
        values["fold_stability"] = _number(extra.get("fold_stability"))
        metrics[symbol] = values
    return metrics


def _elite_by_symbol(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    entries = _read_object(path).get("elite_results") or []
    return {
        str(entry["symbol"]).upper(): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("symbol"), str)
    }


def _number(value: object) -> float | int | None:
    """A finite number for the ledger; anything else (NaN, inf, text, bool) is None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical(
    strategy: Mapping[str, Any],
    config: Mapping[str, Any],
    symbols: tuple[str, ...],
    preset_base: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """The fingerprinted configuration (presets resolved, canonical timeframe).

    Returns ``(None, reason)`` when the run's configuration cannot be rebuilt.
    """
    metadata = dict(strategy)
    try:
        timeframe = metadata.get("timeframe") or config.get("timeframe")
        if timeframe:
            metadata["timeframe"] = Timeframe.from_str(str(timeframe)).canonical
        try:
            resolved = resolve_presets(metadata)
        except FileNotFoundError:
            # Presets moved since the run: resolve against the workspace instead.
            moved = {**metadata, "_strategy_dir": str(preset_base / "configs")}
            resolved = resolve_presets(moved)
        canonical = canonical_config(resolved, symbols)
        digest(canonical)
        return canonical, None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"configuration unresolvable: {exc}".replace("\n", " ")


def _code(config: Mapping[str, Any]) -> dict[str, Any]:
    git = config.get("git")
    if not isinstance(git, dict):
        return {}
    return {key: git[key] for key in _GIT_KEYS if key in git}


def _window(strategy: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "start": (
            strategy.get("start_date") if isinstance(strategy.get("start_date"), str) else None
        ),
        "end": strategy.get("end_date") if isinstance(strategy.get("end_date"), str) else None,
    }
