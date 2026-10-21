"""Read legacy run directories (``test_results/<run_id>``) for the backfill."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from fwbg_sdk.enums import Timeframe

from fwbg.core.presets import PRESET_SECTIONS, resolve_presets
from fwbg.research.fingerprint import canonical_config, digest
from fwbg.research.ledger import Metrics, TrialKind, TrialStatus
from fwbg.statuses import EXECUTED_STATUSES, NOT_EXECUTED_STATUSES

# The ledger records no per-symbol status: a legacy no_edge symbol shows sharpe 0,
# a new one sharpe null. Neither carries return statistics, so both count towards
# N but never enter the verdict's SR variance (stats.return_samples).
METRIC_KEYS = (
    "sharpe",
    "trades",
    "max_drawdown",
    "win_rate",
    "annual_return",
    "calmar",
    "profit_factor",
    "rrr",
    "sr_per_trade",
    "skew",
    "kurtosis",
    "n_returns",
    "test_period_years",
)
# A symbol with one of these statuses ran to a result and counts towards N
# (conservatively, whether or not it found an edge).
EVALUATED_STATUSES = EXECUTED_STATUSES
NOT_EVALUATED_STATUSES = NOT_EXECUTED_STATUSES
_SMOKE_NAME_RE = re.compile(r"m[0-9]+_smoke.*")
_DATE_PREFIX_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_GIT_KEYS = ("commit", "branch", "dirty", "patch", "patch_error")


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
    presets_from_workspace: bool
    code: Mapping[str, Any]
    window: Mapping[str, str | None]


@dataclass(frozen=True)
class _SymbolFiles:
    """Where one symbol's results live, under the name the run stored it with."""

    result: Path
    # The symbol directory (holding unified_metrics.json and fold_results.json);
    # None for February runs, which wrote a single grid_details/<SYMBOL>.json.
    folder: Path | None


@dataclass(frozen=True)
class _Configuration:
    canonical: dict[str, Any] | None
    unresolved: str | None
    from_workspace: bool


def scan_runs(results_dir: Path, preset_base: Path, skip: Collection[str] = ()) -> list[LegacyRun]:
    """Read every run directory under *results_dir* that has a ``strategy.json``.

    Symlinked run directories are skipped, and so are gated runs (their
    ``config.json`` has a ``research`` block): they are already in the ledger.
    Run ids in *skip* (runs that may still be writing their files) are not read.
    *preset_base* is the workspace strategies directory; it stands in for a
    run's recorded ``_strategy_dir`` when that location no longer holds the
    presets the run referenced.
    """
    if not results_dir.is_dir():
        raise LegacyRunError(f"{results_dir}: not a directory")
    return [
        load_run(run_dir, preset_base)
        for run_dir in sorted(results_dir.iterdir())
        if run_dir.name not in skip
        and run_dir.is_dir()
        and not run_dir.is_symlink()
        and (run_dir / "strategy.json").exists()
        and not _gated(run_dir)
    ]


def _gated(run_dir: Path) -> bool:
    """Whether the run passed the research gate; unreadable configs fail later in load_run."""
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(config, dict) and "research" in config


def run_outcome(run_dir: Path, *, lenient: bool = False) -> tuple[TrialStatus, Metrics]:
    """Status and per-symbol metrics of a run directory, by the ledger's rule.

    ``ok`` iff at least one symbol was evaluated; a holdout look's symbol is
    evaluated iff its holdout was looked at (``holdout.evaluated``). An unknown symbol status or
    an unreadable result file raises :class:`LegacyRunError`; with *lenient*
    such a symbol counts as evaluated (with empty metrics if its files cannot
    be read) and an unreadable elite list is ignored (live runs: never drop a
    trial over an unreadable or unknown symbol).
    """
    metrics = _metrics(run_dir, _symbol_files(run_dir), lenient)
    return (TrialStatus.OK if metrics else TrialStatus.FAILED), metrics


def load_run(run_dir: Path, preset_base: Path) -> LegacyRun:
    """Read one run directory; raise :class:`LegacyRunError` if its files are unusable."""
    strategy = _read_object(run_dir / "strategy.json", run_dir)
    if strategy.get("holdout_cutoff") is not None:
        # Only an ungated run gets here (gated ones are skipped); its metrics are holdout
        # metrics and must never enter the register as development data.
        raise LegacyRunError(
            f"{run_dir.name}/strategy.json: an ungated holdout look (holdout_cutoff set); "
            "it cannot be recorded as a development trial"
        )
    config = _read_object(run_dir / "config.json", run_dir)
    ts = config.get("timestamp")
    if not isinstance(ts, str) or not _DATE_PREFIX_RE.match(ts):
        raise LegacyRunError(f"{run_dir.name}/config.json: no usable timestamp ({ts!r})")
    try:
        run_date = date.fromisoformat(ts[:10])
    except ValueError as exc:
        raise LegacyRunError(f"{run_dir.name}/config.json: bad timestamp {ts!r}") from exc
    name = strategy.get("name")
    name = name if isinstance(name, str) and name.strip() else run_dir.name
    symbols = tuple(sorted(_symbol_files(run_dir)))
    status, metrics = run_outcome(run_dir)
    configuration = _configuration(strategy, config, symbols, preset_base)
    # An unresolvable run still counts as a trial, under a stand-in that never
    # matches a real configuration.
    fingerprint = (
        digest(configuration.canonical)
        if configuration.canonical is not None
        else digest({"unresolvable_legacy_run": run_dir.name})
    )
    return LegacyRun(
        run_id=run_dir.name,
        ts=ts,
        run_date=run_date,
        name=name,
        kind=TrialKind.SMOKE if _SMOKE_NAME_RE.fullmatch(name) else TrialKind.LEGACY,
        status=status,
        symbols=symbols,
        metrics=metrics,
        config=configuration.canonical,
        fingerprint=fingerprint,
        unresolved=configuration.unresolved,
        presets_from_workspace=configuration.from_workspace,
        code=run_code(config),
        window=run_window(strategy),
    )


def _read_object(path: Path, run_dir: Path) -> dict[str, Any]:
    """Read a JSON object or raise a legacy error naming its run-relative path."""
    where = f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LegacyRunError(f"{where}: {exc}") from exc
    if not isinstance(data, dict):
        raise LegacyRunError(f"{where}: expected a JSON object")
    return data


def _optional_object(path: Path, run_dir: Path) -> dict[str, Any] | None:
    """Read an optional JSON object, returning None only when its file is absent."""
    return _read_object(path, run_dir) if path.exists() else None


def _symbol_files(run_dir: Path) -> dict[str, _SymbolFiles]:
    """Recorded (upper-case) symbol -> its files, for directories and February files."""
    details = run_dir / "grid_details"
    if not details.is_dir():
        return {}
    files: dict[str, _SymbolFiles] = {}
    for entry in sorted(details.iterdir()):
        if entry.is_symlink():
            continue
        if entry.is_dir():
            files[entry.name.upper()] = _SymbolFiles(result=entry / "config.json", folder=entry)
        elif entry.is_file() and entry.suffix == ".json":
            # A directory for the same symbol wins over a February-format file.
            files.setdefault(entry.stem.upper(), _SymbolFiles(result=entry, folder=None))
    return files


def _metrics(
    run_dir: Path, files: Mapping[str, _SymbolFiles], lenient: bool = False
) -> dict[str, dict[str, float | int | None]]:
    """Per-symbol metrics for every evaluated symbol (not only the reported elite)."""
    try:
        elite = _elite_by_symbol(run_dir)
    except LegacyRunError:
        if not lenient:
            raise
        elite = {}
    metrics: dict[str, dict[str, float | int | None]] = {}
    for symbol, source in files.items():
        if not source.result.exists():
            continue
        try:
            result = _read_object(source.result, run_dir)
            if _counted(result, run_dir.name, symbol, lenient):
                metrics[symbol] = _symbol_metrics(run_dir, source, result, elite.get(symbol, {}))
        except LegacyRunError:
            if not lenient:
                raise
            # A killed run can leave a truncated file; the symbol was still looked at.
            metrics[symbol] = {}
    return metrics


def _counted(result: Mapping[str, Any], run_id: str, symbol: str, lenient: bool) -> bool:
    """Whether a symbol counts towards N: for a holdout look, iff its holdout was looked at."""
    if "holdout" not in result:
        return _evaluated(result.get("status"), run_id, symbol, lenient)
    looked = _mapping(result["holdout"]).get("evaluated")
    if isinstance(looked, bool):
        return looked
    if lenient:
        return True
    raise LegacyRunError(f"{run_id}: symbol {symbol} has an unreadable holdout block")


def _evaluated(status: object, run_id: str, symbol: str, lenient: bool = False) -> bool:
    """Classify known symbol statuses; unknown ones count with *lenient*, else raise."""
    if isinstance(status, str) and status in EVALUATED_STATUSES:
        return True
    if isinstance(status, str) and status in NOT_EVALUATED_STATUSES:
        return False
    if lenient:
        return True
    # Fail closed: whether such a symbol counts towards N is the operator's call.
    raise LegacyRunError(f"{run_id}: symbol {symbol} has unknown status {status!r}")


def _symbol_metrics(
    run_dir: Path,
    source: _SymbolFiles,
    result: Mapping[str, Any],
    elite: Mapping[str, Any],
) -> dict[str, float | int | None]:
    """Merge historical metric formats, preferring per-symbol unified results."""
    unified = fold = None
    if source.folder is not None:
        unified = _optional_object(source.folder / "unified_metrics.json", run_dir)
        fold = _optional_object(source.folder / "fold_results.json", run_dir)
    candidates = (
        unified,
        result.get("metrics"),
        result.get("holdout_metrics"),
        result.get("selected_metrics"),
    )
    base = next((c for c in candidates if isinstance(c, Mapping)), {})
    values = {key: _number(raw) for key, raw in base.items() if key in METRIC_KEYS}
    walk_forward = _mapping(result.get("walk_forward"))
    fold_walk_forward = _mapping(_mapping(fold).get("walk_forward"))
    if values.get("trades") is None:
        trades = _first_number(
            walk_forward.get("total_trades"), fold_walk_forward.get("total_trades")
        )
        if trades is not None:
            values["trades"] = trades
    values["p_value"] = _first_number(
        _mapping(result.get("monte_carlo")).get("p_value"),
        _mapping(elite.get("monte_carlo")).get("p_value"),
    )
    values["fold_stability"] = _first_number(
        fold_walk_forward.get("fold_stability"),
        walk_forward.get("fold_stability"),
        elite.get("fold_stability"),
    )
    return values


def _elite_by_symbol(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    """Index valid elite result entries by their uppercase symbol names."""
    entries = (_optional_object(run_dir / "results.json", run_dir) or {}).get("elite_results")
    if not isinstance(entries, list):
        return {}
    return {
        entry["symbol"].upper(): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("symbol"), str)
    }


def _mapping(value: object) -> Mapping[str, Any]:
    """Return a mapping unchanged, or an empty mapping for any other value."""
    return value if isinstance(value, Mapping) else {}


def _number(value: object) -> float | int | None:
    """A finite number for the ledger; anything else (NaN, inf, text, bool) is None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _first_number(*values: object) -> float | int | None:
    """Return the first finite numeric candidate, or None if none qualifies."""
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def _configuration(
    strategy: Mapping[str, Any],
    config: Mapping[str, Any],
    symbols: tuple[str, ...],
    preset_base: Path,
) -> _Configuration:
    """The fingerprinted configuration (presets resolved, canonical timeframe)."""
    metadata = dict(strategy)
    from_workspace = False
    try:
        timeframe = metadata.get("timeframe") or config.get("timeframe")
        if timeframe:
            metadata["timeframe"] = Timeframe.from_str(str(timeframe)).canonical
        if not metadata.get("_strategy_dir"):
            # Without a recorded directory, presets would resolve against the current one.
            metadata["_strategy_dir"] = str(preset_base / "configs")
            from_workspace = any(isinstance(metadata.get(key), str) for key in PRESET_SECTIONS)
        try:
            resolved = resolve_presets(metadata)
        except FileNotFoundError:
            # ponytail: presets moved since the run, so all of them re-resolve from the
            # workspace (even if only one is missing) with today's content, not the
            # content the run used; flagged per run. Upgrade: store preset content
            # (or its hash) with each run.
            moved = {**metadata, "_strategy_dir": str(preset_base / "configs")}
            resolved = resolve_presets(moved)
            from_workspace = True
        canonical = canonical_config(resolved, symbols)
        digest(canonical)
    except (OSError, ValueError, TypeError) as exc:
        reason = f"configuration unresolvable: {exc}".replace("\n", " ")
        return _Configuration(canonical=None, unresolved=reason, from_workspace=False)
    return _Configuration(canonical=canonical, unresolved=None, from_workspace=from_workspace)


def run_code(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the recorded Git provenance fields supported by the ledger."""
    git = config.get("git")
    if not isinstance(git, dict):
        return {}
    return {key: git[key] for key in _GIT_KEYS if key in git}


def run_window(strategy: Mapping[str, Any]) -> dict[str, str | None]:
    """Extract textual date bounds, using None for absent or nontext bounds."""
    return {
        "start": (
            strategy.get("start_date") if isinstance(strategy.get("start_date"), str) else None
        ),
        "end": strategy.get("end_date") if isinstance(strategy.get("end_date"), str) else None,
    }
