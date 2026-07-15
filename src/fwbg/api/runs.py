"""Run management endpoints."""
import json
import logging
import os
import signal
import statistics
import subprocess
import sys
import hashlib
import threading
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from fwbg.api.deps import get_strategies_dir, get_test_results_dir
from fwbg.api._paths import (
    safe_results_path as _safe_results_path,
    validate_id as _validate_id,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

# Track active background processes. The lock guards every read/write because
# multiple FastAPI worker threads can hit the run endpoints concurrently.
_active_jobs: dict[str, dict] = {}
_active_jobs_lock = threading.Lock()

# Limit concurrent CLI subprocesses to prevent resource exhaustion via spam.
MAX_CONCURRENT_RUNS = int(os.environ.get("FWBG_MAX_CONCURRENT_RUNS", "10"))


def _spawn_cli_process(cmd: list[str], env: dict, run_dir) -> tuple:
    """Start the CLI with stdout/stderr redirected to files in the run dir.

    NEVER use subprocess.PIPE here without a reader: nobody drains the pipes
    while the run is alive, so once the CLI has written ~64KB the OS pipe is
    full and the CLI blocks on write. That freezes its progress display
    thread, which holds the display lock, which stalls the progress-queue
    reader, which fills the workers' progress pipe — the whole backtest
    deadlocks. Long runs hit this reliably; short ones stay under the buffer,
    which is why it went unnoticed.
    """
    from pathlib import Path

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "cli_stdout.log"
    stderr_path = run_dir / "cli_stderr.log"
    with open(stdout_path, "wb") as stdout_f, open(stderr_path, "wb") as stderr_f:
        process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
    return process, stdout_path, stderr_path


def _job_duration_seconds(job: dict) -> Optional[float]:
    """Elapsed time since a job started, for still-running/just-finished jobs."""
    started_at = job.get("started_at")
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    return (datetime.now() - start).total_seconds()


def _job_error_output(job: dict, limit: int = 500) -> str:
    """Tail of the CLI's stderr (or stdout) for failure messages."""
    from pathlib import Path

    for key in ("stderr_path", "stdout_path"):
        path = job.get(key)
        if not path:
            continue
        try:
            output = Path(path).read_text(errors="replace").strip()
        except OSError:
            continue
        if output:
            return output[-limit:]
    return ""


class RunStartRequest(BaseModel):
    """Request body for starting a run."""
    strategy_name: str
    assets: Optional[list[str]] = None
    asset_classes: Optional[list[str]] = None
    description: Optional[str] = None
    preview: Optional[bool] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    cost_multiplier: Optional[float] = None


class PreviewRequest(BaseModel):
    """Request body for signal preview (no optimization)."""
    strategy_name: str
    symbol: str
    datasource: Optional[str] = None
    tp: Optional[float] = None
    sl: Optional[float] = None
    last_n_bars: Optional[int] = Field(default=None, ge=1)


@router.post("/start")
def start_run(body: RunStartRequest) -> dict:
    """Start a strategy optimization run in the background."""
    _validate_id(body.strategy_name, "strategy_name")
    if body.assets:
        for a in body.assets:
            _validate_id(a, "asset")
    if body.asset_classes:
        for c in body.asset_classes:
            _validate_id(c, "asset_class")

    with _active_jobs_lock:
        # Refresh stale statuses first: a job whose process already exited
        # must not occupy a slot. Statuses are otherwise only updated when a
        # status endpoint happens to be polled — with a concurrency limit of
        # 1 a stale "running" would block every future run.
        for j in _active_jobs.values():
            proc = j.get("process")
            if j.get("status") == "running" and proc and proc.poll() is not None:
                j["status"] = "completed" if proc.returncode == 0 else "failed"
        running = sum(1 for j in _active_jobs.values() if j.get("status") == "running")
    if running >= MAX_CONCURRENT_RUNS:
        raise HTTPException(429, f"Too many active runs (limit {MAX_CONCURRENT_RUNS})")

    strategies_dir = get_strategies_dir()
    strategy_file = strategies_dir / f"{body.strategy_name}.json"

    if not strategy_file.exists():
        raise HTTPException(404, f"Strategy not found: {body.strategy_name}")

    # Build the fwbg CLI command
    cmd = [sys.executable, "-m", "fwbg.cli", "--strategy-file", str(strategy_file)]

    if body.assets:
        cmd.extend(["--assets", ",".join(body.assets)])
    if body.asset_classes:
        cmd.extend(["--asset-classes", ",".join(body.asset_classes)])
    if body.description:
        cmd.extend(["-d", body.description])
    if body.start_date:
        cmd.extend(["--start-date", body.start_date])
    if body.end_date:
        cmd.extend(["--end-date", body.end_date])
    if body.cost_multiplier is not None:
        cmd.extend(["--cost-multiplier", str(body.cost_multiplier)])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.md5(timestamp.encode()).hexdigest()[:6]
    job_id = f"{timestamp}_{short_hash}"
    cmd.extend(["--run-id", job_id])

    try:
        from fwbg.api.workspace import get_workspace
        env = os.environ.copy()
        env.setdefault("FWBG_WORKSPACE", str(get_workspace()))
        process, stdout_path, stderr_path = _spawn_cli_process(
            cmd, env, get_test_results_dir() / job_id
        )

        with _active_jobs_lock:
            _active_jobs[job_id] = {
                "job_id": job_id,
                "pid": process.pid,
                "process": process,
                "strategy_name": body.strategy_name,
                "status": "running",
                "started_at": datetime.now().isoformat(),
                "cmd": cmd,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
            }
    except Exception as e:
        log.exception("Failed to start run")
        raise HTTPException(500, "Failed to start run") from e

    return {
        "job_id": job_id,
        "status": "running",
        "strategy_name": body.strategy_name,
        "pid": process.pid,
    }


@router.post("/preview")
def preview_signals(body: PreviewRequest) -> dict:
    """Preview entry/exit signals without running full optimization.

    Returns trades in the same format as /runs/{id}/trades/{symbol}
    so the frontend can display them without format conversion.
    """
    from fwbg.core.config import StrategyConfig
    from fwbg.core.data_sources import get_data_source, CSVSourceConfig
    from fwbg.data.loader import load_data_aligned
    from fwbg.data.assets import get_asset
    from fwbg.data.config import convert_numpy
    from fwbg.pipeline.features import compute_indicator_pool

    strategies_dir = get_strategies_dir()
    strategy_file = strategies_dir / f"{body.strategy_name}.json"
    if not strategy_file.exists():
        raise HTTPException(404, f"Strategy not found: {body.strategy_name}")

    strategy = StrategyConfig.from_json_file(str(strategy_file))
    timeframe = strategy.timeframe or "HOUR"

    # Resolve datasource → CSV path
    ds_name = body.datasource or strategy.datasource
    if not ds_name:
        raise HTTPException(400, "No datasource configured (set in strategy or request)")

    try:
        ds = get_data_source(ds_name)
    except ValueError:
        raise HTTPException(404, f"Datasource not found: {ds_name}")

    if not isinstance(ds, CSVSourceConfig) or not ds.exists():
        raise HTTPException(400, f"Datasource '{ds_name}' is not a CSV source or path missing")

    symbol = body.symbol
    csv_path = ds.get_file_path(symbol, timeframe)
    if not csv_path.exists():
        available = [f.stem.rsplit(f"_{timeframe}", 1)[0]
                     for f in ds.list_files(f"*_{timeframe}.csv")]
        raise HTTPException(404, f"No data for {symbol}/{timeframe}. "
                          f"Available symbols: {sorted(available)}")

    # Load data
    df = load_data_aligned(str(csv_path))
    if df is None:
        raise HTTPException(500, f"Failed to load data from {csv_path}")

    if body.last_n_bars and len(df) > body.last_n_bars:
        df = df.tail(body.last_n_bars)

    # Compute indicators
    indicators = strategy.get_indicators()
    if not indicators:
        raise HTTPException(400, "Strategy has no indicators configured")

    df_ind = compute_indicator_pool(df, indicators=indicators)

    # Collect signal columns: explicit from model config + grid variants
    long_signals, short_signals = _collect_signal_columns(strategy)

    # Filter to columns that actually exist in the computed DataFrame
    all_configured = long_signals | short_signals
    long_signals = {c for c in long_signals if c in df_ind.columns}
    short_signals = {c for c in short_signals if c in df_ind.columns}

    # Fallback: auto-discover signal columns from indicator plugins
    if not long_signals and not short_signals:
        long_signals, short_signals = _discover_signal_columns(indicators, df_ind)

    existing_signals = long_signals | short_signals
    if not existing_signals:
        raise HTTPException(400, f"No signal columns found. "
                          f"Configured: {all_configured or 'none'}")

    # Build trades list (frontend-compatible format)
    trades = _extract_entry_signals(df_ind, long_signals, short_signals)

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_bars": len(df_ind),
        "trades": convert_numpy(trades),
    }

    # Simulate trades with TP/SL if possible
    asset = get_asset(symbol)
    ep = strategy.exit_params
    tp_list = ep.get("tp_mult", [])
    sl_list = ep.get("sl_mult", [])
    tp_val = body.tp or (float(statistics.median(tp_list)) if tp_list else None)
    sl_val = body.sl or (float(statistics.median(sl_list)) if sl_list else None)

    if tp_val is not None and sl_val is not None:
        sim_trades = _simulate_preview_trades(
            df_ind, long_signals, short_signals, tp_val, sl_val, asset, strategy,
        )
        if sim_trades:
            result["trades"] = convert_numpy(sim_trades)
            result["tp_used"] = tp_val
            result["sl_used"] = sl_val

    return result


def _collect_signal_columns(
    strategy,
) -> tuple[set[str], set[str]]:
    """Collect signal columns from signal_rules."""
    long_signals: set[str] = set()
    short_signals: set[str] = set()

    signal_rules = getattr(strategy, "signal_rules", None) or {}
    for direction, target in (("long", long_signals), ("short", short_signals)):
        rules = signal_rules.get(direction) or {}
        for cond in rules.get("conditions", []):
            col = cond.get("column") or cond.get("column_a", "")
            if col:
                target.add(col)

    return long_signals, short_signals


def _discover_signal_columns(
    indicators: list, df_ind,
) -> tuple[set[str], set[str]]:
    """Auto-discover signal columns from DataFrame by suffix heuristic.

    Signal columns have known suffixes (e.g. _retest_bull, _breakout_up).
    Columns may have parameter prefixes (rb1_cf0_prb0_orb_s08_breakout_up).
    """
    _LONG_SUFFIXES = ("_bull", "_breakout_up", "_retest_up")
    _SHORT_SUFFIXES = ("_bear", "_breakout_down", "_retest_down")
    base_cols = {"O", "H", "L", "C", "V"}

    long_signals: set[str] = set()
    short_signals: set[str] = set()

    for col in df_ind.columns:
        if col in base_cols or col.startswith("_"):
            continue
        if any(col.endswith(s) for s in _LONG_SUFFIXES):
            long_signals.add(col)
        elif any(col.endswith(s) for s in _SHORT_SUFFIXES):
            short_signals.add(col)

    return long_signals, short_signals


def _extract_entry_signals(
    df_ind, long_signals: set[str], short_signals: set[str],
) -> list[dict]:
    """Extract entry signal bars as frontend-compatible trade entries."""
    trades = []
    for col in sorted(long_signals | short_signals):
        mask = df_ind[col] > 0.5
        signal_bars = df_ind.index[mask]
        direction = "LONG" if col in long_signals else "SHORT"

        for ts in signal_bars:
            bar_idx = df_ind.index.get_loc(ts)
            if bar_idx + 1 < len(df_ind):
                price = float(df_ind["O"].iloc[bar_idx + 1])
                entry_time = str(df_ind.index[bar_idx + 1])
            else:
                continue  # No next bar for entry

            trades.append({
                "entry_time": entry_time,
                "entry_price": price,
                "direction": direction,
                "signal": col,
            })

    trades.sort(key=lambda t: t["entry_time"])
    return trades


def _simulate_preview_trades(
    df_ind, long_signals: set[str], short_signals: set[str],
    tp: float, sl: float, asset, strategy,
) -> list[dict]:
    """Simulate trades with fixed TP/SL for preview."""
    from fwbg.core.context import SimulationContext
    from fwbg.optimization.targets import _simulate_trades_core

    # Use first long/short signal column for simulation
    sig_long = next(iter(sorted(long_signals)), "")
    sig_short = next(iter(sorted(short_signals)), "")
    if not sig_long and not sig_short:
        return []

    ctx = SimulationContext.create(asset, strategy)
    n = len(df_ind)

    probs_long = None
    probs_short = None
    if sig_long and sig_long in df_ind.columns:
        probs_long = np.zeros((n, 2), dtype=np.float64)
        probs_long[:, 1] = df_ind[sig_long].fillna(0).clip(0, 1).values
        probs_long[:, 0] = 1.0 - probs_long[:, 1]

    if sig_short and sig_short in df_ind.columns:
        probs_short = np.zeros((n, 2), dtype=np.float64)
        probs_short[:, 1] = df_ind[sig_short].fillna(0).clip(0, 1).values
        probs_short[:, 0] = 1.0 - probs_short[:, 1]

    sim_result = _simulate_trades_core(
        df_ind,
        probs_long=probs_long,
        probs_short=probs_short,
        long_win_idx=1 if probs_long is not None else None,
        short_win_idx=1 if probs_short is not None else None,
        ct_long=0.5,
        ct_short=0.5,
        tp=int(tp),
        sl=int(sl),
        ctx=ctx,
        return_detailed=True,
    )

    # Convert to frontend-compatible format
    trades = []
    for t in sim_result.get("trades_detailed", []):
        trades.append({
            "entry_time": str(t.get("entry_time", "")),
            "exit_time": str(t.get("exit_time", "")),
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "direction": "LONG" if t.get("direction") == 1 else "SHORT",
            "result": t.get("result"),
            "pnl_raw": t.get("pnl_raw"),
        })
    return trades


class CompareRequest(BaseModel):
    """Request body for comparing multiple runs."""
    run_ids: list[str]


@router.post("/compare")
def compare_runs(body: CompareRequest) -> dict:
    """Compare multiple runs side-by-side with per-asset metrics."""
    results_dir = get_test_results_dir()

    runs = []
    all_symbols: set[str] = set()

    for run_id in body.run_ids:
        run_dir = results_dir / run_id
        if not run_dir.exists():
            continue

        run_data: dict = {"run_id": run_id}

        # Config
        config_file = run_dir / "config.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
                run_data["timestamp"] = config.get("timestamp")
                run_data["description"] = config.get("description")
                run_data["timeframe"] = config.get("timeframe")
            except (json.JSONDecodeError, IOError):
                pass

        # Strategy
        strategy_file = run_dir / "strategy.json"
        if strategy_file.exists():
            try:
                strategy = json.loads(strategy_file.read_text())
                _resolve_strategy_refs(strategy)
                run_data["strategy"] = strategy
                run_data["strategy_name"] = strategy.get("name", "")
            except (json.JSONDecodeError, IOError):
                pass

        # Per-asset metrics + trades
        grid_dir = run_dir / "grid_details"
        assets: dict[str, dict] = {}
        if grid_dir.exists():
            for sym_dir in sorted(d for d in grid_dir.iterdir() if d.is_dir()):
                symbol = sym_dir.name
                all_symbols.add(symbol)
                asset_data: dict = {"symbol": symbol}

                # Unified metrics
                um_file = sym_dir / "unified_metrics.json"
                if um_file.exists():
                    try:
                        asset_data["metrics"] = json.loads(um_file.read_text())
                    except (json.JSONDecodeError, IOError):
                        pass

                # Config (status, best_config)
                cfg_file = sym_dir / "config.json"
                if cfg_file.exists():
                    try:
                        cfg = json.loads(cfg_file.read_text())
                        asset_data["status"] = cfg.get("status")
                        asset_data["best_config"] = cfg.get("best_config")
                    except (json.JSONDecodeError, IOError):
                        pass

                # Trade trace for equity overlay
                trades_file = sym_dir / "trades.json"
                if trades_file.exists():
                    try:
                        tdata = json.loads(trades_file.read_text())
                        asset_data["tr_trace"] = tdata.get("tr_trace", [])
                    except (json.JSONDecodeError, IOError):
                        pass

                # Fold stability from fold_results
                fold_file = sym_dir / "fold_results.json"
                if fold_file.exists():
                    try:
                        fdata = json.loads(fold_file.read_text())
                        wf = fdata.get("walk_forward", {})
                        asset_data["fold_summary"] = {
                            "n_folds": wf.get("n_folds"),
                            "successful_folds": wf.get("successful_folds"),
                            "profitable_folds": wf.get("profitable_folds"),
                            "fold_stability": wf.get("fold_stability"),
                            "mean_win_rate": wf.get("mean_win_rate"),
                            "std_win_rate": wf.get("std_win_rate"),
                            "mean_pnl": wf.get("mean_pnl"),
                            "std_pnl": wf.get("std_pnl"),
                        }
                    except (json.JSONDecodeError, IOError):
                        pass

                assets[symbol] = asset_data

        run_data["assets"] = assets

        # Aggregate metrics across assets
        metrics_list = [a["metrics"] for a in assets.values() if "metrics" in a]
        if metrics_list:
            run_data["aggregate"] = {
                "total_pnl": sum(m.get("pnl", 0) for m in metrics_list),
                "avg_win_rate": sum(m.get("win_rate", 0) for m in metrics_list) / len(metrics_list),
                "avg_sharpe": sum(m.get("sharpe", 0) for m in metrics_list) / len(metrics_list),
                "avg_calmar": sum(m.get("calmar", 0) for m in metrics_list) / len(metrics_list),
                "avg_profit_factor": sum(m.get("profit_factor", 0) for m in metrics_list) / len(metrics_list),
                "total_trades": sum(m.get("trades", 0) for m in metrics_list),
                "asset_count": len(assets),
                "profitable_count": sum(1 for a in assets.values() if a.get("status") == "ok"),
            }

        runs.append(run_data)

    return {
        "runs": runs,
        "all_symbols": sorted(all_symbols),
    }


@router.get("")
def list_runs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    """List completed and active runs with pagination."""
    results_dir = get_test_results_dir()
    runs = []

    # Active jobs — reap finished processes (snapshot under lock to avoid
    # mutating the dict while iterating from another thread).
    finished_ids = []
    with _active_jobs_lock:
        jobs_snapshot = list(_active_jobs.items())

    for job_id, job in jobs_snapshot:
        proc = job.get("process")
        if proc and proc.poll() is not None:
            if proc.returncode == 0:
                job["status"] = "completed"
            else:
                job["status"] = "failed"
                if "error_message" not in job:
                    output = _job_error_output(job)
                    job["error_message"] = output or f"Process exited with code {proc.returncode}"

            if (results_dir / job_id).exists():
                finished_ids.append(job_id)
                continue

        runs.append({
            "run_id": job_id,
            "status": job["status"],
            "strategy_name": job.get("strategy_name"),
            "started_at": job.get("started_at"),
            "is_active": job["status"] == "running",
            "error_message": job.get("error_message"),
            "duration_seconds": _job_duration_seconds(job),
        })

    with _active_jobs_lock:
        for jid in finished_ids:
            _active_jobs.pop(jid, None)
        active_ids = set(_active_jobs.keys())

    # Collect completed run directory names (cheap: no file reads yet)
    completed_ids: list[str] = []
    if results_dir.exists():
        completed_ids = sorted(
            (d.name for d in results_dir.iterdir()
             if d.is_dir() and d.name not in active_ids),
            reverse=True,
        )

    total = len(runs) + len(completed_ids)

    # Paginate: active jobs come first, then completed runs by name (desc)
    active_count = len(runs)
    if offset < active_count:
        # Page starts within active jobs
        remaining = limit - (active_count - offset)
        runs = runs[offset:]
        if remaining > 0:
            completed_ids = completed_ids[:remaining]
        else:
            completed_ids = []
    else:
        # Page starts within completed runs
        runs = []
        skip = offset - active_count
        completed_ids = completed_ids[skip:skip + limit]

    # Only read JSON files for the paginated completed runs
    for run_id in completed_ids:
        run_dir = results_dir / run_id
        run_info: dict = {"run_id": run_id, "status": "completed"}

        config_file = run_dir / "config.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
                run_info.update({
                    "timestamp": config.get("timestamp"),
                    "description": config.get("description"),
                    "timeframe": config.get("timeframe"),
                })
            except (json.JSONDecodeError, IOError):
                pass

        strategy_file = run_dir / "strategy.json"
        if strategy_file.exists():
            try:
                strategy = json.loads(strategy_file.read_text())
                run_info["strategy_name"] = strategy.get("name", "")
                run_info["tags"] = strategy.get("tags", [])
            except (json.JSONDecodeError, IOError):
                pass

        progress_file = run_dir / "progress.json"
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text())
                run_info["duration_seconds"] = progress.get("elapsed_seconds")
            except (json.JSONDecodeError, IOError):
                pass

        grid_dir = run_dir / "grid_details"
        if grid_dir.exists():
            sym_dirs = [d for d in grid_dir.iterdir() if d.is_dir()]
            run_info["asset_count"] = len(sym_dirs)

            profitable = 0
            for sd in sym_dirs:
                cfg_file = sd / "config.json"
                if cfg_file.exists():
                    try:
                        data = json.loads(cfg_file.read_text())
                        if data.get("status") == "ok":
                            profitable += 1
                    except (json.JSONDecodeError, IOError):
                        pass
            run_info["profitable_count"] = profitable

        runs.append(run_info)

    return {"items": runs, "total": total}


def _resolve_strategy_refs(strategy: dict) -> None:
    """Resolve string references in strategy data to their preset content.

    Mutates the dict in-place.  For example ``"pipeline": "orb_scalping_v1"``
    is replaced with the contents of ``strategies/pipelines/orb_scalping_v1.json``.
    """
    strategies_dir = get_strategies_dir()
    for key, subdir in [
        ("pipeline", "pipelines"),
        ("model", "models"),
        ("validation", "validations"),
        ("filters", "filters"),
        ("resources", "resources"),
        ("risk_params", "risk_params"),
    ]:
        value = strategy.get(key)
        if not isinstance(value, str):
            continue
        preset_dir = strategies_dir.parent / subdir
        preset_path = preset_dir / f"{value}.json"
        if preset_path.exists():
            try:
                strategy[key] = json.loads(preset_path.read_text())
            except (json.JSONDecodeError, IOError):
                pass


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    """Get detailed results for a completed run."""
    _validate_id(run_id, "run_id")
    run_dir = _safe_results_path(run_id)

    if not run_dir.exists():
        # Check active jobs
        with _active_jobs_lock:
            job = _active_jobs.get(run_id)
            if job is not None:
                proc = job.get("process")
                if proc and proc.poll() is not None:
                    job["status"] = "completed" if proc.returncode == 0 else "failed"
                return {
                    "run_id": run_id,
                    "status": job["status"],
                    "strategy_name": job.get("strategy_name"),
                    "started_at": job.get("started_at"),
                }
        raise HTTPException(404, f"Run not found: {run_id}")

    result: dict = {"run_id": run_id, "status": "completed"}

    # Load config
    config_file = run_dir / "config.json"
    if config_file.exists():
        try:
            result["config"] = json.loads(config_file.read_text())
        except (json.JSONDecodeError, IOError):
            pass

    # Load strategy (resolve string references like "pipeline": "orb_scalping_v1")
    strategy_file = run_dir / "strategy.json"
    if strategy_file.exists():
        try:
            strategy = json.loads(strategy_file.read_text())
            _resolve_strategy_refs(strategy)
            result["strategy"] = strategy
        except (json.JSONDecodeError, IOError):
            pass

    # Load per-asset results (subdirectories per symbol)
    grid_dir = run_dir / "grid_details"
    if grid_dir.exists():
        assets = {}
        for sd in sorted(d for d in grid_dir.iterdir() if d.is_dir()):
            symbol = sd.name
            try:
                cfg_data = json.loads((sd / "config.json").read_text()) if (sd / "config.json").exists() else {}
                fold_data = json.loads((sd / "fold_results.json").read_text()) if (sd / "fold_results.json").exists() else {}
                um_data = json.loads((sd / "unified_metrics.json").read_text()) if (sd / "unified_metrics.json").exists() else {}
                assets[symbol] = {
                    "symbol": symbol,
                    "status": cfg_data.get("status", "unknown"),
                    "total_combinations": cfg_data.get("total_combinations", 0),
                    "unified_metrics": um_data,
                    "walk_forward": _summarize_walk_forward(fold_data.get("walk_forward", {})),
                }
            except (json.JSONDecodeError, IOError):
                pass
        result["assets"] = assets

    return result


def _summarize_walk_forward(wf: dict) -> dict:
    """Extract key metrics from walk-forward results."""
    if not wf:
        return {}
    return {
        "n_folds": wf.get("n_folds"),
        "successful_folds": wf.get("successful_folds"),
        "mean_win_rate": wf.get("mean_win_rate"),
        "mean_pnl": wf.get("mean_pnl"),
        "total_trades": wf.get("total_trades"),
    }


@router.get("/{run_id}/grid_details")
def list_grid_details(run_id: str) -> list[str]:
    """List asset symbols with grid details for a completed run."""
    _validate_id(run_id, "run_id")
    grid_dir = _safe_results_path(run_id, "grid_details")

    if not grid_dir.exists():
        raise HTTPException(404, f"No grid details for run: {run_id}")

    return sorted(d.name for d in grid_dir.iterdir() if d.is_dir())


@router.get("/{run_id}/grid_details/{symbol}")
def get_grid_detail(run_id: str, symbol: str) -> dict:
    """Get merged grid detail for a symbol (config + fold_results + grid_results)."""
    _validate_id(run_id, "run_id")
    _validate_id(symbol, "symbol")
    sym_dir = _safe_results_path(run_id, "grid_details", symbol)

    if not sym_dir.exists() or not sym_dir.is_dir():
        raise HTTPException(404, f"Grid detail not found: {run_id}/{symbol}")

    merged = {}
    for fname in ("config.json", "fold_results.json", "grid_results.json", "unified_metrics.json"):
        fpath = sym_dir / fname
        if fpath.exists():
            try:
                merged.update(json.loads(fpath.read_text()))
            except (json.JSONDecodeError, IOError):
                pass

    if not merged:
        raise HTTPException(404, f"Grid detail not found: {run_id}/{symbol}")

    return merged


@router.get("/{run_id}/trades/{symbol}")
def get_run_symbol_trades(run_id: str, symbol: str) -> dict:
    """Return all detailed trades for a specific symbol.

    Returns unified simulation trades from trades.json if available,
    otherwise extracts fold-level trades from fold_results.json.
    """
    _validate_id(run_id, "run_id")
    _validate_id(symbol, "symbol")
    sym_dir = _safe_results_path(run_id, "grid_details", symbol)

    if not sym_dir.exists() or not sym_dir.is_dir():
        raise HTTPException(404, f"No grid detail found for symbol: {symbol}")

    trades: list[dict] = []

    # Prefer unified simulation trades
    trades_file = sym_dir / "trades.json"
    if trades_file.exists():
        try:
            tdata = json.loads(trades_file.read_text())
            trades = tdata.get("trades_detailed", tdata.get("tr_trace", []))
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback: fold-level trades from fold_results.json
    if not trades:
        fold_file = sym_dir / "fold_results.json"
        if fold_file.exists():
            try:
                fdata = json.loads(fold_file.read_text())
                for fold in fdata.get("walk_forward", {}).get("fold_details", []):
                    fold_id = fold.get("fold_id")
                    detail_trades = fold.get("test_trades_detail", [])
                    if detail_trades:
                        for trade in detail_trades:
                            if isinstance(trade, dict) and "entry_time" in trade:
                                trades.append({**trade, "fold_id": fold_id})
                    else:
                        for trade in fold.get("test_trades_trace", []):
                            if isinstance(trade, dict) and "entry_time" in trade:
                                trades.append({**trade, "fold_id": fold_id})
            except (json.JSONDecodeError, IOError):
                pass

    return {"symbol": symbol, "run_id": run_id, "trades": trades}


@router.get("/{run_id}/progress")
def get_run_progress(run_id: str) -> dict:
    """Get progress for an active or completed run.

    Reads the progress.json file written by RunProgressWriter.
    Falls back to basic job info if no progress file exists.
    """
    _validate_id(run_id, "run_id")

    # Try reading progress.json from run directory
    progress_file = _safe_results_path(run_id, "progress.json")
    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text())
            # Stale "running" detection: if status is running but no active job
            # and progress.json hasn't been updated in >2 minutes, the process
            # has exited without writing a "completed" status (e.g. killed).
            with _active_jobs_lock:
                run_active = run_id in _active_jobs
            if data.get("status") == "running" and not run_active:
                updated_at_str = data.get("updated_at")
                if updated_at_str:
                    from datetime import datetime, timezone, timedelta
                    try:
                        updated_at = datetime.fromisoformat(updated_at_str)
                        if updated_at.tzinfo is None:
                            # Legacy progress files wrote naive timestamps (UTC)
                            updated_at = updated_at.replace(tzinfo=timezone.utc)
                        if datetime.now(timezone.utc) - updated_at > timedelta(minutes=2):
                            data["status"] = "completed"
                            data["stale_status_recovered"] = True
                    except ValueError:
                        pass
            return data
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback: check active jobs
    with _active_jobs_lock:
        job = _active_jobs.get(run_id)
    if job is not None:
        proc = job.get("process")
        if proc and proc.poll() is not None:
            if proc.returncode == 0:
                job["status"] = "completed"
            else:
                job["status"] = "failed"
                if "error_message" not in job:
                    output = _job_error_output(job)
                    job["error_message"] = output or f"Process exited with code {proc.returncode}"
        result = {
            "job_id": run_id,
            "status": job["status"],
            "strategy_name": job.get("strategy_name"),
            "started_at": job.get("started_at"),
            "pid": job.get("pid"),
        }
        if job.get("error_message"):
            result["message"] = job["error_message"]
        return result

    raise HTTPException(404, f"No progress data for run: {run_id}")


@router.get("/{run_id}/logs")
def get_run_logs(
    run_id: str,
    symbol: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    stage: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
) -> list[dict]:
    """Get structured logs for a run.

    Reads logs.jsonl and applies optional filters.
    """
    _validate_id(run_id, "run_id")
    if symbol is not None:
        _validate_id(symbol, "symbol")
    logs_file = _safe_results_path(run_id, "logs.jsonl")

    if not logs_file.exists():
        raise HTTPException(404, f"No logs for run: {run_id}")

    entries = []
    try:
        with open(logs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if symbol and entry.get("symbol") != symbol:
                    continue
                if level and entry.get("level") != level:
                    continue
                if stage and entry.get("stage") != stage:
                    continue

                entries.append(entry)
    except IOError as e:
        log.exception("Failed to read logs for run %s", run_id)
        raise HTTPException(500, "Failed to read run logs") from e

    return entries[-limit:]


@router.delete("/{run_id}")
def delete_run(run_id: str) -> dict:
    """Delete all results for a completed run."""
    import shutil

    _validate_id(run_id, "run_id")
    run_dir = _safe_results_path(run_id)

    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")

    try:
        shutil.rmtree(run_dir)
    except Exception as e:
        log.exception("Failed to delete run %s", run_id)
        raise HTTPException(500, "Failed to delete run") from e

    return {"run_id": run_id, "deleted": True}


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    """Cancel an active run."""
    _validate_id(run_id, "run_id")
    with _active_jobs_lock:
        job = _active_jobs.get(run_id)
        if job is None:
            raise HTTPException(404, f"No active job: {run_id}")
        proc = job.get("process")
        if proc and proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                # Process exited between poll() and kill — treat as already
                # finished and fall through to the normal status response.
                pass
            else:
                job["status"] = "cancelled"
                return {"job_id": run_id, "status": "cancelled"}
        return {"job_id": run_id, "status": job["status"], "message": "Job already finished"}
