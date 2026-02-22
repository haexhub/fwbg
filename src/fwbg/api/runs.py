"""Run management endpoints."""
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from fwbg.api.deps import get_strategies_dir, get_test_results_dir

router = APIRouter(prefix="/runs", tags=["runs"])

# Track active background processes
_active_jobs: dict[str, dict] = {}


class RunStartRequest(BaseModel):
    """Request body for starting a run."""
    strategy_name: str
    assets: Optional[list[str]] = None
    asset_classes: Optional[list[str]] = None
    description: Optional[str] = None


@router.post("/start")
def start_run(body: RunStartRequest) -> dict:
    """Start a strategy optimization run in the background."""
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

    job_id = str(uuid.uuid4())[:8]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        _active_jobs[job_id] = {
            "job_id": job_id,
            "pid": process.pid,
            "process": process,
            "strategy_name": body.strategy_name,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "cmd": cmd,
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to start run: {e}")

    return {
        "job_id": job_id,
        "status": "running",
        "strategy_name": body.strategy_name,
        "pid": process.pid,
    }


@router.get("")
def list_runs(limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    """List completed and active runs."""
    results_dir = get_test_results_dir()
    runs = []

    # Completed runs from test_results/
    if results_dir.exists():
        run_dirs = sorted(results_dir.iterdir(), reverse=True)
        for run_dir in run_dirs[:limit]:
            if not run_dir.is_dir():
                continue

            config_file = run_dir / "config.json"
            strategy_file = run_dir / "strategy.json"

            run_info: dict = {
                "run_id": run_dir.name,
                "status": "completed",
            }

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

            if strategy_file.exists():
                try:
                    strategy = json.loads(strategy_file.read_text())
                    run_info["strategy_name"] = strategy.get("name", "")
                    run_info["tags"] = strategy.get("tags", [])
                except (json.JSONDecodeError, IOError):
                    pass

            # Count assets from grid_details
            grid_dir = run_dir / "grid_details"
            if grid_dir.exists():
                asset_files = list(grid_dir.glob("*.json"))
                run_info["asset_count"] = len(asset_files)

                # Summarize asset results
                profitable = 0
                for af in asset_files:
                    try:
                        data = json.loads(af.read_text())
                        if data.get("status") == "ok":
                            profitable += 1
                    except (json.JSONDecodeError, IOError):
                        pass
                run_info["profitable_count"] = profitable

            runs.append(run_info)

    # Active jobs
    for job_id, job in _active_jobs.items():
        proc = job.get("process")
        if proc and proc.poll() is not None:
            job["status"] = "completed" if proc.returncode == 0 else "failed"

        runs.insert(0, {
            "run_id": job_id,
            "status": job["status"],
            "strategy_name": job.get("strategy_name"),
            "started_at": job.get("started_at"),
            "is_active": job["status"] == "running",
        })

    return runs


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    """Get detailed results for a completed run."""
    results_dir = get_test_results_dir()
    run_dir = results_dir / run_id

    if not run_dir.exists():
        # Check active jobs
        if run_id in _active_jobs:
            job = _active_jobs[run_id]
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

    # Load strategy
    strategy_file = run_dir / "strategy.json"
    if strategy_file.exists():
        try:
            result["strategy"] = json.loads(strategy_file.read_text())
        except (json.JSONDecodeError, IOError):
            pass

    # Load per-asset results
    grid_dir = run_dir / "grid_details"
    if grid_dir.exists():
        assets = {}
        for af in sorted(grid_dir.glob("*.json")):
            try:
                data = json.loads(af.read_text())
                symbol = af.stem
                assets[symbol] = {
                    "symbol": symbol,
                    "status": data.get("status", "unknown"),
                    "total_combinations": data.get("total_combinations", 0),
                    "walk_forward": _summarize_walk_forward(data.get("walk_forward", {})),
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
    """List grid detail filenames for a completed run."""
    results_dir = get_test_results_dir()
    grid_dir = results_dir / run_id / "grid_details"

    if not grid_dir.exists():
        raise HTTPException(404, f"No grid details for run: {run_id}")

    return sorted(f.name for f in grid_dir.glob("*.json"))


@router.get("/{run_id}/grid_details/{filename}")
def get_grid_detail(run_id: str, filename: str) -> dict:
    """Get a single grid detail file for a completed run."""
    results_dir = get_test_results_dir()
    filepath = results_dir / run_id / "grid_details" / filename

    if not filepath.exists():
        raise HTTPException(404, f"Grid detail not found: {run_id}/{filename}")

    try:
        return json.loads(filepath.read_text())
    except (json.JSONDecodeError, IOError) as e:
        raise HTTPException(500, f"Failed to read grid detail: {e}")


@router.get("/{run_id}/trades/{symbol}")
def get_run_symbol_trades(run_id: str, symbol: str) -> dict:
    """Return all detailed trades for a specific symbol across all walk-forward folds.

    Reads the stored grid-detail file and extracts full trade records (entry/exit
    timestamps, prices, direction, TP/SL levels) from each fold's test_trades_detail.
    Falls back to test_trades_trace for older runs that don't have test_trades_detail.
    """
    results_dir = get_test_results_dir()
    grid_dir = results_dir / run_id / "grid_details"

    if not grid_dir.exists():
        raise HTTPException(404, f"No grid details for run: {run_id}")

    # Find the grid-detail file for this symbol (matched by the "symbol" field inside)
    trades: list[dict] = []
    found = False
    for filepath in sorted(grid_dir.glob("*.json")):
        try:
            data = json.loads(filepath.read_text())
        except (json.JSONDecodeError, IOError):
            continue

        if data.get("symbol", "").upper() != symbol.upper():
            continue

        found = True
        for fold in data.get("walk_forward", {}).get("fold_details", []):
            fold_id = fold.get("fold_id")
            # Prefer test_trades_detail (full trade info with timing/prices)
            detail_trades = fold.get("test_trades_detail", [])
            if detail_trades:
                for trade in detail_trades:
                    if isinstance(trade, dict) and "entry_time" in trade:
                        trades.append({**trade, "fold_id": fold_id})
            else:
                # Fallback: test_trades_trace (older format, may have entry_time)
                for trade in fold.get("test_trades_trace", []):
                    if isinstance(trade, dict) and "entry_time" in trade:
                        trades.append({**trade, "fold_id": fold_id})
        break

    if not found:
        raise HTTPException(404, f"No grid detail found for symbol: {symbol}")

    return {"symbol": symbol, "run_id": run_id, "trades": trades}


@router.get("/{run_id}/progress")
def get_run_progress(run_id: str) -> dict:
    """Get progress for an active or completed run.

    Reads the progress.json file written by RunProgressWriter.
    Falls back to basic job info if no progress file exists.
    """
    results_dir = get_test_results_dir()

    # Try reading progress.json from run directory
    progress_file = results_dir / run_id / "progress.json"
    if progress_file.exists():
        try:
            return json.loads(progress_file.read_text())
        except (json.JSONDecodeError, IOError):
            pass

    # Fallback: check active jobs
    if run_id in _active_jobs:
        job = _active_jobs[run_id]
        proc = job.get("process")
        if proc and proc.poll() is not None:
            job["status"] = "completed" if proc.returncode == 0 else "failed"
        return {
            "job_id": run_id,
            "status": job["status"],
            "strategy_name": job.get("strategy_name"),
            "started_at": job.get("started_at"),
            "pid": job.get("pid"),
        }

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
    results_dir = get_test_results_dir()
    logs_file = results_dir / run_id / "logs.jsonl"

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
                if len(entries) >= limit:
                    break
    except IOError:
        raise HTTPException(500, f"Failed to read logs for run: {run_id}")

    return entries


@router.delete("/{run_id}")
def delete_run(run_id: str) -> dict:
    """Delete all results for a completed run."""
    import shutil

    results_dir = get_test_results_dir()
    run_dir = results_dir / run_id

    if not run_dir.exists():
        raise HTTPException(404, f"Run not found: {run_id}")

    try:
        shutil.rmtree(run_dir)
    except Exception as e:
        raise HTTPException(500, f"Failed to delete run: {e}")

    return {"run_id": run_id, "deleted": True}


@router.post("/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    """Cancel an active run."""
    if run_id not in _active_jobs:
        raise HTTPException(404, f"No active job: {run_id}")

    job = _active_jobs[run_id]
    proc = job.get("process")

    if proc and proc.poll() is None:
        os.kill(proc.pid, signal.SIGTERM)
        job["status"] = "cancelled"
        return {"job_id": run_id, "status": "cancelled"}

    return {"job_id": run_id, "status": job["status"], "message": "Job already finished"}
