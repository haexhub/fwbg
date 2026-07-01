"""On-demand data provisioning.

POST /api/data/ensure — ensure OHLCV data exists for a (symbol, timeframe).
GET  /api/data/ensure/{task_id} — poll an in-progress download.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fwbg.core.data_sources import _DATA_SOURCES, CSVSourceConfig

log = logging.getLogger(__name__)

router = APIRouter(tags=["data"])

# In-memory task store — mirrors _download_tasks / _prepare_tasks pattern.
_ensure_tasks: dict[str, dict] = {}


class EnsureRequest(BaseModel):
    symbol: str
    timeframe: str = "HOUR_1"
    date_from: Optional[str] = None  # ISO date "YYYY-MM-DD"
    date_to: Optional[str] = None    # ISO date "YYYY-MM-DD"


@router.post("/data/ensure")
def ensure_data(req: EnsureRequest):
    """Ensure OHLCV data exists for (symbol, timeframe).

    - 200 "ready"      — file already present in a configured CSV source.
    - 202 "downloading" — download started; poll GET /api/data/ensure/{task_id}.
    - 404              — symbol not covered by any adapter (non-FX instruments).
    - 422              — unsupported timeframe or invalid date range.
    - 503              — no CSV datasource configured to store the download.

    Currently only FX instruments are downloadable (Dukascopy). Non-FX symbols
    must be uploaded manually via the datasource upload endpoints.
    """
    from fwbg.data.dukascopy import DukascopyError, TIMEFRAMES, download, resolve_instrument

    symbol = _normalize_symbol(req.symbol)

    # 1. Already cached in any CSV source?
    hit = _find_existing_file(symbol, req.timeframe)
    if hit is not None:
        source_name, file_path = hit
        return {
            "status": "ready",
            "symbol": symbol,
            "timeframe": req.timeframe,
            "source": source_name,
            "path": str(file_path),
        }

    # 2. Validate timeframe before touching Dukascopy.
    if req.timeframe not in TIMEFRAMES:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported timeframe {req.timeframe!r}; valid values: {sorted(TIMEFRAMES)}",
        )

    # 3. Can Dukascopy provide this symbol?
    try:
        resolve_instrument(symbol)
    except DukascopyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data available for {symbol!r}. "
                "Dukascopy does not list this instrument — "
                "only FX instruments are currently supported for on-demand download."
            ),
        )

    # 4. Need a CSV source to write into.
    csv_source = _first_csv_source()
    if csv_source is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No CSV datasource is configured. "
                "Create one via POST /api/datasources before requesting on-demand downloads."
            ),
        )

    # 5. Build date range (default: 2020-01-01 … today UTC).
    date_from_str = req.date_from or "2020-01-01"
    date_to_str = req.date_to or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        start_dt = datetime.fromisoformat(date_from_str).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(date_to_str).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid date: {exc}") from exc
    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="date_to must be after date_from")

    # 6. Kick off background download.
    task_id = uuid.uuid4().hex[:12]
    _ensure_tasks[task_id] = {
        "status": "running",
        "symbol": symbol,
        "timeframe": req.timeframe,
        "result": None,
        "error": None,
    }

    def _run():
        try:
            results = download(
                csv_source.path,
                symbols=[symbol],
                timeframe=req.timeframe,
                start=start_dt,
                end=end_dt,
            )
            _ensure_tasks[task_id]["status"] = "ready"
            _ensure_tasks[task_id]["result"] = results[0] if results else None
            log.info("ensure_data: %s %s done (task=%s)", symbol, req.timeframe, task_id)
        except Exception as exc:
            log.error("ensure_data: %s %s failed: %s", symbol, req.timeframe, exc)
            _ensure_tasks[task_id]["status"] = "error"
            _ensure_tasks[task_id]["error"] = str(exc)

    threading.Thread(target=_run, daemon=True, name=f"ensure-{task_id}").start()

    return JSONResponse(
        status_code=202,
        content={
            "status": "downloading",
            "task_id": task_id,
            "symbol": symbol,
            "timeframe": req.timeframe,
            "source": csv_source.name,
            "poll_url": f"/api/data/ensure/{task_id}",
        },
    )


@router.get("/data/ensure/{task_id}")
def ensure_status(task_id: str):
    """Poll the status of an in-progress ensure_data download."""
    task = _ensure_tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"unknown task id: {task_id!r}")
    return {"task_id": task_id, **task}


# ── helpers ───────────────────────────────────────────────────────────────────


def _normalize_symbol(raw: str) -> str:
    """Strip separators and uppercase — matches Dukascopy's internal normalization."""
    return raw.replace("/", "").replace("_", "").replace("-", "").upper()


def _find_existing_file(symbol: str, timeframe: str) -> tuple[str, Path] | None:
    """Return (source_name, path) if any CSV source already has this file."""
    for name, source in _DATA_SOURCES.items():
        if isinstance(source, CSVSourceConfig):
            path = source.get_file_path(symbol, timeframe)
            if path.exists():
                return name, path
    return None


def _first_csv_source() -> CSVSourceConfig | None:
    for source in _DATA_SOURCES.values():
        if isinstance(source, CSVSourceConfig):
            return source
    return None
