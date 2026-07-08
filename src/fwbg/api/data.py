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
# Guarded by _ensure_tasks_lock: written from daemon threads, read from
# request handlers (see runs.py _active_jobs_lock for the same pattern).
_ensure_tasks: dict[str, dict] = {}
_ensure_tasks_lock = threading.Lock()


class EnsureRequest(BaseModel):
    symbol: str
    timeframe: str = "HOUR_1"
    date_from: Optional[str] = None  # ISO date "YYYY-MM-DD"
    date_to: Optional[str] = None    # ISO date "YYYY-MM-DD"


# historyStart granularity per timeframe family (see instrument_catalogue()).
_TIMEFRAME_GRANULARITY = {"MINUTE": "minute", "HOUR": "hourly", "DAY": "daily"}
_FALLBACK_HISTORY_START = "2020-01-01"


def _default_history_start(symbol: str, timeframe: str) -> str:
    """Earliest available date for (symbol, timeframe-granularity).

    Backtests want as much history as possible, so an ensure request without
    an explicit date_from downloads the instrument's FULL available history —
    e.g. EURUSD daily back to 1973, minute data back to 2003. Falls back to
    2020-01-01 when the catalogue has no entry for the symbol.
    """
    from fwbg.data.dukascopy import instrument_catalogue

    granularity = _TIMEFRAME_GRANULARITY.get(timeframe.split("_")[0])
    if granularity is None:
        return _FALLBACK_HISTORY_START
    try:
        for inst in instrument_catalogue():
            if inst["symbol"] == symbol:
                start = (inst.get("historyStart") or {}).get(granularity)
                if start:
                    return start
                break
    except Exception:
        log.warning("catalogue lookup failed for %s; using fallback start", symbol,
                    exc_info=True)
    return _FALLBACK_HISTORY_START


@router.get("/data/timeframes")
def list_timeframes() -> dict:
    """Supported OHLCV timeframes for downloads and strategy configs."""
    from fwbg.data.dukascopy import TIMEFRAMES

    return {"timeframes": list(TIMEFRAMES.keys())}


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

    # Determine the desired date range up front. Without an explicit date_from
    # the full available history is requested (date_from defaults to the
    # instrument's catalogue start).
    explicit_from = req.date_from is not None
    date_from_str = req.date_from or _default_history_start(symbol, req.timeframe)
    date_to_str = req.date_to or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Already cached?  For an explicit date_from the cached file must cover
    #    that start. For an implicit full-history request any existing file is
    #    accepted: re-downloading decades of intraday history just because the
    #    catalogue claims an earlier start (e.g. a 2012 file vs a 2003 catalogue
    #    start) re-fetches years of data on every run for marginal early
    #    coverage — and a single stalled Dukascopy connection then hangs the
    #    whole run. The cached range is enough to backtest on.
    hit = _find_existing_file(
        symbol, req.timeframe, date_from_str if explicit_from else None
    )
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
    try:
        start_dt = datetime.fromisoformat(date_from_str).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(date_to_str).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid date: {exc}") from exc
    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="date_to must be after date_from")

    # 6. Kick off background download.
    task_id = uuid.uuid4().hex[:12]
    with _ensure_tasks_lock:
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
            with _ensure_tasks_lock:
                _ensure_tasks[task_id]["result"] = results[0] if results else None
                _ensure_tasks[task_id]["status"] = "ready"
            log.info("ensure_data: %s %s done (task=%s)", symbol, req.timeframe, task_id)
        except Exception as exc:
            log.error("ensure_data: %s %s failed: %s", symbol, req.timeframe, exc)
            with _ensure_tasks_lock:
                _ensure_tasks[task_id]["error"] = str(exc)
                _ensure_tasks[task_id]["status"] = "error"

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
    with _ensure_tasks_lock:
        task = _ensure_tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"unknown task id: {task_id!r}")
        return {"task_id": task_id, **task}


# ── helpers ───────────────────────────────────────────────────────────────────


def _normalize_symbol(raw: str) -> str:
    """Strip separators and uppercase — matches Dukascopy's internal normalization."""
    return raw.replace("/", "").replace("_", "").replace("-", "").upper()


def _csv_first_date(path: Path) -> str | None:
    """Read the first data row's date (YYYY-MM-DD) from a fwbg CSV (T,O,H,L,C,V)."""
    try:
        with path.open() as f:
            f.readline()  # skip header
            first_line = f.readline()
        if first_line:
            return first_line.split(",")[0][:10]
    except Exception:
        pass
    return None


_HISTORY_START_TOLERANCE_DAYS = 31  # Dukascopy catalogue dates are approximate


def _file_covers_from(path: Path, date_from: str) -> bool:
    """Return True if *path*'s first bar is within tolerance of *date_from*.

    Dukascopy's historyStart is an approximation; the actual first downloadable
    bar may be a few days later. 31 days of tolerance avoids a perpetual
    re-download loop while still catching files that start years too late.
    """
    from datetime import date as _date

    first = _csv_first_date(path)
    if first is None:
        return True  # can't read → assume ok, don't re-download
    try:
        gap = (_date.fromisoformat(first) - _date.fromisoformat(date_from)).days
        return gap <= _HISTORY_START_TOLERANCE_DAYS
    except ValueError:
        return True


def _find_existing_file(
    symbol: str, timeframe: str, date_from: str | None = None
) -> tuple[str, Path] | None:
    """Return (source_name, path) if any CSV source has this file AND it covers
    from *date_from* (within tolerance). If the existing file starts too late
    (e.g. 2012 when full history goes back to 2003) it is skipped so the
    caller triggers a fresh full-history download."""
    for name, source in _DATA_SOURCES.items():
        if isinstance(source, CSVSourceConfig):
            path = source.get_file_path(symbol, timeframe)
            if path.exists():
                if date_from is not None and not _file_covers_from(path, date_from):
                    log.info(
                        "ensure_data: %s starts too late for requested %s — re-downloading",
                        path.name, date_from,
                    )
                    continue
                return name, path
    return None


def _first_csv_source() -> CSVSourceConfig | None:
    for source in _DATA_SOURCES.values():
        if isinstance(source, CSVSourceConfig):
            return source
    return None
