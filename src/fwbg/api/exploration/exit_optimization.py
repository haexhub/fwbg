"""API endpoints for MFE/MAE exit optimization."""

import hashlib
import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fwbg.api._paths import validate_id

router = APIRouter(tags=["exit-optimization"])


class ExitOptimizationRequest(BaseModel):
    source: str
    symbol: str  # e.g. "ASX200"
    timeframe: str = "HOUR"
    exit_strategy: str = "atr_based"
    exit_params: dict = {"atr_period": 14}
    max_bars: int = 48
    # The research register, as for ``fwbg analyze``: the card the analysis serves,
    # one line on what it is for, and why an identical analysis runs again.
    research_id: str | None = None
    note: str | None = None
    reason: str | None = None


def _cache_key(
    req: "ExitOptimizationRequest",
    data_path: str,
    window: dict[str, str | None],
) -> str:
    """Cache file stem for one analysis: symbol and timeframe plus a digest of everything
    else that changes the result (datasource, exit strategy and params, forward bars,
    the data window read, and the data file's size and modification time as its version).
    """
    stat = os.stat(data_path)
    identity: dict[str, Any] = {
        "source": req.source,
        "exit_strategy": req.exit_strategy,
        "exit_params": req.exit_params,
        "max_bars": req.max_bars,
        "window": [window["start_date"], window["end_date"]],
        "data": [stat.st_size, stat.st_mtime_ns],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    return f"{req.symbol}_{req.timeframe}_{digest}"


@router.post("")
def run_exit_optimization(req: ExitOptimizationRequest):
    """Run MFE/MAE analysis for a single asset. Returns full result.

    The analysis is recorded in the research register before it reads data and
    reads only data before the card's holdout cutoff. A refusal is a 409/422
    with ``{"message", "denials"}`` like ``POST /api/runs/start``.
    """
    from fwbg.exploration.exit_analyzer import analyze_asset, write_json
    from fwbg.api.deps import get_test_results_dir
    from fwbg.api.run_preflight import gate_refusal
    from fwbg.core.data_sources import get_data_source, CSVSourceConfig
    from fwbg.research.analysis import analysis_config, record_analysis
    from fwbg.research.runs import GateDenied, denial_payload, gate_enabled

    try:
        ds = get_data_source(req.source)
    except ValueError as e:
        raise HTTPException(404, str(e))

    if not isinstance(ds, CSVSourceConfig):
        raise HTTPException(400, f"Source '{req.source}' is not a CSV source")

    try:
        path = ds.get_file_path(req.symbol, req.timeframe)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not path.exists():
        raise HTTPException(
            404, f"Data file not found: {req.symbol}_{req.timeframe} in {req.source}"
        )

    window = {"start_date": None, "end_date": None}
    if gate_enabled():
        config = analysis_config(
            req.exit_strategy,
            req.exit_params,
            req.max_bars,
            req.timeframe,
            datasource=req.source,
        )
        try:
            _, window = record_analysis(
                req.research_id,
                config,
                [req.symbol.upper()],
                req.note,
                reason=req.reason,
            )
        except GateDenied as exc:
            raise gate_refusal(denial_payload(exc.denials), "analysis") from None

    result = analyze_asset(
        str(path),
        req.exit_strategy,
        req.exit_params,
        req.max_bars,
        window["start_date"],
        window["end_date"],
    )

    # Cache result to disk
    out_dir = os.path.join(get_test_results_dir(), "exploration")
    os.makedirs(out_dir, exist_ok=True)
    cache_key = _cache_key(req, str(path), window)
    result["cache_key"] = cache_key
    write_json(result, os.path.join(out_dir, f"{cache_key}.json"))

    return result


@router.get("")
def list_exit_optimizations():
    """List all cached exit optimization results (summary only)."""
    from fwbg.api.deps import get_test_results_dir

    out_dir = os.path.join(get_test_results_dir(), "exploration")
    if not os.path.isdir(out_dir):
        return []

    results = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".json"):
            continue
        path = os.path.join(out_dir, f)
        with open(path) as fh:
            data = json.load(fh)
        results.append(
            {
                "symbol": data.get("symbol"),
                "timeframe": data.get("timeframe"),
                "exit_strategy": data.get("exit_strategy"),
                "analyzed_at": data.get("analyzed_at"),
                "exit_params": data.get("exit_params"),
                "max_bars_forward": data.get("max_bars_forward"),
                "window": data.get("window"),
                "cache_key": data.get("cache_key"),
                "bars_analyzed": data.get("bars_analyzed"),
                "suggested_grid": data.get("suggested_grid"),
            }
        )
    return results


@router.get("/{symbol}")
def get_exit_optimization(symbol: str, timeframe: str | None = None):
    """Get the most recent cached exit optimization result for a symbol.

    Several results can exist per symbol (one per timeframe, exit strategy,
    parameters and data window); pass ``timeframe`` to narrow them down.
    """
    from fwbg.api.deps import get_test_results_dir

    validate_id(symbol, "symbol")
    out_dir = os.path.join(get_test_results_dir(), "exploration")
    best: tuple[tuple[str, int], dict] | None = None
    if os.path.isdir(out_dir):
        for name in sorted(os.listdir(out_dir)):
            if not name.endswith(".json") or not name.startswith(f"{symbol}_"):
                continue
            file_path = os.path.join(out_dir, name)
            with open(file_path) as fh:
                data = json.load(fh)
            if data.get("symbol") != symbol:
                continue
            if timeframe is not None and data.get("timeframe") != timeframe:
                continue
            # analyzed_at has second resolution; the file time breaks ties.
            stamp = (data.get("analyzed_at") or "", os.stat(file_path).st_mtime_ns)
            if best is None or stamp >= best[0]:
                best = (stamp, data)
    if best is None:
        raise HTTPException(404, f"No exit optimization found for {symbol}")
    return best[1]
