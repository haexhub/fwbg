"""API endpoints for MFE/MAE exit optimization."""

import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["exit-optimization"])


class ExitOptimizationRequest(BaseModel):
    source: str
    symbol: str  # e.g. "ASX200"
    timeframe: str = "HOUR"
    exit_strategy: str = "atr_based"
    exit_params: dict = {"atr_period": 14}
    max_bars: int = 48


@router.post("")
def run_exit_optimization(req: ExitOptimizationRequest):
    """Run MFE/MAE analysis for a single asset. Returns full result."""
    from fwbg.exploration.exit_analyzer import analyze_asset, write_json
    from fwbg.api.deps import get_test_results_dir
    from fwbg.core.data_sources import get_data_source, CSVSourceConfig

    try:
        ds = get_data_source(req.source)
    except ValueError as e:
        raise HTTPException(404, str(e))

    if not isinstance(ds, CSVSourceConfig):
        raise HTTPException(400, f"Source '{req.source}' is not a CSV source")

    path = ds.get_file_path(req.symbol, req.timeframe)
    if not path.exists():
        raise HTTPException(404, f"Data file not found: {req.symbol}_{req.timeframe} in {req.source}")

    result = analyze_asset(str(path), req.exit_strategy, req.exit_params, req.max_bars)

    # Cache result to disk
    out_dir = os.path.join(get_test_results_dir(), "exploration")
    os.makedirs(out_dir, exist_ok=True)
    cache_key = f"{req.symbol}_{req.timeframe}"
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
        results.append({
            "symbol": data.get("symbol"),
            "timeframe": data.get("timeframe"),
            "exit_strategy": data.get("exit_strategy"),
            "analyzed_at": data.get("analyzed_at"),
            "bars_analyzed": data.get("bars_analyzed"),
            "suggested_grid": data.get("suggested_grid"),
        })
    return results


@router.get("/{symbol}")
def get_exit_optimization(symbol: str):
    """Get cached exit optimization result for a symbol."""
    from fwbg.api.deps import get_test_results_dir

    out_dir = os.path.join(get_test_results_dir(), "exploration")
    # Try exact match, then with _HOUR suffix
    for candidate in [f"{symbol}.json", f"{symbol}_HOUR.json"]:
        path = os.path.join(out_dir, candidate)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)

    raise HTTPException(404, f"No exit optimization found for {symbol}")
