"""API endpoints for MFE/MAE exit analysis."""

import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["analyze"])


class AnalyzeRequest(BaseModel):
    asset: str  # e.g. "BRENT_HOUR.csv"
    exit_strategy: str = "atr_based"
    exit_params: dict = {"atr_period": 14}
    max_bars: int = 48


@router.post("/analyze")
def run_analysis(req: AnalyzeRequest):
    """Run MFE/MAE analysis for a single asset. Returns full result."""
    from fwbg.analysis.exit_analyzer import analyze_asset, write_json
    from fwbg.api.deps import get_test_results_dir
    from fwbg.data.config import DATA_PATH

    data_file = os.path.join(DATA_PATH, req.asset)
    if not os.path.exists(data_file):
        raise HTTPException(404, f"Data file not found: {req.asset}")

    result = analyze_asset(data_file, req.exit_strategy, req.exit_params, req.max_bars)

    # Cache result to disk
    out_dir = os.path.join(get_test_results_dir(), "analyze")
    os.makedirs(out_dir, exist_ok=True)
    symbol = req.asset.replace(".csv", "")
    write_json(result, os.path.join(out_dir, f"{symbol}.json"))

    return result


@router.get("/analyze")
def list_analyses():
    """List all cached analysis results (summary only)."""
    from fwbg.api.deps import get_test_results_dir

    out_dir = os.path.join(get_test_results_dir(), "analyze")
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


@router.get("/analyze/{symbol}")
def get_analysis(symbol: str):
    """Get cached analysis result for a symbol."""
    from fwbg.api.deps import get_test_results_dir

    out_dir = os.path.join(get_test_results_dir(), "analyze")
    # Try exact match, then with _HOUR suffix
    for candidate in [f"{symbol}.json", f"{symbol}_HOUR.json"]:
        path = os.path.join(out_dir, candidate)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)

    raise HTTPException(404, f"No analysis found for {symbol}")
