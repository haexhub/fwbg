"""API endpoints for the signal composer (rule builder)."""
import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fwbg.api.deps import get_strategies_dir, get_plugin_registry
from fwbg.core.config import _resolve_section

router = APIRouter(prefix="/signal-composer", tags=["signal-composer"])


def _find_common_prefix_len(columns: list[str]) -> int:
    """Find shared prefix length up to the last '_' boundary.

    Given columns like ["rb1_orb_s08_range", "rb1_orb_s08_position"],
    returns the length of "rb1_orb_" (the shared prefix up to and
    including the last '_' that all columns share).
    """
    if not columns:
        return 0
    if len(columns) == 1:
        return 0

    shortest = min(columns, key=len)
    prefix_len = 0
    for i, ch in enumerate(shortest):
        if not all(c[i] == ch for c in columns):
            break
        if ch == "_":
            prefix_len = i + 1
    else:
        # All chars matched up to len(shortest) — check if it ends with '_'
        if shortest.endswith("_"):
            prefix_len = len(shortest)

    return prefix_len


def _format_column_label(full_name: str, prefix_len: int) -> str:
    """Strip prefix, replace '_' with space, title-case."""
    short = full_name[prefix_len:]
    return short.replace("_", " ").strip().title()


def _price_columns_group() -> dict[str, Any]:
    """Built-in price columns group."""
    return {
        "fqn": "price",
        "label": "Price",
        "group_labels": {},
        "columns": [
            {"name": "close", "full_name": "C", "label": "Close", "type": "feature"},
            {"name": "high", "full_name": "H", "label": "High", "type": "feature"},
            {"name": "low", "full_name": "L", "label": "Low", "type": "feature"},
            {"name": "open", "full_name": "O", "label": "Open", "type": "feature"},
        ],
    }


def _get_plugin_columns(plugin_cls, params=None) -> tuple[list[str], list[str], dict[str, str]]:
    """Get feature columns, signal columns and group labels from a plugin class.

    When *params* is provided it is forwarded to the column methods so they
    return only the columns that match the pipeline configuration (sessions,
    retracement_levels, etc.) instead of the full default set.
    """
    inst = plugin_cls()
    feature_cols = inst.get_feature_columns(params)
    signal_cols = inst.get_signal_columns(params)

    try:
        group_labels = inst.get_column_group_labels()
    except Exception:
        group_labels = {}

    return feature_cols, signal_cols, group_labels


@router.get("/available-columns/{strategy_name}")
def get_available_columns(strategy_name: str) -> dict[str, Any]:
    """Return all columns produced by the indicators in a strategy's pipeline.

    Columns are grouped by indicator with human-readable labels, suitable
    for populating column dropdowns in the signal composer UI.
    """
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{strategy_name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {strategy_name}")

    try:
        data = json.loads(filepath.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON in strategy file: {e}")

    # Resolve pipeline reference (string -> dict)
    pipeline = _resolve_section(
        data.get("pipeline"),
        "pipelines",
        str(filepath.parent.resolve()),
    )

    if not pipeline:
        raise HTTPException(400, f"Strategy has no pipeline configured")

    indicator_configs = pipeline.get("indicators", [])
    if not indicator_configs:
        raise HTTPException(400, f"Pipeline has no indicators configured")

    registry = get_plugin_registry()
    groups: list[dict[str, Any]] = [_price_columns_group()]

    for ind_cfg in indicator_configs:
        name = ind_cfg.get("name", "")
        if not name:
            continue

        try:
            fqn = name if ":" in name else registry.resolve_name(name)
            plugin_cls = registry.get(fqn)
        except (ValueError, KeyError):
            # Skip unresolvable plugins
            continue

        ind_params = ind_cfg.get("params", {})
        feature_cols, signal_cols, group_labels = _get_plugin_columns(plugin_cls, ind_params)

        if not feature_cols:
            continue

        signal_set = set(signal_cols)
        prefix_len = _find_common_prefix_len(feature_cols)

        # Build human-readable indicator label from plugin name
        plugin_name = fqn.split(":", 1)[1] if ":" in fqn else fqn
        indicator_label = plugin_name.replace("_", " ").title()

        columns: list[dict[str, str]] = []
        for col in feature_cols:
            short_name = col[prefix_len:] if prefix_len else col
            columns.append({
                "name": short_name,
                "full_name": col,
                "label": _format_column_label(col, prefix_len),
                "type": "signal" if col in signal_set else "feature",
            })

        groups.append({
            "fqn": fqn,
            "label": indicator_label,
            "group_labels": group_labels,
            "columns": columns,
        })

    return {"groups": groups}


# ---------------------------------------------------------------------------
# POST /signal-composer/preview — evaluate rules against live data
# ---------------------------------------------------------------------------

class SignalPreviewRequest(BaseModel):
    strategy_name: str
    symbol: str
    timeframe: str = "HOUR"
    source: str = "forexsb"
    rules: dict  # {operator, conditions}
    direction: str = "long"
    limit: int = 5000


@router.post("/preview")
def preview_signal(req: SignalPreviewRequest) -> dict:
    """Evaluate composed signal rules against OHLCV data with indicators.

    Loads the strategy's indicator pipeline, computes all indicators on the
    requested symbol/timeframe data, then evaluates the rule tree.  Returns
    match count, total bars, and the matching timestamps.
    """
    from fwbg.api.chart import _best_native_file, _resample_ohlcv
    from fwbg.core.data_sources import get_data_source, CSVSourceConfig
    from fwbg.data.loader import load_data_aligned
    from fwbg.pipeline.features import compute_indicator_pool
    from fwbg.signals.evaluator import evaluate_rules

    # --- Load strategy & resolve indicator configs ---
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{req.strategy_name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {req.strategy_name}")

    try:
        data = json.loads(filepath.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON in strategy file: {e}")

    pipeline = _resolve_section(
        data.get("pipeline"),
        "pipelines",
        str(filepath.parent.resolve()),
    )

    if not pipeline:
        raise HTTPException(400, "Strategy has no pipeline configured")

    indicator_configs = pipeline.get("indicators", [])
    if not indicator_configs:
        raise HTTPException(400, "Pipeline has no indicators configured")

    # --- Load OHLCV data (same pattern as chart.py GET /ohlcv) ---
    try:
        ds = get_data_source(req.source)
    except ValueError as e:
        raise HTTPException(404, str(e))

    if not isinstance(ds, CSVSourceConfig):
        raise HTTPException(400, f"Source '{req.source}' is not a CSV source")

    path = ds.get_file_path(req.symbol, req.timeframe)
    native_tf = req.timeframe
    if not path.exists():
        path, native_tf = _best_native_file(ds, req.symbol, req.timeframe)
        if not path:
            raise HTTPException(
                404,
                f"Data not found: {req.symbol}_{req.timeframe} in {req.source}",
            )

    df = load_data_aligned(str(path))
    if df is None or df.empty:
        raise HTTPException(500, f"Failed to load data: {req.symbol}_{req.timeframe}")

    if native_tf != req.timeframe:
        df = _resample_ohlcv(df, req.timeframe)

    # Apply limit (take the most recent bars)
    if req.limit and len(df) > req.limit:
        df = df.iloc[-req.limit:]

    # --- Compute indicators ---
    df = compute_indicator_pool(df, indicator_configs)

    # --- Evaluate rules ---
    try:
        mask = evaluate_rules(req.rules, df)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"Rule evaluation error: {e}")

    match_count = int(mask.sum())
    total_bars = len(df)
    timestamps = [
        int(ts.timestamp() * 1000) for ts in df.index[mask]
    ]

    return {
        "match_count": match_count,
        "total_bars": total_bars,
        "timestamps": timestamps,
    }
