"""FWBG MCP Server.

Exposes the FWBG strategy optimizer to AI agents via the Model Context Protocol.
Agents can autonomously: discover indicators → build strategy configs → start runs
→ wait for results → evaluate → adjust → retry.

Environment variables:
  FWBG_API_URL  Base URL of the FWBG API (default: http://localhost:8420)
"""

import os
import time
from typing import Optional

import httpx
from fastmcp import FastMCP

API_URL = os.environ.get("FWBG_API_URL", "http://localhost:8420").rstrip("/") + "/api"

mcp = FastMCP(
    "fwbg",
    instructions=(
        "You are connected to the FWBG walk-forward trading strategy optimizer. "
        "Typical autonomous loop: "
        "1. list_indicators / list_exit_strategies to discover available building blocks. "
        "2. get_strategy (or build from scratch) to read/create a strategy config. "
        "3. save_strategy to persist the config. "
        "4. start_run to launch the optimization. "
        "5. wait_for_run to block until completion (returns full results). "
        "6. Analyse PF, WR, fold_stability, CAGR per asset. "
        "7. Adjust config and repeat. "
        "Key concepts: "
        "- signal_rules pre-filter bars before ML model; without them the model sees all bars → 0 OOS trades. "
        "- CT (confidence threshold) must be <= inner-CV optimal CT; start at [0.35, 0.4, 0.45]. "
        "- regime_filter_grid adds directional market-regime conditions (bitmask 4=Long, 2=Short, 6=Both). "
        "- indicator_grid varies indicator params across combinations. "
        "- Keep total combinations (exit × CT × regime × indicator_grid) <= 50 to avoid overfit. "
    ),
)


def _get(path: str, **params) -> dict | list:
    r = httpx.get(f"{API_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, body: dict) -> dict:
    r = httpx.post(f"{API_URL}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _put(path: str, body: dict) -> dict:
    r = httpx.put(f"{API_URL}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Strategy management
# ---------------------------------------------------------------------------

@mcp.tool()
def list_strategies() -> list[dict]:
    """List all available strategy configs.

    Returns each strategy's filename (use as `name` in other calls),
    display name, description, and tags.
    """
    return _get("/strategies")


@mcp.tool()
def get_strategy(name: str) -> dict:
    """Load a strategy config by filename (without .json extension).

    Preset references (e.g. `"pipeline": "sr_trend_v1"`) are resolved to
    their full inline content so you see the complete configuration.
    The `_refs` key preserves the original preset names.
    """
    return _get(f"/strategies/{name}")


@mcp.tool()
def save_strategy(name: str, config: dict) -> dict:
    """Create or update a strategy config file.

    `name` is the filename (without .json). `config` is the full strategy dict.
    Use this to persist changes before calling start_run.

    Config skeleton:
    {
      "name": "my_strategy",
      "description": "...",
      "datasource": "dukascopy",
      "timeframe": "MINUTE_15",
      "assets": {"filter": ["NAS100"]},
      "pipeline": "sr_trend_v1",          // preset name OR inline dict
      "model": "xgboost_depth3_v1",       // preset name OR inline dict
      "validation": "walk_forward_sr_trend_v1",
      "filters": "sr_strict_v1",
      "signal_rules": {...},
      "exit_strategies": [...],
      "optimization": {"regime_filter_grid": {...}, "indicator_grid": {...}}
    }
    """
    return _put(f"/strategies/{name}", config)


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@mcp.tool()
def start_run(
    strategy_name: str,
    assets: Optional[list[str]] = None,
    description: Optional[str] = None,
) -> dict:
    """Start a strategy optimization run in the background.

    Returns immediately with a `run_id`. Use wait_for_run or get_run_status
    to track progress.

    Args:
        strategy_name: Filename of the strategy config (without .json).
        assets: Optional list of asset symbols to override the config filter,
                e.g. ["NAS100", "DAX"].
        description: Optional human-readable note attached to this run.
    """
    body: dict = {"strategy_name": strategy_name}
    if assets:
        body["assets"] = assets
    if description:
        body["description"] = description
    return _post("/runs/start", body)


@mcp.tool()
def wait_for_run(run_id: str, timeout_minutes: int = 120) -> dict:
    """Block until a run completes, then return full results.

    Polls every 30 seconds. Returns the same payload as get_run_results.
    Raises an error if the run fails or the timeout is exceeded.

    Args:
        run_id: The run ID returned by start_run.
        timeout_minutes: Maximum time to wait (default: 120 min).
    """
    deadline = time.time() + timeout_minutes * 60
    poll_interval = 30

    while time.time() < deadline:
        progress = _get(f"/runs/{run_id}/progress")
        status = progress.get("status", "unknown")

        if status == "completed":
            return get_run_results(run_id)

        if status == "failed":
            raise RuntimeError(
                f"Run {run_id} failed: {progress.get('message', 'unknown error')}"
            )

        # Log current stage for observability
        current_stage = progress.get("current_stage", "")
        fraction = progress.get("progress_fraction", 0)
        print(f"[fwbg-mcp] {run_id}: {status} – {current_stage} ({fraction:.0%})")

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Run {run_id} did not complete within {timeout_minutes} minutes."
    )


@mcp.tool()
def get_run_status(run_id: str) -> dict:
    """Get the current status and progress of a run without blocking.

    Returns: status (running/completed/failed), progress_fraction,
    current_stage, started_at, and per-asset stage details.
    """
    return _get(f"/runs/{run_id}/progress")


@mcp.tool()
def get_run_results(run_id: str) -> dict:
    """Get structured results for a completed run.

    Returns:
    - summary: overall KPIs across all assets
    - assets: per-asset dict with status, unified_metrics (PF, WR, CAGR,
      max_drawdown, sharpe), and walk_forward summary (n_folds,
      successful_folds, mean_win_rate, total_trades)
    - config: run configuration snapshot
    """
    run = _get(f"/runs/{run_id}")

    # Enrich with grid details per asset (contains fold_results + unified_metrics)
    try:
        asset_symbols = _get(f"/runs/{run_id}/grid_details")
        details = {}
        for symbol in asset_symbols:
            details[symbol] = _get(f"/runs/{run_id}/grid_details/{symbol}")
        run["grid_details"] = details
    except httpx.HTTPStatusError:
        pass

    return run


@mcp.tool()
def get_run_logs(
    run_id: str,
    level: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Get structured logs for a run – useful for diagnosing failures.

    Args:
        run_id: The run ID.
        level: Optional filter: "info", "debug", "warning", "error".
        limit: Max number of log entries to return (default: 200, max: 5000).
    """
    params: dict = {"limit": limit}
    if level:
        params["level"] = level
    return _get(f"/runs/{run_id}/logs", **params)


@mcp.tool()
def cancel_run(run_id: str) -> dict:
    """Cancel an active run."""
    return _post(f"/runs/{run_id}/cancel", {})


@mcp.tool()
def list_recent_runs(
    limit: int = 20,
    strategy: Optional[str] = None,
) -> list[dict]:
    """List recent runs with their status and outcome summary.

    Args:
        limit: Number of runs to return (default: 20, max: 100).
        strategy: Optional strategy name filter (substring match on strategy_name).
    """
    result = _get("/runs", limit=min(limit, 100))
    items = result.get("items", result) if isinstance(result, dict) else result
    if strategy:
        items = [r for r in items if strategy.lower() in (r.get("strategy_name") or "").lower()]
    return items


# ---------------------------------------------------------------------------
# Discovery: indicators, exit strategies, presets
# ---------------------------------------------------------------------------

@mcp.tool()
def list_indicators() -> list[dict]:
    """List all available indicator plugins.

    Returns name, fqn (use in pipeline config), description,
    signal_columns (use in signal_rules), and feature_columns.
    """
    plugins = _get("/plugins", phase="indicator")
    return [
        {
            "name": p["name"],
            "fqn": p["fqn"],
            "description": p.get("description", ""),
            "signal_columns": p.get("signal_columns", []),
            "feature_columns": p.get("feature_columns", [])[:10],  # truncate
        }
        for p in plugins
    ]


@mcp.tool()
def get_indicator_schema(name: str) -> dict:
    """Get full schema for an indicator: param types, defaults, signal/feature columns.

    `name` can be the short name (e.g. "opening_range") or full fqn
    (e.g. "fwbg-core:opening_range"). Use the fqn from list_indicators.
    """
    # Try both namespaces
    for fqn in [f"fwbg-core:{name}", f"fwbg-premium:{name}", name]:
        try:
            return _get(f"/plugins/{fqn}")
        except httpx.HTTPStatusError:
            continue
    raise ValueError(f"Indicator not found: {name}")


@mcp.tool()
def list_exit_strategies() -> list[dict]:
    """List all available exit strategy plugins.

    Returns name, fqn, description, and param_schema.
    Key params for grid optimization: tp_mult (TP as ATR multiple),
    sl_mult (SL as ATR multiple), timeout_bars (max bars in trade).
    """
    plugins = _get("/plugins", phase="exit_strategy")
    return [
        {
            "name": p["name"],
            "fqn": p["fqn"],
            "description": p.get("description", ""),
            "param_schema": p.get("param_schema", {}),
            "defaults": p.get("defaults", {}),
        }
        for p in plugins
    ]


@mcp.tool()
def list_presets(preset_type: str) -> list[str]:
    """List available preset names for a given type.

    Args:
        preset_type: One of: "pipelines", "models", "validations",
                     "filters", "resources", "grids", "regime_filters".

    Returns list of preset names (use as string values in strategy config,
    e.g. `"pipeline": "sr_trend_v1"`).
    """
    import pathlib

    strategies_dir = pathlib.Path("/home/haex/Projekte/fwbg/strategies")
    preset_dir = strategies_dir / preset_type
    if not preset_dir.exists():
        return []
    return sorted(p.stem for p in preset_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
