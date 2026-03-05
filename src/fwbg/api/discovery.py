"""Feature discovery — find indicators that distinguish winning from losing trades.

Pipeline:
1. Load trades from fold_results (entry_time + pnl)
2. Load OHLCV data, compute ALL available indicators
3. Sample indicator values at each trade's entry_time
4. Split into wins vs losses, run statistical tests
5. Return ranking by discriminative power

Uses SSE (Server-Sent Events) to stream progress and partial results.
"""
import json
import logging
import math
import threading
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from fwbg.api.deps import get_test_results_dir
from fwbg.data import load_data_aligned
from fwbg.pipeline.features import compute_indicator_pool

log = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])

# Indicators to skip (need external data, special config, or internal-only)
_SKIP_INDICATORS = frozenset({
    "autoencoder_features",   # Needs training data
    "adversarial_validation", # Needs train/test split
    "calendar_events",        # Needs external data
    "cross_features",         # Needs specific feature pairs
    "computed_signal",        # Internal (signal composition)
    "cusum_events",           # Needs specific config
    "distribution",           # Extremely slow (O(n²) rolling)
    "multi_timeframe",        # Needs multi-TF data
    "macro_surprise",         # Needs external macro data
    "market_regime",          # Duplicate of regime
    "regime_cluster",         # Depends on regime
    "topological_features",   # Very slow, niche
})

# Per-indicator wall-clock timeout (seconds) — computed in a daemon thread
_INDICATOR_TIMEOUT = 120


def _load_run_config(run_id: str) -> dict:
    results_dir = get_test_results_dir()
    config_file = results_dir / run_id / "config.json"
    if not config_file.exists():
        raise HTTPException(404, f"Run config not found: {run_id}")
    return json.loads(config_file.read_text())


def _load_strategy_config(run_id: str) -> dict:
    results_dir = get_test_results_dir()
    strat_file = results_dir / run_id / "strategy.json"
    if not strat_file.exists():
        raise HTTPException(404, f"Strategy config not found: {run_id}")
    return json.loads(strat_file.read_text())


def _load_trades(run_id: str, symbol: str) -> list[dict]:
    results_dir = get_test_results_dir()
    fold_file = results_dir / run_id / "grid_details" / symbol / "fold_results.json"
    if not fold_file.exists():
        raise HTTPException(404, f"No fold results for {run_id}/{symbol}")

    fdata = json.loads(fold_file.read_text())
    trades = []
    for fold in fdata.get("walk_forward", {}).get("fold_details", []):
        for t in fold.get("test_trades_detail", []):
            if isinstance(t, dict) and "pnl_raw" in t and "entry_time" in t:
                trades.append(t)
    return trades


def _resolve_csv_path(strategy: dict, symbol: str, timeframe: str) -> Path:
    """Resolve the OHLCV CSV path from datasource config."""
    ds_name = strategy.get("datasource")
    if not ds_name:
        raise HTTPException(400, "Strategy has no datasource configured")

    from fwbg.core.data_sources import get_data_source, CSVSourceConfig
    try:
        ds = get_data_source(ds_name)
    except ValueError:
        raise HTTPException(404, f"Datasource not found: {ds_name}")

    if not isinstance(ds, CSVSourceConfig):
        raise HTTPException(400, f"Datasource '{ds_name}' is not a CSV source")

    csv_path = ds.get_file_path(symbol, timeframe)
    if not csv_path.exists():
        raise HTTPException(404, f"No data file for {symbol}/{timeframe}")
    return csv_path


def _get_available_indicators() -> list[str]:
    """Get all indicator plugin names that can be computed."""
    from fwbg.pipeline.registry import get_registry, PluginPhase
    registry = get_registry()
    registry.auto_discover()
    all_plugins = registry.list_plugins(phase=PluginPhase.INDICATORS)
    names = []
    for fqn in all_plugins:
        short = fqn.split(":")[-1] if ":" in fqn else fqn
        if short not in _SKIP_INDICATORS:
            names.append(short)
    return sorted(names)


def _compute_indicator_with_timeout(
    df: pd.DataFrame, ind_name: str, timeout: int = _INDICATOR_TIMEOUT,
) -> pd.DataFrame | None:
    """Compute a single indicator in a daemon thread with wall-clock timeout."""
    result: list[pd.DataFrame | None] = [None]
    error: list[Exception | None] = [None]

    def worker():
        try:
            result[0] = compute_indicator_pool(df, indicators=[ind_name])
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(f"{ind_name} exceeded {timeout}s timeout")
    if error[0]:
        raise error[0]
    return result[0]


def _compute_effect_size(wins: np.ndarray, losses: np.ndarray) -> float:
    """Cohen's d — standardized mean difference."""
    n1, n2 = len(wins), len(losses)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(wins), np.mean(losses)
    s1, s2 = np.var(wins, ddof=1), np.var(losses, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((m1 - m2) / pooled_std)


def _compute_auc(wins: np.ndarray, losses: np.ndarray) -> float:
    """AUC via Mann-Whitney U statistic (probability that a random win > random loss)."""
    from scipy.stats import mannwhitneyu
    try:
        u_stat, _ = mannwhitneyu(wins, losses, alternative="two-sided")
        return float(u_stat / (len(wins) * len(losses)))
    except ValueError:
        return 0.5


def _analyze_features(
    sampled: pd.DataFrame, feature_cols: list[str],
    win_mask: np.ndarray, loss_mask: np.ndarray,
    indicator_map: dict[str, str] | None = None,
) -> list[dict]:
    """Run statistical tests on all feature columns."""
    from scipy.stats import mannwhitneyu

    results = []
    for col in feature_cols:
        vals = sampled[col].values
        if np.isnan(vals).sum() > len(vals) * 0.5:
            continue

        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        median_val = np.median(valid)
        vals_clean = np.where(np.isnan(vals), median_val, vals)

        win_vals = vals_clean[win_mask]
        loss_vals = vals_clean[loss_mask]

        if len(win_vals) < 5 or len(loss_vals) < 5:
            continue

        d = _compute_effect_size(win_vals, loss_vals)

        try:
            _, p_value = mannwhitneyu(win_vals, loss_vals, alternative="two-sided")
        except ValueError:
            p_value = 1.0

        auc = _compute_auc(win_vals, loss_vals)

        entry = {
            "feature": col,
            "effect_size": round(d, 4),
            "abs_effect_size": round(abs(d), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < 0.05),
            "auc": round(auc, 4),
            "win_mean": round(float(np.mean(win_vals)), 6),
            "loss_mean": round(float(np.mean(loss_vals)), 6),
            "win_std": round(float(np.std(win_vals)), 6),
            "loss_std": round(float(np.std(loss_vals)), 6),
            "n_valid": int(np.sum(~np.isnan(vals))),
        }
        if indicator_map and col in indicator_map:
            entry["indicator"] = indicator_map[col]
        results.append(entry)

    results.sort(key=lambda x: x["abs_effect_size"], reverse=True)
    return results


def _analyze_direction(
    sampled: pd.DataFrame, feature_cols: list[str],
    directions: list[str], pnls: np.ndarray,
    win_mask: np.ndarray, loss_mask: np.ndarray,
    indicator_map: dict[str, str] | None = None,
) -> dict:
    """Per-direction discovery (long/short)."""
    from scipy.stats import mannwhitneyu

    direction_results = {}
    for dir_name, dir_label in [("LONG", "long"), ("SHORT", "short")]:
        dir_mask = np.array([d == dir_name for d in directions])
        dir_pnls = pnls[dir_mask]
        if len(dir_pnls) < 10:
            direction_results[dir_label] = []
            continue

        dir_win = dir_mask & win_mask
        dir_loss = dir_mask & loss_mask
        if dir_win.sum() < 3 or dir_loss.sum() < 3:
            direction_results[dir_label] = []
            continue

        dir_features = []
        for col in feature_cols:
            vals = sampled[col].values
            if np.isnan(vals).sum() > len(vals) * 0.5:
                continue
            valid = vals[~np.isnan(vals)]
            if len(valid) < 10:
                continue
            median_val = np.median(valid)
            vals_clean = np.where(np.isnan(vals), median_val, vals)

            w = vals_clean[dir_win]
            l = vals_clean[dir_loss]
            if len(w) < 3 or len(l) < 3:
                continue

            d = _compute_effect_size(w, l)
            try:
                _, p = mannwhitneyu(w, l, alternative="two-sided")
            except ValueError:
                p = 1.0

            entry = {
                "feature": col,
                "effect_size": round(d, 4),
                "abs_effect_size": round(abs(d), 4),
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05),
                "win_mean": round(float(np.mean(w)), 6),
                "loss_mean": round(float(np.mean(l)), 6),
            }
            if indicator_map and col in indicator_map:
                entry["indicator"] = indicator_map[col]
            dir_features.append(entry)

        dir_features.sort(key=lambda x: x["abs_effect_size"], reverse=True)
        direction_results[dir_label] = dir_features[:50]

    return direction_results


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _discovery_stream(run_id: str, symbol: str) -> Generator[str, None, None]:
    """Generator that yields SSE events during discovery."""
    strategy = _load_strategy_config(run_id)
    run_config = _load_run_config(run_id)
    timeframe = run_config.get("timeframe") or strategy.get("timeframe") or "HOUR"

    # Load trades
    trades = _load_trades(run_id, symbol)
    if len(trades) < 20:
        yield _sse_event("error", {"message": f"Too few trades ({len(trades)}) for meaningful discovery"})
        return

    entry_times = pd.to_datetime([t["entry_time"] for t in trades])
    pnls = np.array([t["pnl_raw"] for t in trades])
    directions = [t.get("direction", "LONG") for t in trades]

    win_mask = pnls > 0
    loss_mask = pnls < 0

    # Load OHLCV data
    csv_path = _resolve_csv_path(strategy, symbol, timeframe)
    df = load_data_aligned(str(csv_path))
    if df is None or df.empty:
        yield _sse_event("error", {"message": f"Failed to load data from {csv_path}"})
        return

    # Trim to trade window
    lookback_bars = 500
    first_trade = entry_times.min()
    last_trade = entry_times.max()
    start_idx = max(0, df.index.searchsorted(first_trade) - lookback_bars)
    end_idx = min(len(df), df.index.searchsorted(last_trade) + 1)
    df = df.iloc[start_idx:end_idx].copy()

    indicator_names = _get_available_indicators()
    total_steps = len(indicator_names)

    yield _sse_event("init", {
        "total_trades": len(trades),
        "wins": int(win_mask.sum()),
        "losses": int(loss_mask.sum()),
        "total_indicators": total_steps,
        "bars": len(df),
    })

    # Compute indicators one by one, yielding progress + partial results
    computed_indicators = []
    indicator_map: dict[str, str] = {}  # column → indicator name
    df_full = df.copy()
    base_cols = {"O", "H", "L", "C", "V"}

    for step, ind_name in enumerate(indicator_names, 1):
        yield _sse_event("progress", {
            "step": step,
            "total": total_steps,
            "indicator": ind_name,
        })

        t0 = time.monotonic()
        try:
            result = _compute_indicator_with_timeout(df, ind_name)
            elapsed = time.monotonic() - t0
            if result is None:
                continue
            new_cols = [c for c in result.columns if c not in df_full.columns]
            if new_cols:
                df_full = df_full.join(result[new_cols])
                computed_indicators.append(ind_name)
                for col in new_cols:
                    indicator_map[col] = ind_name

                # Analyze new features immediately
                new_feature_cols = [c for c in new_cols if not c.startswith("_")]
                if new_feature_cols:
                    sampled = df_full.reindex(entry_times, method="nearest", tolerance=pd.Timedelta("2h"))
                    partial_results = _analyze_features(
                        sampled, new_feature_cols, win_mask, loss_mask, indicator_map,
                    )
                    significant = [r for r in partial_results if r["significant"]]

                    yield _sse_event("indicator_done", {
                        "indicator": ind_name,
                        "columns": len(new_cols),
                        "elapsed": round(elapsed, 1),
                        "features_analyzed": len(partial_results),
                        "significant_count": len(significant),
                        "top_features": partial_results[:5],
                    })
                else:
                    yield _sse_event("indicator_done", {
                        "indicator": ind_name,
                        "columns": len(new_cols),
                        "elapsed": round(elapsed, 1),
                        "features_analyzed": 0,
                        "significant_count": 0,
                        "top_features": [],
                    })
        except Exception as e:
            yield _sse_event("indicator_skip", {
                "indicator": ind_name,
                "reason": str(e),
            })

    # Final analysis across all features
    feature_cols = [c for c in df_full.columns
                    if c not in base_cols and not c.startswith("_")]

    sampled = df_full.reindex(entry_times, method="nearest", tolerance=pd.Timedelta("2h"))
    all_results = _analyze_features(sampled, feature_cols, win_mask, loss_mask, indicator_map)
    direction_results = _analyze_direction(
        sampled, feature_cols, directions, pnls, win_mask, loss_mask, indicator_map,
    )

    yield _sse_event("done", {
        "run_id": run_id,
        "symbol": symbol,
        "strategy_name": strategy.get("name", ""),
        "total_trades": len(trades),
        "wins": int(win_mask.sum()),
        "losses": int(loss_mask.sum()),
        "total_features": len(feature_cols),
        "analyzed_features": len(all_results),
        "indicators_computed": computed_indicators,
        "results": all_results[:100],
        "direction": direction_results,
    })


@router.get("/{run_id}/discovery/{symbol}")
def get_feature_discovery(run_id: str, symbol: str):
    """Feature discovery via SSE stream."""
    return StreamingResponse(
        _discovery_stream(run_id, symbol),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
