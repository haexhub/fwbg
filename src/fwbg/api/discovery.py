"""Feature discovery — find indicators that distinguish winning from losing trades.

Pipeline:
1. Load trades from fold_results (entry_time + pnl)
2. Load OHLCV data, compute ALL available indicators
3. Sample indicator values at each trade's entry_time
4. Split into wins vs losses, run statistical tests
5. Return ranking by discriminative power

Uses SSE (Server-Sent Events) to stream progress and partial results.
"""
import asyncio
import json
import logging
import math
import os
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
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
    "multi_timeframe",        # Needs multi-TF data
    "macro_surprise",         # Needs external macro data
    "market_regime",          # Duplicate of regime
    "regime_cluster",         # Depends on regime
})

# Per-indicator timeout (seconds) for the thread pool future
_INDICATOR_TIMEOUT = 120
# Max parallel workers for indicator computation
_MAX_WORKERS = min(8, (os.cpu_count() or 4))


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


def _ensure_registry_initialized() -> None:
    """Pre-initialize the plugin registry once (avoids repeated auto_discover)."""
    from fwbg.pipeline.registry import get_registry
    registry = get_registry()
    registry.auto_discover()


def _compute_single_indicator(
    df: pd.DataFrame, ind_name: str,
    params: dict | None = None,
) -> tuple[str, pd.DataFrame, float]:
    """Compute a single indicator, return (name, result_df, elapsed_seconds).

    When *params* is provided the indicator is computed with those params
    (matching the strategy pipeline config) instead of plugin defaults.
    """
    t0 = time.monotonic()
    if params is not None:
        result = compute_indicator_pool(
            df, indicators=[{"name": ind_name, "params": params}],
        )
    else:
        result = compute_indicator_pool(df, indicators=[ind_name])
    elapsed = time.monotonic() - t0
    return ind_name, result, elapsed


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
            lo = vals_clean[dir_loss]
            if len(w) < 3 or len(lo) < 3:
                continue

            d = _compute_effect_size(w, lo)
            try:
                _, p = mannwhitneyu(w, lo, alternative="two-sided")
            except ValueError:
                p = 1.0

            entry = {
                "feature": col,
                "effect_size": round(d, 4),
                "abs_effect_size": round(abs(d), 4),
                "p_value": round(float(p), 6),
                "significant": bool(p < 0.05),
                "win_mean": round(float(np.mean(w)), 6),
                "loss_mean": round(float(np.mean(lo)), 6),
            }
            if indicator_map and col in indicator_map:
                entry["indicator"] = indicator_map[col]
            dir_features.append(entry)

        dir_features.sort(key=lambda x: x["abs_effect_size"], reverse=True)
        direction_results[dir_label] = dir_features[:50]

    return direction_results


def _analyze_combinations_for_mask(
    sampled: pd.DataFrame, top_features: list[dict],
    trade_mask: np.ndarray, win_mask: np.ndarray,
    indicator_map: dict[str, str] | None = None,
    max_features: int = 20,
    min_trades: int = 15,
) -> list[dict]:
    """Find synergistic feature pairs for a given subset of trades.

    ``trade_mask`` selects which trades to consider (e.g. all, long-only,
    short-only).  ``win_mask`` marks which of those are wins (same length
    as ``sampled``; entries outside ``trade_mask`` are ignored).
    """
    from scipy.stats import fisher_exact

    candidates = top_features[:max_features]
    if len(candidates) < 2:
        return []

    # Restrict to the trade subset
    sub_win = win_mask & trade_mask
    total_trades = int(trade_mask.sum())
    total_wins = int(sub_win.sum())
    base_wr = total_wins / total_trades if total_trades else 0

    feature_masks: dict[str, np.ndarray] = {}
    feature_wr: dict[str, float] = {}
    feature_info: dict[str, dict] = {}

    for feat in candidates:
        col = feat["feature"]
        if col not in sampled.columns:
            continue
        vals = sampled[col].values
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        median_val = np.median(valid)
        vals_clean = np.where(np.isnan(vals), median_val, vals)

        threshold = (feat["win_mean"] + feat["loss_mean"]) / 2
        if feat["effect_size"] < 0:
            val_mask = vals_clean < threshold
            op = "<"
        else:
            val_mask = vals_clean > threshold
            op = ">"

        # Intersect with trade subset
        mask = val_mask & trade_mask
        n_pass = int(mask.sum())
        if n_pass < min_trades:
            continue

        wr = float(sub_win[mask].sum()) / n_pass
        feature_masks[col] = mask
        feature_wr[col] = wr
        feature_info[col] = {
            "threshold": round(threshold, 4),
            "op": op,
            "indicator": indicator_map.get(col, "") if indicator_map else "",
        }

    cols = list(feature_masks.keys())
    results = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a, col_b = cols[i], cols[j]
            ind_a = feature_info[col_a]["indicator"]
            ind_b = feature_info[col_b]["indicator"]
            if ind_a and ind_b and ind_a == ind_b:
                continue

            combined = feature_masks[col_a] & feature_masks[col_b]
            n_combined = int(combined.sum())
            if n_combined < min_trades:
                continue

            combined_wins = int(sub_win[combined].sum())
            combined_wr = combined_wins / n_combined

            best_individual_wr = max(feature_wr[col_a], feature_wr[col_b])
            synergy = combined_wr - best_individual_wr

            combined_losses = n_combined - combined_wins
            other_mask = trade_mask & ~combined
            other_wins = int(sub_win[other_mask].sum())
            other_losses = int(other_mask.sum()) - other_wins

            try:
                _, p_value = fisher_exact([
                    [combined_wins, combined_losses],
                    [other_wins, other_losses],
                ])
            except ValueError:
                p_value = 1.0

            results.append({
                "feature_a": col_a,
                "feature_b": col_b,
                "indicator_a": ind_a,
                "indicator_b": ind_b,
                "op_a": feature_info[col_a]["op"],
                "op_b": feature_info[col_b]["op"],
                "threshold_a": feature_info[col_a]["threshold"],
                "threshold_b": feature_info[col_b]["threshold"],
                "wr_a": round(feature_wr[col_a], 4),
                "wr_b": round(feature_wr[col_b], 4),
                "wr_combined": round(combined_wr, 4),
                "base_wr": round(base_wr, 4),
                "synergy": round(synergy, 4),
                "lift": round(combined_wr - base_wr, 4),
                "n_trades": n_combined,
                "n_wins": combined_wins,
                "p_value": round(float(p_value), 6),
                "significant": bool(p_value < 0.05),
            })

    results.sort(key=lambda x: x["synergy"], reverse=True)

    # Deduplicate: keep only the best synergy per indicator pair
    seen_pairs: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for r in results:
        pair = tuple(sorted([r["indicator_a"], r["indicator_b"]]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append(r)

    return deduped[:50]


def _analyze_combinations(
    sampled: pd.DataFrame, direction_features: dict[str, list[dict]],
    directions: list[str], pnls: np.ndarray,
    win_mask: np.ndarray,
    indicator_map: dict[str, str] | None = None,
    max_features: int = 20,
    min_trades: int = 15,
) -> dict[str, list[dict]]:
    """Compute pairwise feature combinations per direction.

    Returns ``{"long": [...], "short": [...]}``.  Each direction uses its
    own top features (from ``direction_features``) and only considers
    trades of that direction.
    """
    dir_arrays = np.array(directions)
    result: dict[str, list[dict]] = {}

    for dir_name, dir_label in [("LONG", "long"), ("SHORT", "short")]:
        dir_mask = dir_arrays == dir_name
        if int(dir_mask.sum()) < min_trades:
            result[dir_label] = []
            continue

        top = direction_features.get(dir_label, [])
        if len(top) < 2:
            result[dir_label] = []
            continue

        result[dir_label] = _analyze_combinations_for_mask(
            sampled, top, dir_mask, win_mask,
            indicator_map, max_features, min_trades,
        )

    return result


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, cls=_NumpyEncoder)}\n\n"


def _cache_path(run_id: str, symbol: str) -> Path:
    return get_test_results_dir() / run_id / "discovery" / f"{symbol}.json"


def _load_cached_discovery(run_id: str, symbol: str) -> dict | None:
    path = _cache_path(run_id, symbol)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _save_discovery_cache(run_id: str, symbol: str, result: dict) -> None:
    path = _cache_path(run_id, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, cls=_NumpyEncoder, ensure_ascii=False, indent=2))


async def _discovery_stream(run_id: str, symbol: str, force: bool = False) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events during discovery.

    Indicators are computed in parallel via asyncio.run_in_executor so that
    cancellation propagates cleanly when the client disconnects or the server
    is shut down with Ctrl+C — no thread-join deadlock.
    """
    if not force:
        cached = _load_cached_discovery(run_id, symbol)
        if cached is not None:
            cached["cached"] = True
            yield _sse_event("done", cached)
            return

    try:
        strategy = _load_strategy_config(run_id)
        run_config = _load_run_config(run_id)
        timeframe = run_config.get("timeframe") or strategy.get("timeframe") or "HOUR"
        trades = _load_trades(run_id, symbol)
        csv_path = _resolve_csv_path(strategy, symbol, timeframe)
    except HTTPException as exc:
        yield _sse_event("error", {"message": exc.detail})
        return

    if len(trades) < 20:
        yield _sse_event("error", {"message": f"Too few trades ({len(trades)}) for meaningful discovery"})
        return

    entry_times = pd.to_datetime([t["entry_time"] for t in trades])
    pnls = np.array([t["pnl_raw"] for t in trades])
    directions = [t.get("direction", "LONG") for t in trades]

    win_mask = pnls > 0
    loss_mask = pnls < 0

    # Load OHLCV data
    df = load_data_aligned(str(csv_path))
    if df is None or df.empty:
        yield _sse_event("error", {"message": f"Failed to load data from {csv_path}"})
        return

    # Trim to trade window (with lookback for indicator warm-up)
    lookback_bars = 500
    first_trade = entry_times.min()
    last_trade = entry_times.max()
    start_idx = max(0, df.index.searchsorted(first_trade) - lookback_bars)
    end_idx = min(len(df), df.index.searchsorted(last_trade) + 1)
    df = df.iloc[start_idx:end_idx].copy()

    # Pre-initialize registry once before spawning workers
    _ensure_registry_initialized()

    # Build params lookup from the strategy's pipeline config so that
    # indicators already in the pipeline are computed with the same params
    # (producing the same column names the strategy actually uses).
    pipeline_params: dict[str, dict] = {}
    for ind_cfg in strategy.get("pipeline", {}).get("indicators", []):
        if isinstance(ind_cfg, dict) and ind_cfg.get("name"):
            pipeline_params[ind_cfg["name"]] = ind_cfg.get("params", {})

    indicator_names = _get_available_indicators()
    total_steps = len(indicator_names)

    yield _sse_event("init", {
        "total_trades": len(trades),
        "wins": int(win_mask.sum()),
        "losses": int(loss_mask.sum()),
        "total_indicators": total_steps,
        "bars": len(df),
        "workers": _MAX_WORKERS,
    })

    # ── Async parallel indicator computation ──
    computed_indicators: list[str] = []
    indicator_map: dict[str, str] = {}  # column → indicator name
    base_cols = {"O", "H", "L", "C", "V"}
    all_new_cols: dict[str, pd.DataFrame] = {}  # ind_name → result df
    done_count = 0

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
    tasks: list[asyncio.Task] = []

    async def _compute_safe(name: str) -> tuple[str, pd.DataFrame | None, float, str | None]:
        """Run one indicator; always returns (name, result, elapsed, error_msg)."""
        try:
            params = pipeline_params.get(name)
            ind_name, result, elapsed = await asyncio.wait_for(
                loop.run_in_executor(
                    pool, _compute_single_indicator, df, name, params,
                ),
                timeout=_INDICATOR_TIMEOUT,
            )
            return ind_name, result, elapsed, None
        except TimeoutError:
            return name, None, 0.0, f"Timeout after {_INDICATOR_TIMEOUT}s"
        except Exception as exc:
            return name, None, 0.0, str(exc)

    try:
        tasks = [asyncio.create_task(_compute_safe(name)) for name in indicator_names]
        for coro in asyncio.as_completed(tasks):
            ind_name, result, elapsed, err = await coro
            done_count += 1

            if err:
                yield _sse_event("indicator_skip", {
                    "step": done_count,
                    "total": total_steps,
                    "indicator": ind_name,
                    "reason": err,
                })
                continue

            new_cols = [c for c in result.columns if c not in base_cols and c not in df.columns]
            if new_cols:
                all_new_cols[ind_name] = result[new_cols]
                computed_indicators.append(ind_name)
                for col in new_cols:
                    indicator_map[col] = ind_name

            yield _sse_event("indicator_done", {
                "step": done_count,
                "total": total_steps,
                "indicator": ind_name,
                "columns": len(new_cols),
                "elapsed": round(elapsed, 1),
            })

    finally:
        for task in tasks:
            task.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    # ── Merge all indicator results at once ──
    df_full = df
    if all_new_cols:
        df_full = pd.concat([df, *all_new_cols.values()], axis=1)

    feature_cols = [c for c in df_full.columns
                    if c not in base_cols and not c.startswith("_")]

    yield _sse_event("progress", {
        "step": total_steps,
        "total": total_steps,
        "indicator": "Finale Analyse & Kombinationen...",
    })

    # ── Statistical analysis (single pass) ──
    sampled = df_full.reindex(entry_times, method="nearest", tolerance=pd.Timedelta("2h"))
    all_results = _analyze_features(sampled, feature_cols, win_mask, loss_mask, indicator_map)
    direction_results = _analyze_direction(
        sampled, feature_cols, directions, pnls, win_mask, loss_mask, indicator_map,
    )
    combinations = _analyze_combinations(
        sampled, direction_results, directions, pnls, win_mask, indicator_map,
    )

    done_payload = {
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
        "combinations": combinations,
    }

    try:
        _save_discovery_cache(run_id, symbol, done_payload)
    except Exception as e:
        log.warning("Failed to cache discovery results for %s/%s: %s", run_id, symbol, e)

    yield _sse_event("done", done_payload)


@router.get("/{run_id}/discovery/{symbol}")
async def get_feature_discovery(run_id: str, symbol: str, force: bool = False):
    """Feature discovery via SSE stream.

    Results are cached in test_results/{run_id}/discovery/{symbol}.json.
    Pass ?force=true to recompute even if a cache exists.
    """
    return StreamingResponse(
        _discovery_stream(run_id, symbol, force=force),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
