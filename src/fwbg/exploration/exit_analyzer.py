"""MFE/MAE Exit Analyzer — Orchestration, Grid Suggestion, Output.

Analyzes an asset's price data to determine optimal TP/SL ranges via
Maximum Favorable/Adverse Excursion analysis.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tabulate import tabulate

from fwbg.exploration.mfe_mae import compute_capture_rates, compute_mfe_mae

_PERCENTILES = [10, 25, 50, 60, 75, 85, 90, 95]
_SCAN_RANGE = np.round(np.arange(0.2, 4.1, 0.1), 1)


def analyze_asset(data_file, exit_strategy="atr_based", exit_params=None, max_bars=48):
    """Run full MFE/MAE analysis on a single asset.

    Returns a dict with mfe_mae stats, capture_matrix, and suggested_grid.
    """
    if exit_params is None:
        exit_params = {"atr_period": 14}

    df = _load_ohlcv(data_file)
    open_ = df["O"].values.astype(np.float64)
    high = df["H"].values.astype(np.float64)
    low = df["L"].values.astype(np.float64)
    close = df["C"].values.astype(np.float64)

    atr_period = exit_params.get("atr_period", 14)
    atr = _compute_atr(high, low, close, atr_period)

    # --- MFE/MAE ---
    mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars)

    # Normalize by ATR (skip bars with ATR=0 or NaN MFE)
    valid = np.isfinite(mfe_l) & (atr > 0)
    mfe_l_norm = mfe_l[valid] / atr[valid]
    mae_l_norm = mae_l[valid] / atr[valid]
    mfe_s_norm = mfe_s[valid] / atr[valid]
    mae_s_norm = mae_s[valid] / atr[valid]

    mfe_mae = {
        "long": {
            "mfe_percentiles": _compute_percentiles(mfe_l_norm),
            "mae_percentiles": _compute_percentiles(mae_l_norm),
        },
        "short": {
            "mfe_percentiles": _compute_percentiles(mfe_s_norm),
            "mae_percentiles": _compute_percentiles(mae_s_norm),
        },
    }

    # --- Capture Rates ---
    tp_scan = _SCAN_RANGE.copy()
    sl_scan = _SCAN_RANGE.copy()
    wr_long, wr_short, trade_counts = compute_capture_rates(
        open_, high, low, atr, tp_scan, sl_scan, max_bars
    )
    capture_matrix = _build_capture_matrix(wr_long, wr_short, trade_counts, tp_scan, sl_scan)

    # --- Grid Suggestion ---
    suggested_grid = _suggest_grid(
        mfe_mae["long"]["mfe_percentiles"],
        mfe_mae["short"]["mfe_percentiles"],
        mfe_mae["long"]["mae_percentiles"],
        mfe_mae["short"]["mae_percentiles"],
    )

    # --- Metadata ---
    basename = os.path.basename(data_file).replace(".csv", "")
    parts = basename.rsplit("_", 1)
    symbol = parts[0] if len(parts) > 1 else basename
    timeframe = parts[1] if len(parts) > 1 else "UNKNOWN"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": os.path.basename(data_file),
        "bars_analyzed": int(np.sum(valid)),
        "max_bars_forward": max_bars,
        "exit_strategy": exit_strategy,
        "exit_params": exit_params,
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mfe_mae": mfe_mae,
        "capture_matrix": capture_matrix,
        "suggested_grid": suggested_grid,
    }


def format_terminal_output(result):
    """Format analysis result as readable terminal output."""
    lines = []

    # MFE/MAE Percentile Table
    for direction in ("long", "short"):
        data = result["mfe_mae"][direction]
        rows = []
        for p in _PERCENTILES:
            sp = str(p)
            rows.append([
                f"P{p}",
                f"{data['mfe_percentiles'].get(sp, 0):.2f}",
                f"{data['mae_percentiles'].get(sp, 0):.2f}",
            ])
        lines.append(f"\n  MFE/MAE Percentiles ({direction.upper()}) [ATR multiples]")
        lines.append(tabulate(rows, headers=["Percentile", "MFE", "MAE"], tablefmt="simple"))

    # Top Capture Matrix entries
    matrix = result["capture_matrix"][:15]
    if matrix:
        rows = []
        for entry in matrix:
            rows.append([
                f"{entry['tp']:.1f}",
                f"{entry['sl']:.1f}",
                f"{entry['rrr']:.2f}",
                f"{entry['win_rate_long']:.1%}",
                f"{entry['win_rate_short']:.1%}",
                f"{entry['edge_long']:+.3f}",
                f"{entry['edge_short']:+.3f}",
            ])
        lines.append("\n  Top TP/SL Combinations (by long edge)")
        lines.append(tabulate(
            rows,
            headers=["TP", "SL", "RRR", "WR Long", "WR Short", "Edge L", "Edge S"],
            tablefmt="simple",
        ))

    # Suggested Grid
    grid = result["suggested_grid"]
    lines.append("\n  Suggested Grid:")
    lines.append(f"    TP: {grid['tp']}")
    lines.append(f"    SL: {grid['sl']}")
    lines.append(f"    Reasoning: {grid['reasoning']}")

    return "\n".join(lines)


def write_json(result, output_path):
    """Write result dict to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_ohlcv(data_file):
    """Load OHLCV from CSV. Expects columns T,O,H,L,C,V or positional."""
    df = pd.read_csv(data_file)
    if "O" not in df.columns:
        # Positional: Date, Open, High, Low, Close, Volume
        df.columns = ["T", "O", "H", "L", "C", "V"][:len(df.columns)]
    return df


def _compute_atr(high, low, close, period):
    """Compute ATR via exponential moving average of True Range."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.empty(n)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    # Fill leading NaN with first valid ATR
    first_valid = atr[period - 1]
    atr[:period - 1] = first_valid

    return atr


def _compute_percentiles(values):
    """Compute percentiles, return as {str_pct: float} dict."""
    if len(values) == 0:
        return {str(p): 0.0 for p in _PERCENTILES}
    return {str(p): round(float(np.percentile(values, p)), 4) for p in _PERCENTILES}


def _build_capture_matrix(wr_long, wr_short, trade_counts, tp_vals, sl_vals):
    """Build sorted capture matrix from grid results."""
    entries = []
    for ti, tp in enumerate(tp_vals):
        for si, sl in enumerate(sl_vals):
            if trade_counts[ti, si] == 0:
                continue
            rrr = tp / sl if sl > 0 else 0
            wl = wr_long[ti, si]
            ws = wr_short[ti, si]
            # Edge = E[PnL] per trade in ATR units: WR*TP - (1-WR)*SL
            edge_l = wl * tp - (1 - wl) * sl
            edge_s = ws * tp - (1 - ws) * sl
            entries.append({
                "tp": round(float(tp), 1),
                "sl": round(float(sl), 1),
                "rrr": round(rrr, 2),
                "win_rate_long": round(float(wl), 4),
                "win_rate_short": round(float(ws), 4),
                "edge_long": round(float(edge_l), 4),
                "edge_short": round(float(edge_s), 4),
                "resolved_trades": int(trade_counts[ti, si]),
            })

    entries.sort(key=lambda e: e["edge_long"], reverse=True)
    return entries


def _suggest_grid(mfe_pct_long, mfe_pct_short, mae_pct_long, mae_pct_short):
    """Derive TP/SL grid suggestion from MFE/MAE percentiles."""
    # TP candidates from MFE: P50, P60, P75 (catch 50-75% of favorable moves)
    tp_candidates = set()
    for pct in mfe_pct_long, mfe_pct_short:
        for key in ("50", "60", "75"):
            val = pct.get(key, 0)
            if val > 0:
                tp_candidates.add(round(val, 1))

    # SL candidates from MAE: P75, P85, P90 (protect from 75-90% of adverse moves)
    sl_candidates = set()
    for pct in mae_pct_long, mae_pct_short:
        for key in ("75", "85", "90"):
            val = pct.get(key, 0)
            if val > 0:
                sl_candidates.add(round(val, 1))

    tp_sorted = sorted(tp_candidates)
    sl_sorted = sorted(sl_candidates)

    # Ensure at least 2 values per dimension
    if len(tp_sorted) < 2:
        tp_sorted = [0.5, 1.0, 1.5]
    if len(sl_sorted) < 2:
        sl_sorted = [0.8, 1.0, 1.5]

    # Limit to 4 values max
    tp_sorted = tp_sorted[:4]
    sl_sorted = sl_sorted[:4]

    mfe_p50_l = mfe_pct_long.get("50", 0)
    mfe_p75_l = mfe_pct_long.get("75", 0)
    mae_p75_l = mae_pct_long.get("75", 0)
    mae_p90_l = mae_pct_long.get("90", 0)

    reasoning = (
        f"MFE P50={mfe_p50_l:.2f}, P75={mfe_p75_l:.2f} -> TP range {tp_sorted[0]}-{tp_sorted[-1]}. "
        f"MAE P75={mae_p75_l:.2f}, P90={mae_p90_l:.2f} -> SL range {sl_sorted[0]}-{sl_sorted[-1]}"
    )

    return {
        "tp": tp_sorted,
        "sl": sl_sorted,
        "reasoning": reasoning,
    }


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
