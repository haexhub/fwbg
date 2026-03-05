"""Statistical analysis for completed backtesting runs."""
import json
import math

import numpy as np
from fastapi import APIRouter, HTTPException

from fwbg.api.deps import get_test_results_dir

router = APIRouter(prefix="/runs", tags=["runs"])


def _load_trades(run_id: str, symbol: str) -> list[dict]:
    """Load detailed trades for a symbol from fold_results.json."""
    results_dir = get_test_results_dir()
    sym_dir = results_dir / run_id / "grid_details" / symbol

    if not sym_dir.exists():
        raise HTTPException(404, f"No data for {run_id}/{symbol}")

    fold_file = sym_dir / "fold_results.json"
    if not fold_file.exists():
        raise HTTPException(404, f"No fold results for {run_id}/{symbol}")

    fdata = json.loads(fold_file.read_text())
    trades: list[dict] = []
    for fold in fdata.get("walk_forward", {}).get("fold_details", []):
        fold_id = fold.get("fold_id")
        for t in fold.get("test_trades_detail", []):
            if isinstance(t, dict) and "pnl_raw" in t:
                trades.append({**t, "fold_id": fold_id})
    return trades


def _direction_stats(trades: list[dict], direction: str) -> dict:
    filtered = [t for t in trades if t.get("direction") == direction]
    if not filtered:
        return {"count": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0}

    pnls = [t["pnl_raw"] for t in filtered]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": len(filtered),
        "win_rate": round(wins / len(filtered) * 100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 4),
        "total_pnl": round(sum(pnls), 4),
    }


def _consecutive_streaks(pnls: list[float]) -> dict:
    """Max consecutive wins and losses."""
    max_wins = max_losses = cur_wins = cur_losses = 0
    for p in pnls:
        if p > 0:
            cur_wins += 1
            cur_losses = 0
        elif p < 0:
            cur_losses += 1
            cur_wins = 0
        else:
            cur_wins = cur_losses = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return {"max_consecutive_wins": max_wins, "max_consecutive_losses": max_losses}


def _drawdown_analysis(pnls: list[float]) -> dict:
    """Drawdown duration and depth from PnL sequence."""
    cumsum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumsum)
    drawdown = peak - cumsum

    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

    # Longest drawdown duration (in trades)
    in_dd = drawdown > 0
    longest_dd = 0
    current_dd = 0
    for v in in_dd:
        if v:
            current_dd += 1
            longest_dd = max(longest_dd, current_dd)
        else:
            current_dd = 0

    return {
        "max_drawdown": round(max_dd, 4),
        "longest_drawdown_trades": longest_dd,
    }


def _hourly_distribution(trades: list[dict]) -> list[dict]:
    """Win rate and trade count by hour."""
    hours: dict[int, list[float]] = {}
    for t in trades:
        h = t.get("hour")
        if h is not None:
            hours.setdefault(h, []).append(t["pnl_raw"])

    result = []
    for h in sorted(hours.keys()):
        pnls = hours[h]
        wins = sum(1 for p in pnls if p > 0)
        result.append({
            "hour": h,
            "count": len(pnls),
            "win_rate": round(wins / len(pnls) * 100, 1),
            "avg_pnl": round(float(np.mean(pnls)), 4),
            "total_pnl": round(sum(pnls), 4),
        })
    return result


def _fold_stability(trades: list[dict]) -> list[dict]:
    """Per-fold performance for temporal stability analysis."""
    folds: dict[int, list[float]] = {}
    for t in trades:
        fid = t.get("fold_id")
        if fid is not None:
            folds.setdefault(fid, []).append(t["pnl_raw"])

    result = []
    for fid in sorted(folds.keys()):
        pnls = folds[fid]
        wins = sum(1 for p in pnls if p > 0)
        result.append({
            "fold_id": fid,
            "count": len(pnls),
            "win_rate": round(wins / len(pnls) * 100, 1),
            "avg_pnl": round(float(np.mean(pnls)), 4),
            "total_pnl": round(sum(pnls), 4),
            "profit_factor": round(
                sum(p for p in pnls if p > 0) / max(sum(abs(p) for p in pnls if p < 0), 1e-9),
                2,
            ),
        })
    return result


def _trade_quality(pnls: list[float]) -> dict:
    """Expectancy, payoff ratio, edge metrics."""
    if not pnls:
        return {"expectancy": 0, "payoff_ratio": 0, "sqn": 0, "kelly_pct": 0}

    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    avg_win = float(np.mean(wins)) if wins else 0
    avg_loss = float(np.mean(losses)) if losses else 0
    win_rate = len(wins) / len(pnls)

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # System Quality Number (Van Tharp)
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 1
    sqn = (mean_pnl / std_pnl) * math.sqrt(len(pnls)) if std_pnl > 0 else 0

    # Kelly criterion
    kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio if payoff_ratio > 0 else 0

    return {
        "expectancy": round(expectancy, 4),
        "payoff_ratio": round(payoff_ratio, 2),
        "sqn": round(sqn, 2),
        "kelly_pct": round(max(kelly * 100, 0), 1),
    }


def _bars_held_stats(trades: list[dict]) -> dict:
    """Trade duration statistics."""
    durations = [t["bars_held"] for t in trades if "bars_held" in t]
    if not durations:
        return {"avg": 0, "median": 0, "min": 0, "max": 0}
    return {
        "avg": round(float(np.mean(durations)), 1),
        "median": int(np.median(durations)),
        "min": int(np.min(durations)),
        "max": int(np.max(durations)),
    }


def _significance_test(pnls: list[float]) -> dict:
    """One-sample t-test: is mean PnL significantly > 0?"""
    if len(pnls) < 5:
        return {"t_stat": 0, "p_value": 1.0, "significant": False}

    mean = float(np.mean(pnls))
    std = float(np.std(pnls, ddof=1))
    n = len(pnls)

    if std == 0:
        return {"t_stat": 0, "p_value": 1.0, "significant": False}

    t_stat = mean / (std / math.sqrt(n))

    # Approximate one-tailed p-value using normal distribution for large n
    # For n >= 30 this is a reasonable approximation
    from scipy.stats import t as t_dist
    p_value = float(1 - t_dist.cdf(t_stat, df=n - 1))

    return {
        "t_stat": round(t_stat, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n": n,
    }


def _equity_curve(pnls: list[float]) -> list[float]:
    """Cumulative PnL series for equity curve charting."""
    if not pnls:
        return []
    return [round(float(v), 4) for v in np.cumsum(pnls)]


def _direction_detail(trades: list[dict], direction: str) -> dict:
    """Full analysis for a single direction (LONG or SHORT)."""
    filtered = [t for t in trades if t.get("direction") == direction]
    if not filtered:
        return {
            "count": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0,
            "quality": _trade_quality([]),
            "streaks": {"max_consecutive_wins": 0, "max_consecutive_losses": 0},
            "drawdown": {"max_drawdown": 0, "longest_drawdown_trades": 0},
            "bars_held": {"avg": 0, "median": 0, "min": 0, "max": 0},
            "equity_curve": [],
            "hourly": [],
        }

    pnls = [t["pnl_raw"] for t in filtered]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "count": len(filtered),
        "win_rate": round(wins / len(filtered) * 100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 4),
        "total_pnl": round(sum(pnls), 4),
        "quality": _trade_quality(pnls),
        "streaks": _consecutive_streaks(pnls),
        "drawdown": _drawdown_analysis(pnls),
        "bars_held": _bars_held_stats(filtered),
        "equity_curve": _equity_curve(pnls),
        "hourly": _hourly_distribution(filtered),
    }


def compute_analysis(run_id: str, symbol: str) -> dict:
    """Compute full statistical analysis for a symbol's trades."""
    trades = _load_trades(run_id, symbol)

    if not trades:
        raise HTTPException(404, f"No trades found for {run_id}/{symbol}")

    pnls = [t["pnl_raw"] for t in trades]

    return {
        "run_id": run_id,
        "symbol": symbol,
        "total_trades": len(trades),
        "direction": {
            "long": _direction_detail(trades, "LONG"),
            "short": _direction_detail(trades, "SHORT"),
        },
        "quality": _trade_quality(pnls),
        "streaks": _consecutive_streaks(pnls),
        "drawdown": _drawdown_analysis(pnls),
        "significance": _significance_test(pnls),
        "bars_held": _bars_held_stats(trades),
        "equity_curve": _equity_curve(pnls),
        "hourly": _hourly_distribution(trades),
        "fold_stability": _fold_stability(trades),
    }


@router.get("/{run_id}/analysis/{symbol}")
def get_run_analysis(run_id: str, symbol: str) -> dict:
    """Statistical analysis of trades for a specific symbol."""
    return compute_analysis(run_id, symbol)
