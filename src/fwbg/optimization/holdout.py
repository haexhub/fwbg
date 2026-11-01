"""Holdout look: develop before the card's cutoff, evaluate once after it.

Option A (research register step 5): walk-forward, selection, unified
simulation and risk calibration see only [start, cutoff) — process_symbol with
end_date = cutoff. The frozen unified setting is then retrained once on every
development bar except the embargo and evaluated once on [cutoff, end), sized
with the development's fk, circuit breaker and loss scale. Reported metrics
come from the holdout trades alone; the development outcome is kept apart.

Causality: raw (stationary-free) indicators and the data-loading pipeline run
on the whole window [start, end) so the first holdout bars have their warm-up.
That gives development rows the same values as the development run only
because registered indicators and data loaders are prefix-stable: computed on
[start, end) they equal their values computed on [start, cutoff) (pinned by
tests/test_prefix_stability.py; tests/test_lookahead_bias.py checks they are
causal). Preprocessors and stateful indicators are fit on the training frame
only (process_fold).
"""

import dataclasses
import traceback
from functools import partial

import pandas as pd

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.data import config as data_config
from fwbg.data.assets import get_asset
from fwbg.data.validation import DataCheckError
from fwbg.research.dsr import return_statistics
from fwbg.simulation.trade import calculate_calmar_from_returns, calculate_sharpe_ratio
from fwbg.utils.logging import log
from fwbg.utils.progress import report_done, report_phase

from .causal_reporting import equity_replay, monte_carlo_summary, sized_returns, slim_trades
from .indicator_grid import expand_indicator_grid
from .process import load_symbol_frame, process_symbol
from .process_fold import precompute_indicators
from .robust_validation import WalkForwardFold
from .selection_protocol import SELECTION_PROTOCOL
from .signal_fold import prepare_signal_fold_data
from .unified_simulation import run_unified_simulation

# ponytail: data loaded and raw indicators precomputed a second time for the
# full window; cache only if holdout runs get frequent.

# Statuses the holdout phase itself assigns (a development that is not "ok"
# passes its own status through).
HOLDOUT_STATUSES = frozenset(
    {"ok", "not_significant", "no_unified_trades", "insufficient_data", "no_data", "data_invalid", "error"}
)
_DEVELOPMENT_KEYS = (
    "status",
    "reason",
    "error",
    "pnl",
    "win_rate",
    "rrr",
    "sharpe",
    "calmar",
    "fold_stability",
    "test_period_years",
    "monte_carlo",
    "return_stats",
    "overfitting",
    "config",
    "best_config",
    "indicator_variant",
    "unified_candidate",  # the frozen setting the holdout evaluated
    "selection_protocol",
)


def symbol_worker(strategy: StrategyConfig) -> partial:
    """The per-symbol worker of a run: a holdout look when the gate set a cutoff."""
    worker = process_symbol_holdout if strategy.holdout_cutoff else process_symbol
    return partial(worker, strategy=strategy)


def process_symbol_holdout(csv_path: str, strategy: StrategyConfig) -> dict:
    """Develop *csv_path* on [start, cutoff), then evaluate the frozen setting once after it."""
    result = _holdout_look(csv_path, strategy)
    result["selection_protocol"] = SELECTION_PROTOCOL  # the development's
    return result


def _holdout_look(csv_path: str, strategy: StrategyConfig) -> dict:
    cutoff = strategy.holdout_cutoff
    development = process_symbol(
        csv_path, dataclasses.replace(strategy, end_date=cutoff, holdout_cutoff=None)
    )
    status = development.get("status")
    if status != "ok":
        return _not_evaluated(development, cutoff, f"development ended in {status}")
    sym = development["symbol"]
    report_phase(sym, "Holdout evaluation...")
    try:
        result = _evaluate(csv_path, strategy, development)
    except DataCheckError as exc:  # the data after the cutoff failed its checks
        result = _not_evaluated(development, cutoff, str(exc), status="data_invalid")
        result["data_checks"] = exc.report.to_dict()
    except Exception as exc:  # before any holdout trade exists: not evaluated
        result = _failed(development, {"cutoff": cutoff, "evaluated": False}, exc)
    report_done(sym, result["status"])
    return result


def holdout_fold(df: pd.DataFrame, n_dev: int, embargo_bars: int) -> WalkForwardFold:
    """Train on the development bars minus the embargo, test on every bar from *n_dev*."""
    train_end = max(0, n_dev - embargo_bars)
    return WalkForwardFold(
        fold_id=0,
        train_start=0,
        train_end=train_end,
        test_start=n_dev,
        test_end=len(df),
        train_df=df.iloc[:train_end].copy(),
        test_df=df.iloc[n_dev:].copy(),
    )


def development_summary(development: dict) -> dict:
    """The development outcome kept beside a holdout result (no per-fold details)."""
    summary = {key: development[key] for key in _DEVELOPMENT_KEYS if key in development}
    if "walk_forward" in development:
        summary["walk_forward"] = {
            key: value
            for key, value in development["walk_forward"].items()
            if key != "fold_details"
        }
    return summary


def _evaluate(csv_path: str, strategy: StrategyConfig, development: dict) -> dict:
    sym = development["symbol"]
    cutoff = strategy.holdout_cutoff
    df = load_symbol_frame(csv_path, strategy, sym)
    n_dev = 0 if df is None else int(df.index.searchsorted(pd.Timestamp(cutoff)))
    if df is None or n_dev == len(df):
        return _not_evaluated(development, cutoff, "no data after the cutoff", status="no_data")

    variant = _variant(strategy, development.get("indicator_variant"))
    ctx = SimulationContext.create(get_asset(sym), variant)
    if n_dev <= ctx.embargo_bars:
        return _not_evaluated(
            development,
            cutoff,
            "not enough development bars after the embargo",
            status="insufficient_data",
        )
    fold = holdout_fold(df, n_dev, ctx.embargo_bars)
    fold_indicators, precomputed_raw_df, _ = precompute_indicators(df, variant, sym)
    prepare = prepare_signal_fold_data if ctx.model_type == "signal" else None
    # ponytail: one-fold run_unified_simulation (logs say "Unified Simulation") so the
    # holdout builds its context exactly like development; a dedicated function only if it diverges.
    evaluations = run_unified_simulation(
        [fold],
        [development["unified_candidate"]],
        fold_indicators,
        precomputed_raw_df,
        variant.get_preprocessing(),
        ctx,
        sym,
        prepare_data_fn=prepare,
    )
    if not evaluations:
        return _not_evaluated(
            development, cutoff, "holdout window too short to evaluate", status="insufficient_data"
        )
    holdout = _look(fold, evaluations[0], cutoff, ctx.embargo_bars)
    try:
        return _holdout_result(development, evaluations[0], fold, holdout)
    except Exception as exc:  # the holdout trades exist: the data was looked at
        return _failed(development, holdout, exc)


def _variant(strategy: StrategyConfig, label: str | None) -> StrategyConfig:
    """The indicator-grid variant development selected (its label is set only for grids)."""
    variants = expand_indicator_grid(strategy)
    if label is None and len(variants) == 1:
        return variants[0][1]
    for variant_label, variant in variants:
        if variant_label == label:
            return variant
    raise ValueError(f"indicator variant {label!r} not in the grid")


def _look(fold: WalkForwardFold, evaluation: dict, cutoff: str | None, embargo: int) -> dict:
    """What the holdout look touched: its windows, trades and the features it could not use."""
    return {
        "cutoff": cutoff,
        "evaluated": True,
        "train_start": fold.train_df.index[0].isoformat(),
        "train_end": fold.train_df.index[-1].isoformat(),
        "test_start": fold.test_df.index[0].isoformat(),
        "test_end": fold.test_df.index[-1].isoformat(),
        "bars": len(fold.test_df),
        "embargo_bars": embargo,
        "trades": len(evaluation["trades"]),
        # Candidate features the preparation dropped (e.g. all-NaN after the cutoff).
        "dropped_features": evaluation.get("dropped_features", []),
    }


def _holdout_result(
    development: dict, evaluation: dict, fold: WalkForwardFold, holdout: dict
) -> dict:
    sym = development["symbol"]
    config = development["config"]
    trades = evaluation["trades"]
    detailed = evaluation.get("trades_detailed") or []
    returns, pnl, taken = sized_returns(
        trades, config["risk_per_trade"], config["circuit_breaker"], development["tr_trace"]
    )
    years = len(fold.test_df) / (data_config.tf_cfg["bars_per_hour"] * 24 * 250)
    holdout["sized_trades"] = len(pnl)
    if not trades:
        log(1, "HOLDOUT - no trades after the cutoff", sym)
        return {
            "symbol": sym,
            "status": "no_unified_trades",
            "holdout": holdout,
            "development": development_summary(development),
        }

    mc = monte_carlo_summary(pnl, returns)
    sharpe = calculate_sharpe_ratio(
        returns, trades_per_year=len(pnl) / years if years > 0 else len(pnl)
    )
    aligned = len(detailed) == len(trades) and all(t.get("exit_time") for t in detailed)
    equity = equity_replay([detailed[i]["exit_time"] for i in taken], returns) if aligned else None
    status = "ok" if mc["is_significant"] else "not_significant"
    win_rate = sum(1 for t in trades if t["result"] == 1.0) / len(trades)
    log(
        1,
        f"HOLDOUT {status.upper()} - WR={win_rate:.1%} Sharpe={sharpe:.2f} "
        f"p={mc['p_value']:.3f} Trades={len(pnl)}/{len(trades)}",
        sym,
    )
    return {
        "symbol": sym,
        "status": status,
        "pnl": sum(pnl),
        "win_rate": win_rate,
        "rrr": development["rrr"],
        "sharpe": sharpe,
        "calmar": calculate_calmar_from_returns(returns, years=years),
        "tr_trace": [t["pnl_raw"] for t in trades],
        "test_period_years": years,
        "config": config,
        "currencies": development["currencies"],
        "monte_carlo": mc,
        "return_stats": return_statistics(returns),
        "reporting": {"causal": True, "n_trades": len(pnl), "pnl_trace": pnl, "equity": equity},
        "trades_detailed": slim_trades(detailed) if len(detailed) == len(trades) else [],
        "holdout": holdout,
        "development": development_summary(development),
    }


def _failed(development: dict, holdout: dict, exc: Exception) -> dict:
    """A crash in the holdout phase is this symbol's error, like in process_symbol."""
    log(1, f"Holdout evaluation failed: {exc}", development["symbol"])
    return {
        "symbol": development["symbol"],
        "status": "error",
        "holdout": {**holdout, "reason": f"holdout evaluation failed: {exc}"},
        "development": development_summary(development),
        "error": f"{exc}\n{traceback.format_exc()}",
    }


def _not_evaluated(
    development: dict, cutoff: str | None, reason: str, status: str | None = None
) -> dict:
    result = {
        "symbol": development["symbol"],
        "status": status or development["status"],
        "holdout": {"cutoff": cutoff, "evaluated": False, "reason": reason},
        "development": development_summary(development),
    }
    if status is None and "error" in development:
        result["error"] = development["error"]
    return result
