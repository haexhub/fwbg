"""
Walk-Forward Optimierung und Symbol-Verarbeitung.

Hauptmodul für die Verarbeitung einzelner Symbole mit Walk-Forward Optimierung.
Fold-Verarbeitung ist in process_fold.py ausgelagert.
Grid-Search Funktionen sind in grid_search.py ausgelagert.
"""
import functools
import gc
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from fwbg.data import config as data_config
from fwbg.core.config import StrategyConfig
from fwbg.data.assets import get_asset
from fwbg.core.context import SimulationContext
from fwbg.data.loader import load_data_aligned, run_data_loading
from fwbg.simulation.trade import (
    calculate_sharpe_ratio, calculate_calmar_from_returns, pnl_to_returns,
)
from fwbg.core import get_risk_manager
from fwbg.research.dsr import return_statistics
from fwbg.utils.progress import report_done, report_phase
from fwbg.utils.logging import log, start_log_capture, stop_log_capture
from .robust_validation import create_walk_forward_folds, plan_walk_forward
from .bias_checks import check_asset_bias, fold_bias_ratios
from .causal_reporting import equity_replay, monte_carlo_summary, slim_trades
from .causal_variants import VariantRecord, replay_prefix_variants, replay_view, variant_score
from .process_fold import precompute_indicators, process_single_fold
from .selection_protocol import SELECTION_PROTOCOL


def _compute_trade_analytics(trades_detailed):
    """Compute MAE/MFE and SL-potential statistics from detailed trades."""
    if not trades_detailed:
        return None

    winners = [t for t in trades_detailed if t.get("result", 0) > 0]
    losers = [t for t in trades_detailed if t.get("result", 0) < 0]

    def _mae_stats(trades):
        maes = [t["mae"] for t in trades if "mae" in t]
        if not maes:
            return None
        return {
            "mean": float(np.mean(maes)),
            "median": float(np.median(maes)),
            "max": float(max(maes)),
            "p75": float(np.percentile(maes, 75)),
            "p90": float(np.percentile(maes, 90)),
        }

    def _mfe_stats(trades):
        mfes = [t["mfe"] for t in trades if "mfe" in t]
        if not mfes:
            return None
        return {
            "mean": float(np.mean(mfes)),
            "median": float(np.median(mfes)),
            "max": float(max(mfes)),
        }

    analytics = {
        "n_winners": len(winners),
        "n_losers": len(losers),
        "mae_winners": _mae_stats(winners),
        "mae_losers": _mae_stats(losers),
        "mfe_winners": _mfe_stats(winners),
        "mfe_losers": _mfe_stats(losers),
    }

    # SL potential analysis: for losers, how many would have reached TP with wider SL?
    losers_with_potential = [t for t in losers if "potential_tp_reached" in t]
    if losers_with_potential:
        would_win = [t for t in losers_with_potential if t["potential_tp_reached"]]
        analytics["sl_potential"] = {
            "losers_analyzed": len(losers_with_potential),
            "would_reach_tp": len(would_win),
            "recovery_rate": len(would_win) / len(losers_with_potential),
        }
        if would_win:
            required_maes = [t["required_mae"] for t in would_win]
            analytics["sl_potential"]["required_mae"] = {
                "mean": float(np.mean(required_maes)),
                "median": float(np.median(required_maes)),
                "max": float(max(required_maes)),
                "p75": float(np.percentile(required_maes, 75)),
                "p90": float(np.percentile(required_maes, 90)),
            }
            bars_to_tp = [t["bars_to_potential_tp"] for t in would_win]
            analytics["sl_potential"]["bars_to_tp"] = {
                "mean": float(np.mean(bars_to_tp)),
                "median": float(np.median(bars_to_tp)),
                "max": int(max(bars_to_tp)),
            }

    # TP potential analysis: for winners, how much further did the price run?
    winners_with_continuation = [t for t in winners if "continuation_mfe" in t]
    if winners_with_continuation:
        cont_mfes = [t["continuation_mfe"] for t in winners_with_continuation]
        cont_maes = [t["continuation_mae"] for t in winners_with_continuation]
        # How many winners had significant continuation (>50% of TP distance)?
        with_continuation = [t for t in winners_with_continuation
                             if t["continuation_mfe"] > t.get("tp_distance", 1) * 0.5]
        analytics["tp_potential"] = {
            "winners_analyzed": len(winners_with_continuation),
            "with_significant_continuation": len(with_continuation),
            "continuation_rate": len(with_continuation) / len(winners_with_continuation),
            "continuation_mfe": {
                "mean": float(np.mean(cont_mfes)),
                "median": float(np.median(cont_mfes)),
                "max": float(max(cont_mfes)),
                "p75": float(np.percentile(cont_mfes, 75)),
                "p90": float(np.percentile(cont_mfes, 90)),
            },
            "reversal_depth": {
                "mean": float(np.mean(cont_maes)),
                "median": float(np.median(cont_maes)),
            },
        }

    # Per-CT statistics (from unified trades — all share same CT, but useful for
    # separate_long_short where long_ct != short_ct)
    ct_groups = {}
    for t in trades_detailed:
        ct_val = t.get("ct")
        if ct_val is not None:
            ct_key = str(ct_val)
            if ct_key not in ct_groups:
                ct_groups[ct_key] = {"wins": 0, "losses": 0, "pnl": 0.0}
            if t.get("result", 0) > 0:
                ct_groups[ct_key]["wins"] += 1
            else:
                ct_groups[ct_key]["losses"] += 1
            ct_groups[ct_key]["pnl"] += t.get("pnl_raw", 0)

    if ct_groups:
        for key in ct_groups:
            total = ct_groups[key]["wins"] + ct_groups[key]["losses"]
            ct_groups[key]["total"] = total
            ct_groups[key]["win_rate"] = ct_groups[key]["wins"] / total if total > 0 else 0
        analytics["ct_breakdown"] = ct_groups

    # Per-direction breakdown
    for direction in ("LONG", "SHORT"):
        dir_trades = [t for t in trades_detailed if t.get("direction") == direction]
        if dir_trades:
            dir_wins = sum(1 for t in dir_trades if t.get("result", 0) > 0)
            dir_pnl = sum(t.get("pnl_raw", 0) for t in dir_trades)
            analytics[f"{direction.lower()}_stats"] = {
                "total": len(dir_trades),
                "wins": dir_wins,
                "win_rate": dir_wins / len(dir_trades),
                "pnl": float(dir_pnl),
                "mae": _mae_stats(dir_trades),
                "mfe": _mfe_stats(dir_trades),
            }

    return analytics


def _parse_ct_value(ct_value):
    """Extract (ct_long, ct_short, ct_display) from config ct value."""
    if isinstance(ct_value, tuple):
        ct_long, ct_short = ct_value
        return ct_long, ct_short, ct_long
    return ct_value, ct_value, ct_value


def _build_walk_forward_summary(all_fold_results, win_rates, pnls, total_trades,
                                 sample_bias_detected, bias_ratios, mean_bias_ratio,
                                 config_inconsistent=False, folds_without_bias_ratio=0):
    """Build walk_forward summary dict shared by all result types."""
    profitable_folds = sum(1 for p in pnls if p > 0)
    summary = {
        "n_folds": len(all_fold_results),
        "successful_folds": len(all_fold_results),
        "profitable_folds": profitable_folds,
        "fold_stability": profitable_folds / len(all_fold_results) if all_fold_results else 0.0,
        # win_rates: folds that traded; None when no fold traded.
        "mean_win_rate": np.mean(win_rates) if win_rates else None,
        "std_win_rate": np.std(win_rates) if win_rates else None,
        "min_win_rate": min(win_rates) if win_rates else None,
        "max_win_rate": max(win_rates) if win_rates else None,
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "min_pnl": min(pnls),
        "max_pnl": max(pnls),
        "total_trades": total_trades,
        "sample_bias_detected": sample_bias_detected,
        "bias_ratios": bias_ratios,
        "mean_bias_ratio": mean_bias_ratio,
        "folds_without_bias_ratio": folds_without_bias_ratio,
        "selection_protocol": SELECTION_PROTOCOL,
        "fold_details": all_fold_results,
    }

    if config_inconsistent:
        summary["config_inconsistent"] = True

    return summary


def decode_signal_meta(signal_col: str) -> dict:
    """Decode signal column name into readable metadata."""
    if not signal_col:
        return {}
    meta = {}
    if "pdl_" in signal_col:
        m = re.match(r"^(.*?)rl(\d+)_", signal_col)
        if m:
            prefix, rl = m.group(1), int(m.group(2))
            meta["retracement_level"] = rl / 100
            meta["candle_span"] = "body" if prefix.startswith("body_") else "hl"
            meta["range_scope"] = "all" if "all_" in prefix else "session"
            meta["break_mode"] = "session_only" if "sesbrk_" in prefix else "all_hours"
            meta["retest_mode"] = "session_only" if "sesret_" in prefix else "all_hours"
    elif "orb_" in signal_col:
        m = re.search(r"rb(\d+)_cf(\d+)_prb(\d+)_orb_s(\d+)_rl(\d+)", signal_col)
        if m:
            meta["range_bars"] = int(m.group(1))
            meta["carry_forward"] = int(m.group(2))
            meta["pre_range_bars"] = int(m.group(3))
            meta["session_hour"] = int(m.group(4))
            meta["retracement_level"] = int(m.group(5)) / 100
    return meta


def _run_folds_parallel(wf_folds, max_parallel_folds,
                        fold_indicators, precomputed_raw_df, preprocessing_configs,
                        ctx, sym, total_indicators,
                        all_fold_results, accumulated_grid_results):
    """Run walk-forward folds in parallel using ThreadPoolExecutor.

    XGBoost n_jobs is reduced proportionally to avoid CPU over-subscription.
    Progress is aggregated across folds before reporting.
    """
    from fwbg.utils.xgb_config import get_xgboost_n_jobs, set_xgboost_n_jobs

    n_folds = len(wf_folds)
    workers = min(max_parallel_folds, n_folds)

    # Reduce XGBoost threads to share CPU across parallel folds
    original_n_jobs = get_xgboost_n_jobs()
    cpu_count = os.cpu_count() or 4
    if original_n_jobs == -1:
        effective_n_jobs = max(1, cpu_count // workers)
    else:
        effective_n_jobs = max(1, original_n_jobs // workers)
    set_xgboost_n_jobs(effective_n_jobs)

    log(1, f"Parallel folds: {workers} workers, XGBoost n_jobs={effective_n_jobs}", sym)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_fold,
                    fold, fold_idx, n_folds,
                    fold_indicators, precomputed_raw_df, preprocessing_configs,
                    ctx, sym, total_indicators,
                ): fold_idx
                for fold_idx, fold in enumerate(wf_folds)
            }
            for future in as_completed(futures):
                fold_result, grid_results = future.result()
                accumulated_grid_results.extend(grid_results)
                if fold_result:
                    all_fold_results.append(fold_result)
    finally:
        set_xgboost_n_jobs(original_n_jobs)
        gc.collect()


def _process_single_variant(variant_idx, variant_label, variant_strategy,
                             df, wf_folds, asset, sym, n_variants):
    """Process one indicator grid variant (thread-safe)."""
    if n_variants > 1:
        log(1, f"--- Variant {variant_idx+1}/{n_variants}: {variant_label} ---", sym)

    variant_ctx = SimulationContext.create(asset, variant_strategy)

    grid_per_fold = variant_ctx.total_grid_combinations()
    total_trainings = grid_per_fold * len(wf_folds)
    if n_variants == 1:
        log(1, f"Grid: {grid_per_fold} combos/fold × {len(wf_folds)} folds = {total_trainings} total trainings", sym)

    fold_indicators, precomputed_raw_df, total_indicators = precompute_indicators(
        df, variant_strategy, sym,
    )
    preprocessing_configs = variant_strategy.get_preprocessing()

    all_fold_results = []
    accumulated_grid_results = []
    max_parallel_folds = variant_strategy.resources.max_parallel_folds
    n_folds = len(wf_folds)

    if max_parallel_folds <= 1:
        for fold_idx, fold in enumerate(wf_folds):
            fold_result, grid_results = process_single_fold(
                fold, fold_idx, n_folds,
                fold_indicators, precomputed_raw_df, preprocessing_configs,
                variant_ctx, sym, total_indicators,
            )
            accumulated_grid_results.extend(grid_results)
            if fold_result:
                all_fold_results.append(fold_result)
            gc.collect()
    else:
        _run_folds_parallel(
            wf_folds, max_parallel_folds,
            fold_indicators, precomputed_raw_df, preprocessing_configs,
            variant_ctx, sym, total_indicators,
            all_fold_results, accumulated_grid_results,
        )
    # Parallel folds arrive in completion order; per-fold stats, the HP majority and
    # the merges must not depend on thread timing.
    all_fold_results.sort(key=lambda r: r["fold_id"])

    score = variant_score(all_fold_results)
    n_successful, mean_pnl = score

    if n_variants > 1:
        log(1, f"  Variant result: {n_successful}/{len(wf_folds)} folds"
               + (f", mean_inner_pnl={mean_pnl:.1f}" if n_successful else ""), sym)

    return {
        "score": score,
        "label": variant_label,
        "strategy": variant_strategy,
        "ctx": variant_ctx,
        "all_fold_results": all_fold_results,
        "accumulated_grid_results": accumulated_grid_results,
        "fold_indicators": fold_indicators,
        "precomputed_raw_df": precomputed_raw_df,
        "preprocessing_configs": preprocessing_configs,
    }


def _run_indicator_variants(variants, df, wf_folds, asset, sym):
    """Run indicator grid variants sequentially, keeping only the best.

    Sequential processing avoids multiplying RAM by the number of variants.
    Each variant's precomputed_raw_df (~180MB+) is released before the next
    variant starts, unless it's the current best. A grid's result also carries
    a light record of every variant ("variants") and the best's index
    ("frozen_index") for the causal replay.
    """
    n_variants = len(variants)

    if n_variants == 1:
        label, strategy = variants[0]
        return _process_single_variant(0, label, strategy, df, wf_folds, asset, sym, 1)

    log(1, f"Indicator Grid: {n_variants} variants (sequential)", sym)

    best_score = (-1, float("-inf"))
    best_data = None
    best_index = None
    records = []

    for idx, (label, strat) in enumerate(variants):
        result = _process_single_variant(
            idx, label, strat, df, wf_folds, asset, sym, n_variants,
        )
        records.append(VariantRecord(
            result["label"], result["strategy"], result["ctx"],
            result["preprocessing_configs"], replay_view(result["all_fold_results"]),
        ))
        if result["score"] > best_score:
            # Release previous best's heavy data
            if best_data is not None:
                best_data.pop("precomputed_raw_df", None)
                best_data.pop("fold_indicators", None)
            best_score = result["score"]
            best_data = result
            best_index = idx
        else:
            # Release this variant's heavy data immediately
            result.pop("precomputed_raw_df", None)
            result.pop("fold_indicators", None)
        gc.collect()

    best_data["variants"] = records
    best_data["frozen_index"] = best_index
    return best_data


def _apply_date_window(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    """Restrict `df` (DatetimeIndex) to a half-open date window [start_date, end_date).

    `start_date` is inclusive, `end_date` is exclusive — this keeps in-sample
    and holdout windows that share a boundary date from overlapping. Either
    bound may be omitted. Raises `ValueError` naming the bad value if a date
    string cannot be parsed.
    """
    start = None
    end = None
    if start_date is not None:
        try:
            start = pd.Timestamp(start_date)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid start_date {start_date!r}: {exc}") from exc
    if end_date is not None:
        try:
            end = pd.Timestamp(end_date)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid end_date {end_date!r}: {exc}") from exc

    if start is not None and end is not None:
        return df.loc[(df.index >= start) & (df.index < end)]
    if start is not None:
        return df.loc[df.index >= start]
    return df.loc[df.index < end]


def load_symbol_frame(csv_path: str, strategy: StrategyConfig, sym: str) -> pd.DataFrame | None:
    """Load *csv_path* as the run sees it: resampled, date-windowed, flat bars
    dropped, data-loading pipeline applied. None without data (in the window).
    """
    t0 = time.time()
    df = load_data_aligned(csv_path)
    if df is None:
        log(1, "SKIP - Keine Daten", sym)
        return None

    # Resample to target timeframe if loaded from a lower one
    if data_config.RESAMPLE_FROM:
        from fwbg.data.resample import resample_ohlcv
        n_before = len(df)
        df = resample_ohlcv(df, data_config.TIMEFRAME)
        log(2, f"Resampled {data_config.RESAMPLE_FROM} → {data_config.TIMEFRAME} "
                f"({n_before} → {len(df)} bars)", sym)

    # Restrict to the configured backtest window (ISO start/end dates).
    # The DatetimeIndex makes all downstream fold/holdout splitting (which is
    # purely positional) operate on the window automatically — no fold-logic
    # change needed. Used to reserve a holdout tail or run a holdout window.
    # Half-open: start_date inclusive, end_date exclusive (see _apply_date_window).
    if strategy.start_date or strategy.end_date:
        n_before = len(df)
        df = _apply_date_window(df, strategy.start_date, strategy.end_date)
        log(2, f"Datumsfenster {strategy.start_date}..{strategy.end_date}: "
                f"{n_before} → {len(df)} Zeilen", sym)
        if df.empty:
            log(1, "SKIP - Keine Daten im Datumsfenster", sym)
            return None

    # Drop flat bars (O==H==L==C, weekends/holidays for index data)
    if strategy.assets.get("drop_flat_bars"):
        n_before = len(df)
        df = df[~((df["O"] == df["H"]) & (df["H"] == df["L"]) & (df["L"] == df["C"]))]
        log(2, f"Dropped {n_before - len(df)} flat bars "
                f"({n_before} → {len(df)})", sym)

    log(2, f"Daten geladen: {len(df)} Zeilen ({time.time()-t0:.1f}s)", sym)

    # === DATA LOADING (generic orchestrator) ===
    data_loading_configs = strategy.get_data_loading()
    if data_loading_configs:
        t0 = time.time()
        report_phase(sym, "Data Loading...")
        df = run_data_loading(df, data_loading_configs)
        log(2, f"Data Loading abgeschlossen ({time.time()-t0:.1f}s)", sym)
    return df


def _records_selection_protocol(func):
    """Every result of *func*, early returns and errors included, records its selection protocol."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        result["selection_protocol"] = SELECTION_PROTOCOL
        return result

    return wrapper


@_records_selection_protocol
def process_symbol(csv_path: str, strategy: StrategyConfig) -> dict:
    """
    Verarbeitet ein einzelnes Symbol mit Walk-Forward Optimierung.

    Args:
        csv_path: Pfad zur CSV-Datei
        strategy: StrategyConfig mit allen Strategie-Parametern

    Note:
        The configured backtest window (`strategy.start_date`/`end_date`) is
        half-open: `start_date` is inclusive, `end_date` is exclusive. See
        `_apply_date_window`.
    """
    sym = os.path.basename(csv_path).split("_")[0]
    t_start = time.time()
    start_log_capture()
    result = {}
    try:
        if sym in ["VIX", "DXY"]:
            log(2, "Übersprungen (Makro-Asset)", sym)
            result = {"symbol": sym, "status": "macro_asset"}
            return result

        log(1, "START", sym)
        report_phase(sym, "Lade Daten...")

        df = load_symbol_frame(csv_path, strategy, sym)
        if df is None:
            result = {"symbol": sym, "status": "no_data"}
            return result

        # === LOOKAHEAD BIAS PREVENTION ===
        # KRITISCH: Walk-Forward Splits VOR Indikator-Berechnung!
        # Rolling Windows in Indikatoren würden sonst Zukunftsdaten sehen.

        if len(df) < data_config.MIN_TRADES * 8:
            log(1, f"SKIP - Zu wenig Daten für Walk-Forward ({len(df)} < {data_config.MIN_TRADES * 8})", sym)
            result = {"symbol": sym, "status": "insufficient_data", "rows": len(df)}
            return result

        # Asset-Konfiguration laden
        asset = get_asset(sym)

        # SimulationContext erstellen (wird durch alle Funktionen gereicht)
        ctx = SimulationContext.create(asset, strategy)

        # === MODEL DEPENDENCY VALIDATION ===
        from fwbg.core.registry import get_model
        model_class = get_model(ctx.model_type)
        required = model_class.get_required_indicators()
        if required:
            configured = {ind["name"] for ind in strategy.get_indicators()}
            missing = [r for r in required if r not in configured]
            if missing:
                log(1, f"SKIP - Model '{ctx.model_type}' requires indicators: {missing}", sym)
                result = {
                    "symbol": sym,
                    "status": "missing_model_dependencies",
                    "error": f"Model '{ctx.model_type}' requires indicators not in config: {missing}",
                }
                return result

        # === SIGNAL SOURCE VALIDATION ===
        # Signal models read their entry column from full_pool, which is
        # required_features ∩ columns (signal_fold.py). That pool is only
        # non-empty when signal_rules-with-conditions or time filters produce
        # _composed_signal_*, or model.required_features names a real column.
        # Without any of those, every fold is skipped with an empty pool and
        # the run silently "completes" as no_successful_folds. Fail fast and
        # loudly instead. (strategy.model.required_features, not ctx's, since
        # signal_fold appends the composed signal columns to ctx's.)
        if ctx.model_type == "signal":
            has_signal_rules = bool(ctx.signal_rules) and any(
                (ctx.signal_rules.get(d) or {}).get("conditions")
                for d in ("long", "short")
            )
            has_time_filter = bool(ctx.allowed_hours) or bool(ctx.allowed_days)
            has_required_features = bool(strategy.model.required_features)
            if not (has_signal_rules or has_time_filter or has_required_features):
                log(1, "SKIP - Signal model has no entry-signal source", sym)
                result = {
                    "symbol": sym,
                    "status": "no_signal_source",
                    "error": (
                        "Signal model has no entry-signal source: needs "
                        "signal_rules with conditions, non-empty "
                        "model.required_features, or an allowed_hours/"
                        "allowed_days time filter."
                    ),
                }
                return result

        # === WALK-FORWARD FOLDS ERSTELLEN ===
        # Fold-Dimensionen adaptiv an die tatsächlich vorhandene Historie
        # anpassen: die Test-Fenster füllen die verfügbaren Daten, bei knapper
        # Historie werden Training/Fold-Zahl reduziert statt hart zu skippen.
        target_folds = strategy.validation.folds
        target_min_train = data_config.WINDOW_SIZE // 2
        # OOS darf beliebig klein werden, muss aber je Fold aussagekräftig
        # bleiben — Boden relativ zum Timeframe-Ziel.
        min_oos = max(60, data_config.OOS_SIZE // 10)
        plan = plan_walk_forward(len(df), target_folds, target_min_train, min_oos)
        if plan is None:
            reason = (
                f"only {len(df)} bars — too little history for a valid "
                f"walk-forward split (need at least "
                f"~{min_oos * 3 + max(200, target_min_train // 4)})"
            )
            log(1, f"SKIP - {reason}", sym)
            return {"symbol": sym, "status": "insufficient_data_for_folds", "error": reason}
        n_folds, oos_size, min_train = plan
        if n_folds < target_folds:
            log(1, f"Adaptive folds: {n_folds} instead of {target_folds} "
                   f"({len(df)} bars available)", sym)
        report_phase(sym, f"Creating {n_folds} walk-forward folds (oos_size={oos_size})...")
        try:
            wf_folds = create_walk_forward_folds(
                df,
                n_folds=n_folds,
                test_size=oos_size,
                min_train_size=min_train,
                anchored=True,
                embargo_bars=ctx.embargo_bars,
            )
        except ValueError as e:
            log(1, f"SKIP - {str(e)}", sym)
            result = {"symbol": sym, "status": "insufficient_data_for_folds", "error": str(e)}
            return result

        log(1, f"Walk-Forward: {len(wf_folds)} folds created (prevents sample bias)", sym)
        for fold in wf_folds:
            log(2, f"  Fold {fold.fold_id}: Train[{fold.train_end - fold.train_start}] Test[{fold.test_end - fold.test_start}]", sym)

        # === INDICATOR GRID: expand variants ===
        from .indicator_grid import expand_indicator_grid

        variants = expand_indicator_grid(strategy)
        n_variants = len(variants)

        if n_variants > 1:
            log(1, f"Indicator Grid: {n_variants} variants", sym)

        best_variant_data = _run_indicator_variants(
            variants, df, wf_folds, asset, sym,
        )

        # Use best variant for post-processing
        if len(variants) > 1:
            score = best_variant_data["score"]
            log(1, f"Best variant: {best_variant_data['label']} "
                   f"({score[0]}/{len(wf_folds)} folds, mean_inner_pnl={score[1]:.1f})", sym)

        ctx = best_variant_data["ctx"]
        all_fold_results = best_variant_data["all_fold_results"]
        accumulated_grid_results = best_variant_data["accumulated_grid_results"]
        fold_indicators = best_variant_data["fold_indicators"]
        precomputed_raw_df = best_variant_data["precomputed_raw_df"]
        preprocessing_configs = best_variant_data["preprocessing_configs"]
        indicator_variant_label = best_variant_data["label"] if len(variants) > 1 else None

        # === END OF WALK-FORWARD LOOP ===

        if len(all_fold_results) == 0:
            log(1, "SKIP - No successful folds", sym)
            report_done(sym, "no_successful_folds")
            result = {"symbol": sym, "status": "no_successful_folds", "grid_results": accumulated_grid_results}
            if indicator_variant_label:
                result["indicator_variant"] = indicator_variant_label
            # Include best grid result for diagnostics
            valid = [g for g in accumulated_grid_results
                     if g.get("inner_val_pnl") is not None
                     and g["inner_val_pnl"] != float("-inf")]
            if valid:
                best_grid = max(valid, key=lambda g: g["inner_val_pnl"])
                result["best_grid_result"] = best_grid
                log(1, f"  Best grid result: TP={best_grid['tp_mult']}, SL={best_grid['sl_mult']}, "
                       f"CT={best_grid.get('conf_thresh', best_grid.get('ct', '?'))}, "
                       f"PnL={best_grid['inner_val_pnl']:.1f}", sym)
            return result

        log(1, f"=== Walk-Forward Complete: {len(all_fold_results)}/{len(wf_folds)} successful folds ===", sym)

        # Aggregate metrics across all folds
        # A fold whose chosen setting traded nothing (kept: the test set must not filter
        # folds) has no win rate; it still counts as a fold and in fold_stability.
        win_rates = [r["test_win_rate"] for r in all_fold_results if r["test_trades"] > 0]
        pnls = [r["test_pnl"] for r in all_fold_results]
        trades_counts = [r["test_trades"] for r in all_fold_results]
        bias = fold_bias_ratios(all_fold_results)
        bias_ratios = bias["bias_ratios"]

        mean_wr = np.mean(win_rates) if win_rates else 0.0  # replaced by the unified trades' WR
        std_wr = np.std(win_rates) if win_rates else 0.0
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        total_trades = sum(trades_counts)
        per_fold_total_trades = total_trades
        mean_bias_ratio = bias["mean_bias_ratio"]

        # Detect sample bias
        sample_bias_detected = any(ratio > 2.0 for ratio in bias_ratios if ratio > 0)

        log(1, "Walk-Forward Results:", sym)
        if win_rates:
            log(1, f"  Win-Rate: {mean_wr*100:.1f}% ± {std_wr*100:.1f}% (range: {min(win_rates)*100:.1f}%-{max(win_rates)*100:.1f}%)", sym)
        else:
            log(1, "  Win-Rate: n/a (no fold traded)", sym)
        log(1, f"  PnL: {mean_pnl:.1f} ± {std_pnl:.1f} (range: {min(pnls):.1f}-{max(pnls):.1f})", sym)
        log(1, f"  Total Trades: {total_trades}", sym)
        log(1, f"  Bias Ratios: {[f'{r:.2f}x' for r in bias_ratios]}"
               f" ({bias['folds_without_bias_ratio']} folds without: inner PnL <= 0)", sym)

        if sample_bias_detected:
            log(1, "  WARNING: Sample bias detected in some folds (>2x ratio)", sym)

        # Check if fold configs are consistent (CV = coefficient of variation)
        configs = [r["best_config"] for r in all_fold_results]
        tp_values = [c["tp"] for c in configs]
        sl_values = [c["sl"] for c in configs]
        rrr_values = [c.get("rrr", 1.0) for c in configs]

        tp_mean = np.mean(tp_values)
        sl_mean = np.mean(sl_values)
        tp_cv = np.std(tp_values) / tp_mean if tp_mean > 0 else 0
        sl_cv = np.std(sl_values) / sl_mean if sl_mean > 0 else 0
        rrr_std = np.std(rrr_values) if len(rrr_values) > 1 else 0

        # CV > 0.5 means TP/SL vary substantially across folds (informational warning only)
        is_consistent = tp_cv <= 0.5 and sl_cv <= 0.5 and rrr_std <= 0.5
        config_inconsistent = not is_consistent and len(all_fold_results) > 1

        if config_inconsistent:
            log(1, "  WARNING: Fold configs vary across folds (aggregating all trades)", sym)
            log(1, f"    TP: {tp_values} (CV={tp_cv:.2f})", sym)
            log(1, f"    SL: {sl_values} (CV={sl_cv:.2f})", sym)
            log(1, f"    RRR: {[f'{r:.2f}' for r in rrr_values]} (std={rrr_std:.3f})", sym)

        # === MODEL HYPERPARAMETERS CONSISTENCY ===
        # For rule-based signal strategies, model_hyperparameters define the strategy
        # (rl level, hour window). Different HP = different strategy. The frozen merge
        # uses only the majority-HP folds; each causal prefix applies the same rule.
        from .unified_simulation import (
            causal_candidates, consistent_folds, hp_key, merge_unified_settings,
            run_unified_simulation,
        )

        majority_folds = consistent_folds(all_fold_results)
        if len(majority_folds) < len(all_fold_results):
            majority_hp_key = hp_key(majority_folds[0]["best_config"].get("model_hyperparameters"))
            # Show what each fold picked
            for r in all_fold_results:
                hp = r["best_config"].get("model_hyperparameters") or {}
                is_majority = "✓" if hp_key(hp) == majority_hp_key else "✗"
                log(2, f"    Fold {r['fold_id'] + 1}: hp={dict(hp) if hp else '{}'} {is_majority}", sym)
            log(1, f"  Model HP inconsistent: {len(majority_folds)}/{len(all_fold_results)} folds agree; "
                   "the frozen setting merges only those.", sym)

        # === UNIFIED SIMULATION ===
        # The frozen (deployment) setting merges all folds. The replay behind the
        # reported metrics is causal: fold k uses the merge of folds 0..k only and,
        # for an indicator grid, the variant best on folds 0..k (causal_variants.py).
        unified_candidate = merge_unified_settings(majority_folds, all_fold_results)
        tp_unified, sl_unified, ct_unified = unified_candidate["params"]
        features_long = unified_candidate.get("selected_features_long") or []
        features_short = unified_candidate.get("selected_features_short") or []
        log(1, f"  Frozen setting: TP={tp_unified:.2f} SL={sl_unified:.2f} CT={ct_unified} "
               f"timeout={unified_candidate.get('timeout_bars')} "
               f"features={len(features_long)} long, {len(features_short)} short", sym)
        if unified_candidate.get("model_hyperparameters"):
            log(2, f"    Model HP: {unified_candidate['model_hyperparameters']}", sym)
        if unified_candidate.get("exit_modifier_params"):
            log(2, f"    Exit modifier: {unified_candidate['exit_modifier_params']}", sym)
        if best_variant_data.get("variants"):
            unified_fold_results = replay_prefix_variants(
                wf_folds, best_variant_data["variants"], best_variant_data["frozen_index"],
                fold_indicators, precomputed_raw_df, df, sym,
            )
        else:
            replay_candidates = causal_candidates(wf_folds, all_fold_results)

            # Pass signal data prep for signal models
            if ctx.model_type == "signal":
                from .signal_fold import prepare_signal_fold_data
                prepare_fn = prepare_signal_fold_data
            else:
                prepare_fn = None  # uses default (prepare_fold_data)

            unified_fold_results = run_unified_simulation(
                wf_folds, replay_candidates,
                fold_indicators, precomputed_raw_df, preprocessing_configs,
                ctx, sym, prepare_data_fn=prepare_fn,
            )

        # Release heavy data no longer needed after unified simulation
        del fold_indicators, precomputed_raw_df, best_variant_data, df
        gc.collect()

        b_config = {
            "tp": tp_unified,
            "sl": sl_unified,
            "ct": ct_unified,
            "rrr": tp_unified / sl_unified if sl_unified > 0 else 1.0,
            "timeout_bars": unified_candidate.get("timeout_bars"),
            "model_hyperparameters": unified_candidate.get("model_hyperparameters"),
            "exit_modifier_params": unified_candidate.get("exit_modifier_params"),
        }

        # Aggregate trades from unified simulation
        all_trades = []
        all_trades_detailed = []
        for unified_fold_result in unified_fold_results:
            all_trades.extend(unified_fold_result["trades"])
            all_trades_detailed.extend(unified_fold_result.get("trades_detailed") or [])

        n_unified_folds = len(unified_fold_results)

        all_trades_pnl = [t["pnl_raw"] for t in all_trades]
        all_trades_binary = [t["result"] for t in all_trades]
        rv_values = [t["rv_at_entry"] for t in all_trades if "rv_at_entry" in t]

        # Recalculate metrics from unified trades
        total_trades = len(all_trades)
        mean_wr = sum(1 for t in all_trades if t["result"] == 1.0) / total_trades if total_trades > 0 else 0.0
        mean_pnl = sum(all_trades_pnl) / n_unified_folds if n_unified_folds else 0.0

        if total_trades == 0:
            log(1, "SKIP - Unified simulation produced no trades", sym)
            report_done(sym, "no_unified_trades")
            result = {"symbol": sym, "status": "no_unified_trades", "grid_results": accumulated_grid_results}
            return result

        # === RISK MANAGEMENT PLUGIN ===
        # Computes risk_per_trade, circuit_breaker, risk_adjustment.
        # Must run BEFORE Monte Carlo so MC uses the correct risk value.
        rrr = b_config["rrr"]
        risk_mgr_cls = get_risk_manager(strategy.risk_management)
        risk_mgr = risk_mgr_cls()
        risk_result = risk_mgr.compute_risk_params(
            all_trades_binary, mean_wr, rrr,
            rv_values=rv_values if len(rv_values) == len(all_trades_binary) else None,
            **strategy.risk_params
        )
        fk = risk_result["risk_per_trade"]
        circuit_breaker = risk_result["circuit_breaker"]
        risk_adjustment = risk_result["risk_adjustment"]

        if risk_adjustment["scale_factor"] < 1.0:
            log(2, f"Risk adjusted: scale_factor={risk_adjustment['scale_factor']:.2f}", sym)
        if circuit_breaker["enabled"]:
            log(2, f"Circuit Breaker: Pause after {circuit_breaker['pause_after_losses']} losses "
                   f"for {circuit_breaker['pause_bars']} bars", sym)

        # === CAUSAL REPORTING CALIBRATION ===
        # Der Deployment-Call oben kalibriert auf ALLEN Folds — korrekt fürs
        # Live-Sizing (assets.json/bot.py), aber Hindsight-Optimierung fürs
        # Reporting. Für die reported Metriken wird Fold i deshalb nur mit
        # einer auf Folds 0..i-1 kalibrierten Sizing bewertet (Fold 0 entfällt).
        from .causal_reporting import compute_causal_reporting
        causal = compute_causal_reporting(
            unified_fold_results, risk_mgr, rrr, strategy.risk_params
        )
        del unified_fold_results
        causal_ok = len(causal["causal_pnl"]) >= 10
        if not causal_ok:
            log(1, f"  Causal reporting: nur {len(causal['causal_pnl'])} kausale "
                   f"Trades — Fallback auf naive (nicht-kausale) Metriken", sym)

        # === OVERFITTING METRICS (DSR + PBO) ===
        grid_results_by_fold = {}
        for gr in accumulated_grid_results:
            fid = gr.get("fold_id")
            if fid is not None:
                grid_results_by_fold.setdefault(fid, []).append(gr)

        # Per-trade returns for reporting: kausal kalibriert (Fold i mit
        # Prior-Fold-Sizing), Fallback auf naive fk-Skalierung wenn zu wenige
        # kausale Trades vorhanden sind.
        if causal_ok:
            reporting_returns = causal["causal_returns"]
            reporting_pnl = causal["causal_pnl"]
        else:
            reporting_returns = pnl_to_returns(all_trades_pnl, fk)
            reporting_pnl = all_trades_pnl
        reporting_returns_arr = np.array(reporting_returns)
        non_ann_sr = float(np.mean(reporting_returns_arr) / np.std(reporting_returns_arr)) if len(reporting_returns_arr) > 1 and np.std(reporting_returns_arr) > 0 else 0.0
        # Per-trade return statistics of the reporting series (the verdict's DSR inputs).
        return_stats = return_statistics(reporting_returns)

        from .overfitting import compute_overfitting_metrics
        try:
            overfitting = compute_overfitting_metrics(
                trade_returns=reporting_returns,
                observed_sr=non_ann_sr,
                n_strategies=len(accumulated_grid_results),
                grid_results_by_fold=grid_results_by_fold,
                n_trades=len(reporting_pnl),
            )
        except Exception as e:
            log(1, f"  Overfitting metrics failed: {e}", sym)
            overfitting = {
                "dsr": {"dsr": 0.0, "observed_sr": non_ann_sr, "expected_max_sr": 0.0,
                         "n_strategies": len(accumulated_grid_results), "is_significant": False},
                "pbo": {"pbo": None, "n_cscv_splits": 0, "is_overfit": None,
                         "degradation": None, "logit_mean": None},
            }
        pbo_val = overfitting["pbo"]["pbo"]
        log(1, f"  DSR={overfitting['dsr']['dsr']:.3f}"
               f" PBO={pbo_val:.2f}" if pbo_val is not None else
               f"  DSR={overfitting['dsr']['dsr']:.3f} PBO=n/a", sym)

        del grid_results_by_fold

        # === FEATURE STABILITY ANALYSIS ===
        feature_counts = {}
        n_successful_folds = len(all_fold_results)
        for fr in all_fold_results:
            for feat in (fr.get("selected_features_long") or []) + (fr.get("selected_features_short") or []):
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        feature_stability_details = {
            feat: {"count": count, "stability": count / n_successful_folds}
            for feat, count in sorted(feature_counts.items(), key=lambda x: -x[1])
        }
        stable_features = [f for f, s in feature_stability_details.items() if s["stability"] >= 0.5]
        unstable_features = [f for f, s in feature_stability_details.items() if s["stability"] < 0.5]

        if unstable_features:
            log(2, f"  Unstable features ({len(unstable_features)}): {unstable_features[:5]}", sym)

        feature_stability = {
            "stable_count": len(stable_features),
            "unstable_count": len(unstable_features),
            "details": feature_stability_details,
        }

        # Shared data for result building
        wf_summary = _build_walk_forward_summary(
            all_fold_results,
            win_rates,
            pnls,
            per_fold_total_trades,
            sample_bias_detected, bias_ratios, mean_bias_ratio,
            config_inconsistent=config_inconsistent,
            folds_without_bias_ratio=bias["folds_without_bias_ratio"],
        )
        # Override total_trades with unified simulation count
        wf_summary["total_trades"] = total_trades
        wf_summary["causal_reporting"] = {
            "causal": causal_ok,
            "fold_0_excluded": True,
            "excluded_trades": causal["excluded_trades"],
            "untraded_folds": causal["untraded_folds"],
            "per_fold_risk": causal["per_fold_risk"],
            "n_causal_trades": len(causal["causal_pnl"]),
        }
        features_list = (unified_candidate.get("selected_features_long") or []) + (unified_candidate.get("selected_features_short") or [])

        # Trade analytics (MAE/MFE, SL potential, direction breakdown)
        trade_analytics = _compute_trade_analytics(all_trades_detailed)
        if trade_analytics:
            sl_pot = trade_analytics.get("sl_potential")
            if sl_pot:
                log(1, f"  SL Potential: {sl_pot['would_reach_tp']}/{sl_pot['losers_analyzed']} "
                       f"losers would reach TP ({sl_pot['recovery_rate']:.0%})"
                       + (f", required MAE median={sl_pot['required_mae']['median']:.1f}" if sl_pot.get('required_mae') else ""),
                    sym)
            tp_pot = trade_analytics.get("tp_potential")
            if tp_pot:
                log(1, f"  TP Potential: {tp_pot['with_significant_continuation']}/{tp_pot['winners_analyzed']} "
                       f"winners had >50% continuation ({tp_pot['continuation_rate']:.0%})"
                       f", median continuation={tp_pot['continuation_mfe']['median']:.1f}",
                    sym)

        # Test period duration (used for annual return, sharpe, etc.)
        # Use all wf_folds (not just successful ones) because unified simulation
        # replays every fold with its causal candidate; untraded leading folds
        # still count in the test period.
        bars_per_year = data_config.tf_cfg["bars_per_hour"] * 24 * 250
        total_test_bars = sum(f.test_end - f.test_start for f in wf_folds)
        test_period_years = total_test_bars / bars_per_year if bars_per_year > 0 else 1
        # Kausales Reporting bewertet nur Folds i>0 → Fold-0-Testbars ausnehmen
        if causal_ok and len(wf_folds) > 1:
            reporting_test_bars = sum(f.test_end - f.test_start for f in wf_folds[1:])
        else:
            reporting_test_bars = total_test_bars

        # === NO EDGE ===
        if fk <= 0:
            ct_long, ct_short, ct_display = _parse_ct_value(b_config["ct"])

            result = {
                "symbol": sym,
                "status": "no_edge",
                "pnl": mean_pnl,
                "win_rate": mean_wr,
                "rrr": rrr,
                # no edge, no sizing: no Sharpe (a 0 would enter statistics)
                "sharpe": None,
                "calmar": 0,
                "tr_trace": all_trades_pnl,
                "test_period_years": test_period_years,
                "best_config": {
                    "tp_mult": b_config["tp"],
                    "sl_mult": b_config["sl"],
                    "conf_thresh": ct_display,
                    "ct_long": ct_long,
                    "ct_short": ct_short,
                    "features": features_list,
                    "model_hyperparameters": b_config.get("model_hyperparameters"),
                },
                "walk_forward": wf_summary,
                "overfitting": overfitting,
                "feature_stability": feature_stability,
                "trade_analytics": trade_analytics,
                "reason": f"No profitable edge (WR={mean_wr*100:.1f}%, RRR={rrr:.2f})"
            }

            bias_check_result = check_asset_bias(result, verbose=True)
            result["bias_check"] = bias_check_result

            log(1, f"NO_EDGE - WR={mean_wr:.1%}±{std_wr:.1%} RRR={rrr:.2f} "
                   f"TP={b_config['tp']} SL={b_config['sl']} CT={ct_display:.2f} "
                   f"Trades={total_trades}", sym)

            report_done(sym, "no_edge")
            return result

        # === MONTE CARLO TESTS (on aggregated trades from all folds) ===
        report_phase(sym, "Monte Carlo Validierung...")

        t_mc = time.time()
        mc_summary = monte_carlo_summary(reporting_pnl, reporting_returns)

        log(2, f"  Monte Carlo: p={mc_summary['p_value']:.3f}, "
               f"Equity median={mc_summary['equity_median']:.1f}, "
               f"bankruptcy={mc_summary['bankruptcy_rate']:.1%} ({time.time()-t_mc:.1f}s)", sym)

        # Sharpe/Calmar (needed for both not_significant and ok paths)
        actual_trades_per_year = len(reporting_pnl) * bars_per_year / reporting_test_bars if reporting_test_bars > 0 else len(reporting_pnl)
        sharpe = calculate_sharpe_ratio(reporting_returns, trades_per_year=actual_trades_per_year)
        calmar = calculate_calmar_from_returns(reporting_returns)

        # === TIME-BASED EQUITY REPLAY (Reporting, see equity_replay) ===
        detailed_aligned = (
            len(all_trades_detailed) == total_trades
            and all(t.get("exit_time") for t in all_trades_detailed)
        )
        reporting_equity = None
        if causal_ok and detailed_aligned:
            exit_times = [
                all_trades_detailed[i]["exit_time"]
                for i in causal["causal_trade_indices"]
            ]
            reporting_equity = equity_replay(exit_times, causal["causal_returns"])

        reporting = {
            "causal": causal_ok,
            "n_trades": len(reporting_pnl),
            "pnl_trace": reporting_pnl,
            "equity": reporting_equity,
        }
        slim_trades_detailed = (
            slim_trades(all_trades_detailed) if len(all_trades_detailed) == total_trades else []
        )

        ct_long, ct_short, ct_display = _parse_ct_value(b_config["ct"])

        # === NOT SIGNIFICANT ===
        if not mc_summary["is_significant"]:
            log(1, f"SKIP - Not significant (p={mc_summary['p_value']:.3f})", sym)

            result = {
                "symbol": sym,
                "status": "not_significant",
                "pnl": mean_pnl,
                "win_rate": mean_wr,
                "rrr": rrr,
                "sharpe": sharpe,
                "calmar": calmar,
                "risk_per_trade": fk,
                "tr_trace": all_trades_pnl,
                "test_period_years": test_period_years,
                "best_config": {
                    "risk_per_trade": fk,
                    "tp_mult": b_config["tp"],
                    "sl_mult": b_config["sl"],
                    "conf_thresh": ct_display,
                    "ct_long": ct_long,
                    "ct_short": ct_short,
                    "features": features_list,
                    "model_hyperparameters": b_config.get("model_hyperparameters"),
                },
                "walk_forward": wf_summary,
                "monte_carlo": mc_summary,
                "overfitting": overfitting,
                "return_stats": return_stats,
                "feature_stability": feature_stability,
                "trade_analytics": trade_analytics,
                "reporting": reporting,
                "trades_detailed": slim_trades_detailed,
                "reason": f"Not statistically significant (p={mc_summary['p_value']:.3f})",
                "grid_results": accumulated_grid_results,
            }

            bias_check_result = check_asset_bias(result, verbose=True)
            result["bias_check"] = bias_check_result

            log(1, f"NOT_SIGNIFICANT - WR={mean_wr:.1%}±{std_wr:.1%} Sharpe={sharpe:.2f} "
                   f"p={mc_summary['p_value']:.3f} TP={b_config['tp']} SL={b_config['sl']} "
                   f"CT={ct_display:.2f} Trades={total_trades}", sym)

            report_done(sym, "not_significant")
            return result

        # === OK ===
        if mc_summary["bankruptcy_rate"] > 0.1:
            log(1, f"WARNING: {mc_summary['bankruptcy_rate']:.1%} Bankruptcy-Rate", sym)

        # OOS fold stability: fraction of walk-forward folds with positive OOS PnL
        oos_fold_stability = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0

        result = {
            "symbol": sym,
            "status": "ok",
            "pnl": mean_pnl,
            "fold_stability": oos_fold_stability,
            "config": {
                "risk_per_trade": fk,
                "point_value": asset.point,
                "spread": ctx.spread,
                "tp_mult": b_config["tp"],
                "sl_mult": b_config["sl"],
                "conf_thresh": ct_display,
                "ct_long": ct_long,
                "ct_short": ct_short,
                "separate_long_short": bool(ct_long or ct_short),
                "features": features_list,
                "circuit_breaker": circuit_breaker,
                "risk_adjustment": risk_adjustment,
                "vol_targeting": risk_result.get("vol_targeting"),
                "model_hyperparameters": b_config.get("model_hyperparameters"),
                "signal_meta": {},
            },
            "tr_trace": all_trades_pnl,
            "test_period_years": test_period_years,
            "rrr": rrr,
            "win_rate": mean_wr,
            "sharpe": sharpe,
            "calmar": calmar,
            "currencies": asset.currencies,
            "walk_forward": wf_summary,
            "monte_carlo": mc_summary,
            "reporting": reporting,
            "trades_detailed": slim_trades_detailed,
            "overfitting": overfitting,
            "return_stats": return_stats,
            "feature_stability": feature_stability,
            "trade_analytics": trade_analytics,
            "grid_results": accumulated_grid_results,
            "unified_candidate": unified_candidate,
        }
        if indicator_variant_label:
            result["indicator_variant"] = indicator_variant_label

        log(1, f"OK (Walk-Forward) - WR={mean_wr:.1%}±{std_wr:.1%} Sharpe={sharpe:.2f} "
               f"p={mc_summary['p_value']:.3f} Trades={total_trades} ({time.time()-t_start:.1f}s)", sym)

        if sample_bias_detected:
            log(1, f"  NOTE: Sample bias detected in {sum(1 for r in bias_ratios if r > 2.0)} folds", sym)

        # === LIVE BIAS CHECK ===
        bias_check_result = check_asset_bias(result, verbose=True)
        result["bias_check"] = bias_check_result

        report_done(sym, "ok")
        return result

    except Exception as e:
        import traceback
        import sys
        from fwbg.utils.progress import report_log
        tb = traceback.format_exc()
        print(f"[{sym}] FEHLER: {e}\n{tb}", file=sys.stderr, flush=True)
        report_log(sym, "processing", "error", f"{e}", traceback=tb)
        report_done(sym, "error")
        result = {"symbol": sym, "status": "error", "error": f"{e}\n{tb}"}
        return result
    finally:
        result["logs"] = stop_log_capture()
