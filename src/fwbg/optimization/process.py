"""
Walk-Forward Optimierung und Symbol-Verarbeitung.

Hauptmodul für die Verarbeitung einzelner Symbole mit Walk-Forward Optimierung.
Fold-Verarbeitung ist in process_fold.py ausgelagert.
Grid-Search Funktionen sind in grid_search.py ausgelagert.
"""
import gc
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from fwbg.data import config as data_config
from fwbg.core.config import StrategyConfig
from fwbg.data.assets import get_asset
from fwbg.core.context import SimulationContext
from fwbg.data.loader import load_data_aligned, run_data_loading
from fwbg.simulation.trade import (
    calculate_sharpe_ratio, calculate_calmar_from_returns,
    monte_carlo_permutation_test, monte_carlo_equity_from_returns,
    pnl_to_returns,
)
from fwbg.core import get_risk_manager
from fwbg.utils.progress import report_done, report_phase
from fwbg.utils.logging import log, start_log_capture, stop_log_capture
from .robust_validation import create_walk_forward_folds
from .bias_checks import check_asset_bias
from .process_fold import precompute_indicators, process_single_fold


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
                                 config_inconsistent=False):
    """Build walk_forward summary dict shared by all result types."""
    profitable_folds = sum(1 for p in pnls if p > 0)
    summary = {
        "n_folds": len(all_fold_results),
        "successful_folds": len(all_fold_results),
        "profitable_folds": profitable_folds,
        "fold_stability": profitable_folds / len(all_fold_results) if all_fold_results else 0.0,
        "mean_win_rate": np.mean(win_rates),
        "std_win_rate": np.std(win_rates),
        "min_win_rate": min(win_rates),
        "max_win_rate": max(win_rates),
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "min_pnl": min(pnls),
        "max_pnl": max(pnls),
        "total_trades": total_trades,
        "sample_bias_detected": sample_bias_detected,
        "bias_ratios": bias_ratios,
        "mean_bias_ratio": mean_bias_ratio,
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

    n_successful = len(all_fold_results)
    mean_pnl = float(np.mean([r["test_pnl"] for r in all_fold_results])) if n_successful else float("-inf")

    if n_variants > 1:
        log(1, f"  Variant result: {n_successful}/{len(wf_folds)} folds"
               + (f", mean_pnl={mean_pnl:.1f}" if n_successful else ""), sym)

    return {
        "score": (n_successful, mean_pnl),
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
    variant starts, unless it's the current best.
    """
    n_variants = len(variants)

    if n_variants == 1:
        label, strategy = variants[0]
        return _process_single_variant(0, label, strategy, df, wf_folds, asset, sym, 1)

    log(1, f"Indicator Grid: {n_variants} variants (sequential)", sym)

    best_score = (-1, float("-inf"))
    best_data = None

    for idx, (label, strat) in enumerate(variants):
        result = _process_single_variant(
            idx, label, strat, df, wf_folds, asset, sym, n_variants,
        )
        if result["score"] > best_score:
            # Release previous best's heavy data
            if best_data is not None:
                best_data.pop("precomputed_raw_df", None)
                best_data.pop("fold_indicators", None)
            best_score = result["score"]
            best_data = result
        else:
            # Release this variant's heavy data immediately
            result.pop("precomputed_raw_df", None)
            result.pop("fold_indicators", None)
        gc.collect()

    return best_data


def process_symbol(csv_path: str, strategy: StrategyConfig) -> dict:
    """
    Verarbeitet ein einzelnes Symbol mit Walk-Forward Optimierung.

    Args:
        csv_path: Pfad zur CSV-Datei
        strategy: StrategyConfig mit allen Strategie-Parametern
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

        t0 = time.time()
        df = load_data_aligned(csv_path)
        if df is None:
            log(1, "SKIP - Keine Daten", sym)
            result = {"symbol": sym, "status": "no_data"}
            return result

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
        if strategy.start_date or strategy.end_date:
            n_before = len(df)
            df = df.loc[strategy.start_date:strategy.end_date]
            log(2, f"Datumsfenster {strategy.start_date}..{strategy.end_date}: "
                    f"{n_before} → {len(df)} Zeilen", sym)
            if df.empty:
                log(1, "SKIP - Keine Daten im Datumsfenster", sym)
                result = {"symbol": sym, "status": "no_data"}
                return result

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

        # === WALK-FORWARD FOLDS ERSTELLEN ===
        n_folds = strategy.validation.folds
        min_train = data_config.WINDOW_SIZE // 2
        # OOS size: split all available data (after min training) into n_folds
        oos_size = max(data_config.OOS_SIZE, (len(df) - min_train) // n_folds)
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
            score = (len(best_variant_data["all_fold_results"]),
                     float(np.mean([r["test_pnl"] for r in best_variant_data["all_fold_results"]]))
                     if best_variant_data["all_fold_results"] else float("-inf"))
            log(1, f"Best variant: {best_variant_data['label']} "
                   f"({score[0]}/{len(wf_folds)} folds, pnl={score[1]:.1f})", sym)

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
                       f"CT={best_grid.get('conf_thresh', '?')}, PnL={best_grid['inner_val_pnl']:.1f}", sym)
            return result

        log(1, f"=== Walk-Forward Complete: {len(all_fold_results)}/{len(wf_folds)} successful folds ===", sym)

        # Aggregate metrics across all folds
        win_rates = [r["test_win_rate"] for r in all_fold_results]
        pnls = [r["test_pnl"] for r in all_fold_results]
        trades_counts = [r["test_trades"] for r in all_fold_results]
        bias_ratios = [r["test_pnl"] / r["inner_val_pnl"] if r["inner_val_pnl"] > 0 else 0
                      for r in all_fold_results]

        mean_wr = np.mean(win_rates)
        std_wr = np.std(win_rates)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        total_trades = sum(trades_counts)
        per_fold_total_trades = total_trades
        mean_bias_ratio = np.mean(bias_ratios)

        # Detect sample bias
        sample_bias_detected = any(ratio > 2.0 for ratio in bias_ratios if ratio > 0)

        log(1, "Walk-Forward Results:", sym)
        log(1, f"  Win-Rate: {mean_wr*100:.1f}% ± {std_wr*100:.1f}% (range: {min(win_rates)*100:.1f}%-{max(win_rates)*100:.1f}%)", sym)
        log(1, f"  PnL: {mean_pnl:.1f} ± {std_pnl:.1f} (range: {min(pnls):.1f}-{max(pnls):.1f})", sym)
        log(1, f"  Total Trades: {total_trades}", sym)
        log(1, f"  Bias Ratios: {[f'{r:.2f}x' for r in bias_ratios]}", sym)

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
        # (rl level, hour window). Different HP = different strategy. Only aggregate
        # trades from folds that agree on the same HP to avoid mixing strategies.
        hp_list = [r["best_config"].get("model_hyperparameters") or {} for r in all_fold_results]

        def _hp_key(hp):
            """Hashable key for model_hyperparameters (ignoring None values)."""
            if not hp:
                return ()
            def _make_hashable(v):
                if isinstance(v, list):
                    return tuple(v)
                return v
            return tuple(sorted((k, _make_hashable(v)) for k, v in hp.items() if v is not None))

        hp_keys = [_hp_key(hp) for hp in hp_list]
        hp_counts = {}
        for key in hp_keys:
            hp_counts[key] = hp_counts.get(key, 0) + 1
        majority_hp_key = max(hp_counts, key=hp_counts.get) if hp_counts else ()
        majority_hp_count = hp_counts.get(majority_hp_key, 0)
        hp_consistent = majority_hp_count == len(all_fold_results)

        if not hp_consistent:
            # Only aggregate trades from folds with majority model_hyperparameters
            consistent_folds = [
                r for r, key in zip(all_fold_results, hp_keys)
                if key == majority_hp_key
            ]
            inconsistent_count = len(all_fold_results) - len(consistent_folds)
            # Show what each fold picked
            for i, (r, hp) in enumerate(zip(all_fold_results, hp_list)):
                is_majority = "✓" if hp_keys[i] == majority_hp_key else "✗"
                log(2, f"    Fold {i+1}: hp={dict(hp) if hp else '{}'} {is_majority}", sym)
            log(1, f"  Model HP inconsistent: {majority_hp_count}/{len(all_fold_results)} folds agree. "
                   f"Dropping {inconsistent_count} inconsistent fold(s).", sym)
        else:
            consistent_folds = all_fold_results

        # === UNIFIED SIMULATION ===
        # Derive one setting from consistent folds, re-simulate all folds.
        from .unified_simulation import merge_unified_settings, run_unified_simulation

        unified_candidate = merge_unified_settings(consistent_folds, all_fold_results)

        # Pass signal data prep for signal models
        if ctx.model_type == "signal":
            from .signal_fold import prepare_signal_fold_data
            prepare_fn = prepare_signal_fold_data
        else:
            prepare_fn = None  # uses default (prepare_fold_data)

        unified_fold_results = run_unified_simulation(
            wf_folds, unified_candidate,
            fold_indicators, precomputed_raw_df, preprocessing_configs,
            ctx, sym, prepare_data_fn=prepare_fn,
        )

        # Release heavy data no longer needed after unified simulation
        del fold_indicators, precomputed_raw_df, best_variant_data, df
        gc.collect()

        tp_unified, sl_unified, ct_unified = unified_candidate["params"]
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
        del unified_fold_results

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

        # === OVERFITTING METRICS (DSR + PBO) ===
        grid_results_by_fold = {}
        for gr in accumulated_grid_results:
            fid = gr.get("fold_id")
            if fid is not None:
                grid_results_by_fold.setdefault(fid, []).append(gr)

        # Per-trade returns from actual pnl_raw (not binary Kelly).
        # Scaled so avg loss return = -fk — consistent with what MC permutation test sees.
        pnl_returns = pnl_to_returns(all_trades_pnl, fk)
        pnl_returns_arr = np.array(pnl_returns)
        non_ann_sr = float(np.mean(pnl_returns_arr) / np.std(pnl_returns_arr)) if len(pnl_returns_arr) > 1 and np.std(pnl_returns_arr) > 0 else 0.0

        from .overfitting import compute_overfitting_metrics
        try:
            overfitting = compute_overfitting_metrics(
                trade_returns=pnl_returns,
                observed_sr=non_ann_sr,
                n_strategies=len(accumulated_grid_results),
                grid_results_by_fold=grid_results_by_fold,
                n_trades=total_trades,
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
        )
        # Override total_trades with unified simulation count
        wf_summary["total_trades"] = total_trades
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
        # re-simulates every fold with the merged config.
        bars_per_year = data_config.tf_cfg["bars_per_hour"] * 24 * 250
        total_test_bars = sum(f.test_end - f.test_start for f in wf_folds)
        test_period_years = total_test_bars / bars_per_year if bars_per_year > 0 else 1

        # === NO EDGE ===
        if fk <= 0:
            ct_long, ct_short, ct_display = _parse_ct_value(b_config["ct"])

            result = {
                "symbol": sym,
                "status": "no_edge",
                "pnl": mean_pnl,
                "win_rate": mean_wr,
                "rrr": rrr,
                "sharpe": 0,
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
        mc_perm = monte_carlo_permutation_test(all_trades_pnl, n_permutations=1000)
        mc_equity = monte_carlo_equity_from_returns(pnl_returns, n_simulations=500)

        log(2, f"  Monte Carlo: p={mc_perm['p_value']:.3f}, "
               f"Equity median={mc_equity['median_equity']:.1f}, "
               f"bankruptcy={mc_equity['bankruptcy_rate']:.1%} ({time.time()-t_mc:.1f}s)", sym)

        mc_summary = {
            "p_value": mc_perm["p_value"],
            "is_significant": mc_perm["is_significant"],
            "percentile": mc_perm["percentile"],
            "equity_median": mc_equity["median_equity"],
            "equity_p5": mc_equity["p5_equity"],
            "equity_p95": mc_equity["p95_equity"],
            "bankruptcy_rate": mc_equity["bankruptcy_rate"],
        }

        # Sharpe/Calmar (needed for both not_significant and ok paths)
        actual_trades_per_year = total_trades * bars_per_year / total_test_bars if total_test_bars > 0 else total_trades
        sharpe = calculate_sharpe_ratio(pnl_returns, trades_per_year=actual_trades_per_year)
        calmar = calculate_calmar_from_returns(pnl_returns)

        ct_long, ct_short, ct_display = _parse_ct_value(b_config["ct"])

        # === NOT SIGNIFICANT ===
        if not mc_perm["is_significant"]:
            log(1, f"SKIP - Not significant (p={mc_perm['p_value']:.3f})", sym)

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
                "feature_stability": feature_stability,
                "trade_analytics": trade_analytics,
                "reason": f"Not statistically significant (p={mc_perm['p_value']:.3f})",
                "grid_results": accumulated_grid_results,
            }

            bias_check_result = check_asset_bias(result, verbose=True)
            result["bias_check"] = bias_check_result

            log(1, f"NOT_SIGNIFICANT - WR={mean_wr:.1%}±{std_wr:.1%} Sharpe={sharpe:.2f} "
                   f"p={mc_perm['p_value']:.3f} TP={b_config['tp']} SL={b_config['sl']} "
                   f"CT={ct_display:.2f} Trades={total_trades}", sym)

            report_done(sym, "not_significant")
            return result

        # === OK ===
        if mc_equity["bankruptcy_rate"] > 0.1:
            log(1, f"WARNING: {mc_equity['bankruptcy_rate']:.1%} Bankruptcy-Rate", sym)

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
            "overfitting": overfitting,
            "feature_stability": feature_stability,
            "trade_analytics": trade_analytics,
            "grid_results": accumulated_grid_results,
        }
        if indicator_variant_label:
            result["indicator_variant"] = indicator_variant_label

        log(1, f"OK (Walk-Forward) - WR={mean_wr:.1%}±{std_wr:.1%} Sharpe={sharpe:.2f} "
               f"p={mc_perm['p_value']:.3f} Trades={total_trades} ({time.time()-t_start:.1f}s)", sym)

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
