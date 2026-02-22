"""
Grid-Search Funktionen für Walk-Forward Optimierung.

Enthält:
- _process_single_grid_combo: Einzelne Grid-Kombination verarbeiten
- _process_tp_sl_combo_wrapper: Wrapper für parallele Verarbeitung
- run_grid_search: Grid-Search über TP/SL/Timeout Kombinationen
"""
import dataclasses
import math
from typing import Tuple, Optional, List

import numpy as np

from fwbg.utils.logging import log
from .nested_cv import (
    run_inner_cv, select_features_from_fold,
    _evaluate_single_fold, _aggregate_cv_folds,
)


def select_features(
    inner_folds: list,
    features: list,
    ctx,
    sym: str,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Führt Feature Selection einmal durch (via Plugin-Interface).

    Verwendet den ERSTEN Inner Fold für Feature Selection.
    Dies passiert VOR dem Grid-Search und reduziert Feature-Selection-Läufe.

    Args:
        inner_folds: Liste von (train_df, val_df) Tuples
        features: Verfügbare Feature-Spalten
        ctx: SimulationContext
        sym: Symbol für Logging

    Returns:
        Tuple von (selected_features_long, selected_features_short)
    """
    from .targets import compute_targets_cached, _validate_targets

    if not inner_folds or len(features) < 3:
        return None, None

    # Verwende ersten Inner Fold für Feature Selection
    train_df, _ = inner_folds[0]

    # Berechne Targets mit Default TP/SL (Median der Grid-Werte)
    # Feature Selection ist unabhängig von TP/SL!
    default_tp = ctx.grid_tp[len(ctx.grid_tp) // 2] if ctx.grid_tp else 20
    default_sl = ctx.grid_sl[len(ctx.grid_sl) // 2] if ctx.grid_sl else 30

    # Use compute_targets_cached which dispatches to the exit strategy plugin.
    # This ensures ATR-based exits compute targets with ATR distances,
    # not spread-based distances.
    result = compute_targets_cached(
        train_df, default_tp, default_sl, ctx, timeout_bars=None,
        exit_strategy_mode=ctx.exit_strategy,
    )
    targets_long, targets_short = result[0], result[1]
    has_long, has_short = _validate_targets(targets_long, targets_short, ctx)

    selected_long = None
    selected_short = None

    if has_long:
        selected_long, _ = select_features_from_fold(
            train_df, targets_long, features, ctx.min_trades,
            feature_selection_plugins=ctx.feature_selection_plugins,
        )
        if selected_long:
            log(2, f"  Feature Selection (Long): {len(selected_long)} Features ausgewählt", sym)

    if has_short:
        selected_short, _ = select_features_from_fold(
            train_df, targets_short, features, ctx.min_trades,
            feature_selection_plugins=ctx.feature_selection_plugins,
        )
        if selected_short:
            log(2, f"  Feature Selection (Short): {len(selected_short)} Features ausgewählt", sym)

    return selected_long, selected_short


def _compute_cached_targets(tp, sl, timeout_bars, inner_folds, inner_df, ctx):
    """Pre-compute and slice targets for all folds of a TP/SL combo."""
    from .targets import compute_targets_cached, slice_targets_for_fold

    if inner_df is None:
        return None

    use_durations = ctx.sample_weights
    if use_durations:
        full_tgt_l, full_tgt_s, full_dur_l, full_dur_s = compute_targets_cached(
            inner_df, tp, sl, ctx, timeout_bars,
            exit_strategy_mode=ctx.exit_strategy,
            return_durations=True,
        )
    else:
        full_tgt_l, full_tgt_s = compute_targets_cached(
            inner_df, tp, sl, ctx, timeout_bars,
            exit_strategy_mode=ctx.exit_strategy,
        )

    cached_targets = {}
    for fold_idx, (train_df, _) in enumerate(inner_folds):
        fold_tgt_l, fold_tgt_s, _, _ = slice_targets_for_fold(
            full_tgt_l, full_tgt_s, inner_df, train_df, ctx
        )
        if use_durations:
            fold_dur_l, fold_dur_s, _, _ = slice_targets_for_fold(
                full_dur_l, full_dur_s, inner_df, train_df, ctx
            )
            cached_targets[fold_idx] = (fold_tgt_l, fold_tgt_s, fold_dur_l, fold_dur_s)
        else:
            cached_targets[fold_idx] = (fold_tgt_l, fold_tgt_s)

    return cached_targets


def _build_candidate_and_grid_result(inner_result, tp, sl, timeout_bars, regime_config, ctx):
    """Build candidate and grid_result dicts from inner CV result."""
    if not inner_result["success"]:
        return None, None

    rrr = tp / sl
    candidate = {
        "inner_val_pnl": inner_result["avg_val_pnl"],
        "params": (tp, sl, inner_result["best_ct"]),
        "timeout_bars": timeout_bars,
        "feats": inner_result["selected_features"],
        "rrr": rrr,
        "selected_features_long": inner_result["selected_features_long"],
        "selected_features_short": inner_result["selected_features_short"],
        "fold_stability": inner_result.get("fold_stability", 0),
        "regime_filter": regime_config,
        "exit_modifier_params": ctx.exit_modifier_params,
    }

    if ctx.separate_long_short and "ct_long" in inner_result:
        candidate["ct_long"] = inner_result["ct_long"]
        candidate["ct_short"] = inner_result["ct_short"]

    conf_thresh = inner_result["best_ct"]
    grid_result = {
        "tp_mult": tp,
        "sl_mult": sl,
        "timeout_bars": timeout_bars,
        "conf_thresh": conf_thresh,
        "rrr": rrr,
        "inner_val_pnl": inner_result["avg_val_pnl"],
        "fold_stability": inner_result.get("fold_stability", 0),
        "features": inner_result["selected_features"],
        "regime_filter": regime_config,
        "exit_modifier_params": ctx.exit_modifier_params,
    }
    if isinstance(conf_thresh, tuple):
        grid_result["ct_long"] = conf_thresh[0]
        grid_result["ct_short"] = conf_thresh[1]

    return candidate, grid_result


def _process_single_grid_combo(
    tp: int,
    sl: int,
    timeout_bars,
    features: list,
    inner_folds: list,
    ctx,
    regime_config: dict,
    global_grid_pos: int,
    total_grid_combos: int,
    cached_targets: dict,
    selected_features_long: list = None,
    selected_features_short: list = None,
) -> tuple:
    """
    Verarbeitet eine einzelne Grid-Kombination (TP/SL/Timeout).

    Returns:
        Tuple von (candidate_or_none, grid_result_or_none)
    """
    inner_result = run_inner_cv(
        inner_folds, features, tp, sl, ctx,
        global_grid_pos, total_grid_combos,
        timeout_bars=timeout_bars,
        cached_targets=cached_targets,
        selected_features_long=selected_features_long,
        selected_features_short=selected_features_short,
    )
    return _build_candidate_and_grid_result(inner_result, tp, sl, timeout_bars, regime_config, ctx)


def _process_tp_sl_combo_wrapper(args):
    """
    Wrapper für sequentielle Verarbeitung einer TP/SL+timeout Kombination.

    Returns:
        Tuple von (candidate_or_none, grid_result_or_none, combo_idx)
    """
    (tp, sl, timeout_bars, combo_idx, features, inner_folds, ctx, regime_config,
     grid_offset, total_grid_combos, inner_df,
     selected_features_long, selected_features_short) = args

    global_grid_pos = grid_offset + combo_idx + 1
    cached_targets = _compute_cached_targets(tp, sl, timeout_bars, inner_folds, inner_df, ctx)

    candidate, grid_result = _process_single_grid_combo(
        tp, sl, timeout_bars,
        features, inner_folds, ctx, regime_config,
        global_grid_pos, total_grid_combos,
        cached_targets,
        selected_features_long=selected_features_long,
        selected_features_short=selected_features_short,
    )

    return candidate, grid_result, combo_idx


def _build_combo_tuples(
    grid, ctx, timeout_values, features, inner_folds, regime_config,
    total_grid_combos, inner_df,
    selected_features_long, selected_features_short, sym,
):
    """Build combo tuples for grid search. Returns (combos, skipped_count)."""
    combos = []
    combo_idx = 0
    skipped_count = 0

    for modifier_params in ctx.grid_exit_modifier_params:
        # Create a per-modifier ctx so each combo carries the right exit_modifier_params
        if modifier_params is not None:
            combo_ctx = dataclasses.replace(ctx, exit_modifier_params=modifier_params)
        else:
            combo_ctx = ctx

        for tp in grid.tp:
            for sl in grid.sl:
                rrr = tp / sl
                if ctx.min_rrr > 0 and rrr < ctx.min_rrr:
                    skipped_count += len(timeout_values)
                    log(2, f"  Grid (TP={tp}, SL={sl}) - SKIP (RRR {rrr:.2f} < {ctx.min_rrr})", sym)
                    continue

                for timeout_bars in timeout_values:
                    combos.append((
                        tp, sl, timeout_bars, combo_idx,
                        features, inner_folds, combo_ctx, regime_config,
                        0, total_grid_combos, inner_df,
                        selected_features_long, selected_features_short
                    ))
                    combo_idx += 1

    return combos, skipped_count


def _run_with_successive_halving(
    combos, inner_folds, ctx,
    features, regime_config, inner_df,
    selected_features_long, selected_features_short,
    sym, progress_callback, progress_reported, grid_total,
):
    """Fold-by-fold grid search with successive halving between folds."""
    n_folds = len(inner_folds)
    n_combos = len(combos)

    # Pre-compute cached targets for all combos
    combo_targets = {}
    for combo_idx, combo in enumerate(combos):
        tp, sl, timeout_bars, combo_ctx = combo[0], combo[1], combo[2], combo[6]
        try:
            combo_targets[combo_idx] = _compute_cached_targets(
                tp, sl, timeout_bars, inner_folds, inner_df, combo_ctx
            )
        except (ImportError, ModuleNotFoundError):
            # Re-raise setup errors — they indicate broken dependencies,
            # not something we can recover from per-combo.
            raise
        except Exception as tgt_e:
            import sys, traceback
            print(f"\n[ERROR] {sym}: target precompute failed for combo={combo_idx} "
                  f"tp={tp} sl={sl}: {type(tgt_e).__name__}: {tgt_e}\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)
            combo_targets[combo_idx] = None

    # State per combo: list of fold results
    combo_fold_results = {i: [] for i in range(n_combos)}
    active_indices = set(range(n_combos))

    log(2, f"  Successive Halving: {n_combos} combos × {n_folds} folds", sym)

    for fold_idx in range(n_folds):
        train_df, val_df = inner_folds[fold_idx]

        for combo_idx in list(active_indices):
            combo = combos[combo_idx]
            tp, sl, timeout_bars, combo_ctx = combo[0], combo[1], combo[2], combo[6]

            fold_result = _evaluate_single_fold(
                fold_idx, train_df, val_df,
                features, tp, sl, combo_ctx, timeout_bars,
                cached_targets=combo_targets[combo_idx],
                selected_features_long=selected_features_long,
                selected_features_short=selected_features_short,
            )
            combo_fold_results[combo_idx].append(fold_result)

        # Prune after each fold except the last, but not before the ratio of folds completed
        ratio = getattr(ctx, "early_pruning_min_folds_before_pruning_ratio", 0.3)
        min_folds = max(1, math.ceil(n_folds * ratio))
        if fold_idx < n_folds - 1 and fold_idx >= min_folds - 1:
            scores = []
            for idx in active_indices:
                pnls = [r["pnl"] for r in combo_fold_results[idx] if r.get("success")]
                avg_pnl = np.mean(pnls) if pnls else float("-inf")
                scores.append((idx, avg_pnl))

            scores.sort(key=lambda x: x[1], reverse=True)
            n_keep = min(
                len(scores),
                max(ctx.early_pruning_min_survivors,
                    int(len(scores) * ctx.early_pruning_keep_ratio)),
            )

            new_active = {idx for idx, _ in scores[:n_keep]}
            pruned = active_indices - new_active

            if pruned:
                threshold_pnl = scores[n_keep - 1][1] if n_keep <= len(scores) else float("-inf")
                log(2, f"  Fold {fold_idx}: {len(new_active)} survivors, "
                       f"{len(pruned)} pruned (threshold PnL={threshold_pnl:.1f})", sym)

                # Report pruned combos as done
                for _ in pruned:
                    progress_reported += 1
                    if progress_callback:
                        progress_callback(progress_reported, grid_total)

            active_indices = new_active

    # Aggregate survivors and report progress
    candidates = []
    grid_results = []

    for combo_idx in range(n_combos):
        if combo_idx not in active_indices:
            continue

        combo = combos[combo_idx]
        tp, sl, timeout_bars, combo_ctx = combo[0], combo[1], combo[2], combo[6]

        result = _aggregate_cv_folds(
            combo_fold_results[combo_idx], n_folds, combo_ctx,
            selected_features_long, selected_features_short,
        )

        candidate, grid_result = _build_candidate_and_grid_result(
            result, tp, sl, timeout_bars, regime_config, combo_ctx,
        )
        if candidate:
            candidates.append(candidate)
        if grid_result:
            grid_results.append(grid_result)

        progress_reported += 1
        if progress_callback:
            progress_callback(progress_reported, grid_total)

    return candidates, grid_results, progress_reported


def run_grid_search(
    full_pool: list,
    inner_folds: list,
    grid,
    ctx,
    regime_config: dict,
    sym: str,
    progress_callback=None,
    inner_df=None,
    preselected_features_long=None,
    preselected_features_short=None,
) -> tuple:
    """
    Führt Grid-Search über TP/SL/Timeout Kombinationen durch.

    Beinhaltet:
    1. Feature-Bereinigung (inf/nan Filter)
    2. Feature Selection (falls nicht pre-selected)
    3. Sequentielle Verarbeitung aller TP/SL-Kombinationen

    Args:
        full_pool: Alle verfügbaren Feature-Spalten
        inner_folds: Liste von (train_df, val_df) Tuples
        grid: GridConfig mit TP/SL/CT Werten
        ctx: SimulationContext
        regime_config: Regime-Filter Konfiguration
        sym: Symbol für Logging
        progress_callback: Optional callback(grid_count, grid_total)
        inner_df: DataFrame für Target-Caching
        preselected_features_long: Pre-computed feature selection (skip internal)
        preselected_features_short: Pre-computed feature selection (skip internal)

    Returns:
        Tuple von (candidates_list, grid_results_list)
    """
    features = list(full_pool)

    # Filtere Features mit inf/nan-Werten im inner_df heraus
    # Dies verhindert XGBoost-Fehler: "Input data contains `inf` or a value too large"
    if inner_df is not None and len(features) > 0:
        clean_features = []
        for feat in features:
            if feat in inner_df.columns:
                col = inner_df[feat]
                has_inf = np.isinf(col).any()
                nan_ratio = col.isna().mean()
                if has_inf:
                    log(2, f"  Feature '{feat}' hat inf-Werte - übersprungen", sym)
                elif nan_ratio > 0.5:
                    log(2, f"  Feature '{feat}' hat {nan_ratio*100:.0f}% NaN - übersprungen", sym)
                else:
                    clean_features.append(feat)
            else:
                log(2, f"  Feature '{feat}' nicht in inner_df - übersprungen", sym)

        if len(clean_features) < len(features):
            log(2, f"  {len(features) - len(clean_features)} Features mit inf/nan gefiltert", sym)
        features = clean_features

    if len(features) < 1:
        log(2, f"  Keine Features verfügbar - übersprungen", sym)
        if progress_callback:
            grid_total = ctx.total_grid_combinations()
            for i in range(grid_total):
                progress_callback(i + 1, grid_total)
        return [], []

    log(2, f"  {len(features)} Features verfügbar: {features[:5]}...", sym)

    grid_total = ctx.total_grid_combinations()
    total_grid_combos = ctx.total_grid_combinations()

    # === FEATURE SELECTION ===
    # Use pre-selected features if provided (hoisted out of regime loop)
    if preselected_features_long is not None or preselected_features_short is not None:
        selected_features_long = preselected_features_long
        selected_features_short = preselected_features_short
    else:
        selected_features_long, selected_features_short = select_features(
            inner_folds, features, ctx, sym
        )

    if not selected_features_long and not selected_features_short:
        log(2, f"  Keine Features selektiert - übersprungen", sym)
        if progress_callback:
            for i in range(grid_total):
                progress_callback(i + 1, grid_total)
        return [], []

    # Reduzierte Feature-Liste für Grid-Search
    effective_features = selected_features_long or selected_features_short or features
    log(2, f"  {len(effective_features)} selektierte Features für Grid-Search", sym)

    # Timeout-Werte: Bei adaptive_timeout nur [None], sonst Grid-Werte
    adaptive_timeout = ctx.exit_params.get("adaptive_timeout", False)
    if adaptive_timeout:
        timeout_values = [None]
    else:
        timeout_values = grid.timeout_bars if grid.timeout_bars else [None]

    # Erstelle alle Kombinationen
    combos, skipped_count = _build_combo_tuples(
        grid, ctx, timeout_values, features, inner_folds, regime_config,
        total_grid_combos, inner_df,
        selected_features_long, selected_features_short, sym,
    )

    candidates = []
    grid_results = []

    log(2, f"  Verarbeite {len(combos)} TP/SL-Kombinationen", sym)

    # SEQUENTIELLE Verarbeitung der TP/SL-Kombinationen
    # XGBoost nutzt intern bereits n_jobs für Threading
    progress_reported = skipped_count

    # Report übersprungene Combos (RRR-Filter) am Anfang
    if skipped_count > 0 and progress_callback:
        for i in range(1, skipped_count + 1):
            progress_callback(i, grid_total)

    # Early Pruning: Successive Halving (fold-by-fold with pruning)
    pruning_active = (
        ctx.early_pruning_enabled
        and len(inner_folds) >= 2
        and len(combos) > ctx.early_pruning_min_survivors
    )

    try:
        if pruning_active:
            candidates, grid_results, progress_reported = _run_with_successive_halving(
                combos, inner_folds, ctx,
                features, regime_config, inner_df,
                selected_features_long, selected_features_short,
                sym, progress_callback, progress_reported, grid_total,
            )
        else:
            for i, combo in enumerate(combos):
                try:
                    candidate, grid_result, _ = _process_tp_sl_combo_wrapper(combo)
                    progress_reported += 1

                    if progress_callback:
                        progress_callback(progress_reported, grid_total)

                    if candidate:
                        candidates.append(candidate)
                    if grid_result:
                        grid_results.append(grid_result)
                except (ImportError, ModuleNotFoundError) as e:
                    raise  # Let outer handler deal with it
                except Exception as e:
                    import sys, traceback
                    tb = traceback.format_exc()
                    print(f"\n[ERROR] {sym}: combo {i} failed: {type(e).__name__}: {e}\n{tb}",
                          file=sys.stderr, flush=True)
                    log(1, f"  ERROR in combo {i}: {type(e).__name__}: {e}", sym)
                    progress_reported += 1
                    if progress_callback:
                        progress_callback(progress_reported, grid_total)
    except (ImportError, ModuleNotFoundError) as outer_e:
        # Setup/environment errors must never be silently swallowed.
        # They indicate broken dependencies or stale Numba caches.
        import sys, traceback
        tb = traceback.format_exc()
        print(f"\n[ERROR] {sym}: {type(outer_e).__name__}: {outer_e}\n{tb}",
              file=sys.stderr, flush=True)
        log(1, f"  FATAL: {type(outer_e).__name__}: {outer_e}\n{tb}", sym)
        raise
    except Exception as outer_e:
        import sys, traceback
        tb = traceback.format_exc()
        print(f"\n[ERROR] {sym}: Grid-Search failed: {type(outer_e).__name__}: {outer_e}\n{tb}",
              file=sys.stderr, flush=True)
        log(1, f"  OUTER ERROR: {type(outer_e).__name__}: {outer_e}", sym)

    log(2, f"  {len(candidates)} Kandidaten aus {len(combos)} Kombinationen", sym)

    return candidates, grid_results
