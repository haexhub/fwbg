"""Unified simulation — derive one setting from all consistent folds and re-simulate."""
import dataclasses
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from fwbg.utils.logging import log
from fwbg.utils.progress import report_phase
from .nested_cv import evaluate_on_holdout
from .process_fold import prepare_fold_data


def merge_unified_settings(
    consistent_folds: List[Dict[str, Any]],
    all_fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Derive one unified setting from all consistent fold results.

    Merging strategy:
    - TP, SL, CT: median across folds
    - timeout_bars: median (None treated as infinity, excluded from median)
    - model_hyperparameters: taken from first fold (all consistent folds share same HP)
    - exit_modifier_params: majority vote (whole dict)
    - Features: stability >= 50% across ALL successful folds (not just consistent)

    Args:
        consistent_folds: fold results that passed the HP consistency check.
        all_fold_results: ALL successful fold results (for feature stability).

    Returns:
        candidate dict compatible with evaluate_on_holdout.
    """
    configs = [fold_result["best_config"] for fold_result in consistent_folds]

    # TP / SL: median
    tp_median = float(np.median([config["tp"] for config in configs]))
    sl_median = float(np.median([config["sl"] for config in configs]))

    # CT: median (handle tuple CT for separate long/short thresholds)
    ct_values = [config["ct"] for config in configs]
    if isinstance(ct_values[0], tuple):
        ct_long_median = float(np.median([ct[0] for ct in ct_values]))
        ct_short_median = float(np.median([ct[1] for ct in ct_values]))
        ct_median = (ct_long_median, ct_short_median)
    else:
        ct_median = float(np.median(ct_values))

    # timeout_bars: median of non-None values (None = no timeout)
    timeout_values = [config.get("timeout_bars") for config in configs]
    non_none_timeouts = [t for t in timeout_values if t is not None]
    if non_none_timeouts:
        timeout_median = int(np.median(non_none_timeouts))
    else:
        timeout_median = None

    # model_hyperparameters: all consistent folds share the same HP (by definition)
    model_hyperparameters = configs[0].get("model_hyperparameters")

    # exit_modifier_params: majority vote (whole dict as hashable key)
    exit_modifier_params_list = [
        config.get("exit_modifier_params") for config in configs
    ]
    exit_modifier_params = _majority_vote_dict(exit_modifier_params_list)

    # Features: stability >= 50% across ALL successful folds
    n_successful_folds = len(all_fold_results)
    long_counts: Dict[str, int] = {}
    short_counts: Dict[str, int] = {}
    for fold_result in all_fold_results:
        for feature in fold_result.get("selected_features_long") or []:
            long_counts[feature] = long_counts.get(feature, 0) + 1
        for feature in fold_result.get("selected_features_short") or []:
            short_counts[feature] = short_counts.get(feature, 0) + 1

    stable_features_long = [
        feature for feature, count in long_counts.items()
        if count / n_successful_folds >= 0.5
    ]
    stable_features_short = [
        feature for feature, count in short_counts.items()
        if count / n_successful_folds >= 0.5
    ]

    return {
        "params": (tp_median, sl_median, ct_median),
        "timeout_bars": timeout_median,
        "selected_features_long": stable_features_long,
        "selected_features_short": stable_features_short,
        "model_hyperparameters": model_hyperparameters,
        "exit_modifier_params": exit_modifier_params,
    }


def _majority_vote_dict(dict_list: List[Optional[Dict]]) -> Optional[Dict]:
    """Pick the most common dict from a list (majority vote)."""
    if not dict_list:
        return None

    def _hashable(d):
        if d is None:
            return None
        return tuple(sorted(d.items()))

    counter = Counter(_hashable(d) for d in dict_list)
    winner_key = counter.most_common(1)[0][0]
    if winner_key is None:
        return None
    return dict(winner_key)


def run_unified_simulation(
    wf_folds,
    unified_candidate: Dict[str, Any],
    fold_indicators,
    precomputed_raw_df,
    preprocessing_configs,
    ctx,
    sym: str,
) -> List[Dict[str, Any]]:
    """Re-simulate all folds with the unified setting.

    For each fold: prepares data, then calls evaluate_on_holdout with the
    unified candidate. For Signal models this is effectively instant (no
    trained weights). For XGBoost the model is retrained per fold with
    the unified TP/SL/features.

    Returns:
        List of per-fold result dicts from evaluate_on_holdout.
    """
    report_phase(sym, "Unified simulation...")
    log(1, "=== Unified Simulation ===", sym)

    tp, sl, ct = unified_candidate["params"]
    timeout = unified_candidate.get("timeout_bars")
    log(1, f"  Settings: TP={tp:.2f} SL={sl:.2f} CT={ct} timeout={timeout}", sym)

    model_hp = unified_candidate.get("model_hyperparameters")
    exit_mod = unified_candidate.get("exit_modifier_params")
    if model_hp:
        log(2, f"  Model HP: {model_hp}", sym)
    if exit_mod:
        log(2, f"  Exit modifier: {exit_mod}", sym)

    features_long = unified_candidate.get("selected_features_long") or []
    features_short = unified_candidate.get("selected_features_short") or []
    log(1, f"  Features: {len(features_long)} long, {len(features_short)} short", sym)

    # Build holdout context with unified params
    holdout_context = ctx
    if model_hp and model_hp != ctx.model_hyperparameters:
        holdout_context = dataclasses.replace(
            holdout_context, model_hyperparameters=model_hp,
        )
    if exit_mod and exit_mod != ctx.exit_modifier_params:
        holdout_context = dataclasses.replace(
            holdout_context, exit_modifier_params=exit_mod,
        )

    unified_fold_results = []
    total_trades = 0

    for fold_index, fold in enumerate(wf_folds):
        report_phase(sym, f"Unified sim fold {fold_index + 1}/{len(wf_folds)}...")

        fold_data = prepare_fold_data(
            fold, fold_indicators, precomputed_raw_df,
            preprocessing_configs, ctx, sym,
        )
        if fold_data is None:
            log(2, f"  Unified fold {fold_index + 1}: SKIP (insufficient data)", sym)
            continue

        train_df, test_df, _full_pool = fold_data

        result = evaluate_on_holdout(
            test_df, train_df, unified_candidate, holdout_context,
        )

        total_trades += result["n_trades"]
        pnl = result["pnl"]
        log(2, f"  Unified fold {fold_index + 1}: "
               f"PnL={pnl:.1f} Trades={result['n_trades']} "
               f"WR={result['win_rate']:.1%}", sym)

        unified_fold_results.append(result)

    log(1, f"  Unified total: {total_trades} trades across "
           f"{len(unified_fold_results)} folds", sym)

    return unified_fold_results
