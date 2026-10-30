"""Unified simulation — merge fold settings and replay the walk-forward folds.

The replay is causal: test fold k is replayed with the setting merged from the
successful folds 0..k, each chosen on training data that ends before fold k's
test data. The frozen (deployment) setting is the merge over all folds.
"""
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from fwbg.utils.logging import log
from fwbg.utils.progress import report_phase
from fwbg.research.fingerprint import canonical_json
from .nested_cv import evaluate_on_holdout
from .process_fold import apply_candidate_regime, candidate_context, prepare_fold_data


def merge_unified_settings(
    majority_folds: List[Dict[str, Any]],
    all_fold_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Derive one unified setting from the folds of the model-HP majority.

    Merging strategy:
    - TP, SL, CT: median across folds
    - timeout_bars: median (None treated as infinity, excluded from median)
    - model_hyperparameters: taken from first fold (all majority folds share same HP)
    - exit/entry modifier with their params and the regime filter: majority vote
      over the combination one fold chose, so a modifier never meets another
      fold's params
    - Features: stability >= 50% across ALL successful folds (not just the majority)

    Args:
        majority_folds: fold results that share the majority model HP
            (:func:`consistent_folds`).
        all_fold_results: ALL successful fold results (for feature stability).

    Returns:
        candidate dict compatible with evaluate_on_holdout.
    """
    configs = [fold_result["best_config"] for fold_result in majority_folds]

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

    # model_hyperparameters: all majority folds share the same HP (by definition)
    model_hyperparameters = configs[0].get("model_hyperparameters")

    voted_keys = (
        "exit_modifier", "exit_modifier_params",
        "entry_modifier", "entry_modifier_params", "regime_filter",
    )
    voted = _majority_vote([{key: config.get(key) for key in voted_keys} for config in configs])

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

    # exit_strategy / exit_params: all majority folds should agree
    # (they were selected by the same exit strategy combo).
    exit_strategy = configs[0].get("exit_strategy")
    exit_params = configs[0].get("exit_params")

    return {
        "params": (tp_median, sl_median, ct_median),
        "timeout_bars": timeout_median,
        "selected_features_long": stable_features_long,
        "selected_features_short": stable_features_short,
        "model_hyperparameters": model_hyperparameters,
        **voted,
        "exit_strategy": exit_strategy,
        "exit_params": exit_params,
    }


def _majority_vote(values: List[Any]) -> Any:
    """Most common value (first seen wins ties); dicts and lists compare by content."""
    if not values:
        return None
    keys = [canonical_json(value, default=repr) for value in values]
    winner = Counter(keys).most_common(1)[0][0]
    return values[keys.index(winner)]


def hp_key(hp: Optional[Dict[str, Any]]) -> tuple:
    """Hashable key for model_hyperparameters (ignoring None values)."""
    if not hp:
        return ()
    def _make_hashable(v):
        if isinstance(v, list):
            return tuple(v)
        return v
    return tuple(sorted((k, _make_hashable(v)) for k, v in hp.items() if v is not None))


def consistent_folds(fold_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The fold results that share the majority model_hyperparameters.

    For rule-based signal strategies the HP define the strategy; folds that disagree
    are left out of the merge. Ties go to the HP seen first.
    """
    hp_keys = [hp_key(r["best_config"].get("model_hyperparameters")) for r in fold_results]
    hp_counts = {}
    for key in hp_keys:
        hp_counts[key] = hp_counts.get(key, 0) + 1
    majority_hp_key = max(hp_counts, key=hp_counts.get) if hp_counts else ()
    return [r for r, key in zip(fold_results, hp_keys) if key == majority_hp_key]


def causal_candidates(
    wf_folds, fold_results: List[Dict[str, Any]]
) -> List[Optional[Dict[str, Any]]]:
    """Per walk-forward fold, the unified setting known before its test period.

    Fold k gets merge_unified_settings over the successful folds with fold_id <= k: each
    chose its setting on its own training data, which ends before fold k's test data.
    None while no fold up to k has succeeded (that fold is not traded).
    Fold results are ordered by fold_id first: parallel folds arrive in completion
    order, and the merge's tie-breaks and first-fold choices depend on order.
    """
    ordered = sorted(fold_results, key=lambda r: r["fold_id"])
    candidates = []
    for fold in wf_folds:
        prefix = [r for r in ordered if r["fold_id"] <= fold.fold_id]
        candidates.append(
            merge_unified_settings(consistent_folds(prefix), prefix) if prefix else None
        )
    return candidates


def run_unified_simulation(
    wf_folds,
    candidates: List[Optional[Dict[str, Any]]],
    fold_indicators,
    precomputed_raw_df,
    preprocessing_configs,
    ctx,
    sym: str,
    prepare_data_fn=None,  # caller passes the right prepare function
) -> List[Dict[str, Any]]:
    """Re-simulate each fold with its own candidate (one per fold).

    For each fold: prepares data, then calls evaluate_on_holdout with that
    fold's candidate. For Signal models this is effectively instant (no
    trained weights). For XGBoost the model is retrained per fold with
    the candidate's TP/SL/features. A None candidate means the fold is not
    traded: it gets an empty result without preparing data.

    Returns:
        List of per-fold result dicts from evaluate_on_holdout, each with its fold_id.
    """
    if len(candidates) != len(wf_folds):
        raise ValueError(
            f"run_unified_simulation needs one candidate per fold: "
            f"{len(candidates)} candidates for {len(wf_folds)} folds"
        )

    report_phase(sym, "Unified simulation...")
    log(1, "=== Unified Simulation ===", sym)

    unified_fold_results = []
    total_trades = 0

    for fold_index, (fold, candidate) in enumerate(zip(wf_folds, candidates)):
        report_phase(sym, f"Unified sim fold {fold_index + 1}/{len(wf_folds)}...")

        if candidate is None:
            log(
                2,
                f"  Unified fold {fold.fold_id + 1}: "
                "not traded (no successful fold up to this one)",
                sym,
            )
            unified_fold_results.append({
                "trades": [], "trades_detailed": [], "pnl": 0.0, "win_rate": 0.0,
                "n_trades": 0, "dropped_features": [], "fold_id": fold.fold_id,
            })
            continue

        tp, sl, ct = candidate["params"]
        features_long = candidate.get("selected_features_long") or []
        features_short = candidate.get("selected_features_short") or []
        log(2, f"  Unified fold {fold.fold_id + 1} settings: TP={tp:.2f} SL={sl:.2f} CT={ct} "
               f"timeout={candidate.get('timeout_bars')} "
               f"features={len(features_long)} long, {len(features_short)} short", sym)
        if candidate.get("model_hyperparameters"):
            log(2, f"    Model HP: {candidate['model_hyperparameters']}", sym)
        if candidate.get("exit_modifier_params"):
            log(2, f"    Exit modifier: {candidate['exit_modifier_params']}", sym)

        _prepare = prepare_data_fn or prepare_fold_data
        fold_data = _prepare(
            fold, fold_indicators, precomputed_raw_df,
            preprocessing_configs, ctx, sym,
        )
        if fold_data is None:
            log(2, f"  Unified fold {fold.fold_id + 1}: SKIP (insufficient data)", sym)
            continue

        train_df, test_df, _full_pool = fold_data

        # Filter the candidate's features to columns available in this fold.
        # NaN-cleaning in prepare_fold_data may drop columns that were
        # stable across most folds but have >10% NaN in this specific fold.
        available = set(train_df.columns)
        fold_candidate = candidate
        dropped = set()
        for key in ("selected_features_long", "selected_features_short"):
            feats = candidate.get(key)
            if feats and not all(f in available for f in feats):
                if fold_candidate is candidate:
                    fold_candidate = {**candidate}
                fold_candidate[key] = [f for f in feats if f in available]
                dropped.update(f for f in feats if f not in available)

        apply_candidate_regime(train_df, test_df, candidate)
        result = evaluate_on_holdout(
            test_df, train_df, fold_candidate, candidate_context(ctx, candidate),
        )
        result["dropped_features"] = sorted(dropped)
        # The RRR this fold was traded with: causal reporting sizes the fold with it.
        result["rrr"] = tp / sl if sl > 0 else 1.0
        # Lets a replay split across indicator variants be put back in fold order.
        result["fold_id"] = fold.fold_id

        total_trades += result["n_trades"]
        pnl = result["pnl"]
        log(2, f"  Unified fold {fold.fold_id + 1}: "
               f"PnL={pnl:.1f} Trades={result['n_trades']} "
               f"WR={result['win_rate']:.1%}", sym)

        unified_fold_results.append(result)

    log(1, f"  Unified total: {total_trades} trades across "
           f"{len(unified_fold_results)} folds", sym)

    return unified_fold_results
