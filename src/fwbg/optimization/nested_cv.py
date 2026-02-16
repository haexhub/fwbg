"""
Nested Cross-Validation für unbiased Optimizer-Evaluation.

Struktur:
[=============== INNER (80%) ===============][== HOLDOUT (20%) ==]
                    ↓                                     ↓
            Grid-Search hier                    Finale Evaluation
            (Walk-Forward Folds)                (NIE während Optimierung gesehen!)
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from xgboost import XGBClassifier

from fwbg.core.context import SimulationContext
from fwbg.core import get_feature_selector
from fwbg.utils.xgb_config import get_xgboost_n_jobs, get_xgboost_params

from .targets import (
    _validate_targets,
    compute_targets,
    _get_probs,
    evaluate_on_validation,
    simulate_trades_sequential,
    simulate_trades_sequential_separate_ct,
)


def nested_cv_split(
    df: pd.DataFrame,
    holdout_ratio: float = 0.20,
    n_inner_folds: int = 5,
    oos_size: int = 4000,
    embargo_bars: int = 0,
) -> Dict[str, Any]:
    """
    Nested Cross-Validation Split für unbiased Evaluation.

    OPTIMIERUNG: Verwendet Views statt Copies wo möglich.
    .copy() nur für inner_df und holdout_df (die modifiziert werden können).
    Fold-DataFrames sind Views (read-only).

    Returns:
        dict mit inner_folds, holdout_df, inner_df
    """
    total_len = len(df)
    holdout_size = int(total_len * holdout_ratio)
    inner_size = total_len - holdout_size

    # Copy nur für die Haupt-DataFrames (werden später modifiziert: _regime)
    inner_df = df.iloc[:inner_size].copy()
    holdout_df = df.iloc[inner_size:].copy()

    inner_folds = []
    val_size = min(oos_size, inner_size // (n_inner_folds + 2))

    for i in range(n_inner_folds):
        val_end = inner_size - (i * val_size)
        val_start = val_end - val_size
        train_end = val_start - embargo_bars

        if train_end < val_size * 2:
            continue

        # Views statt Copies - DataFrame-Slices sind read-only in der Simulation
        train_df = inner_df.iloc[:train_end]
        val_df = inner_df.iloc[val_start:val_end]
        inner_folds.append((train_df, val_df))

    return {
        "inner_folds": list(reversed(inner_folds)),
        "holdout_df": holdout_df,
        "inner_df": inner_df,
    }


def select_features_from_fold(
    train_df: pd.DataFrame,
    targets: np.ndarray,
    group_features: List[str],
    min_trades: int,
    feature_selection_plugins: List[Dict] = None,
) -> Tuple[Optional[List[str]], Dict]:
    """
    Wählt Features über das Plugin-Interface aus.

    Iteriert über konfigurierte Feature-Selection-Plugins und kettet sie:
    Jedes Plugin operiert auf dem Output des vorherigen.

    Args:
        train_df: Training DataFrame
        targets: Target Array
        group_features: Features der aktuellen Gruppe
        min_trades: Minimum Trades
        feature_selection_plugins: Liste von Plugin-Configs
            [{"name": "boruta", "params": {"min_z_score": 0.5}}, ...]

    Returns:
        (selected_features, metadata) oder (None, {})
    """
    if np.count_nonzero(targets) < min_trades // 2:
        return None, {}

    available_features = [f for f in group_features if f in train_df.columns]
    if not available_features:
        return None, {}

    if not feature_selection_plugins:
        return available_features, {}

    selected = available_features
    metadata = {}

    for plugin_config in feature_selection_plugins:
        name = plugin_config["name"]
        params = plugin_config.get("params", {}).copy()

        selector_cls = get_feature_selector(name)
        selector = selector_cls()

        max_features = params.pop("max_features", None)
        selected, meta = selector.select_features(
            train_df[selected], targets,
            max_features=max_features, **params
        )

        if not selected or len(selected) < 2:
            return None, metadata
        metadata.update(meta)

    return selected, metadata


def train_model(
    train_df: pd.DataFrame,
    targets: np.ndarray,
    features: Optional[List[str]],
    min_trades: int,
    ctx: SimulationContext,
    use_reduced_params: bool = False,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[XGBClassifier]:
    """
    Trainiert ein XGBoost-Modell.

    Args:
        use_reduced_params: Wenn True, werden Hyperparameter halbiert (für Inner CV)
        sample_weight: Optional - Uniqueness-basierte Sample Weights (AFML Ch. 4)
    """
    if features is None or np.count_nonzero(targets) < min_trades // 2:
        return None

    params = ctx.model_hyperparameters.copy()

    if use_reduced_params:
        params["n_estimators"] = max(10, params.get("n_estimators", 100) // 2)

    params.setdefault("random_state", 42)
    params.setdefault("verbosity", 0)
    params["n_jobs"] = get_xgboost_n_jobs()

    params.update(get_xgboost_params())

    model = XGBClassifier(**params)

    if ctx.probability_calibration:
        from sklearn.calibration import CalibratedClassifierCV
        model = CalibratedClassifierCV(model, method=ctx.calibration_method, cv=3)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    try:
        model.fit(train_df[features], targets, **fit_kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        if "cuda" in error_msg or "gpu" in error_msg or "device" in error_msg:
            from fwbg.utils.xgb_config import disable_gpu
            disable_gpu()
            cpu_params = {k: v for k, v in params.items()
                         if k not in ("device", "tree_method")}
            cpu_params["tree_method"] = "hist"
            cpu_params["device"] = "cpu"
            cpu_model = XGBClassifier(**cpu_params)
            if ctx.probability_calibration:
                from sklearn.calibration import CalibratedClassifierCV
                cpu_model = CalibratedClassifierCV(cpu_model, method=ctx.calibration_method, cv=3)
            cpu_model.fit(train_df[features], targets, **fit_kwargs)
            return cpu_model
        else:
            raise

    return model


def _generate_oof_predictions(
    df: pd.DataFrame,
    targets: np.ndarray,
    features: List[str],
    ctx: SimulationContext,
    n_splits: int = 3,
) -> np.ndarray:
    """
    Generate out-of-fold probability predictions (AFML Ch. 3).

    Uses time-series KFold (no shuffle) to avoid data leakage.
    Each fold trains a reduced-param XGBoost and predicts on held-out bars.

    Returns:
        Array of shape (n,) with OOF win-probabilities for each bar.
    """
    from sklearn.model_selection import KFold

    oof_probs = np.zeros(len(df))
    X = df[features].values
    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_idx, val_idx in kf.split(X):
        if len(np.unique(targets[train_idx])) < 2:
            continue

        params = ctx.model_hyperparameters.copy()
        params["n_estimators"] = max(10, params.get("n_estimators", 100) // 2)
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", 0)
        params["n_jobs"] = get_xgboost_n_jobs()
        params.update(get_xgboost_params())

        model = XGBClassifier(**params)
        model.fit(X[train_idx], targets[train_idx])

        if 1 in model.classes_:
            win_idx = np.where(model.classes_ == 1)[0][0]
            oof_probs[val_idx] = model.predict_proba(X[val_idx])[:, win_idx]

    return oof_probs


def _train_meta_model(
    df: pd.DataFrame,
    targets: np.ndarray,
    features: List[str],
    oof_probs: np.ndarray,
    ctx: SimulationContext,
) -> Optional[XGBClassifier]:
    """
    Train a meta-model that predicts whether the primary signal is profitable (AFML Ch. 3).

    Meta-features = original features + primary model's OOF probability.
    Returns None when insufficient positive targets for training.
    """
    if np.count_nonzero(targets) < ctx.min_trades // 2:
        return None
    if len(np.unique(targets)) < 2:
        return None

    X_meta = np.column_stack([df[features].values, oof_probs])

    params = ctx.model_hyperparameters.copy()
    params["n_estimators"] = max(10, params.get("n_estimators", 100) // 2)
    params.setdefault("random_state", 42)
    params.setdefault("verbosity", 0)
    params["n_jobs"] = get_xgboost_n_jobs()
    params.update(get_xgboost_params())

    meta_model = XGBClassifier(**params)
    meta_model.fit(X_meta, targets)
    return meta_model


def _evaluate_single_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_features: List[str],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
    cached_targets: Optional[Dict] = None,
    selected_features_long: Optional[List[str]] = None,
    selected_features_short: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single combo on a single inner fold.

    Used by run_inner_cv (standard path) and successive halving (pruning path).
    Returns fold result with pnl, best_ct, and updated feature selections.
    """
    weights = None
    if cached_targets is not None and fold_idx in cached_targets:
        entry = cached_targets[fold_idx]
        if len(entry) == 4:
            targets_long, targets_short, dur_long, dur_short = entry
            if ctx.sample_weights:
                from .purging import compute_sample_weights
                weights = compute_sample_weights(dur_long, dur_short, len(train_df))
        else:
            targets_long, targets_short = entry
        has_long, has_short = _validate_targets(targets_long, targets_short, ctx)
    else:
        targets_long, targets_short, has_long, has_short = compute_targets(
            train_df, tp, sl, ctx, timeout_bars
        )

    if not has_long and not has_short:
        return {"success": False}

    feat_long = selected_features_long
    feat_short = selected_features_short
    if feat_long is None and feat_short is None:
        if has_long:
            feat_long, _ = select_features_from_fold(
                train_df, targets_long, group_features, ctx.min_trades,
                feature_selection_plugins=ctx.feature_selection_plugins,
            )
        if has_short:
            feat_short, _ = select_features_from_fold(
                train_df, targets_short, group_features, ctx.min_trades,
                feature_selection_plugins=ctx.feature_selection_plugins,
            )

    if not feat_long and not feat_short:
        return {"success": False}

    mod_long = train_model(
        train_df, targets_long, feat_long, ctx.min_trades, ctx,
        use_reduced_params=True, sample_weight=weights,
    ) if has_long else None
    mod_short = train_model(
        train_df, targets_short, feat_short, ctx.min_trades, ctx,
        use_reduced_params=True, sample_weight=weights,
    ) if has_short else None

    # Meta-Labeling: train meta-models to filter primary predictions
    meta_mod_long = None
    meta_mod_short = None
    if ctx.meta_labeling:
        if mod_long is not None and feat_long:
            oof_long = _generate_oof_predictions(train_df, targets_long, feat_long, ctx)
            meta_mod_long = _train_meta_model(train_df, targets_long, feat_long, oof_long, ctx)
        if mod_short is not None and feat_short:
            oof_short = _generate_oof_predictions(train_df, targets_short, feat_short, ctx)
            meta_mod_short = _train_meta_model(train_df, targets_short, feat_short, oof_short, ctx)

    best_fold_ct, best_fold_pnl, trades_by_ct = evaluate_on_validation(
        val_df, mod_long, mod_short,
        feat_long, feat_short,
        tp, sl, ctx, timeout_bars,
        meta_mod_long=meta_mod_long,
        meta_mod_short=meta_mod_short,
    )

    if not best_fold_ct:
        return {"success": False}

    return {
        "success": True,
        "pnl": best_fold_pnl,
        "best_ct": best_fold_ct,
        "trades_by_ct": trades_by_ct,
        "selected_features_long": feat_long,
        "selected_features_short": feat_short,
    }


def _aggregate_cv_folds(
    fold_results: List[Dict[str, Any]],
    total_folds: int,
    ctx: SimulationContext,
    selected_features_long: Optional[List[str]] = None,
    selected_features_short: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate per-fold results into final inner CV result.

    Computes average PnL, majority CT vote, fold stability, and merged feature list.
    """
    inner_val_pnls = []
    best_ct_votes = {}

    for fr in fold_results:
        if not fr.get("success"):
            continue
        inner_val_pnls.append(fr["pnl"])
        ct = fr["best_ct"]
        best_ct_votes[ct] = best_ct_votes.get(ct, 0) + 1

    if not inner_val_pnls or not best_ct_votes:
        return {"success": False}

    selected_features = []
    if selected_features_long:
        selected_features.extend(selected_features_long)
    if selected_features_short:
        selected_features.extend(
            [f for f in selected_features_short if f not in selected_features]
        )

    if not selected_features:
        return {"success": False}

    profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)
    fold_stability = profitable_folds / total_folds if total_folds > 0 else 0

    best_ct = max(best_ct_votes.keys(), key=lambda x: best_ct_votes[x])

    result = {
        "success": True,
        "avg_val_pnl": np.mean(inner_val_pnls),
        "best_ct": best_ct,
        "selected_features_long": selected_features_long,
        "selected_features_short": selected_features_short,
        "selected_features": selected_features,
        "fold_stability": fold_stability,
        "fold_pnls": inner_val_pnls,
    }

    if ctx.separate_long_short and isinstance(best_ct, tuple):
        result["ct_long"] = best_ct[0]
        result["ct_short"] = best_ct[1]

    return result


def run_inner_cv(
    inner_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    group_features: List[str],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    global_grid_pos: int,
    total_grid_combos: int,
    timeout_bars: int = None,
    cached_targets: Optional[Dict] = None,
    selected_features_long: Optional[List[str]] = None,
    selected_features_short: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Führt Inner Cross-Validation für eine Grid-Kombination durch.

    Args:
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen
        cached_targets: Optional - vorberechnete Targets {fold_idx: (targets_long, targets_short)}

    Returns:
        dict mit success, avg_val_pnl, best_ct, selected_features etc.

    Early Termination:
        Wenn early_termination aktiviert ist und der Kandidat mathematisch
        nicht mehr min_fold_stability erreichen kann, wird vorzeitig abgebrochen.
    """
    total_folds = len(inner_folds)
    min_fold_stability = getattr(ctx, 'min_fold_stability', 0.5)
    early_termination_enabled = getattr(ctx, 'early_termination', True)
    min_profitable = int(np.ceil(total_folds * min_fold_stability))
    profitable_count = 0
    failed_count = 0

    first_fold_sanity_check = getattr(ctx, 'first_fold_sanity_check', True)
    first_fold_min_win_rate = getattr(ctx, 'first_fold_min_win_rate', 0.25)
    first_fold_min_pnl = getattr(ctx, 'first_fold_min_pnl', -10.0)
    first_fold_min_trades = getattr(ctx, 'first_fold_min_trades', 5)

    fold_results = []

    for fold_idx, (train_df, val_df) in enumerate(inner_folds):
        # Early Termination Check
        if early_termination_enabled and min_profitable > 0:
            remaining_folds = total_folds - fold_idx
            max_possible_profitable = profitable_count + remaining_folds
            if max_possible_profitable < min_profitable:
                return {"success": False, "early_terminated": True, "failed_folds": failed_count}

        fold_result = _evaluate_single_fold(
            fold_idx, train_df, val_df,
            group_features, tp, sl, ctx, timeout_bars,
            cached_targets=cached_targets,
            selected_features_long=selected_features_long,
            selected_features_short=selected_features_short,
        )
        fold_results.append(fold_result)

        if fold_result["success"]:
            if selected_features_long is None:
                selected_features_long = fold_result.get("selected_features_long")
            if selected_features_short is None:
                selected_features_short = fold_result.get("selected_features_short")

            if fold_result["pnl"] > 0:
                profitable_count += 1
            else:
                failed_count += 1

            # First-Fold Sanity Check
            if fold_idx == 0 and first_fold_sanity_check:
                trades_by_ct = fold_result["trades_by_ct"]
                best_ct = fold_result["best_ct"]
                fold_trades = trades_by_ct.get(best_ct, [])
                n_fold_trades = len(fold_trades)

                if n_fold_trades > 0:
                    fold_win_rate = sum(1 for t in fold_trades if t["result"] == 1.0) / n_fold_trades
                    is_catastrophic = (
                        fold_win_rate < first_fold_min_win_rate and
                        fold_result["pnl"] < first_fold_min_pnl and
                        n_fold_trades >= first_fold_min_trades
                    )
                    if is_catastrophic:
                        return {"success": False, "first_fold_failed": True, "reason": "catastrophic_first_fold"}
                elif n_fold_trades < first_fold_min_trades:
                    return {"success": False, "first_fold_failed": True, "reason": "catastrophic_first_fold"}
        else:
            failed_count += 1

    return _aggregate_cv_folds(
        fold_results, total_folds, ctx,
        selected_features_long, selected_features_short,
    )


def evaluate_on_holdout(
    holdout_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    candidate: Dict[str, Any],
    ctx: SimulationContext
) -> Dict[str, Any]:
    """
    Finale Evaluation auf dem Holdout-Set.
    Trainiert Modell auf GESAMTEM Inner-Set und testet auf Holdout.

    Returns:
        dict mit trades, trades_detailed, pnl, win_rate, n_trades
    """
    tp, sl, ct = candidate["params"]
    timeout_bars = candidate.get("timeout_bars")
    features_long = candidate.get("selected_features_long")
    features_short = candidate.get("selected_features_short")

    # Berechne Targets (und optional Durations für Sample Weights)
    weights = None
    if ctx.sample_weights:
        from .targets import compute_targets_cached
        result = compute_targets_cached(
            inner_df, tp, sl, ctx, timeout_bars,
            exit_strategy_mode=ctx.exit_strategy,
            return_durations=True,
        )
        targets_long, targets_short, dur_long, dur_short = result
        has_long, has_short = _validate_targets(targets_long, targets_short, ctx)

        from .purging import compute_sample_weights
        weights = compute_sample_weights(dur_long, dur_short, len(inner_df))
    else:
        targets_long, targets_short, has_long, has_short = compute_targets(inner_df, tp, sl, ctx, timeout_bars)

    mod_long = train_model(inner_df, targets_long, features_long, ctx.min_trades, ctx, use_reduced_params=False, sample_weight=weights) if has_long and features_long else None
    mod_short = train_model(inner_df, targets_short, features_short, ctx.min_trades, ctx, use_reduced_params=False, sample_weight=weights) if has_short and features_short else None

    if not mod_long and not mod_short:
        return {"trades": [], "trades_detailed": [], "pnl": 0, "win_rate": 0, "n_trades": 0}

    probs_long, long_win_idx = _get_probs(mod_long, holdout_df, features_long)
    probs_short, short_win_idx = _get_probs(mod_short, holdout_df, features_short)

    if isinstance(ct, tuple):
        ct_long, ct_short = ct
        result = simulate_trades_sequential_separate_ct(
            holdout_df, probs_long, probs_short, long_win_idx, short_win_idx,
            ct_long, ct_short, tp, sl, ctx, return_detailed=True,
            timeout_bars=timeout_bars
        )
    else:
        result = simulate_trades_sequential(
            holdout_df, probs_long, probs_short, long_win_idx, short_win_idx,
            ct, tp, sl, ctx, return_detailed=True, timeout_bars=timeout_bars
        )

    trades = result["trades"]
    trades_detailed = result["trades_detailed"]
    pnl = sum(t["pnl_raw"] for t in trades) if trades else 0
    win_rate = sum(1 for t in trades if t["result"] == 1.0) / len(trades) if trades else 0

    output = {
        "trades": trades,
        "trades_detailed": trades_detailed,
        "pnl": pnl,
        "win_rate": win_rate,
        "n_trades": len(trades),
    }

    if ctx.separate_long_short and trades_detailed:
        long_trades = [t for t in trades_detailed if t.get("direction") == "LONG"]
        short_trades = [t for t in trades_detailed if t.get("direction") == "SHORT"]

        output["long_stats"] = {
            "n_trades": len(long_trades),
            "wins": sum(1 for t in long_trades if t.get("result") == 1.0),
            "pnl": sum(t.get("pnl_raw", 0) for t in long_trades),
            "win_rate": sum(1 for t in long_trades if t.get("result") == 1.0) / len(long_trades) if long_trades else 0,
        }
        output["short_stats"] = {
            "n_trades": len(short_trades),
            "wins": sum(1 for t in short_trades if t.get("result") == 1.0),
            "pnl": sum(t.get("pnl_raw", 0) for t in short_trades),
            "win_rate": sum(1 for t in short_trades if t.get("result") == 1.0) / len(short_trades) if short_trades else 0,
        }

    return output
