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
from fwbg.pipeline.features import select_plateau_features, select_features_boruta
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

    # Copy nur für die Haupt-DataFrames (werden später modifiziert: _regime_ok)
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
    feature_selection: str = "boruta",
    max_features: int = 0,
    min_z_score: float = 0.3,
) -> Tuple[Optional[List[str]], Dict[str, float]]:
    """
    Wählt Features basierend auf einem Training-Fold.

    Args:
        train_df: Training DataFrame
        targets: Target Array
        group_features: Features der aktuellen Gruppe
        min_trades: Minimum Trades
        feature_selection:
            - "boruta" (default): Boruta findet alle relevanten Features
            - "boruta_plateau": Boruta + Plateau-Validierung (kombiniert)
            - "importance_based": Altes Verhalten mit top_n=5
        max_features: Maximum Features pro Modell (0 = Default 15)
        min_z_score: Minimum Z-Score für Boruta Feature-Akzeptanz (Default 0.3)

    Returns:
        (selected_features, importances_dict) oder (None, {})
    """
    if np.count_nonzero(targets) < min_trades // 2:
        return None, {}

    # Nur verfügbare Features nutzen
    available_features = [f for f in group_features if f in train_df.columns]
    if not available_features:
        return None, {}

    if feature_selection == "boruta":
        # Boruta: Findet relevante Features, begrenzt durch max_features
        return select_features_boruta(
            train_df, targets, available_features,
            min_trades=min_trades,
            min_z_score=min_z_score,
            max_features=max_features,
        )

    elif feature_selection == "boruta_plateau":
        # Kombination: Boruta findet relevante Features, Plateau filtert danach
        boruta_features, importances = select_features_boruta(
            train_df, targets, available_features,
            min_trades=min_trades,
            min_z_score=min_z_score * 0.8,
            max_features=max_features,
        )

        if boruta_features and len(boruta_features) >= 2:
            from .plateau import calculate_feature_plateau_score
            plateau_results = calculate_feature_plateau_score(importances, boruta_features)

            stable_features = [
                f for f in boruta_features
                if f in plateau_results and (
                    plateau_results[f]["is_plateau"] or
                    len(plateau_results[f]["neighbors"]) == 0
                )
            ]

            if len(stable_features) >= 2:
                return stable_features, importances
            return boruta_features, importances

        return boruta_features, importances

    else:
        # Altes Verhalten: Importance + Plateau mit top_n=5
        params = {
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": get_xgboost_n_jobs(),
        }

        model = XGBClassifier(**params)
        model.fit(train_df[available_features], targets)
        importances = pd.Series(model.feature_importances_, index=available_features)

        plateau_features = select_plateau_features(
            importances.to_dict(), available_features,
            top_n=5, min_importance=0
        )

        if len(plateau_features) >= 2:
            return plateau_features, importances.to_dict()

        return None, importances.to_dict()


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
            model = XGBClassifier(**cpu_params)
            model.fit(train_df[features], targets, **fit_kwargs)
        else:
            raise

    return model


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
    inner_val_pnls = []
    best_ct_votes = {}

    # Early Termination Setup
    total_folds = len(inner_folds)
    min_fold_stability = getattr(ctx, 'min_fold_stability', 0.5)
    early_termination_enabled = getattr(ctx, 'early_termination', True)
    min_profitable = int(np.ceil(total_folds * min_fold_stability))
    profitable_count = 0
    failed_count = 0
    early_terminated = False
    first_fold_failed = False

    # First-Fold Sanity Check Setup
    first_fold_sanity_check = getattr(ctx, 'first_fold_sanity_check', True)
    first_fold_min_win_rate = getattr(ctx, 'first_fold_min_win_rate', 0.25)
    first_fold_min_pnl = getattr(ctx, 'first_fold_min_pnl', -10.0)
    first_fold_min_trades = getattr(ctx, 'first_fold_min_trades', 5)

    for fold_idx, (train_df, val_df) in enumerate(inner_folds):
        # Early Termination Check
        if early_termination_enabled and min_profitable > 0:
            remaining_folds = total_folds - fold_idx
            max_possible_profitable = profitable_count + remaining_folds
            if max_possible_profitable < min_profitable:
                early_terminated = True
                break

        # Use cached targets if available
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
            failed_count += 1
            continue

        # Feature-Auswahl auf erstem Fold (mit Fallback auf spätere Folds)
        if selected_features_long is None and selected_features_short is None:
            if has_long:
                selected_features_long, _ = select_features_from_fold(
                    train_df, targets_long, group_features, ctx.min_trades,
                    feature_selection=ctx.feature_selection,
                    max_features=ctx.max_features,
                    min_z_score=ctx.min_z_score,
                )
            if has_short:
                selected_features_short, _ = select_features_from_fold(
                    train_df, targets_short, group_features, ctx.min_trades,
                    feature_selection=ctx.feature_selection,
                    max_features=ctx.max_features,
                    min_z_score=ctx.min_z_score,
                )

        if not selected_features_long and not selected_features_short:
            failed_count += 1
            continue

        mod_long = train_model(train_df, targets_long, selected_features_long, ctx.min_trades, ctx, use_reduced_params=True, sample_weight=weights) if has_long else None
        mod_short = train_model(train_df, targets_short, selected_features_short, ctx.min_trades, ctx, use_reduced_params=True, sample_weight=weights) if has_short else None

        best_fold_ct, best_fold_pnl, trades_by_ct = evaluate_on_validation(
            val_df, mod_long, mod_short,
            selected_features_long, selected_features_short,
            tp, sl, ctx, timeout_bars
        )

        if best_fold_ct:
            inner_val_pnls.append(best_fold_pnl)
            best_ct_votes[best_fold_ct] = best_ct_votes.get(best_fold_ct, 0) + 1

            if best_fold_pnl > 0:
                profitable_count += 1
            else:
                failed_count += 1

            # First-Fold Sanity Check
            if fold_idx == 0 and first_fold_sanity_check:
                fold_trades = trades_by_ct.get(best_fold_ct, [])
                n_fold_trades = len(fold_trades)

                if n_fold_trades > 0:
                    fold_win_rate = fold_trades.count(1.0) / n_fold_trades

                    is_catastrophic = (
                        fold_win_rate < first_fold_min_win_rate and
                        best_fold_pnl < first_fold_min_pnl and
                        n_fold_trades >= first_fold_min_trades
                    )

                    if is_catastrophic:
                        first_fold_failed = True
                        break
                elif n_fold_trades < first_fold_min_trades:
                    first_fold_failed = True
                    break
        else:
            failed_count += 1

    if early_terminated:
        return {"success": False, "early_terminated": True, "failed_folds": failed_count}

    if first_fold_failed:
        return {"success": False, "first_fold_failed": True, "reason": "catastrophic_first_fold"}

    if not inner_val_pnls or not best_ct_votes:
        return {"success": False}

    selected_features = []
    if selected_features_long:
        selected_features.extend(selected_features_long)
    if selected_features_short:
        selected_features.extend([f for f in selected_features_short if f not in selected_features])

    if not selected_features:
        return {"success": False}

    profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)
    fold_stability = profitable_folds / len(inner_val_pnls) if inner_val_pnls else 0

    if best_ct_votes:
        best_ct = max(best_ct_votes.keys(), key=lambda x: best_ct_votes[x])
    else:
        best_ct = ctx.grid_ct[len(ctx.grid_ct) // 2] if ctx.grid_ct else 0.5

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
    pnl = sum(trades) if trades else 0
    win_rate = trades.count(1.0) / len(trades) if trades else 0

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
            "pnl": sum(t.get("result", 0) for t in long_trades),
            "win_rate": sum(1 for t in long_trades if t.get("result") == 1.0) / len(long_trades) if long_trades else 0,
        }
        output["short_stats"] = {
            "n_trades": len(short_trades),
            "wins": sum(1 for t in short_trades if t.get("result") == 1.0),
            "pnl": sum(t.get("result", 0) for t in short_trades),
            "win_rate": sum(1 for t in short_trades if t.get("result") == 1.0) / len(short_trades) if short_trades else 0,
        }

    return output
