"""
Nested Cross-Validation für unbiased Optimizer-Evaluation.

Struktur:
[=============== INNER (80%) ===============][== HOLDOUT (20%) ==]
                    ↓                                     ↓
            Grid-Search hier                    Finale Evaluation
            (Walk-Forward Folds)                (NIE während Optimierung gesehen!)
"""
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import MAX_TRADE_BARS, MIN_TRADES, OOS_SIZE, RELEVANCE_THRESHOLD
from .simulation import simulate_pro_trade
from .plateau import select_plateau_features
from .progress import report_progress
from .logging_utils import log


def simulate_trades_sequential(df, probs_long, probs_short, long_win_idx, short_win_idx,
                                ct, tp, sl, spread, sym, return_detailed=False):
    """
    Simuliert Trades sequentiell (nur ein Trade gleichzeitig).

    Diese Funktion zentralisiert die Trade-Simulation für Validation und Holdout,
    um Code-Duplikation zu vermeiden (DRY-Prinzip).

    Args:
        df: DataFrame mit OHLC-Daten und _regime_ok
        probs_long: Wahrscheinlichkeiten für Long-Trades (oder None)
        probs_short: Wahrscheinlichkeiten für Short-Trades (oder None)
        long_win_idx: Index der Win-Klasse im Long-Modell
        short_win_idx: Index der Win-Klasse im Short-Modell
        ct: Confidence Threshold
        tp: Take-Profit Multiplikator
        sl: Stop-Loss Multiplikator
        spread: Asset Spread
        sym: Symbol Name
        return_detailed: Wenn True, auch volle Trade-Details zurückgeben

    Returns:
        dict mit:
            - trades: Liste von Trade-Results (1.0/-1.0)
            - trades_detailed: Liste von vollen Trade-Dicts (wenn return_detailed=True)
    """
    opn = df["O"].values
    cls = df["C"].values
    hgh = df["H"].values
    low = df["L"].values
    atr = df["_atr"].values
    regime = df["_regime_ok"].values
    timestamps = df.index.values

    trades = []
    trades_detailed = [] if return_detailed else None
    next_allowed_entry = 0  # Kein neuer Trade bevor dieser Index erreicht ist

    for i in range(len(df) - MAX_TRADE_BARS):
        # Warte bis vorheriger Trade beendet ist (nur ein Trade gleichzeitig)
        if i < next_allowed_entry:
            continue

        if not regime[i]:
            continue

        # Bestimme Trade-Richtung
        direction = None
        if probs_long is not None and probs_long[i, long_win_idx] >= ct:
            direction = 1
        elif probs_short is not None and probs_short[i, short_win_idx] >= ct:
            direction = -1

        if direction:
            trade = simulate_pro_trade(
                cls, hgh, low, atr, i, direction, tp, sl, spread,
                timestamps=timestamps, symbol=sym, opens=opn
            )
            if trade:
                trades.append(trade["result"])
                next_allowed_entry = trade["exit_idx"] + 1

                if return_detailed:
                    trade["ct"] = ct
                    trade["hour"] = df.index[i].hour
                    trades_detailed.append(trade)

    result = {"trades": trades}
    if return_detailed:
        result["trades_detailed"] = trades_detailed
    return result


def nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=5, oos_size=OOS_SIZE):
    """
    Nested Cross-Validation Split für unbiased Evaluation.

    Args:
        df: DataFrame mit allen Daten
        holdout_ratio: Anteil für finales Holdout (default 20%)
        n_inner_folds: Anzahl Inner Folds für Grid-Search
        oos_size: OOS-Größe pro Inner Fold

    Returns:
        dict mit:
            - inner_folds: [(train_df, val_df), ...] für Grid-Search
            - holdout_df: DataFrame für finale Evaluation
            - inner_df: DataFrame für Inner CV (zum Re-Training)
    """
    total_len = len(df)
    holdout_size = int(total_len * holdout_ratio)
    inner_size = total_len - holdout_size

    # Holdout: Die letzten 20% - werden NIE während Grid-Search gesehen
    inner_df = df.iloc[:inner_size].copy()
    holdout_df = df.iloc[inner_size:].copy()

    # Inner Folds: Walk-Forward auf den ersten 80%
    inner_folds = []
    val_size = min(oos_size, inner_size // (n_inner_folds + 2))

    for i in range(n_inner_folds):
        # Validation Bereich (rollend)
        val_end = inner_size - (i * val_size)
        val_start = val_end - val_size

        # Training: Alles vor Validation
        train_end = val_start

        if train_end < val_size * 2:
            continue

        train_df = inner_df.iloc[:train_end].copy()
        val_df = inner_df.iloc[val_start:val_end].copy()

        inner_folds.append((train_df, val_df))

    return {
        "inner_folds": list(reversed(inner_folds)),
        "holdout_df": holdout_df,
        "inner_df": inner_df,
    }


def compute_targets(df, tp, sl, spread, sym):
    """
    Berechnet Long/Short Targets für einen DataFrame.

    Returns:
        (targets_long, targets_short, has_long, has_short)
    """
    targets_long = np.zeros(len(df))
    targets_short = np.zeros(len(df))

    opn_v = df["O"].values
    cls_v = df["C"].values
    hgh_v = df["H"].values
    low_v = df["L"].values
    atr_v = df["_atr"].values
    timestamps = df.index.values

    sim_count = len(df) - MAX_TRADE_BARS
    for i in range(sim_count):
        trade_long = simulate_pro_trade(
            cls_v, hgh_v, low_v, atr_v, i, 1, tp, sl, spread,
            timestamps=timestamps, symbol=sym, opens=opn_v
        )
        trade_short = simulate_pro_trade(
            cls_v, hgh_v, low_v, atr_v, i, -1, tp, sl, spread,
            timestamps=timestamps, symbol=sym, opens=opn_v
        )
        if trade_long and trade_long["result"] == 1.0:
            targets_long[i] = 1
        if trade_short and trade_short["result"] == 1.0:
            targets_short[i] = 1

    min_per_direction = MIN_TRADES // 2
    n_long = np.count_nonzero(targets_long)
    n_short = np.count_nonzero(targets_short)
    has_long = n_long >= min_per_direction
    has_short = n_short >= min_per_direction

    return targets_long, targets_short, has_long, has_short


def select_features_from_fold(train_df, targets, group_features, direction_name):
    """
    Wählt Features basierend auf einem Training-Fold.

    Args:
        train_df: Training DataFrame
        targets: Target-Array
        group_features: Liste der zu testenden Features
        direction_name: "long" oder "short" (für Logging)

    Returns:
        (selected_features, importances_dict) oder (None, {})
    """
    if np.count_nonzero(targets) < MIN_TRADES // 2:
        return None, {}

    model = XGBClassifier(
        n_estimators=100, max_depth=5, n_jobs=1,
        random_state=42, verbosity=0,
    )
    model.fit(train_df[group_features], targets)
    importances = pd.Series(model.feature_importances_, index=group_features)

    plateau_features = select_plateau_features(
        importances.to_dict(), group_features,
        top_n=5, min_importance=RELEVANCE_THRESHOLD
    )

    if len(plateau_features) >= 2:
        return plateau_features, importances.to_dict()

    return None, importances.to_dict()


def train_model(train_df, targets, features):
    """
    Trainiert ein XGBoost-Modell.

    Returns:
        Trainiertes Modell oder None
    """
    if features is None or np.count_nonzero(targets) < MIN_TRADES // 2:
        return None

    model = XGBClassifier(
        n_estimators=100, max_depth=5, n_jobs=1,
        random_state=42, verbosity=0,
    )
    model.fit(train_df[features], targets)
    return model


def evaluate_on_validation(val_df, mod_long, mod_short, features_long, features_short,
                           tp, sl, spread, sym, grid_ct):
    """
    Evaluiert Modelle auf Validation-Set und findet besten CT.

    Returns:
        (best_ct, best_pnl, trades_by_ct)
    """
    # Wahrscheinlichkeiten berechnen
    probs_long, long_win_idx = _get_probs(mod_long, val_df, features_long)
    probs_short, short_win_idx = _get_probs(mod_short, val_df, features_short)

    best_ct = None
    best_pnl = float("-inf")
    trades_by_ct = {}

    for ct in grid_ct:
        result = simulate_trades_sequential(
            val_df, probs_long, probs_short, long_win_idx, short_win_idx,
            ct, tp, sl, spread, sym, return_detailed=False
        )
        ct_trades = result["trades"]
        trades_by_ct[ct] = ct_trades

        if len(ct_trades) >= 10:
            ct_pnl = sum(ct_trades)
            if ct_pnl > best_pnl:
                best_pnl = ct_pnl
                best_ct = ct

    return best_ct, best_pnl, trades_by_ct


def _get_probs(model, df, features):
    """Berechnet Wahrscheinlichkeiten für ein Modell. Hilfsfunktion für DRY."""
    if not features or model is None:
        return None, None
    probs = model.predict_proba(df[features])
    if 1 in model.classes_:
        win_idx = np.where(model.classes_ == 1)[0][0]
        return probs, win_idx
    return None, None


def run_inner_cv(inner_folds, group_features, tp, sl, spread, sym, grid_ct, global_grid_pos, total_grid_combos):
    """
    Führt Inner Cross-Validation für eine Grid-Kombination durch.

    Returns:
        dict mit:
            - success: bool
            - avg_val_pnl: float
            - best_ct: float
            - selected_features_long: list
            - selected_features_short: list
            - selected_features: list (kombiniert)
    """
    inner_val_pnls = []
    selected_features_long = None
    selected_features_short = None
    best_ct_votes = {}

    for fold_idx, (train_df, val_df) in enumerate(inner_folds):
        report_progress(sym, fold_idx + 1, len(inner_folds), "inner_cv", global_grid_pos, total_grid_combos)

        # Targets berechnen
        targets_long, targets_short, has_long, has_short = compute_targets(
            train_df, tp, sl, spread, sym
        )

        if not has_long and not has_short:
            continue

        # Feature-Auswahl nur auf erstem Fold
        if fold_idx == 0:
            if has_long:
                selected_features_long, _ = select_features_from_fold(
                    train_df, targets_long, group_features, "long"
                )
            if has_short:
                selected_features_short, _ = select_features_from_fold(
                    train_df, targets_short, group_features, "short"
                )

        if not selected_features_long and not selected_features_short:
            continue

        # Modelle trainieren
        mod_long = train_model(train_df, targets_long, selected_features_long) if has_long else None
        mod_short = train_model(train_df, targets_short, selected_features_short) if has_short else None

        # CT-Optimierung auf Validation
        best_fold_ct, best_fold_pnl, _ = evaluate_on_validation(
            val_df, mod_long, mod_short,
            selected_features_long, selected_features_short,
            tp, sl, spread, sym, grid_ct
        )

        if best_fold_ct:
            inner_val_pnls.append(best_fold_pnl)
            best_ct_votes[best_fold_ct] = best_ct_votes.get(best_fold_ct, 0) + 1

    # Ergebnis zusammenstellen
    if not inner_val_pnls or not best_ct_votes:
        return {"success": False}

    # Kombiniere Features
    selected_features = []
    if selected_features_long:
        selected_features.extend(selected_features_long)
    if selected_features_short:
        selected_features.extend([f for f in selected_features_short if f not in selected_features])

    if not selected_features:
        return {"success": False}

    # Fold-Stabilität: Wie viele Folds waren profitabel?
    profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)
    fold_stability = profitable_folds / len(inner_val_pnls) if inner_val_pnls else 0

    return {
        "success": True,
        "avg_val_pnl": np.mean(inner_val_pnls),
        "best_ct": max(best_ct_votes.keys(), key=lambda x: best_ct_votes[x]),
        "selected_features_long": selected_features_long,
        "selected_features_short": selected_features_short,
        "selected_features": selected_features,
        "fold_stability": fold_stability,
        "fold_pnls": inner_val_pnls,  # Für Debugging
    }


def evaluate_on_holdout(holdout_df, inner_df, candidate, spread, sym):
    """
    Finale Evaluation auf dem Holdout-Set.
    Trainiert Modell auf GESAMTEM Inner-Set und testet auf Holdout.

    Args:
        holdout_df: Holdout DataFrame (nie vorher gesehen!)
        inner_df: Gesamtes Inner DataFrame für finales Training
        candidate: Dict mit params, features etc.
        spread: Asset Spread
        sym: Symbol Name

    Returns:
        dict mit:
            - trades: Liste von Trade-Results
            - trades_detailed: Liste von vollen Trade-Dicts
            - pnl: Gesamt-PnL
            - win_rate: Win-Rate
            - n_trades: Anzahl Trades
    """
    tp, sl, ct = candidate["params"]
    features_long = candidate.get("selected_features_long")
    features_short = candidate.get("selected_features_short")

    # Targets auf Inner-Set berechnen
    targets_long, targets_short, has_long, has_short = compute_targets(
        inner_df, tp, sl, spread, sym
    )

    # Finale Modelle trainieren (auf gesamtem Inner-Set)
    mod_long = train_model(inner_df, targets_long, features_long) if has_long and features_long else None
    mod_short = train_model(inner_df, targets_short, features_short) if has_short and features_short else None

    if not mod_long and not mod_short:
        return {"trades": [], "trades_detailed": [], "pnl": 0, "win_rate": 0, "n_trades": 0}

    # Wahrscheinlichkeiten berechnen
    probs_long, long_win_idx = _get_probs(mod_long, holdout_df, features_long)
    probs_short, short_win_idx = _get_probs(mod_short, holdout_df, features_short)

    # Trades simulieren (nur ein Trade gleichzeitig)
    result = simulate_trades_sequential(
        holdout_df, probs_long, probs_short, long_win_idx, short_win_idx,
        ct, tp, sl, spread, sym, return_detailed=True
    )

    trades = result["trades"]
    trades_detailed = result["trades_detailed"]
    pnl = sum(trades) if trades else 0
    win_rate = trades.count(1.0) / len(trades) if trades else 0

    return {
        "trades": trades,
        "trades_detailed": trades_detailed,
        "pnl": pnl,
        "win_rate": win_rate,
        "n_trades": len(trades),
    }
