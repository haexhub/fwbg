"""
Fold-Worker für parallelisierte Walk-Forward Validierung.
"""
import os
import time
import numpy as np
from xgboost import XGBClassifier

from .config import MAX_TRADE_BARS, MIN_TRADES
from .simulation import simulate_pro_trade, generate_combined_labels
from .progress import report_progress
from .logging_utils import log


def process_fold(
    fold_idx,
    fold_data,
    shared_config,
    selected_features_long,
    selected_features_short,
):
    """
    Verarbeitet einen einzelnen Fold (für Parallelisierung).

    Args:
        fold_idx: Index des Folds (0-basiert)
        fold_data: Tuple (train_df, val_df, test_df)
        shared_config: Dict mit {
            "sym": str,
            "tp": int,
            "sl": int,
            "spread": float,
            "group_features": list,
            "grid_ct": list,
            "global_grid_pos": int,
            "total_grid_combos": int,
        }
        selected_features_long: Liste der Features für Long-Modell (bereits bestimmt)
        selected_features_short: Liste der Features für Short-Modell (bereits bestimmt)

    Returns:
        dict mit:
            - fold_idx: int
            - oos_trades: List[dict]
            - importances_long: dict
            - importances_short: dict
            - best_ct: float
            - fold_pnl: float
            - fold_wr: float
            - skipped: bool
    """
    train_df, val_df, test_df = fold_data
    sym = shared_config["sym"]
    tp = shared_config["tp"]
    sl = shared_config["sl"]
    spread = shared_config["spread"]
    group_features = shared_config["group_features"]
    grid_ct = shared_config["grid_ct"]

    t_fold = time.time()
    log(2, f"  Fold {fold_idx+1} (parallel)", sym)

    # Progress-Report für UI
    report_progress(
        sym, fold_idx + 1, shared_config.get("total_folds", 8),
        "train", shared_config.get("global_grid_pos", 0),
        shared_config.get("total_grid_combos", 1)
    )

    # === TARGETS BERECHNEN (OHNE LOOK-AHEAD BIAS!) ===
    # Wir verwenden jetzt Labels basierend auf VERGANGENEN Trends,
    # nicht auf zukünftigen Trade-Ergebnissen.
    label_method = os.environ.get("OPTIMIZER_LABEL_METHOD", "trend")
    label_lookback = int(os.environ.get("OPTIMIZER_LABEL_LOOKBACK", "24"))
    label_threshold = float(os.environ.get("OPTIMIZER_LABEL_THRESHOLD", "0.3"))

    train_targs_long, train_targs_short = generate_combined_labels(
        train_df,
        method=label_method,
        lookback=label_lookback,
        threshold_pct=label_threshold
    )

    min_per_direction = MIN_TRADES // 2
    n_long = np.count_nonzero(train_targs_long)
    n_short = np.count_nonzero(train_targs_short)
    has_long = n_long >= min_per_direction
    has_short = n_short >= min_per_direction

    if not has_long and not has_short:
        log(3, f"    SKIP Fold {fold_idx} - zu wenig Targets", sym)
        return {"fold_idx": fold_idx, "skipped": True}

    # === LONG MODELL ===
    mod_long = None
    importances_long = {}
    if has_long and selected_features_long:
        mod_long = XGBClassifier(
            n_estimators=100, max_depth=5, n_jobs=1,
            random_state=42, verbosity=0,
        )
        mod_long.fit(train_df[selected_features_long], train_targs_long)
        importances_long = dict(zip(selected_features_long, mod_long.feature_importances_))

    # === SHORT MODELL ===
    mod_short = None
    importances_short = {}
    if has_short and selected_features_short:
        mod_short = XGBClassifier(
            n_estimators=100, max_depth=5, n_jobs=1,
            random_state=42, verbosity=0,
        )
        mod_short.fit(train_df[selected_features_short], train_targs_short)
        importances_short = dict(zip(selected_features_short, mod_short.feature_importances_))

    # === VALIDATION: CT-Optimierung ===
    report_progress(
        sym, fold_idx + 1, shared_config.get("total_folds", 8),
        "validate", shared_config.get("global_grid_pos", 0),
        shared_config.get("total_grid_combos", 1)
    )

    val_opn = val_df["O"].values
    val_cls = val_df["C"].values
    val_hgh = val_df["H"].values
    val_low = val_df["L"].values
    val_atr = val_df["_atr"].values
    val_regime = val_df["_regime_ok"].values
    val_timestamps = val_df.index.values

    probs_long_val = None
    probs_short_val = None
    long_win_idx = None
    short_win_idx = None

    if selected_features_long and mod_long:
        probs_long_val = mod_long.predict_proba(val_df[selected_features_long])
        if 1 in mod_long.classes_:
            long_win_idx = np.where(mod_long.classes_ == 1)[0][0]
        else:
            probs_long_val = None

    if selected_features_short and mod_short:
        probs_short_val = mod_short.predict_proba(val_df[selected_features_short])
        if 1 in mod_short.classes_:
            short_win_idx = np.where(mod_short.classes_ == 1)[0][0]
        else:
            probs_short_val = None

    # Teste alle CTs auf VALIDATION Set
    val_trades_by_ct = {ct: [] for ct in grid_ct}
    for ct in grid_ct:
        for i in range(len(val_df) - MAX_TRADE_BARS):
            if not val_regime[i]:
                continue

            direction = None
            if probs_long_val is not None and probs_long_val[i, long_win_idx] >= ct:
                direction = 1
            elif probs_short_val is not None and probs_short_val[i, short_win_idx] >= ct:
                direction = -1

            if direction:
                trade = simulate_pro_trade(
                    val_cls, val_hgh, val_low, val_atr,
                    i, direction, tp, sl, spread,
                    timestamps=val_timestamps, symbol=sym,
                    opens=val_opn
                )
                if trade:
                    val_trades_by_ct[ct].append(trade["result"])

    # Wähle besten CT
    best_ct = None
    best_val_pnl = float("-inf")
    for ct, trades in val_trades_by_ct.items():
        if len(trades) >= 10:
            pnl = sum(trades)
            if pnl > best_val_pnl:
                best_val_pnl = pnl
                best_ct = ct

    if best_ct is None:
        log(3, f"    SKIP Fold {fold_idx} - kein valider CT", sym)
        return {"fold_idx": fold_idx, "skipped": True}

    # === OOS PREDICTION ===
    report_progress(
        sym, fold_idx + 1, shared_config.get("total_folds", 8),
        "oos", shared_config.get("global_grid_pos", 0),
        shared_config.get("total_grid_combos", 1)
    )

    test_opn = test_df["O"].values
    test_cls = test_df["C"].values
    test_hgh = test_df["H"].values
    test_low = test_df["L"].values
    test_atr = test_df["_atr"].values
    test_regime = test_df["_regime_ok"].values
    test_timestamps = test_df.index.values

    probs_long_oos = None
    probs_short_oos = None

    if selected_features_long and mod_long:
        probs_long_oos = mod_long.predict_proba(test_df[selected_features_long])

    if selected_features_short and mod_short:
        probs_short_oos = mod_short.predict_proba(test_df[selected_features_short])

    fold_oos_trades = []
    for i in range(len(test_df) - MAX_TRADE_BARS):
        if not test_regime[i]:
            continue

        hour = test_df.index[i].hour
        direction = None

        if probs_long_oos is not None and probs_long_oos[i, long_win_idx] >= best_ct:
            direction = 1
        elif probs_short_oos is not None and probs_short_oos[i, short_win_idx] >= best_ct:
            direction = -1

        if direction:
            trade = simulate_pro_trade(
                test_cls, test_hgh, test_low, test_atr,
                i, direction, tp, sl, spread,
                timestamps=test_timestamps, symbol=sym,
                opens=test_opn
            )
            if trade:
                # Füge zusätzliche Kontext-Infos hinzu
                trade["ct"] = best_ct
                trade["hour"] = hour
                fold_oos_trades.append(trade)

    # Ergebnis
    fold_pnl = sum(t["result"] for t in fold_oos_trades) if fold_oos_trades else 0
    fold_wr = sum(1 for t in fold_oos_trades if t["result"] > 0) / len(fold_oos_trades) if fold_oos_trades else 0

    log(2, f"    Fold {fold_idx+1} done ({time.time()-t_fold:.1f}s, {len(fold_oos_trades)} trades)", sym)

    return {
        "fold_idx": fold_idx,
        "skipped": False,
        "oos_trades": fold_oos_trades,
        "importances_long": importances_long,
        "importances_short": importances_short,
        "best_ct": best_ct,
        "fold_pnl": fold_pnl,
        "fold_wr": fold_wr,
    }
