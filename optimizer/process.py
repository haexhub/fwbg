"""
Walk-Forward Optimierung und Symbol-Verarbeitung
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import (
    DATA_PATH, MAX_TRADE_BARS, MIN_TRADES, WALK_FORWARD_FOLDS, OOS_SIZE,
    RELEVANCE_THRESHOLD, FEATURE_STABILITY_MIN, CLASS_GRIDS,
    MACRO_INDICATORS, LOOKBACKS_HOURS, LOOKBACKS_DAYS,
    get_asset_config
)
from .data_loader import load_data_aligned
from .indicators import compute_indicator_pool, get_feature_columns, compute_regime_filter
from .simulation import (
    simulate_pro_trade, calculate_sharpe_ratio, calculate_calmar_ratio,
    check_feature_stability
)
from .plateau import (
    calculate_param_plateau_score, select_plateau_features,
    select_best_plateau_candidate
)


def walk_forward_split(df, n_folds=WALK_FORWARD_FOLDS, oos_size=OOS_SIZE):
    """Generiert Walk-Forward Fenster: [(train_df, test_df), ...]"""
    total_len = len(df)
    min_train = total_len - (n_folds * oos_size)

    if min_train < oos_size * 2:
        # Nicht genug Daten für Walk-Forward, fallback auf Single Split
        split_idx = total_len - oos_size
        return [(df.iloc[:split_idx], df.iloc[split_idx:])]

    folds = []
    for i in range(n_folds):
        test_end = total_len - (i * oos_size)
        test_start = test_end - oos_size
        train_end = test_start
        folds.append((df.iloc[:train_end].copy(), df.iloc[test_start:test_end].copy()))

    return list(reversed(folds))  # Chronologisch sortieren


def process_symbol(csv_path):
    """Verarbeitet ein einzelnes Symbol mit Walk-Forward Optimierung."""
    sym = os.path.basename(csv_path).split("_")[0]
    if sym in ["VIX", "DXY"]:
        return None
    try:
        df = load_data_aligned(csv_path)
        if df is None:
            return None

        # === ALLE MAKRO-INDIKATOREN LADEN ===
        df["_date"] = df.index.date

        for filename, prefix in MACRO_INDICATORS.items():
            macro_path = f"{DATA_PATH}/{filename}.csv"
            if os.path.exists(macro_path):
                try:
                    macro_df = pd.read_csv(macro_path, parse_dates=["DATE"], index_col="DATE")
                    macro_lookup = macro_df["Close"].to_dict()

                    col_name = f"macro_{prefix}"
                    df[col_name] = df["_date"].map(lambda d: macro_lookup.get(pd.Timestamp(d), np.nan))
                    df[col_name] = df[col_name].ffill()

                    # Stunden-basierte Lookbacks
                    for lb_h in LOOKBACKS_HOURS:
                        df[f"{col_name}_chg_{lb_h}h"] = df[col_name].pct_change(lb_h) * 100

                    # Tages-basierte Lookbacks
                    for lb_d in LOOKBACKS_DAYS:
                        df[f"{col_name}_chg_{lb_d}d"] = df[col_name].pct_change(24 * lb_d) * 100

                except Exception:
                    pass

        df = df.drop(columns=["_date"], errors="ignore")

        # === ABGELEITETE FEATURES (Spreads & Ratios) ===
        if "macro_tnx" in df.columns and "macro_irx" in df.columns:
            df["macro_yield_curve_10y_3m"] = df["macro_tnx"] - df["macro_irx"]
        if "macro_tnx" in df.columns and "macro_fvx" in df.columns:
            df["macro_yield_curve_10y_5y"] = df["macro_tnx"] - df["macro_fvx"]

        if "macro_vix" in df.columns and "macro_vvix" in df.columns:
            df["macro_vix_vvix_ratio"] = df["macro_vix"] / (df["macro_vvix"] + 1e-10)

        if "macro_spx" in df.columns and "macro_tlt" in df.columns:
            df["macro_risk_ratio_spx_tlt"] = df["macro_spx"] / (df["macro_tlt"] + 1e-10)
        if "macro_hyg" in df.columns and "macro_lqd" in df.columns:
            df["macro_credit_spread_proxy"] = df["macro_hyg"] / (df["macro_lqd"] + 1e-10)

        if "macro_russell" in df.columns and "macro_spx" in df.columns:
            df["macro_smallcap_ratio"] = df["macro_russell"] / (df["macro_spx"] + 1e-10)

        if "macro_xlk" in df.columns and "macro_xlu" in df.columns:
            df["macro_tech_defensive_ratio"] = df["macro_xlk"] / (df["macro_xlu"] + 1e-10)

        # Zinsdaten laden
        for rate_name, rate_file in [("fed", "FED_RATE.csv"), ("ecb", "ECB_RATE.csv")]:
            rate_path = f"{DATA_PATH}/{rate_file}"
            if os.path.exists(rate_path):
                try:
                    rate_df = pd.read_csv(rate_path, parse_dates=["Date"], index_col="Date")
                    rate_series = rate_df["Rate"].reindex(df.index, method="ffill")
                    df[f"macro_{rate_name}_rate"] = rate_series
                    for lb in [30, 90, 180]:
                        df[f"macro_{rate_name}_chg_{lb}d"] = df[f"macro_{rate_name}_rate"].diff(24 * lb)
                except Exception:
                    pass

        if "macro_fed_rate" in df.columns and "macro_ecb_rate" in df.columns:
            df["macro_rate_diff_usd_eur"] = df["macro_fed_rate"] - df["macro_ecb_rate"]

        df = compute_indicator_pool(df).dropna()

        # Regime-Filter berechnen
        has_vix = "sent_vix" in df.columns
        df["_regime_ok"] = compute_regime_filter(df, has_vix)

        full_pool = get_feature_columns(df)
        a_class, p_val, spread, currencies = get_asset_config(sym)

        grid = CLASS_GRIDS.get(a_class, CLASS_GRIDS["FOREX"])
        candidates = []

        for tp in grid["tp"]:
            for sl in grid["sl"]:
                # Walk-Forward Validierung
                folds = walk_forward_split(df)
                all_trades = []
                selected_features_long = None
                selected_features_short = None
                fold_importances_long = []
                fold_importances_short = []

                for fold_idx, (train_df, test_df) in enumerate(folds):
                    # === SEPARATE TARGETS FÜR LONG UND SHORT ===
                    train_targs_long = np.zeros(len(train_df))
                    train_targs_short = np.zeros(len(train_df))

                    cls_v = train_df["C"].values
                    hgh_v = train_df["H"].values
                    low_v = train_df["L"].values
                    atr_v = train_df["_atr"].values

                    for i in range(len(train_df) - MAX_TRADE_BARS):
                        res_long, _ = simulate_pro_trade(
                            cls_v, hgh_v, low_v, atr_v, i, 1, tp, sl, spread
                        )
                        res_short, _ = simulate_pro_trade(
                            cls_v, hgh_v, low_v, atr_v, i, -1, tp, sl, spread
                        )
                        if res_long == 1.0:
                            train_targs_long[i] = 1
                        if res_short == 1.0:
                            train_targs_short[i] = 1

                    min_per_direction = MIN_TRADES // 2
                    has_long = np.count_nonzero(train_targs_long) >= min_per_direction
                    has_short = np.count_nonzero(train_targs_short) >= min_per_direction

                    if not has_long and not has_short:
                        continue

                    # === LONG MODELL ===
                    mod_long = None
                    if has_long:
                        mod_long = XGBClassifier(
                            n_estimators=100, max_depth=5, n_jobs=1,
                            random_state=42, verbosity=0,
                        )
                        mod_long.fit(train_df[full_pool], train_targs_long)
                        imps_long = pd.Series(mod_long.feature_importances_, index=full_pool)
                        fold_importances_long.append(imps_long.to_dict())

                        if fold_idx == 0:
                            # Plateau-basierte Feature-Auswahl statt nur Top-Importance
                            # Bevorzugt Features, deren Nachbarn ähnliche Importance haben
                            plateau_features_long = select_plateau_features(
                                imps_long.to_dict(),
                                full_pool,
                                top_n=5,
                                min_importance=RELEVANCE_THRESHOLD
                            )
                            if len(plateau_features_long) >= 2:
                                selected_features_long = plateau_features_long

                    # === SHORT MODELL ===
                    mod_short = None
                    if has_short:
                        mod_short = XGBClassifier(
                            n_estimators=100, max_depth=5, n_jobs=1,
                            random_state=42, verbosity=0,
                        )
                        mod_short.fit(train_df[full_pool], train_targs_short)
                        imps_short = pd.Series(mod_short.feature_importances_, index=full_pool)
                        fold_importances_short.append(imps_short.to_dict())

                        if fold_idx == 0:
                            # Plateau-basierte Feature-Auswahl
                            plateau_features_short = select_plateau_features(
                                imps_short.to_dict(),
                                full_pool,
                                top_n=5,
                                min_importance=RELEVANCE_THRESHOLD
                            )
                            if len(plateau_features_short) >= 2:
                                selected_features_short = plateau_features_short

                    # Feature Stability Check
                    if fold_idx == len(folds) - 1:
                        if selected_features_long and len(fold_importances_long) >= FEATURE_STABILITY_MIN:
                            stable = check_feature_stability(fold_importances_long)
                            selected_features_long = [f for f in selected_features_long if f in stable]
                            if len(selected_features_long) < 2:
                                selected_features_long = None
                        if selected_features_short and len(fold_importances_short) >= FEATURE_STABILITY_MIN:
                            stable = check_feature_stability(fold_importances_short)
                            selected_features_short = [f for f in selected_features_short if f in stable]
                            if len(selected_features_short) < 2:
                                selected_features_short = None

                    # Re-Training mit selected features
                    if selected_features_long and mod_long:
                        mod_long.fit(train_df[selected_features_long], train_targs_long)
                    if selected_features_short and mod_short:
                        mod_short.fit(train_df[selected_features_short], train_targs_short)

                    # === OOS PREDICTION ===
                    test_cls = test_df["C"].values
                    test_hgh = test_df["H"].values
                    test_low = test_df["L"].values
                    test_atr = test_df["_atr"].values
                    test_regime = test_df["_regime_ok"].values

                    probs_long = None
                    probs_short = None
                    long_win_idx = None
                    short_win_idx = None

                    if selected_features_long and mod_long:
                        probs_long = mod_long.predict_proba(test_df[selected_features_long])
                        if 1 in mod_long.classes_:
                            long_win_idx = np.where(mod_long.classes_ == 1)[0][0]
                        else:
                            probs_long = None

                    if selected_features_short and mod_short:
                        probs_short = mod_short.predict_proba(test_df[selected_features_short])
                        if 1 in mod_short.classes_:
                            short_win_idx = np.where(mod_short.classes_ == 1)[0][0]
                        else:
                            probs_short = None

                    for ct in grid["ct"]:
                        for i in range(len(test_df) - MAX_TRADE_BARS):
                            if not test_regime[i]:
                                continue

                            res = 0
                            hour = test_df.index[i].hour
                            direction = None

                            if probs_long is not None and probs_long[i, long_win_idx] >= ct:
                                direction = 1
                            elif probs_short is not None and probs_short[i, short_win_idx] >= ct:
                                direction = -1

                            if direction:
                                res, _ = simulate_pro_trade(
                                    test_cls, test_hgh, test_low, test_atr,
                                    i, direction, tp, sl, spread
                                )
                            if res != 0:
                                all_trades.append({"res": res, "ct": ct, "hour": hour, "dir": direction})

                # Kombiniere Features
                selected_features = []
                if selected_features_long:
                    selected_features.extend(selected_features_long)
                if selected_features_short:
                    selected_features.extend([f for f in selected_features_short if f not in selected_features])

                if not selected_features:
                    continue

                # Aggregiere pro Confidence-Threshold
                for ct in grid["ct"]:
                    trades_ct = [t for t in all_trades if t["ct"] == ct]
                    tr = [t["res"] for t in trades_ct]
                    if len(tr) >= MIN_TRADES:
                        rrr = tp / sl
                        preliminary_kelly = max(0, min(0.05, (
                            (tr.count(1.0) / len(tr) * rrr - (1 - tr.count(1.0) / len(tr))) / rrr
                        ) / 4)) if len(tr) > 0 else 0.01

                        trade_returns = [preliminary_kelly * rrr if r > 0 else -preliminary_kelly for r in tr]
                        sharpe = calculate_sharpe_ratio(trade_returns)
                        calmar = calculate_calmar_ratio(tr, preliminary_kelly, rrr)

                        hour_pnl = {}
                        for t in trades_ct:
                            h = t["hour"]
                            hour_pnl[h] = hour_pnl.get(h, 0) + t["res"]
                        good_hours = [h for h, pnl in hour_pnl.items() if pnl > 0]

                        candidates.append({
                            "pnl": sum(tr),
                            "tr": tr,
                            "params": (tp, sl, ct),
                            "feats": selected_features,
                            "rrr": rrr,
                            "sharpe": sharpe,
                            "calmar": calmar,
                            "good_hours": good_hours if good_hours else list(range(24)),
                        })

        if not candidates:
            return None

        # Sortiere nach kombinierter Metrik
        for c in candidates:
            c["score"] = c["pnl"] * (1 + max(0, c["sharpe"]) / 10)

        # === PLATEAU-BASIERTE AUSWAHL ===
        # Statt einfach den höchsten Score zu nehmen, bevorzugen wir
        # Konfigurationen, deren Nachbarn ähnlich gut performen (Plateau)
        candidates = calculate_param_plateau_score(
            candidates,
            grid["tp"],
            grid["sl"],
            grid["ct"]
        )

        # Wähle besten Plateau-Kandidaten
        b = select_best_plateau_candidate(
            candidates,
            grid["tp"],
            grid["sl"],
            grid["ct"],
            min_neighbors=2
        )

        if not b:
            return None

        # Ensemble: Top-3 nach Plateau-Score
        candidates.sort(key=lambda x: x.get("plateau_score", x["score"]), reverse=True)
        top_n = min(3, len(candidates))
        ensemble_configs = candidates[:top_n]

        total_score = sum(c.get("plateau_score", c["score"]) for c in ensemble_configs)
        if total_score <= 0:
            return None
        wr = b["tr"].count(1.0) / len(b["tr"]) if b["tr"] else 0

        # 1/4 Kelly
        p = wr
        q = 1 - p
        rrr = b["rrr"]
        full_kelly = (p * rrr - q) / rrr if rrr > 0 else 0
        fk = max(0, min(0.05, full_kelly / 4))

        if fk <= 0:
            return None

        # Ensemble-Gewichte (basierend auf Plateau-Score)
        ensemble_weights = []
        if top_n > 1:
            for c in ensemble_configs[1:]:
                c_wr = c["tr"].count(1.0) / len(c["tr"]) if c["tr"] else 0
                c_kelly = max(0, min(0.05, ((c_wr * c["rrr"] - (1 - c_wr)) / c["rrr"]) / 4))
                if c_kelly > 0:
                    c_plateau_score = c.get("plateau_score", c["score"])
                    ensemble_weights.append({
                        "tp_mult": c["params"][0],
                        "sl_mult": c["params"][1],
                        "conf_thresh": c["params"][2],
                        "weight": c_plateau_score / total_score,
                        "stability": c.get("stability_score", 0),
                    })

        return {
            "symbol": sym,
            "pnl": b["pnl"],
            "config": {
                "kelly_risk": fk,
                "point_value": p_val,
                "spread": spread,
                "tp_mult": b["params"][0],
                "sl_mult": b["params"][1],
                "conf_thresh": b["params"][2],
                "features": b["feats"],
                "good_hours": b.get("good_hours", list(range(24))),
                "ensemble": ensemble_weights if ensemble_weights else None,
                "dd_scaling": {"10": 0.5, "20": 0.25},
            },
            "tr_trace": b["tr"],
            "rrr": b["rrr"],
            "win_rate": wr,
            "sharpe": b["sharpe"],
            "calmar": b["calmar"],
            "currencies": currencies,
        }
    except Exception as e:
        print(f"Fehler {sym}: {e}")
        import traceback
        traceback.print_exc()
        return None
