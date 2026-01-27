"""
Walk-Forward Optimierung und Symbol-Verarbeitung
"""
import os
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import (
    DATA_PATH, MAX_TRADE_BARS, MIN_TRADES, WALK_FORWARD_FOLDS, OOS_SIZE,
    RELEVANCE_THRESHOLD, FEATURE_STABILITY_MIN, CLASS_GRIDS,
    MACRO_INDICATORS, LOOKBACKS_HOURS, LOOKBACKS_DAYS,
    DEFAULT_FEATURE_GROUPS,
    get_asset_config
)
from .data_loader import load_data_aligned, load_macro_csv
from .indicators import compute_indicator_pool, get_feature_columns, compute_regime_filter, filter_features_by_group
from .simulation import (
    simulate_pro_trade, calculate_sharpe_ratio, calculate_calmar_ratio,
    check_feature_stability, monte_carlo_permutation_test, monte_carlo_equity_simulation,
    adjust_kelly_for_target_dd, find_optimal_circuit_breaker, calculate_equity_smoothness
)
from .plateau import (
    calculate_param_plateau_score, select_plateau_features,
    select_best_plateau_candidate
)
from .progress import report_progress, report_done
from .logging_utils import log
from .fold_worker import process_fold


def walk_forward_split(df, n_folds=WALK_FORWARD_FOLDS, oos_size=OOS_SIZE):
    """
    Generiert Walk-Forward Fenster: [(train_df, val_df, test_df), ...]

    Jeder Fold hat (60/20/20 Split relativ zum verfügbaren Bereich):
    - train_df: Für Modell-Training (~60%)
    - val_df: Für Hyperparameter-Optimierung (~20%, z.B. Confidence Threshold)
    - test_df: Für finale Out-of-Sample Evaluation (~20%, KEINE Parameterauswahl!)

    Args:
        df: DataFrame mit allen Daten
        n_folds: Anzahl der Walk-Forward Folds
        oos_size: Größe des OOS-Fensters pro Fold (definiert die 20% OOS)
    """
    total_len = len(df)
    min_train = total_len - (n_folds * oos_size)

    # Validation-Größe = gleich wie OOS für 60/20/20 Split
    val_size = oos_size

    if min_train < oos_size * 3:
        # Nicht genug Daten für Walk-Forward mit 60/20/20
        # Fallback: 60/20/20 auf gesamten Datensatz
        test_size = int(total_len * 0.2)
        val_size = int(total_len * 0.2)
        train_size = total_len - test_size - val_size

        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        return [(train_df, val_df, test_df)]

    folds = []
    for i in range(n_folds):
        # OOS (Test) Bereich
        test_end = total_len - (i * oos_size)
        test_start = test_end - oos_size

        # Validation Bereich (gleich groß wie OOS, direkt davor)
        val_end = test_start
        val_start = val_end - val_size

        # Training Bereich (alles davor)
        train_end = val_start

        if train_end < oos_size:
            # Nicht genug Daten für diesen Fold
            continue

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        folds.append((train_df, val_df, test_df))

    return list(reversed(folds))  # Chronologisch sortieren


def process_symbol(csv_path):
    """Verarbeitet ein einzelnes Symbol mit Walk-Forward Optimierung."""
    sym = os.path.basename(csv_path).split("_")[0]
    t_start = time.time()

    if sym in ["VIX", "DXY"]:
        log(2, "Übersprungen (Makro-Asset)", sym)
        return None

    log(1, "START", sym)

    try:
        t0 = time.time()
        df = load_data_aligned(csv_path)
        if df is None:
            log(1, "SKIP - Keine Daten", sym)
            return None
        log(2, f"Daten geladen: {len(df)} Zeilen ({time.time()-t0:.1f}s)", sym)

        # === ALLE MAKRO-INDIKATOREN LADEN ===
        t0 = time.time()
        df["_date"] = df.index.date
        macro_count = 0

        for filename, prefix in MACRO_INDICATORS.items():
            macro_path = f"{DATA_PATH}/{filename}.csv"
            macro_df = load_macro_csv(macro_path)
            if macro_df is not None:
                try:
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

                    macro_count += 1
                except Exception:
                    pass

        df = df.drop(columns=["_date"], errors="ignore")
        log(2, f"Makro-Indikatoren: {macro_count} geladen ({time.time()-t0:.1f}s)", sym)

        # === ABGELEITETE FEATURES (Spreads & Ratios) ===
        t0 = time.time()
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

        log(3, f"Abgeleitete Features berechnet ({time.time()-t0:.1f}s)", sym)

        t0 = time.time()
        df = compute_indicator_pool(df).dropna()
        log(2, f"Indikatoren berechnet: {len(df)} Zeilen nach dropna ({time.time()-t0:.1f}s)", sym)

        if len(df) < MIN_TRADES * 2:
            log(1, f"SKIP - Zu wenig Daten nach dropna ({len(df)} < {MIN_TRADES * 2})", sym)
            return None

        # Regime-Filter berechnen
        has_vix = "sent_vix" in df.columns
        df["_regime_ok"] = compute_regime_filter(df, has_vix)

        full_pool = get_feature_columns(df)
        log(2, f"Feature-Pool: {len(full_pool)} Features", sym)

        # Entferne Features mit inf/nan (XGBoost verträgt keine inf)
        t0 = time.time()
        clean_pool = []
        excluded_inf = 0
        excluded_nan = 0
        for col in full_pool:
            if col in df.columns:
                has_inf = np.isinf(df[col]).any()
                nan_ratio = df[col].isna().sum() / len(df)
                if has_inf:
                    excluded_inf += 1
                elif nan_ratio >= 0.1:
                    excluded_nan += 1
                else:
                    clean_pool.append(col)
        full_pool = clean_pool
        log(2, f"Clean Pool: {len(full_pool)} Features (excl: {excluded_inf} inf, {excluded_nan} nan) ({time.time()-t0:.1f}s)", sym)

        if len(full_pool) < 5:
            log(1, f"SKIP - Zu wenig saubere Features ({len(full_pool)} < 5)", sym)
            return None

        a_class, p_val, spread, currencies = get_asset_config(sym)

        grid = CLASS_GRIDS.get(a_class, CLASS_GRIDS["FOREX"])
        candidates = []
        all_grid_results = []  # Alle Kombinationen tracken

        # Feature-Gruppen für Grid-Search (kann via Umgebungsvariable überschrieben werden)
        custom_groups = os.environ.get("OPTIMIZER_FEATURE_GROUPS", "")
        if custom_groups:
            feature_groups_to_test = [g.strip() for g in custom_groups.split(",") if g.strip()]
        else:
            feature_groups_to_test = DEFAULT_FEATURE_GROUPS
        total_combos = len(feature_groups_to_test) * len(grid["tp"]) * len(grid["sl"]) * len(grid["ct"])
        log(1, f"Grid-Search: {len(feature_groups_to_test)} Feature-Gruppen x {len(grid['tp'])}x{len(grid['sl'])}x{len(grid['ct'])} = {total_combos} Kombinationen", sym)

        # Äußere Schleife: Feature-Gruppen
        for fg_idx, feature_group in enumerate(feature_groups_to_test):
            # Filtere Features nach Gruppe
            group_features = filter_features_by_group(full_pool, feature_group)

            if len(group_features) < 3:
                log(2, f"  Feature-Gruppe '{feature_group}': nur {len(group_features)} Features - übersprungen", sym)
                continue

            log(1, f"Feature-Gruppe {fg_idx+1}/{len(feature_groups_to_test)}: {feature_group} ({len(group_features)} Features)", sym)

            grid_count = 0
            grid_per_fg = len(grid["tp"]) * len(grid["sl"])
            # Gesamte Grid-Kombinationen = Feature-Gruppen * TP * SL
            total_grid_combos = len(feature_groups_to_test) * grid_per_fg
            # Offset für aktuelle Feature-Gruppe
            grid_offset = fg_idx * grid_per_fg

            for tp in grid["tp"]:
                for sl in grid["sl"]:
                    grid_count += 1
                    global_grid_pos = grid_offset + grid_count

                    # Grid-Fortschritt anzeigen (Level 2 für weniger Spam)
                    log(2, f"  Grid {grid_count}/{grid_per_fg} (TP={tp}, SL={sl})", sym)

                    # Walk-Forward Validierung mit Train/Val/Test Split
                    folds = walk_forward_split(df)
                    all_oos_trades = []  # Nur echte OOS Trades (nach CT-Optimierung)
                    selected_features_long = None
                    selected_features_short = None
                    fold_importances_long = []
                    fold_importances_short = []
                    fold_performances = []  # Track per-fold performance

                    # === PHASE 1: Fold 0 zuerst ausführen (Feature-Auswahl) ===
                    train_df_0, val_df_0, test_df_0 = folds[0]
                    t_fold = time.time()
                    log(1, f"  Fold 1/{len(folds)} (Feature-Auswahl)", sym)
                    report_progress(sym, 1, len(folds), "train", global_grid_pos, total_grid_combos)

                    # Targets für Fold 0
                    train_targs_long_0 = np.zeros(len(train_df_0))
                    train_targs_short_0 = np.zeros(len(train_df_0))
                    opn_v = train_df_0["O"].values
                    cls_v = train_df_0["C"].values
                    hgh_v = train_df_0["H"].values
                    low_v = train_df_0["L"].values
                    atr_v = train_df_0["_atr"].values
                    timestamps = train_df_0.index.values

                    sim_count = len(train_df_0) - MAX_TRADE_BARS
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
                            train_targs_long_0[i] = 1
                        if trade_short and trade_short["result"] == 1.0:
                            train_targs_short_0[i] = 1

                    min_per_direction = MIN_TRADES // 2
                    n_long = np.count_nonzero(train_targs_long_0)
                    n_short = np.count_nonzero(train_targs_short_0)
                    has_long = n_long >= min_per_direction
                    has_short = n_short >= min_per_direction

                    if not has_long and not has_short:
                        log(3, f"    SKIP - Fold 0 hat zu wenig Targets", sym)
                        continue

                    # Long-Modell für Feature-Auswahl
                    if has_long:
                        mod_long_0 = XGBClassifier(
                            n_estimators=100, max_depth=5, n_jobs=1,
                            random_state=42, verbosity=0,
                        )
                        mod_long_0.fit(train_df_0[group_features], train_targs_long_0)
                        imps_long = pd.Series(mod_long_0.feature_importances_, index=group_features)
                        fold_importances_long.append(imps_long.to_dict())

                        plateau_features_long = select_plateau_features(
                            imps_long.to_dict(), group_features,
                            top_n=5, min_importance=RELEVANCE_THRESHOLD
                        )
                        if len(plateau_features_long) >= 2:
                            selected_features_long = plateau_features_long
                            log(3, f"    Long Features: {selected_features_long}", sym)

                    # Short-Modell für Feature-Auswahl
                    if has_short:
                        mod_short_0 = XGBClassifier(
                            n_estimators=100, max_depth=5, n_jobs=1,
                            random_state=42, verbosity=0,
                        )
                        mod_short_0.fit(train_df_0[group_features], train_targs_short_0)
                        imps_short = pd.Series(mod_short_0.feature_importances_, index=group_features)
                        fold_importances_short.append(imps_short.to_dict())

                        plateau_features_short = select_plateau_features(
                            imps_short.to_dict(), group_features,
                            top_n=5, min_importance=RELEVANCE_THRESHOLD
                        )
                        if len(plateau_features_short) >= 2:
                            selected_features_short = plateau_features_short
                            log(3, f"    Short Features: {selected_features_short}", sym)

                    if not selected_features_long and not selected_features_short:
                        log(3, f"    SKIP - keine Features ausgewählt", sym)
                        continue

                    # Fold 0 mit process_fold ausführen (mit den neuen Features)
                    shared_config = {
                        "sym": sym,
                        "tp": tp,
                        "sl": sl,
                        "spread": spread,
                        "group_features": group_features,
                        "grid_ct": grid["ct"],
                        "global_grid_pos": global_grid_pos,
                        "total_grid_combos": total_grid_combos,
                        "total_folds": len(folds),
                    }

                    fold_0_result = process_fold(
                        0, (train_df_0, val_df_0, test_df_0),
                        shared_config, selected_features_long, selected_features_short
                    )

                    if not fold_0_result.get("skipped", True):
                        all_oos_trades.extend(fold_0_result.get("oos_trades", []))
                        if fold_0_result.get("importances_long"):
                            fold_importances_long.append(fold_0_result["importances_long"])
                        if fold_0_result.get("importances_short"):
                            fold_importances_short.append(fold_0_result["importances_short"])
                        if fold_0_result.get("fold_pnl") is not None:
                            fold_performances.append({
                                "fold": 0,
                                "ct": fold_0_result.get("best_ct"),
                                "trades": len(fold_0_result.get("oos_trades", [])),
                                "pnl": fold_0_result.get("fold_pnl", 0),
                                "win_rate": fold_0_result.get("fold_wr", 0),
                            })

                    log(1, f"    Fold 1 done ({time.time()-t_fold:.1f}s)", sym)

                    # === PHASE 2: Folds 1-7 sequentiell ausführen ===
                    remaining_folds = folds[1:]
                    if remaining_folds and (selected_features_long or selected_features_short):
                        for fold_idx, fold_data in enumerate(remaining_folds):
                            result = process_fold(
                                fold_idx + 1, fold_data,
                                shared_config, selected_features_long, selected_features_short
                            )
                            if not result.get("skipped", True):
                                all_oos_trades.extend(result.get("oos_trades", []))
                                if result.get("importances_long"):
                                    fold_importances_long.append(result["importances_long"])
                                if result.get("importances_short"):
                                    fold_importances_short.append(result["importances_short"])
                                if result.get("fold_pnl") is not None:
                                    fold_performances.append({
                                        "fold": result["fold_idx"],
                                        "ct": result.get("best_ct"),
                                        "trades": len(result.get("oos_trades", [])),
                                        "pnl": result.get("fold_pnl", 0),
                                        "win_rate": result.get("fold_wr", 0),
                                    })

                    # Feature Stability Check (nach allen Folds)
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

                # Kombiniere Features
                selected_features = []
                if selected_features_long:
                    selected_features.extend(selected_features_long)
                if selected_features_short:
                    selected_features.extend([f for f in selected_features_short if f not in selected_features])

                if not selected_features:
                    continue

                # Aggregiere OOS-Trades (CT wurde bereits auf Validation optimiert)
                tr = [t["result"] for t in all_oos_trades]
                if len(tr) >= MIN_TRADES:
                    rrr = tp / sl

                    # Bestimme den am häufigsten gewählten CT über alle Folds
                    ct_counts = {}
                    for t in all_oos_trades:
                        ct_counts[t["ct"]] = ct_counts.get(t["ct"], 0) + 1
                    best_ct = max(ct_counts.keys(), key=lambda x: ct_counts[x]) if ct_counts else grid["ct"][0]

                    preliminary_kelly = max(0, min(0.05, (
                        (tr.count(1.0) / len(tr) * rrr - (1 - tr.count(1.0) / len(tr))) / rrr
                    ) / 4)) if len(tr) > 0 else 0.01

                    trade_returns = [preliminary_kelly * rrr if r > 0 else -preliminary_kelly for r in tr]
                    sharpe = calculate_sharpe_ratio(trade_returns)
                    calmar = calculate_calmar_ratio(tr, preliminary_kelly, rrr)

                    hour_pnl = {}
                    for t in all_oos_trades:
                        h = t["hour"]
                        hour_pnl[h] = hour_pnl.get(h, 0) + t["result"]
                    good_hours = [h for h, pnl in hour_pnl.items() if pnl > 0]

                    # Fold-Stabilität prüfen (alle Folds sollten profitabel sein)
                    profitable_folds = sum(1 for fp in fold_performances if fp["pnl"] > 0)
                    fold_stability = profitable_folds / len(fold_performances) if fold_performances else 0

                    candidate = {
                        "pnl": sum(tr),
                        "tr": tr,
                        "trades_detailed": all_oos_trades.copy(),  # Volle Trade-Details
                        "params": (tp, sl, best_ct),
                        "feats": selected_features,
                        "feature_group": feature_group,
                        "rrr": rrr,
                        "sharpe": sharpe,
                        "calmar": calmar,
                        "good_hours": good_hours if good_hours else list(range(24)),
                        "fold_performances": fold_performances,
                        "fold_stability": fold_stability,
                    }
                    candidates.append(candidate)

                    # Grid-Results
                    all_grid_results.append({
                        "feature_group": feature_group,
                        "tp_mult": tp,
                        "sl_mult": sl,
                        "conf_thresh": best_ct,
                        "rrr": rrr,
                        "pnl": sum(tr),
                        "trades": len(tr),
                        "win_rate": tr.count(1.0) / len(tr),
                        "sharpe": sharpe,
                        "calmar": calmar,
                        "features": selected_features if selected_features else [],
                        "fold_stability": fold_stability,
                    })

                # Reset für nächste TP/SL Kombination
                all_oos_trades = []
                fold_performances = []

        log(2, f"Grid-Search fertig: {len(candidates)} Kandidaten gefunden ({time.time()-t_start:.1f}s)", sym)

        # grid_results enthält alle Kombinationen mit >= MIN_TRADES
        grid_results = all_grid_results

        if not candidates:
            log(1, f"SKIP - Keine profitablen Kandidaten", sym)
            report_done(sym, "no_candidates")
            return {"symbol": sym, "status": "no_candidates", "grid_results": grid_results}

        # Sortiere nach kombinierter Metrik mit Smoothness-Bonus
        for c in candidates:
            # Berechne Smoothness für jeden Kandidaten
            preliminary_kelly = max(0, min(0.05, (
                (c["tr"].count(1.0) / len(c["tr"]) * c["rrr"] - (1 - c["tr"].count(1.0) / len(c["tr"]))) / c["rrr"]
            ) / 4)) if len(c["tr"]) > 0 else 0.01

            smoothness = calculate_equity_smoothness(c["tr"], preliminary_kelly, c["rrr"])
            c["smoothness"] = smoothness

            # Score: PnL * Sharpe-Bonus * Smoothness-Bonus
            # Smoothness-Bonus: 0.8 bis 1.2 basierend auf Score
            smoothness_bonus = 0.8 + (smoothness["smoothness_score"] * 0.4)
            c["score"] = c["pnl"] * (1 + max(0, c["sharpe"]) / 10) * smoothness_bonus

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
            report_done(sym, "no_plateau")
            return {"symbol": sym, "status": "no_plateau", "grid_results": grid_results}

        # Ensemble: Top-3 nach Plateau-Score
        candidates.sort(key=lambda x: x.get("plateau_score", x["score"]), reverse=True)
        top_n = min(3, len(candidates))
        ensemble_configs = candidates[:top_n]

        total_score = sum(c.get("plateau_score", c["score"]) for c in ensemble_configs)
        if total_score <= 0:
            report_done(sym, "no_score")
            return {"symbol": sym, "status": "no_score", "grid_results": grid_results}
        wr = b["tr"].count(1.0) / len(b["tr"]) if b["tr"] else 0

        # 1/4 Kelly
        p = wr
        q = 1 - p
        rrr = b["rrr"]
        full_kelly = (p * rrr - q) / rrr if rrr > 0 else 0
        fk = max(0, min(0.05, full_kelly / 4))

        if fk <= 0:
            report_done(sym, "no_kelly")
            return {"symbol": sym, "status": "no_kelly", "grid_results": grid_results}

        # === MONTE CARLO TESTS ===
        # Prüfe ob Ergebnisse statistisch signifikant sind
        t_mc = time.time()
        mc_perm = monte_carlo_permutation_test(b["tr"], n_permutations=1000)
        mc_equity = monte_carlo_equity_simulation(b["tr"], fk, rrr, n_simulations=500)

        log(2, f"Monte Carlo: p={mc_perm['p_value']:.3f}, "
               f"Equity median={mc_equity['median_equity']:.1f}, "
               f"bankruptcy={mc_equity['bankruptcy_rate']:.1%} ({time.time()-t_mc:.1f}s)", sym)

        # Filtere nicht-signifikante Ergebnisse
        if not mc_perm["is_significant"]:
            log(1, f"SKIP - Nicht signifikant (p={mc_perm['p_value']:.3f} >= 0.05)", sym)
            report_done(sym, "not_significant")
            return {
                "symbol": sym,
                "status": "not_significant",
                "p_value": mc_perm["p_value"],
                "grid_results": grid_results
            }

        # Warne bei hoher Bankruptcy-Rate
        if mc_equity["bankruptcy_rate"] > 0.1:
            log(1, f"WARNUNG: {mc_equity['bankruptcy_rate']:.1%} Bankruptcy-Rate in MC-Simulation", sym)

        # === KELLY-ANPASSUNG FÜR ZIEL-DRAWDOWN ===
        # Passe Kelly an, um Max DD auf ~30% zu begrenzen
        kelly_adjustment = adjust_kelly_for_target_dd(b["tr"], fk, rrr, target_max_dd=0.30)
        if kelly_adjustment["scale_factor"] < 1.0:
            log(2, f"Kelly angepasst: {fk*100:.2f}% -> {kelly_adjustment['adjusted_kelly']*100:.2f}% "
                   f"(DD: {kelly_adjustment['original_dd']*100:.0f}% -> {kelly_adjustment['adjusted_dd']*100:.0f}%)", sym)
            fk = kelly_adjustment["adjusted_kelly"]

        # === CIRCUIT BREAKER OPTIMIERUNG ===
        # Finde optimale Pause-Parameter
        circuit_breaker = find_optimal_circuit_breaker(
            b["tr"], fk, rrr,
            loss_range=(3, 8),      # Pausiere nach 3-8 Verlusten
            pause_range=(5, 30)     # Pause für 5-30 Trades
        )

        if circuit_breaker["optimal_pause_after_losses"] > 0:
            log(2, f"Circuit Breaker: Pause nach {circuit_breaker['optimal_pause_after_losses']} Verlusten "
                   f"für {circuit_breaker['optimal_pause_bars']} Trades "
                   f"(DD: {circuit_breaker['baseline_dd']*100:.0f}% -> {circuit_breaker['optimized_dd']*100:.0f}%)", sym)

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

        result = {
            "symbol": sym,
            "status": "ok",
            "pnl": b["pnl"],
            "config": {
                "kelly_risk": fk,
                "point_value": p_val,
                "spread": spread,
                "tp_mult": b["params"][0],
                "sl_mult": b["params"][1],
                "conf_thresh": b["params"][2],
                "feature_group": b.get("feature_group", "unknown"),
                "features": b["feats"],
                "good_hours": b.get("good_hours", list(range(24))),
                "ensemble": ensemble_weights if ensemble_weights else None,
                "dd_scaling": {"10": 0.5, "20": 0.25},
                # Circuit Breaker Parameter (von KI optimiert)
                "circuit_breaker": {
                    "pause_after_losses": circuit_breaker["optimal_pause_after_losses"],
                    "pause_bars": circuit_breaker["optimal_pause_bars"],
                    "enabled": circuit_breaker["optimal_pause_after_losses"] > 0,
                },
                # Kelly-Anpassung Info
                "kelly_adjustment": {
                    "original_kelly": kelly_adjustment["adjusted_kelly"] / kelly_adjustment["scale_factor"] if kelly_adjustment["scale_factor"] > 0 else fk,
                    "scale_factor": kelly_adjustment["scale_factor"],
                    "target_dd": 0.30,
                },
            },
            "tr_trace": b["tr"],
            "trades_detailed": b.get("trades_detailed", []),  # Volle Trade-Details mit Zeiten, Preisen etc.
            "rrr": b["rrr"],
            "win_rate": wr,
            "sharpe": b["sharpe"],
            "calmar": b["calmar"],
            "currencies": currencies,
            "grid_results": grid_results,  # Alle getesteten Kombinationen
            # Monte Carlo Statistiken
            "monte_carlo": {
                "p_value": mc_perm["p_value"],
                "is_significant": mc_perm["is_significant"],
                "percentile": mc_perm["percentile"],
                "equity_median": mc_equity["median_equity"],
                "equity_p5": mc_equity["p5_equity"],
                "equity_p95": mc_equity["p95_equity"],
                "bankruptcy_rate": mc_equity["bankruptcy_rate"],
            },
            # Fold-Stabilität
            "fold_stability": b.get("fold_stability", 0),
            "fold_performances": b.get("fold_performances", []),
            # Equity-Smoothness
            "smoothness": b.get("smoothness", {}),
        }

        smoothness_info = b.get("smoothness", {})
        smoothness_score = smoothness_info.get("smoothness_score", 0)
        log(1, f"OK - WR={wr:.1%} Sharpe={b['sharpe']:.2f} Smooth={smoothness_score:.2f} p={mc_perm['p_value']:.3f} Trades={len(b['tr'])} ({time.time()-t_start:.1f}s)", sym)
        report_done(sym, "ok")
        return result

    except Exception as e:
        log(1, f"FEHLER: {e}", sym)
        if LOG_LEVEL >= 2:
            import traceback
            traceback.print_exc()
        report_done(sym, "error")
        return None
