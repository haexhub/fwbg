"""
Walk-Forward Optimierung und Symbol-Verarbeitung
"""
import os
import time
import numpy as np
import pandas as pd

from .config import (
    DATA_PATH, MACRO_INDICATORS, LOOKBACKS_HOURS, LOOKBACKS_DAYS,
    OOS_SIZE, tf_cfg, MIN_TRADES, WALK_FORWARD_FOLDS
)
from .strategy_config import StrategyConfig
from .asset_config import get_asset
from .simulation_context import SimulationContext
from .data_loader import load_data_aligned, load_macro_csv
from .indicators import compute_indicator_pool, get_feature_columns, compute_regime_filter, filter_features_by_group
from .simulation import (
    calculate_sharpe_ratio, calculate_calmar_ratio,
    monte_carlo_permutation_test, monte_carlo_equity_simulation,
    adjust_kelly_for_target_dd, find_optimal_circuit_breaker, calculate_equity_smoothness
)
from .plateau import calculate_param_plateau_score, select_best_plateau_candidate
from .progress import report_done
from .logging_utils import log
from .nested_cv import (
    nested_cv_split, run_inner_cv, evaluate_on_holdout
)


def walk_forward_split(df, n_folds=WALK_FORWARD_FOLDS, oos_size=OOS_SIZE):
    """
    Generiert Walk-Forward Fenster: [(train_df, val_df, test_df), ...]

    DEPRECATED: Nutze nested_cv_split() für unbiased Evaluation.
    Diese Funktion wird nur noch für Abwärtskompatibilität beibehalten.

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


def process_symbol(csv_path: str, strategy: StrategyConfig) -> dict:
    """
    Verarbeitet ein einzelnes Symbol mit Walk-Forward Optimierung.

    Args:
        csv_path: Pfad zur CSV-Datei
        strategy: StrategyConfig mit allen Strategie-Parametern
    """
    sym = os.path.basename(csv_path).split("_")[0]
    t_start = time.time()

    if sym in ["VIX", "DXY"]:
        log(2, "Übersprungen (Makro-Asset)", sym)
        return {"symbol": sym, "status": "macro_asset"}

    log(1, "START", sym)

    try:
        t0 = time.time()
        df = load_data_aligned(csv_path)
        if df is None:
            log(1, "SKIP - Keine Daten", sym)
            return {"symbol": sym, "status": "no_data"}
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
            return {"symbol": sym, "status": "insufficient_data", "rows": len(df)}

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
            return {"symbol": sym, "status": "insufficient_features", "features": len(full_pool)}

        # Asset-Konfiguration laden
        asset = get_asset(sym)

        # SimulationContext erstellen (wird durch alle Funktionen gereicht)
        ctx = SimulationContext.create(asset, strategy)

        # Kurzreferenzen für lokale Verwendung
        grid = strategy.get_grid_for_class(asset.asset_class)

        candidates = []
        all_grid_results = []  # Alle Kombinationen tracken

        # Feature-Gruppen aus Strategy-Config
        feature_groups_to_test = ctx.feature_groups
        total_combos = ctx.total_grid_combinations()
        log(1, f"Grid-Search: {len(feature_groups_to_test)} Feature-Gruppen x {len(grid.tp)}x{len(grid.sl)}x{len(grid.ct)} = {total_combos} Kombinationen", sym)
        if ctx.min_rrr > 0:
            log(1, f"Min RRR Filter: {ctx.min_rrr} (Scalping-Strategien mit RRR < {ctx.min_rrr} werden gefiltert)", sym)
        if ctx.max_trade_bars:
            log(1, f"Max Trade Bars: {ctx.max_trade_bars} ({ctx.max_trade_bars / 24:.0f} Tage)", sym)

        # === NESTED CV: Holdout Split ===
        # Die letzten 20% werden KOMPLETT zurückgehalten für finale Evaluation
        cv_split = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=5)
        inner_folds = cv_split["inner_folds"]
        holdout_df = cv_split["holdout_df"]
        inner_df = cv_split["inner_df"]

        log(1, f"Nested CV: {len(inner_df)} Inner / {len(holdout_df)} Holdout (nie gesehen während Grid-Search)", sym)

        # Äußere Schleife: Feature-Gruppen
        for fg_idx, feature_group in enumerate(feature_groups_to_test):
            # Filtere Features nach Gruppe
            group_features = filter_features_by_group(full_pool, feature_group)

            if len(group_features) < 3:
                log(2, f"  Feature-Gruppe '{feature_group}': nur {len(group_features)} Features - übersprungen", sym)
                continue

            log(1, f"Feature-Gruppe {fg_idx+1}/{len(feature_groups_to_test)}: {feature_group} ({len(group_features)} Features)", sym)

            grid_count = 0
            grid_per_fg = ctx.grid_combinations_per_feature_group()
            total_grid_combos = ctx.total_grid_combinations()
            grid_offset = fg_idx * grid_per_fg

            for tp in grid.tp:
                for sl in grid.sl:
                    grid_count += 1
                    global_grid_pos = grid_offset + grid_count

                    # RRR-Filter: Überspringe Kombinationen mit zu niedrigem RRR
                    rrr = tp / sl
                    if ctx.min_rrr > 0 and rrr < ctx.min_rrr:
                        log(2, f"  Grid {grid_count}/{grid_per_fg} (TP={tp}, SL={sl}) - SKIP (RRR {rrr:.2f} < {ctx.min_rrr})", sym)
                        continue

                    log(2, f"  Grid {grid_count}/{grid_per_fg} (TP={tp}, SL={sl}, RRR={rrr:.2f})", sym)

                    # === INNER CV: Grid-Search auf Inner Folds ===
                    inner_result = run_inner_cv(
                        inner_folds, group_features, tp, sl, ctx,
                        global_grid_pos, total_grid_combos
                    )

                    if not inner_result["success"]:
                        continue

                    # Kandidat speichern (noch OHNE Holdout-Evaluation!)
                    candidate = {
                        "inner_val_pnl": inner_result["avg_val_pnl"],
                        "params": (tp, sl, inner_result["best_ct"]),
                        "feats": inner_result["selected_features"],
                        "feature_group": feature_group,
                        "rrr": rrr,
                        "selected_features_long": inner_result["selected_features_long"],
                        "selected_features_short": inner_result["selected_features_short"],
                        "fold_stability": inner_result.get("fold_stability", 0),
                    }

                    # Bei separater L/S Optimierung: CT-Tuple aufschlüsseln
                    if ctx.separate_long_short and "ct_long" in inner_result:
                        candidate["ct_long"] = inner_result["ct_long"]
                        candidate["ct_short"] = inner_result["ct_short"]

                    candidates.append(candidate)

                    # Grid-Result mit CT (kann Tuple sein)
                    conf_thresh = inner_result["best_ct"]
                    grid_result = {
                        "feature_group": feature_group,
                        "tp_mult": tp,
                        "sl_mult": sl,
                        "conf_thresh": conf_thresh,
                        "rrr": rrr,
                        "inner_val_pnl": inner_result["avg_val_pnl"],
                        "fold_stability": inner_result.get("fold_stability", 0),
                        "features": inner_result["selected_features"],
                    }
                    # Bei separater Optimierung: CTs aufschlüsseln für Grid-Results
                    if isinstance(conf_thresh, tuple):
                        grid_result["ct_long"] = conf_thresh[0]
                        grid_result["ct_short"] = conf_thresh[1]
                    all_grid_results.append(grid_result)

        log(2, f"Inner CV fertig: {len(candidates)} Kandidaten ({time.time()-t_start:.1f}s)", sym)

        grid_results = all_grid_results

        if not candidates:
            log(1, f"SKIP - Keine profitablen Kandidaten", sym)
            report_done(sym, "no_candidates")
            return {"symbol": sym, "status": "no_candidates", "grid_results": grid_results}

        # === PLATEAU-BASIERTE AUSWAHL (basierend auf Inner CV PnL) ===
        # Sortiere nach Inner Validation PnL für Plateau-Berechnung
        for c in candidates:
            c["score"] = c["inner_val_pnl"]

        candidates = calculate_param_plateau_score(
            candidates,
            grid.tp,
            grid.sl,
            grid.ct
        )

        # Wähle besten Plateau-Kandidaten
        b = select_best_plateau_candidate(
            candidates,
            grid.tp,
            grid.sl,
            grid.ct,
            min_neighbors=2
        )

        if not b:
            # Fallback: Bester nach Inner Val PnL
            candidates.sort(key=lambda x: x["inner_val_pnl"], reverse=True)
            b = candidates[0] if candidates else None

        if not b:
            report_done(sym, "no_plateau")
            return {"symbol": sym, "status": "no_plateau", "grid_results": grid_results}

        # === TOP-N KANDIDATEN SAMMELN (für Vergleich) ===
        # Sortiere alle Kandidaten nach Inner Val PnL und behalte Top 5
        # mit unterschiedlichen RRR-Werten für Diversität
        TOP_N = 5
        MIN_ANNUAL_RETURN = 10.0  # Mindestens 10%/Jahr
        candidates.sort(key=lambda x: x.get("inner_val_pnl", 0), reverse=True)

        # Berechne Inner CV Zeitraum für Jahresrendite-Schätzung
        n_inner_folds = len(inner_folds)
        inner_val_size = len(inner_folds[0][1]) if inner_folds else OOS_SIZE
        inner_total_bars = n_inner_folds * inner_val_size
        bars_per_year = tf_cfg["bars_per_hour"] * 24 * 250  # 250 Trading-Tage pro Jahr
        inner_years = inner_total_bars / bars_per_year if bars_per_year > 0 else 1

        def estimate_annual_return(inner_val_pnl):
            """Schätzt Jahresrendite aus Inner CV PnL."""
            if inner_val_pnl <= 0 or inner_years <= 0:
                return -100
            final_equity = 100 + inner_val_pnl
            return ((final_equity / 100.0) ** (1 / inner_years) - 1) * 100

        # Sammle Top-N mit RRR-Diversität und Mindest-Rendite
        top_candidates_for_export = []
        seen_rrr = set()
        skipped_low_return = 0
        for c in candidates:
            # Prüfe Mindest-Rendite
            est_return = estimate_annual_return(c.get("inner_val_pnl", 0))
            if est_return < MIN_ANNUAL_RETURN:
                skipped_low_return += 1
                continue

            rrr_bucket = round(c["rrr"], 1)  # Bucket auf 0.1 gerundet
            if rrr_bucket not in seen_rrr or len(top_candidates_for_export) < 3:
                top_candidates_for_export.append({
                    "rank": len(top_candidates_for_export) + 1,
                    "params": c["params"],
                    "rrr": c["rrr"],
                    "inner_val_pnl": c.get("inner_val_pnl", 0),
                    "est_annual_return": est_return,
                    "feature_group": c.get("feature_group", "unknown"),
                    "feats": c.get("feats", []),
                    "plateau_score": c.get("plateau_score", 0),
                    "selected_features_long": c.get("selected_features_long", []),
                    "selected_features_short": c.get("selected_features_short", []),
                })
                seen_rrr.add(rrr_bucket)
            if len(top_candidates_for_export) >= TOP_N:
                break

        if skipped_low_return > 0:
            log(2, f"  {skipped_low_return} Kandidaten übersprungen (< {MIN_ANNUAL_RETURN}%/Jahr)", sym)
        log(2, f"Top-{len(top_candidates_for_export)} Kandidaten gesammelt (RRR: {[c['rrr'] for c in top_candidates_for_export]})", sym)

        # === HOLDOUT EVALUATION ===
        # JETZT erst evaluieren wir auf dem Holdout-Set (nie vorher gesehen!)
        ct_param = b['params'][2]
        if isinstance(ct_param, tuple):
            ct_str = f"CT_L={ct_param[0]:.2f}/CT_S={ct_param[1]:.2f}"
        else:
            ct_str = f"CT={ct_param:.2f}"
        log(1, f"Holdout-Evaluation für besten Kandidaten (TP={b['params'][0]}, SL={b['params'][1]}, {ct_str})", sym)

        holdout_result = evaluate_on_holdout(holdout_df, inner_df, b, ctx)

        if holdout_result["n_trades"] < ctx.min_trades:
            log(1, f"SKIP - Zu wenig Holdout-Trades ({holdout_result['n_trades']} < {ctx.min_trades})", sym)
            report_done(sym, "insufficient_holdout_trades")
            return {"symbol": sym, "status": "insufficient_holdout_trades", "grid_results": grid_results}

        # Füge Holdout-Ergebnisse zum Kandidaten hinzu
        b["tr"] = holdout_result["trades"]
        b["trades_detailed"] = holdout_result["trades_detailed"]
        b["pnl"] = holdout_result["pnl"]
        b["holdout_win_rate"] = holdout_result["win_rate"]

        # Berechne Metriken auf Holdout-Trades
        wr = holdout_result["win_rate"]
        tr = holdout_result["trades"]

        preliminary_kelly = max(0, min(0.05, (
            (wr * b["rrr"] - (1 - wr)) / b["rrr"]
        ) / 4)) if wr > 0 else 0.01

        trade_returns = [preliminary_kelly * b["rrr"] if r > 0 else -preliminary_kelly for r in tr]

        # Trades pro Jahr = (Anzahl Trades / Anzahl Bars) * Bars pro Jahr
        holdout_bars = len(holdout_df)
        trades_per_year = (len(tr) / holdout_bars) * bars_per_year if holdout_bars > 0 else len(tr)
        b["sharpe"] = calculate_sharpe_ratio(trade_returns, trades_per_year=trades_per_year)
        b["calmar"] = calculate_calmar_ratio(tr, preliminary_kelly, b["rrr"])
        b["smoothness"] = calculate_equity_smoothness(tr, preliminary_kelly, b["rrr"])

        # Good hours aus Holdout
        hour_pnl = {}
        for t in holdout_result["trades_detailed"]:
            h = t["hour"]
            hour_pnl[h] = hour_pnl.get(h, 0) + t["result"]
        b["good_hours"] = [h for h, pnl in hour_pnl.items() if pnl > 0] or list(range(24))

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

        # Ensemble-Gewichte (Top-3 nach Inner Val PnL, ohne den Besten)
        ensemble_weights = []
        candidates.sort(key=lambda x: x.get("inner_val_pnl", 0), reverse=True)
        top_candidates = [c for c in candidates[:3] if c != b]
        total_inner_pnl = sum(c.get("inner_val_pnl", 0) for c in top_candidates) + b.get("inner_val_pnl", 0)

        if total_inner_pnl > 0:
            for c in top_candidates:
                c_inner_pnl = c.get("inner_val_pnl", 0)
                if c_inner_pnl > 0:
                    ensemble_weights.append({
                        "tp_mult": c["params"][0],
                        "sl_mult": c["params"][1],
                        "conf_thresh": c["params"][2],
                        "weight": c_inner_pnl / total_inner_pnl,
                    })

        # CT-Werte extrahieren (kann Tuple sein bei separate_long_short)
        ct_value = b["params"][2]
        if isinstance(ct_value, tuple):
            ct_long, ct_short = ct_value
            ct_display = ct_long  # Für Kompatibilität mit altem Code
        else:
            ct_long = ct_short = ct_value
            ct_display = ct_value

        result = {
            "symbol": sym,
            "status": "ok",
            "pnl": b["pnl"],
            "config": {
                "kelly_risk": fk,
                "point_value": asset.point,
                "spread": ctx.spread,
                "tp_mult": b["params"][0],
                "sl_mult": b["params"][1],
                "conf_thresh": ct_display,
                # Separate CTs bei long_short_separate
                "ct_long": ct_long,
                "ct_short": ct_short,
                "separate_long_short": ctx.separate_long_short,
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
            "currencies": asset.currencies,
            "grid_results": grid_results,  # Alle getesteten Kombinationen
            # Top-N Kandidaten für Export (mit RRR-Diversität)
            "top_candidates": top_candidates_for_export,
            # Fold-Stabilität (aus Inner CV)
            "fold_stability": b.get("fold_stability", 0),
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
            # Equity-Smoothness
            "smoothness": b.get("smoothness", {}),
            # Nested CV Info
            "nested_cv": {
                "inner_samples": len(inner_df),
                "holdout_samples": len(holdout_df),
                "inner_val_pnl": b.get("inner_val_pnl", 0),
                "holdout_pnl": holdout_result["pnl"],
                "fold_stability": b.get("fold_stability", 0),
            },
        }

        # Bei separater L/S Optimierung: Statistiken pro Richtung hinzufügen
        if ctx.separate_long_short:
            long_stats = holdout_result.get("long_stats", {})
            short_stats = holdout_result.get("short_stats", {})
            result["long_short_stats"] = {
                "long": {
                    "n_trades": long_stats.get("n_trades", 0),
                    "win_rate": long_stats.get("win_rate", 0),
                    "pnl": long_stats.get("pnl", 0),
                    "ct": ct_long,
                },
                "short": {
                    "n_trades": short_stats.get("n_trades", 0),
                    "win_rate": short_stats.get("win_rate", 0),
                    "pnl": short_stats.get("pnl", 0),
                    "ct": ct_short,
                },
            }

        smoothness_info = b.get("smoothness", {})
        smoothness_score = smoothness_info.get("smoothness_score", 0)
        log(1, f"OK (Holdout) - WR={wr:.1%} Sharpe={b['sharpe']:.2f} Smooth={smoothness_score:.2f} "
               f"p={mc_perm['p_value']:.3f} Trades={len(b['tr'])} ({time.time()-t_start:.1f}s)", sym)
        report_done(sym, "ok")
        return result

    except Exception as e:
        log(1, f"FEHLER: {e}", sym)
        import traceback
        traceback.print_exc()
        report_done(sym, "error")
        return {"symbol": sym, "status": "error", "error": str(e)}
