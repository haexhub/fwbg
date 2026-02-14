"""
Walk-Forward Optimierung und Symbol-Verarbeitung.

Hauptmodul für die Verarbeitung einzelner Symbole mit Walk-Forward Optimierung.
Grid-Search Funktionen sind in grid_search.py ausgelagert.
"""
import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from fwbg.data.config import (
    DATA_PATH, MACRO_INDICATORS, LOOKBACKS_HOURS, LOOKBACKS_DAYS,
    OOS_SIZE, tf_cfg, MIN_TRADES, WALK_FORWARD_FOLDS
)
from fwbg.core.config import StrategyConfig, RegimeFilterConfig
from fwbg.data.assets import get_asset
from fwbg.core.context import SimulationContext
from fwbg.data.loader import load_data_aligned, load_macro_csv
from fwbg.pipeline import (
    compute_indicator_pool, get_feature_columns, compute_regime_filter,
    calculate_param_plateau_score, select_best_plateau_candidate
)
from fwbg.pipeline.features import split_indicators_by_stationarity
from fwbg.simulation.trade import (
    calculate_sharpe_ratio, calculate_calmar_ratio,
    monte_carlo_permutation_test, monte_carlo_equity_simulation,
    calculate_equity_smoothness,
)
from fwbg.core import get_risk_manager
from fwbg.utils.progress import report_done, report_phase
from fwbg.utils.logging import log
from .nested_cv import nested_cv_split, evaluate_on_holdout
from .grid_search import run_grid_search
from .robust_validation import create_walk_forward_folds
from .bias_checks import check_asset_bias


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
    report_phase(sym, "Lade Daten...")

    try:
        t0 = time.time()
        df = load_data_aligned(csv_path)
        if df is None:
            log(1, "SKIP - Keine Daten", sym)
            return {"symbol": sym, "status": "no_data"}
        log(2, f"Daten geladen: {len(df)} Zeilen ({time.time()-t0:.1f}s)", sym)
        report_phase(sym, "Makro-Indikatoren...")

        # === ALLE MAKRO-INDIKATOREN PARALLEL LADEN ===
        t0 = time.time()
        date_series = df.index.date

        # 1. Parallel alle Makro-Dateien laden
        def _load_macro(item):
            filename, prefix = item
            macro_path = f"{DATA_PATH}/{filename}.csv"
            macro_df = load_macro_csv(macro_path)
            if macro_df is not None:
                return prefix, macro_df["Close"].to_dict()
            return None

        macro_items = list(MACRO_INDICATORS.items())
        macro_data = {}

        with ThreadPoolExecutor(max_workers=min(8, len(macro_items))) as executor:
            results = list(executor.map(_load_macro, macro_items))

        for result in results:
            if result is not None:
                prefix, lookup = result
                macro_data[prefix] = lookup

        # 2. Alle Spalten in einem Batch erstellen (effizienter als einzeln)
        new_columns = {}
        date_timestamps = pd.Series(date_series, index=df.index).map(pd.Timestamp)

        for prefix, lookup in macro_data.items():
            col_name = f"macro_{prefix}"
            base_values = date_timestamps.map(lambda d: lookup.get(d, np.nan)).ffill()
            new_columns[col_name] = base_values

            # Stunden-basierte Lookbacks
            for lb_h in LOOKBACKS_HOURS:
                new_columns[f"{col_name}_chg_{lb_h}h"] = base_values.pct_change(lb_h) * 100

            # Tages-basierte Lookbacks
            for lb_d in LOOKBACKS_DAYS:
                new_columns[f"{col_name}_chg_{lb_d}d"] = base_values.pct_change(24 * lb_d) * 100

        # 3. Alle Spalten auf einmal hinzufügen
        if new_columns:
            df = df.assign(**new_columns)

        log(2, f"Makro-Indikatoren: {len(macro_data)} geladen ({time.time()-t0:.1f}s)", sym)

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

        # === LOOKAHEAD BIAS PREVENTION ===
        # KRITISCH: Walk-Forward Splits VOR Indikator-Berechnung!
        # Rolling Windows in Indikatoren würden sonst Zukunftsdaten sehen.
        #
        # Reihenfolge:
        # 1. OHLC + Makro laden (OK, Makro ist täglich ohne Rolling)
        # 2. Create Walk-Forward Folds (auf OHLC-Basis)
        # 3. Für jeden Fold: Indikatoren SEPARAT berechnen

        if len(df) < MIN_TRADES * 8:  # Need enough data for walk-forward
            log(1, f"SKIP - Zu wenig Daten für Walk-Forward ({len(df)} < {MIN_TRADES * 8})", sym)
            return {"symbol": sym, "status": "insufficient_data", "rows": len(df)}

        # Asset-Konfiguration laden
        asset = get_asset(sym)

        # SimulationContext erstellen (wird durch alle Funktionen gereicht)
        ctx = SimulationContext.create(asset, strategy)

        # Kurzreferenzen für lokale Verwendung
        grid = strategy.get_grid_for_class(asset.asset_class)

        # === WALK-FORWARD FOLDS ERSTELLEN ===
        report_phase(sym, f"Creating {WALK_FORWARD_FOLDS} walk-forward folds...")
        try:
            wf_folds = create_walk_forward_folds(
                df,
                n_folds=WALK_FORWARD_FOLDS,
                test_size=4000,
                min_train_size=20000,
                anchored=True,
                embargo_bars=ctx.embargo_bars,
            )
        except ValueError as e:
            log(1, f"SKIP - {str(e)}", sym)
            return {"symbol": sym, "status": "insufficient_data_for_folds", "error": str(e)}

        log(1, f"Walk-Forward: {len(wf_folds)} folds created (prevents sample bias)", sym)
        for fold in wf_folds:
            log(2, f"  Fold {fold.fold_id}: Train[{fold.train_end - fold.train_start}] Test[{fold.test_end - fold.test_start}]", sym)

        # Grid-Summary
        grid_per_fold = ctx.total_grid_combinations()
        total_trainings = grid_per_fold * len(wf_folds)
        log(1, f"Grid: {grid_per_fold} combos/fold × {len(wf_folds)} folds = {total_trainings} total trainings", sym)

        # === WALK-FORWARD LOOP: Process each fold ===
        # For each fold, we:
        # 1. Compute indicators separately (no lookahead!)
        # 2. Run grid search on train
        # 3. Evaluate on test
        # 4. Store results

        all_fold_results = []
        accumulated_grid_results = []  # Sammelt grid_results über alle Folds

        # === INDICATOR PRECOMPUTATION ===
        # Split indicators: those that benefit from stationary input compute per fold,
        # the rest are precomputed once on raw data (saves ~60% indicator computation).
        indicators = strategy.get_indicators()
        preprocessing_configs = strategy.get_preprocessing()
        has_preprocessing = bool(preprocessing_configs)

        stationary_indicators, raw_indicators = split_indicators_by_stationarity(
            indicators, has_preprocessing=has_preprocessing
        )

        precomputed_raw_df = None
        if raw_indicators:
            t0 = time.time()
            report_phase(sym, "Precomputing raw indicators...")
            precomputed_raw_df = compute_indicator_pool(
                df, indicators=raw_indicators, progress_callback=None
            )
            raw_feature_cols = [c for c in precomputed_raw_df.columns
                                if c not in ['O', 'H', 'L', 'C', 'V']
                                and not c.startswith('_') and not c.startswith('macro_')]
            precomputed_raw_df = precomputed_raw_df[raw_feature_cols]
            log(1, f"Precomputed {len(raw_indicators)} raw indicators: "
                   f"{len(raw_feature_cols)} features ({time.time()-t0:.1f}s)", sym)

        fold_indicators = stationary_indicators if has_preprocessing else []
        log(2, f"Per-fold indicators: {len(fold_indicators)}, Precomputed: {len(raw_indicators)}", sym)

        for fold_idx, fold in enumerate(wf_folds):
            log(1, f"=== Processing Fold {fold.fold_id + 1}/{len(wf_folds)} ===", sym)
            report_phase(sym, f"Fold {fold.fold_id + 1}/{len(wf_folds)}: Computing indicators...")

            def indicator_progress(name, idx, total):
                report_phase(sym, f"Fold {fold.fold_id + 1}: Indicators {name} ({idx}/{total})")

            # Preprocessing VOR Indikatoren anwenden (López de Prado):
            # Fracdiff macht OHLC stationär → stationary-benefiting Indikatoren darauf.
            # Original-OHLC wird für Targets/Trade-Simulation aufbewahrt.
            t0 = time.time()

            pp_train_raw = fold.train_df
            pp_test_raw = fold.test_df
            orig_train_ohlc = None
            orig_test_ohlc = None
            ohlc_cols = ['O', 'H', 'L', 'C']

            if preprocessing_configs:
                from fwbg.core import get_preprocessor
                from fwbg.pipeline.context import PipelineContext

                orig_train_ohlc = {col: fold.train_df[col].copy() for col in ohlc_cols}
                orig_test_ohlc = {col: fold.test_df[col].copy() for col in ohlc_cols}

                for pp_config in preprocessing_configs:
                    pp_name = pp_config.get("name", "")
                    pp_params = pp_config.get("params", {})
                    try:
                        pp_cls = get_preprocessor(pp_name)
                        pp = pp_cls()

                        # Fit auf Train-Daten (kein Lookahead)
                        train_ctx = PipelineContext(
                            df=pp_train_raw.copy(), symbol=sym, asset_class=ctx.asset_class
                        )
                        pp.fit(train_ctx, **pp_params)

                        # Execute auf Train
                        train_ctx = PipelineContext(
                            df=pp_train_raw.copy(), symbol=sym, asset_class=ctx.asset_class
                        )
                        train_ctx = pp.execute(train_ctx, **pp_params)
                        pp_train_raw = train_ctx.df

                        # Execute auf Test (nutzt History vom Train-Fit)
                        test_ctx = PipelineContext(
                            df=pp_test_raw.copy(), symbol=sym, asset_class=ctx.asset_class
                        )
                        test_ctx = pp.execute(test_ctx, **pp_params)
                        pp_test_raw = test_ctx.df
                    except Exception as e:
                        log(1, f"  Preprocessing {pp_name} failed: {e}", sym)

                log(2, f"  Preprocessing: Train {len(fold.train_df)}→{len(pp_train_raw)}, "
                       f"Test {len(fold.test_df)}→{len(pp_test_raw)}", sym)

            # Compute per-fold indicators (only stationary-benefiting, or none if no preprocessing)
            if fold_indicators:
                train_df = compute_indicator_pool(
                    pp_train_raw, indicators=fold_indicators, progress_callback=indicator_progress
                )
                test_df = compute_indicator_pool(
                    pp_test_raw, indicators=fold_indicators, progress_callback=None
                )
            else:
                train_df = pp_train_raw.copy()
                test_df = pp_test_raw.copy()

            # Merge precomputed raw indicator features
            if precomputed_raw_df is not None:
                raw_train = precomputed_raw_df.reindex(train_df.index)
                raw_test = precomputed_raw_df.reindex(test_df.index)
                train_df = pd.concat([train_df, raw_train], axis=1)
                test_df = pd.concat([test_df, raw_test], axis=1)

            train_df = train_df.copy()
            train_df = train_df.dropna()
            test_df = test_df.copy()
            test_df = test_df.dropna()

            # Original-OHLC wiederherstellen (für Targets und Trade-Simulation)
            if orig_train_ohlc:
                for col in ohlc_cols:
                    train_df[col] = orig_train_ohlc[col].reindex(train_df.index)
                    test_df[col] = orig_test_ohlc[col].reindex(test_df.index)

            log(2, f"  Fold {fold.fold_id + 1}: Train={train_df.shape} Test={test_df.shape} ({time.time()-t0:.1f}s)", sym)

            if len(train_df) < MIN_TRADES * 2:
                log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Train-Daten ({len(train_df)})", sym)
                continue

            if len(test_df) < MIN_TRADES:
                log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Test-Daten ({len(test_df)})", sym)
                continue

            # Feature-Pool aus Train-Daten
            full_pool = get_feature_columns(train_df)

            # Entferne Features mit inf/nan (XGBoost verträgt keine inf)
            clean_pool = []
            excluded_inf = 0
            excluded_nan = 0
            for col in full_pool:
                if col in train_df.columns:
                    has_inf = np.isinf(train_df[col]).any()
                    nan_ratio = train_df[col].isna().sum() / len(train_df)
                    if has_inf:
                        excluded_inf += 1
                    elif nan_ratio >= 0.1:
                        excluded_nan += 1
                    else:
                        clean_pool.append(col)
            full_pool = clean_pool
            log(2, f"  Fold {fold.fold_id + 1}: {len(full_pool)} clean features (excl: {excluded_inf} inf, {excluded_nan} nan)", sym)

            if len(full_pool) < 5:
                log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Features ({len(full_pool)})", sym)
                continue

            candidates = []
            all_grid_results = []

            # Regime-Filter Kombinationen aus Grid (falls definiert)
            regime_filter_combinations = grid.regime_filter_grid.get_combinations()
            n_regime_combos = len(regime_filter_combinations)

            # Berechne Gesamtzahl der Kombinationen inkl. Regime-Filter
            base_combos = ctx.total_grid_combinations()
            total_combos = base_combos * n_regime_combos

            if fold.fold_id == 0:  # Only log once
                log(1, f"Grid-Search: {len(grid.tp)}x{len(grid.sl)}x{len(grid.ct)} x {n_regime_combos} Regime = {total_combos} Kombinationen", sym)
                if ctx.min_rrr > 0:
                    log(1, f"Min RRR Filter: {ctx.min_rrr}", sym)

            # === NESTED CV: Inner Folds erstellen (auf Train-Daten dieses Folds) ===
            report_phase(sym, f"Fold {fold.fold_id + 1}: Grid-Search...")
            cv_split = nested_cv_split(train_df, holdout_ratio=0.0, n_inner_folds=ctx.n_inner_folds, embargo_bars=ctx.embargo_bars)
            inner_folds = cv_split["inner_folds"]

            log(2, f"  Fold {fold.fold_id + 1}: Nested CV with {len(inner_folds)} inner folds", sym)

            # === REGIME-FILTER KOMBINATIONEN ===
            for rf_idx, regime_config in enumerate(regime_filter_combinations):
                # Erstelle RegimeFilterConfig aus Kombination
                regime_params = RegimeFilterConfig.from_dict(regime_config)

                # Berechne _regime_ok SEPARAT für Train und Test (kein Lookahead!)
                train_df["_regime_ok"] = compute_regime_filter(train_df, regime_params)
                test_df["_regime_ok"] = compute_regime_filter(test_df, regime_params)

                # Update inner_folds mit neuem regime_ok
                for train_df_fold, val_df_fold in inner_folds:
                    train_df_fold["_regime_ok"] = train_df.loc[train_df_fold.index, "_regime_ok"]
                    val_df_fold["_regime_ok"] = train_df.loc[val_df_fold.index, "_regime_ok"]

                # Log Regime-Filter Info
                regime_desc = []
                if regime_params.adx_enabled:
                    regime_desc.append(f"ADX>={regime_params.adx_min}")
                if regime_params.vix_enabled:
                    regime_desc.append(f"VIX<={regime_params.vix_max}")
                if regime_params.hurst_enabled:
                    hurst_parts = []
                    if regime_params.hurst_min is not None:
                        hurst_parts.append(f"H>={regime_params.hurst_min}")
                    if regime_params.hurst_max is not None:
                        hurst_parts.append(f"H<={regime_params.hurst_max}")
                    regime_desc.append(" & ".join(hurst_parts) if hurst_parts else "Hurst")
                regime_str = " + ".join(regime_desc) if regime_desc else "No Filter"

                if n_regime_combos > 1 and fold.fold_id == 0:  # Only log once
                    log(2, f"  Regime {rf_idx+1}/{n_regime_combos}: {regime_str}", sym)

                # Grid-Search mit Progress-Reporting
                total_grid_combos = ctx.total_grid_combinations()
                fold_idx_1based = fold.fold_id + 1
                total_folds = len(wf_folds)

                from fwbg.utils.progress import report_progress
                def grid_progress_callback(grid_count, grid_total):
                    report_progress(sym, fold_idx_1based, total_folds, "grid_search",
                                   grid_count, total_grid_combos)

                report_phase(sym, f"Fold {fold_idx_1based}/{total_folds}: Feature Selection [{ctx.feature_selection}]...")
                gs_candidates, gs_grid_results = run_grid_search(
                    full_pool, inner_folds,
                    grid, ctx, regime_config, sym,
                    inner_df=train_df,
                    progress_callback=grid_progress_callback
                )
                candidates.extend(gs_candidates)
                all_grid_results.extend(gs_grid_results)

            log(2, f"  Fold {fold.fold_id + 1}: Grid search done, {len(candidates)} candidates", sym)

            grid_results = all_grid_results
            accumulated_grid_results.extend(all_grid_results)  # Akkumuliere über alle Folds

            if not candidates:
                log(2, f"  Fold {fold.fold_id + 1}: No profitable candidates, skipping", sym)
                continue

            # === PLATEAU-BASIERTE AUSWAHL ===
            for c in candidates:
                c["score"] = c["inner_val_pnl"]

            candidates = calculate_param_plateau_score(candidates, grid.tp, grid.sl, grid.ct)
            b = select_best_plateau_candidate(candidates, grid.tp, grid.sl, grid.ct, min_neighbors=2)

            if not b:
                candidates.sort(key=lambda x: x["inner_val_pnl"], reverse=True)
                b = candidates[0] if candidates else None

            if not b:
                log(2, f"  Fold {fold.fold_id + 1}: No plateau candidate found", sym)
                continue

            # === TEST EVALUATION (on this fold's test set) ===
            ct_param = b['params'][2]
            if isinstance(ct_param, tuple):
                ct_str = f"CT_L={ct_param[0]:.2f}/CT_S={ct_param[1]:.2f}"
            else:
                ct_str = f"CT={ct_param:.2f}"

            test_result = evaluate_on_holdout(test_df, train_df, b, ctx)

            if test_result["n_trades"] < ctx.min_trades:
                log(2, f"  Fold {fold.fold_id + 1}: Too few test trades ({test_result['n_trades']})", sym)
                continue

            # Store fold results
            fold_result = {
                "fold_id": fold.fold_id,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "inner_val_pnl": b.get("inner_val_pnl", 0),
                "test_pnl": test_result["pnl"],
                "test_win_rate": test_result["win_rate"],
                "test_trades": test_result["n_trades"],
                "test_trades_trace": test_result["trades"],
                "best_config": {
                    "tp": b["params"][0],
                    "sl": b["params"][1],
                    "ct": b["params"][2],
                    "rrr": b["rrr"],
                },
                "selected_features_long": b.get("selected_features_long", []),
                "selected_features_short": b.get("selected_features_short", []),
            }

            all_fold_results.append(fold_result)

            # Calculate bias ratio
            bias_ratio = fold_result["test_pnl"] / fold_result["inner_val_pnl"] if fold_result["inner_val_pnl"] > 0 else 0

            log(1, f"  Fold {fold.fold_id + 1}: WR={test_result['win_rate']:.1%} "
                   f"PnL={test_result['pnl']:.1f} Trades={test_result['n_trades']} "
                   f"Bias={bias_ratio:.2f}x", sym)

        # === END OF WALK-FORWARD LOOP ===
        # Now aggregate results across all folds

        if len(all_fold_results) == 0:
            log(1, "SKIP - No successful folds", sym)
            report_done(sym, "no_successful_folds")
            return {"symbol": sym, "status": "no_successful_folds", "grid_results": accumulated_grid_results}

        log(1, f"=== Walk-Forward Complete: {len(all_fold_results)}/{len(wf_folds)} successful folds ===", sym)

        # Aggregate metrics across all folds
        win_rates = [r["test_win_rate"] for r in all_fold_results]
        pnls = [r["test_pnl"] for r in all_fold_results]
        trades_counts = [r["test_trades"] for r in all_fold_results]
        bias_ratios = [r["test_pnl"] / r["inner_val_pnl"] if r["inner_val_pnl"] > 0 else 0
                      for r in all_fold_results]

        mean_wr = np.mean(win_rates)
        std_wr = np.std(win_rates)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        total_trades = sum(trades_counts)
        mean_bias_ratio = np.mean(bias_ratios)

        # Detect sample bias
        sample_bias_detected = any(ratio > 2.0 for ratio in bias_ratios if ratio > 0)

        log(1, f"Walk-Forward Results:", sym)
        log(1, f"  Win-Rate: {mean_wr*100:.1f}% ± {std_wr*100:.1f}% (range: {min(win_rates)*100:.1f}%-{max(win_rates)*100:.1f}%)", sym)
        log(1, f"  PnL: {mean_pnl:.1f} ± {std_pnl:.1f} (range: {min(pnls):.1f}-{max(pnls):.1f})", sym)
        log(1, f"  Total Trades: {total_trades}", sym)
        log(1, f"  Bias Ratios: {[f'{r:.2f}x' for r in bias_ratios]}", sym)

        if sample_bias_detected:
            log(1, f"  WARNING: Sample bias detected in some folds (>2x ratio)", sym)

        # Use first fold's best config as representative (they should be similar)
        representative_fold = all_fold_results[0]
        b_config = representative_fold["best_config"]

        # Combine all trades from all folds (needed for all branches)
        all_trades = []
        for fold_result in all_fold_results:
            all_trades.extend(fold_result["test_trades_trace"])

        # Extract binary results for Monte Carlo / Sharpe / Calmar (sign-based)
        all_trades_binary = [t["result"] for t in all_trades]

        # Quick Kelly check for early exit (before Monte Carlo)
        rrr = b_config["rrr"]
        full_kelly = (mean_wr * rrr - (1 - mean_wr)) / rrr if rrr > 0 else 0
        kelly_fraction = strategy.risk_params.get("kelly_fraction", 0.25)
        max_risk = strategy.risk_params.get("max_risk", 0.05)
        fk = max(0, min(max_risk, full_kelly * kelly_fraction))

        if fk <= 0:
            # Build FULL result with best config, even though Kelly is negative
            # User wants to see what was found, even if not profitable
            ct_value = b_config["ct"]
            if isinstance(ct_value, tuple):
                ct_long, ct_short = ct_value
                ct_display = ct_long
            else:
                ct_long = ct_short = ct_value
                ct_display = ct_value

            result = {
                "symbol": sym,
                "status": "no_kelly",
                "pnl": mean_pnl,
                "win_rate": mean_wr,
                "rrr": rrr,
                "sharpe": 0,
                "calmar": 0,
                "tr_trace": all_trades_binary,
                "best_config": {
                    "tp_mult": b_config["tp"],
                    "sl_mult": b_config["sl"],
                    "conf_thresh": ct_display,
                    "ct_long": ct_long,
                    "ct_short": ct_short,
                    "features": representative_fold.get("selected_features_long", []) + representative_fold.get("selected_features_short", []),
                },
                "walk_forward": {
                    "n_folds": len(all_fold_results),
                    "successful_folds": len(all_fold_results),
                    "mean_win_rate": mean_wr,
                    "std_win_rate": std_wr,
                    "min_win_rate": min(win_rates),
                    "max_win_rate": max(win_rates),
                    "mean_pnl": mean_pnl,
                    "std_pnl": std_pnl,
                    "min_pnl": min(pnls),
                    "max_pnl": max(pnls),
                    "total_trades": total_trades,
                    "sample_bias_detected": sample_bias_detected,
                    "bias_ratios": bias_ratios,
                    "mean_bias_ratio": mean_bias_ratio,
                    "fold_details": all_fold_results,
                },
                "reason": f"Kelly <= 0 (WR={mean_wr*100:.1f}%, RRR={rrr:.2f})"
            }

            # Run bias check
            bias_check_result = check_asset_bias(result, verbose=True)
            result["bias_check"] = bias_check_result

            log(1, f"NO_KELLY - WR={mean_wr:.1%}±{std_wr:.1%} RRR={rrr:.2f} "
                   f"TP={b_config['tp']} SL={b_config['sl']} CT={ct_display:.2f} "
                   f"Trades={total_trades}", sym)

            report_done(sym, "no_kelly")
            return result

        # === MONTE CARLO TESTS (on aggregated trades from all folds) ===
        report_phase(sym, "Monte Carlo Validierung...")

        t_mc = time.time()
        mc_perm = monte_carlo_permutation_test(all_trades_binary, n_permutations=1000)
        mc_equity = monte_carlo_equity_simulation(all_trades_binary, fk, rrr, n_simulations=500)

        log(2, f"  Monte Carlo: p={mc_perm['p_value']:.3f}, "
               f"Equity median={mc_equity['median_equity']:.1f}, "
               f"bankruptcy={mc_equity['bankruptcy_rate']:.1%} ({time.time()-t_mc:.1f}s)", sym)

        if not mc_perm["is_significant"]:
            log(1, f"SKIP - Not significant (p={mc_perm['p_value']:.3f})", sym)

            # Build FULL result with best config and Monte Carlo results
            # User wants to see what was found, even if not statistically significant
            bars_per_year = tf_cfg["bars_per_hour"] * 24 * 250
            total_test_bars = sum(r["test_size"] for r in all_fold_results)
            actual_trades_per_year = total_trades * bars_per_year / total_test_bars if total_test_bars > 0 else total_trades
            trade_returns = [fk * rrr if r > 0 else -fk for r in all_trades_binary]
            sharpe = calculate_sharpe_ratio(trade_returns, trades_per_year=actual_trades_per_year)
            calmar = calculate_calmar_ratio(all_trades_binary, fk, rrr)

            ct_value = b_config["ct"]
            if isinstance(ct_value, tuple):
                ct_long, ct_short = ct_value
                ct_display = ct_long
            else:
                ct_long = ct_short = ct_value
                ct_display = ct_value

            result = {
                "symbol": sym,
                "status": "not_significant",
                "pnl": mean_pnl,
                "win_rate": mean_wr,
                "rrr": rrr,
                "sharpe": sharpe,
                "calmar": calmar,
                "kelly_risk": fk,
                "tr_trace": all_trades_binary,
                "best_config": {
                    "kelly_risk": fk,
                    "tp_mult": b_config["tp"],
                    "sl_mult": b_config["sl"],
                    "conf_thresh": ct_display,
                    "ct_long": ct_long,
                    "ct_short": ct_short,
                    "features": representative_fold.get("selected_features_long", []) + representative_fold.get("selected_features_short", []),
                },
                "walk_forward": {
                    "n_folds": len(all_fold_results),
                    "successful_folds": len(all_fold_results),
                    "mean_win_rate": mean_wr,
                    "std_win_rate": std_wr,
                    "min_win_rate": min(win_rates),
                    "max_win_rate": max(win_rates),
                    "mean_pnl": mean_pnl,
                    "std_pnl": std_pnl,
                    "min_pnl": min(pnls),
                    "max_pnl": max(pnls),
                    "total_trades": total_trades,
                    "sample_bias_detected": sample_bias_detected,
                    "bias_ratios": bias_ratios,
                    "mean_bias_ratio": mean_bias_ratio,
                    "fold_details": all_fold_results,
                },
                "monte_carlo": {
                    "p_value": mc_perm["p_value"],
                    "is_significant": False,
                    "percentile": mc_perm["percentile"],
                    "equity_median": mc_equity["median_equity"],
                    "equity_p5": mc_equity["p5_equity"],
                    "equity_p95": mc_equity["p95_equity"],
                    "bankruptcy_rate": mc_equity["bankruptcy_rate"],
                },
                "reason": f"Not statistically significant (p={mc_perm['p_value']:.3f})",
                "grid_results": accumulated_grid_results,
            }

            # Run bias check
            bias_check_result = check_asset_bias(result, verbose=True)
            result["bias_check"] = bias_check_result

            log(1, f"NOT_SIGNIFICANT - WR={mean_wr:.1%}±{std_wr:.1%} Sharpe={sharpe:.2f} "
                   f"p={mc_perm['p_value']:.3f} TP={b_config['tp']} SL={b_config['sl']} "
                   f"CT={ct_display:.2f} Trades={total_trades}", sym)

            report_done(sym, "not_significant")
            return result

        if mc_equity["bankruptcy_rate"] > 0.1:
            log(1, f"WARNING: {mc_equity['bankruptcy_rate']:.1%} Bankruptcy-Rate", sym)

        # === RISK MANAGEMENT PLUGIN ===
        risk_mgr_cls = get_risk_manager(strategy.risk_management)
        risk_mgr = risk_mgr_cls()
        risk_result = risk_mgr.compute_risk_params(
            all_trades_binary, mean_wr, rrr, **strategy.risk_params
        )
        fk = risk_result["kelly_risk"]
        circuit_breaker = risk_result["circuit_breaker"]
        kelly_adjustment = risk_result["kelly_adjustment"]

        if kelly_adjustment["scale_factor"] < 1.0:
            log(2, f"Kelly adjusted: scale_factor={kelly_adjustment['scale_factor']:.2f}", sym)
        if circuit_breaker["enabled"]:
            log(2, f"Circuit Breaker: Pause after {circuit_breaker['pause_after_losses']} losses "
                   f"for {circuit_breaker['pause_bars']} trades", sym)

        # Calculate metrics on aggregated trades
        bars_per_year = tf_cfg["bars_per_hour"] * 24 * 250
        total_test_bars = sum(r["test_size"] for r in all_fold_results)
        actual_trades_per_year = total_trades * bars_per_year / total_test_bars if total_test_bars > 0 else total_trades
        trade_returns = [fk * rrr if r > 0 else -fk for r in all_trades_binary]
        sharpe = calculate_sharpe_ratio(trade_returns, trades_per_year=actual_trades_per_year)
        calmar = calculate_calmar_ratio(all_trades_binary, fk, rrr)

        # CT values
        ct_value = b_config["ct"]
        if isinstance(ct_value, tuple):
            ct_long, ct_short = ct_value
            ct_display = ct_long
        else:
            ct_long = ct_short = ct_value
            ct_display = ct_value

        result = {
            "symbol": sym,
            "status": "ok",
            "pnl": mean_pnl,
            "config": {
                "kelly_risk": fk,
                "point_value": asset.point,
                "spread": ctx.spread,
                "tp_mult": b_config["tp"],
                "sl_mult": b_config["sl"],
                "conf_thresh": ct_display,
                "ct_long": ct_long,
                "ct_short": ct_short,
                "separate_long_short": ctx.separate_long_short,
                "features": representative_fold.get("selected_features_long", []) + representative_fold.get("selected_features_short", []),
                "good_hours": list(range(24)),  # Can be computed from aggregated trades if needed
                "dd_scaling": {"10": 0.5, "20": 0.25},
                "circuit_breaker": circuit_breaker,
                "kelly_adjustment": kelly_adjustment,
            },
            "tr_trace": all_trades_binary,
            "rrr": rrr,
            "win_rate": mean_wr,
            "sharpe": sharpe,
            "calmar": calmar,
            "currencies": asset.currencies,
            # Walk-Forward specific results
            "walk_forward": {
                "n_folds": len(all_fold_results),
                "successful_folds": len(all_fold_results),
                "mean_win_rate": mean_wr,
                "std_win_rate": std_wr,
                "min_win_rate": min(win_rates),
                "max_win_rate": max(win_rates),
                "mean_pnl": mean_pnl,
                "std_pnl": std_pnl,
                "min_pnl": min(pnls),
                "max_pnl": max(pnls),
                "total_trades": total_trades,
                "sample_bias_detected": sample_bias_detected,
                "bias_ratios": bias_ratios,
                "mean_bias_ratio": mean_bias_ratio,
                "fold_details": all_fold_results,
            },
            "monte_carlo": {
                "p_value": mc_perm["p_value"],
                "is_significant": mc_perm["is_significant"],
                "percentile": mc_perm["percentile"],
                "equity_median": mc_equity["median_equity"],
                "equity_p5": mc_equity["p5_equity"],
                "equity_p95": mc_equity["p95_equity"],
                "bankruptcy_rate": mc_equity["bankruptcy_rate"],
            },
            "grid_results": accumulated_grid_results,
        }

        log(1, f"OK (Walk-Forward) - WR={mean_wr:.1%}±{std_wr:.1%} Sharpe={sharpe:.2f} "
               f"p={mc_perm['p_value']:.3f} Trades={total_trades} ({time.time()-t_start:.1f}s)", sym)

        if sample_bias_detected:
            log(1, f"  NOTE: Sample bias detected in {sum(1 for r in bias_ratios if r > 2.0)} folds", sym)

        # === LIVE BIAS CHECK ===
        # Check sofort nach jedem Asset für Echtzeit-Feedback
        bias_check_result = check_asset_bias(result, verbose=True)
        result["bias_check"] = bias_check_result

        report_done(sym, "ok")
        return result

    except Exception as e:
        log(1, f"FEHLER: {e}", sym)
        import traceback
        traceback.print_exc()
        report_done(sym, "error")
        return {"symbol": sym, "status": "error", "error": str(e)}
