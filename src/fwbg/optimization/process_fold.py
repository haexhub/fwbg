"""Walk-forward fold processing — extracted from process.py."""
import time

import numpy as np
import pandas as pd

from fwbg.data.config import MIN_TRADES
from fwbg.core.config import RegimeFilterConfig
from fwbg.pipeline import (
    compute_indicator_pool, get_feature_columns, compute_regime_bitmask,
    calculate_param_plateau_score, select_best_plateau_candidate,
)
from fwbg.pipeline.features import split_indicators_by_stationarity
from fwbg.utils.progress import report_phase, report_meta, report_progress
from fwbg.utils.logging import log
from .nested_cv import nested_cv_split, evaluate_on_holdout
from .grid_search import run_grid_search, select_features


def precompute_indicators(df, strategy, sym):
    """Split indicators by stationarity and precompute raw ones.

    Returns:
        (fold_indicators, precomputed_raw_df, total_indicators)
    """
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
        # Only keep NEW indicator features — exclude base columns
        # (OHLCV, internal, and any columns already in df like macro_*)
        base_cols = set(df.columns) | {'O', 'H', 'L', 'C', 'V'}
        raw_feature_cols = [c for c in precomputed_raw_df.columns
                            if c not in base_cols
                            and not c.startswith('_')]
        precomputed_raw_df = precomputed_raw_df[raw_feature_cols]
        log(1, f"Precomputed {len(raw_indicators)} raw indicators: "
               f"{len(raw_feature_cols)} features ({time.time()-t0:.1f}s)", sym)

    fold_indicators = stationary_indicators if has_preprocessing else []
    total_indicators = len(raw_indicators) + len(fold_indicators)
    log(2, f"Per-fold indicators: {len(fold_indicators)}, Precomputed: {len(raw_indicators)}", sym)
    report_meta(sym, indicator_count=total_indicators)

    return fold_indicators, precomputed_raw_df, total_indicators


def process_single_fold(
    fold, fold_idx, n_folds,
    fold_indicators, precomputed_raw_df, preprocessing_configs,
    grid, ctx, sym, total_indicators,
):
    """Process a single walk-forward fold.

    Handles preprocessing, indicator computation, feature cleaning,
    grid search, plateau selection, and test evaluation.

    Returns:
        (fold_result dict or None, grid_results list)
    """
    log(1, f"=== Processing Fold {fold.fold_id + 1}/{n_folds} ===", sym)
    report_phase(sym, f"Fold {fold.fold_id + 1}/{n_folds}: Computing indicators...")

    def indicator_progress(name, idx, total):
        report_phase(sym, f"Fold {fold.fold_id + 1}: Indicators {name} ({idx}/{total})")

    # === PREPROCESSING ===
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
        from fwbg_sdk import PipelineContext

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
    test_df = test_df.copy()

    # === FEATURE POOL CLEANING (before dropna to prevent all-NaN columns from destroying rows) ===
    full_pool = get_feature_columns(train_df)

    # Entferne Features mit inf/nan (XGBoost verträgt keine inf)
    # Never drop required_features (signal columns for SignalModel) — they are
    # essential for the model even if they have moderate NaN ratios.
    protected_cols = set(ctx.required_features) if ctx.required_features else set()
    clean_pool = []
    excluded_inf = 0
    excluded_nan = 0
    drop_cols = []
    for col in full_pool:
        if col in train_df.columns:
            if col in protected_cols:
                clean_pool.append(col)
                continue
            has_inf = np.isinf(train_df[col]).any()
            nan_ratio = train_df[col].isna().sum() / len(train_df)
            if has_inf:
                excluded_inf += 1
                drop_cols.append(col)
            elif nan_ratio >= 0.1:
                excluded_nan += 1
                drop_cols.append(col)
            else:
                clean_pool.append(col)
    full_pool = clean_pool

    # Also check test set for all-NaN columns (e.g. indicators with warmup > test size)
    for col in list(full_pool):
        if col in test_df.columns and test_df[col].isna().all():
            full_pool.remove(col)
            drop_cols.append(col)
            excluded_nan += 1

    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        test_df = test_df.drop(columns=drop_cols, errors="ignore")

    log(2, f"  Fold {fold.fold_id + 1}: {len(full_pool)} clean features (excl: {excluded_inf} inf, {excluded_nan} nan)", sym)

    # dropna AFTER removing bad columns
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Original-OHLC wiederherstellen (für Targets und Trade-Simulation)
    if orig_train_ohlc:
        for col in ohlc_cols:
            train_df[col] = orig_train_ohlc[col].reindex(train_df.index)
            test_df[col] = orig_test_ohlc[col].reindex(test_df.index)

    log(2, f"  Fold {fold.fold_id + 1}: Train={train_df.shape} Test={test_df.shape} ({time.time()-t0:.1f}s)", sym)

    if len(train_df) < MIN_TRADES * 2:
        log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Train-Daten ({len(train_df)})", sym)
        return None, []

    if len(test_df) < MIN_TRADES:
        log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Test-Daten ({len(test_df)})", sym)
        return None, []

    # Regime-Filter Kombinationen aus Grid (falls definiert)
    regime_filter_combinations = grid.regime_filter_grid.get_combinations()
    n_regime_combos = len(regime_filter_combinations)
    base_combos = ctx.total_grid_combinations()
    total_combos = base_combos * n_regime_combos

    # Update meta with actual feature count (first fold only)
    if fold_idx == 0:
        report_meta(sym, indicator_count=total_indicators, feature_count=len(full_pool),
                    regime_combos=n_regime_combos)

    if len(full_pool) < 5:
        log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Features ({len(full_pool)})", sym)
        return None, []

    # === GRID SEARCH OVER REGIME COMBOS ===
    candidates = []
    all_grid_results = []

    if fold.fold_id == 0:  # Only log once
        n_timeout = len(grid.timeout_bars) if grid.timeout_bars else 1
        n_modifier = len(ctx.grid_exit_modifier_params) if ctx.grid_exit_modifier_params else 1
        log(1, f"Grid-Search: {len(grid.tp)}×{len(grid.sl)}×{n_timeout}×{n_modifier} modifier × {n_regime_combos} Regime = {total_combos} Kombinationen", sym)
        if ctx.min_rrr > 0:
            log(1, f"Min RRR Filter: {ctx.min_rrr}", sym)

    # === NESTED CV: Inner Folds erstellen (auf Train-Daten dieses Folds) ===
    report_phase(sym, f"Fold {fold.fold_id + 1}/{n_folds}: Grid-Search...")
    cv_split = nested_cv_split(train_df, holdout_ratio=0.0, n_inner_folds=ctx.n_inner_folds, embargo_bars=ctx.embargo_bars)
    inner_folds = cv_split["inner_folds"]

    log(2, f"  Fold {fold.fold_id + 1}: Nested CV with {len(inner_folds)} inner folds", sym)

    # === FEATURE SELECTION (einmal pro Fold, regime-unabhängig) ===
    fold_idx_1based = fold.fold_id + 1
    fs_names = [p["name"] for p in (ctx.feature_selection_plugins or [])] or ["none"]
    report_phase(sym, f"Fold {fold_idx_1based}/{n_folds}: Feature Selection [{' > '.join(fs_names)}]...")
    selected_long, selected_short = select_features(
        inner_folds, full_pool, ctx, sym
    )

    if not selected_long and not selected_short:
        log(2, f"  Fold {fold_idx_1based}: No features selected, skipping", sym)
        return None, []

    # === REGIME-FILTER KOMBINATIONEN ===
    # Total inkl. aller Regime-Combos für monoton steigende Progress-Anzeige
    total_grid_combos = base_combos * n_regime_combos
    grid_progress_offset = 0

    for rf_idx, regime_config in enumerate(regime_filter_combinations):
        # Erstelle RegimeFilterConfig aus Kombination
        regime_params = RegimeFilterConfig.from_dict(regime_config)

        # Berechne _regime bitmask SEPARAT für Train und Test (kein Lookahead!)
        train_df["_regime"] = compute_regime_bitmask(train_df, regime_params)
        test_df["_regime"] = compute_regime_bitmask(test_df, regime_params)

        # Update inner_folds mit neuem regime bitmask
        for train_df_fold, val_df_fold in inner_folds:
            train_df_fold["_regime"] = train_df.loc[train_df_fold.index, "_regime"]
            val_df_fold["_regime"] = train_df.loc[val_df_fold.index, "_regime"]

        # Log Regime-Filter Info
        regime_desc = [
            f"{c.column}{c.operator}{c.value}" for c in regime_params.conditions
        ]
        regime_str = " + ".join(regime_desc) if regime_desc else "No Filter"

        if n_regime_combos > 1 and fold.fold_id == 0:  # Only log once
            log(2, f"  Regime {rf_idx+1}/{n_regime_combos}: {regime_str}", sym)

        # Grid-Search mit Progress-Reporting (Offset akkumuliert über Regime-Combos)
        current_offset = grid_progress_offset  # capture for closure

        def grid_progress_callback(grid_count, grid_total, _offset=current_offset):
            report_progress(sym, fold_idx_1based, n_folds, "grid_search",
                           _offset + grid_count, total_grid_combos)

        if n_regime_combos > 1:
            report_phase(sym, f"Fold {fold_idx_1based}/{n_folds}: R{rf_idx+1}/{n_regime_combos} Grid-Search...")
        else:
            report_phase(sym, f"Fold {fold_idx_1based}/{n_folds}: Grid-Search...")
        gs_candidates, gs_grid_results = run_grid_search(
            full_pool, inner_folds,
            grid, ctx, regime_config, sym,
            inner_df=train_df,
            progress_callback=grid_progress_callback,
            preselected_features_long=selected_long,
            preselected_features_short=selected_short,
        )
        candidates.extend(gs_candidates)
        for gr in gs_grid_results:
            gr["fold_id"] = fold.fold_id
        all_grid_results.extend(gs_grid_results)

        grid_progress_offset += base_combos

    log(2, f"  Fold {fold.fold_id + 1}: Grid search done, {len(candidates)} candidates", sym)

    if not candidates:
        log(2, f"  Fold {fold.fold_id + 1}: No profitable candidates, skipping", sym)
        return None, all_grid_results

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
        return None, all_grid_results

    # === TEST EVALUATION (on this fold's test set) ===
    # Apply winning candidate's model_hyperparameters and exit_modifier_params
    # so holdout evaluation uses the same params that won the grid search.
    import dataclasses
    holdout_ctx = ctx
    if b.get("model_hyperparameters") and b["model_hyperparameters"] != ctx.model_hyperparameters:
        holdout_ctx = dataclasses.replace(holdout_ctx, model_hyperparameters=b["model_hyperparameters"])
    if b.get("exit_modifier_params") and b["exit_modifier_params"] != ctx.exit_modifier_params:
        holdout_ctx = dataclasses.replace(holdout_ctx, exit_modifier_params=b["exit_modifier_params"])
    test_result = evaluate_on_holdout(test_df, train_df, b, holdout_ctx)

    if test_result["n_trades"] < 5:
        log(2, f"  Fold {fold.fold_id + 1}: Too few test trades ({test_result['n_trades']})", sym)
        return None, all_grid_results

    if test_result["n_trades"] < ctx.min_trades:
        log(1, f"  Fold {fold.fold_id + 1}: Low trade count ({test_result['n_trades']}/{ctx.min_trades})", sym)

    # Store fold results
    fold_result = {
        "fold_id": fold.fold_id,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_start": str(fold.test_df.index[0]),
        "test_end": str(fold.test_df.index[-1]),
        "inner_val_pnl": b.get("inner_val_pnl", 0),
        "test_pnl": test_result["pnl"],
        "test_win_rate": test_result["win_rate"],
        "test_trades": test_result["n_trades"],
        "test_trades_trace": test_result["trades"],
        "test_trades_detail": test_result.get("trades_detailed", []),
        "best_config": {
            "tp": b["params"][0],
            "sl": b["params"][1],
            "ct": b["params"][2],
            "rrr": b["rrr"],
            "model_hyperparameters": b.get("model_hyperparameters"),
            "exit_modifier_params": b.get("exit_modifier_params"),
        },
        "selected_features_long": b.get("selected_features_long", []),
        "selected_features_short": b.get("selected_features_short", []),
    }

    # Calculate bias ratio
    bias_ratio = fold_result["test_pnl"] / fold_result["inner_val_pnl"] if fold_result["inner_val_pnl"] > 0 else 0

    log(1, f"  Fold {fold.fold_id + 1}: WR={test_result['win_rate']:.1%} "
           f"PnL={test_result['pnl']:.1f} Trades={test_result['n_trades']} "
           f"Bias={bias_ratio:.2f}x", sym)

    return fold_result, all_grid_results
