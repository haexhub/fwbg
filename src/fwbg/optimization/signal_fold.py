"""Signal model fold processing — no training; the combo grid is chosen on inner windows.

Signal models evaluate signal_rules into composed signal columns.
This module handles their specific data preparation (keep all rows,
fill NaN with 0, preserve auxiliary columns for exit strategies)
and their fold evaluation: the exit strategy combo is chosen on inner
windows of the fold's training data (signal_selection), and only the
chosen combo is evaluated on the fold's test set.
"""
from fwbg.core.columns import named_columns
from fwbg.utils.logging import log
from .nested_cv import evaluate_on_holdout
from .process_fold import _prepare_fold_common, _finalize_fold_data
from .signal_selection import inner_windows, select_combo, signal_combos


def prepare_signal_fold_data(fold, fold_indicators, precomputed_raw_df,
                             preprocessing_configs, ctx, sym,
                             indicator_progress_callback=None):
    """Prepare train/test DataFrames for signal models.

    Unlike ML models, signal models:
    - Keep ALL columns (exit strategies need auxiliary cols like *_range, *_sl_dist)
    - Fill feature NaN with 0 (NaN = "no signal") instead of dropping rows
    - Restrict feature pool to required_features only
    - Auto-resolve signal columns when indicator params change column prefixes
    """
    common = _prepare_fold_common(
        fold, fold_indicators, precomputed_raw_df,
        preprocessing_configs, ctx, sym, indicator_progress_callback,
    )

    train_df = common["train_df"]
    test_df = common["test_df"]
    full_pool = common["full_pool"]

    # Don't drop columns — exit strategies need auxiliary columns
    # (e.g. *_range, *_sl_dist) that have high NaN outside session windows.

    log(2, f"  Fold {fold.fold_id + 1}: {len(full_pool)} clean features "
           f"(excl: {common['excluded_inf']} inf, {common['excluded_nan']} nan)", sym)

    # Add composed signal columns to required features
    has_time_filters = getattr(ctx, "allowed_hours", None) or getattr(ctx, "allowed_days", None)
    signal_rules = getattr(ctx, "signal_rules", None)
    has_signal_rules = signal_rules and any(
        signal_rules.get(d, {}).get("conditions") for d in ("long", "short")
    )
    if has_signal_rules or has_time_filters:
        for direction in ("long", "short"):
            col_name = f"_composed_signal_{direction}"
            if col_name not in ctx.required_features:
                ctx.required_features.append(col_name)

    # Fill feature NaN with 0 (NaN = "no signal") instead of dropping rows.
    # Dropping bars creates gaps that break sequential trade simulation
    # (a trade opened during session may hit TP/SL outside session).
    # Named exit columns keep NaN: a zero stop distance or level is not "no
    # value" to the exit, while NaN makes it fall back to its ATR/distance stop.
    exit_cols = set(named_columns(train_df.columns, ctx.exit_columns or []))
    for col in full_pool:
        if col in exit_cols:
            continue
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0)

    # Only use required_features (signal columns) as the feature pool
    full_pool = [c for c in ctx.required_features if c in train_df.columns]

    return _finalize_fold_data(
        train_df, test_df, full_pool,
        common["orig_train_ohlc"], common["orig_test_ohlc"], fold, sym,
    )


def process_signal_fold(fold, train_df, test_df, full_pool, ctx, sym):
    """Process a walk-forward fold for signal models.

    Signal models have no trainable parameters — they read pre-computed
    signal columns directly.  Skips feature selection and model training.
    Every exit strategy combo is scored on the inner windows of the fold's
    training data; the best mean inner validation PnL is evaluated once on
    the fold's test set.  Without an eligible combo the test set is never
    evaluated and the fold yields no result.

    Returns:
        (fold_result dict or None, grid_results list)
    """
    fold_idx_1based = fold.fold_id + 1
    features = list(full_pool)

    windows = inner_windows(train_df, ctx)
    choice, all_grid_results = select_combo(signal_combos(ctx, features), windows)
    for grid_entry in all_grid_results:
        grid_entry["fold_id"] = fold.fold_id

    if choice is None:
        log(2, f"  Fold {fold_idx_1based}: No inner trades from signal model "
               f"({len(windows)} inner windows)", sym)
        return None, all_grid_results

    combo, inner = choice
    candidate = combo.candidate
    exit_cfg = combo.config
    tp, sl, ct = candidate["params"]
    # A chosen combo without test trades keeps the fold (test_pnl 0):
    # dropping it would let the test set filter folds.
    test_result = evaluate_on_holdout(test_df, train_df, candidate, combo.ctx)

    fold_result = {
        "fold_id": fold.fold_id,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_start": str(fold.test_df.index[0]),
        "test_end": str(fold.test_df.index[-1]),
        "inner_val_pnl": inner["inner_val_pnl"],
        "inner_fold_stability": inner["fold_stability"],
        "test_pnl": test_result["pnl"],
        "test_win_rate": test_result["win_rate"],
        "test_trades": test_result["n_trades"],
        "test_trades_trace": test_result["trades"],
        "test_trades_detail": test_result.get("trades_detailed", []),
        "best_config": {
            "tp": tp, "sl": sl, "ct": ct,
            "rrr": tp / sl if sl > 0 else 0,
            "timeout_bars": candidate["timeout_bars"],
            "model_hyperparameters": candidate["model_hyperparameters"],
            "exit_modifier": exit_cfg.exit_modifier,
            "exit_modifier_params": exit_cfg.exit_modifier_params,
            "entry_modifier": exit_cfg.entry_modifier,
            "entry_modifier_params": exit_cfg.entry_modifier_params,
            "exit_strategy": exit_cfg.name,
            "exit_params": exit_cfg.params,
        },
        "selected_features_long": features,
        "selected_features_short": features,
        # Eligible combos (at least one inner trade), not every combo of the grid.
        "n_candidates": sum(1 for e in all_grid_results if e["inner_trades"] > 0),
    }

    log(1, f"  Fold {fold_idx_1based}: inner PnL={inner['inner_val_pnl']:.1f} "
           f"WR={test_result['win_rate']:.1%} "
           f"PnL={test_result['pnl']:.1f} Trades={test_result['n_trades']}", sym)

    return fold_result, all_grid_results
