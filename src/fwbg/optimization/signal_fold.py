"""Signal model fold processing — no training, no grid search.

Signal models evaluate signal_rules into composed signal columns.
This module handles their specific data preparation (keep all rows,
fill NaN with 0, preserve auxiliary columns for exit strategies)
and direct evaluation (no inner CV needed).
"""
import dataclasses

from fwbg.utils.logging import log
from .nested_cv import evaluate_on_holdout
from .process_fold import _prepare_fold_common, _finalize_fold_data


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
    signal_rules = getattr(ctx, "signal_rules", None)
    if signal_rules:
        for direction in ("long", "short"):
            rules = signal_rules.get(direction)
            if rules and rules.get("conditions"):
                col_name = f"_composed_signal_{direction}"
                if col_name not in ctx.required_features:
                    ctx.required_features.append(col_name)

    # Fill feature NaN with 0 (NaN = "no signal") instead of dropping rows.
    # Dropping bars creates gaps that break sequential trade simulation
    # (a trade opened during session may hit TP/SL outside session).
    for col in full_pool:
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
    signal columns directly.  Skips feature selection, inner CV, and
    grid search.  Evaluates each exit strategy combo directly on the
    out-of-sample test set.

    Returns:
        (fold_result dict or None, grid_results list)
    """
    fold_idx_1based = fold.fold_id + 1
    features = list(full_pool)

    best_result = None
    best_pnl = float("-inf")
    best_config = None
    all_grid_results = []

    for exit_cfg in ctx.exit_strategies:
        tp = exit_cfg.params.get("tp_mult", 1.0)
        sl = exit_cfg.params.get("sl_mult", 1.0)
        timeout_bars = exit_cfg.params.get("timeout_bars", None)

        combo_ctx = dataclasses.replace(
            ctx,
            exit_strategy=exit_cfg.name,
            exit_params=exit_cfg.params,
            exit_modifier=exit_cfg.exit_modifier,
            exit_modifier_params=exit_cfg.exit_modifier_params,
            entry_modifier=exit_cfg.entry_modifier,
            entry_modifier_params=exit_cfg.entry_modifier_params,
            separate_long_short=bool(exit_cfg.long_ct or exit_cfg.short_ct),
        )

        for model_hp_variant in ctx.grid_model_hyperparameters:
            hp_ctx = combo_ctx
            merged_hp = combo_ctx.model_hyperparameters
            if model_hp_variant is not None:
                merged_hp = {**combo_ctx.model_hyperparameters, **model_hp_variant}
                hp_ctx = dataclasses.replace(combo_ctx, model_hyperparameters=merged_hp)

            ct_list = exit_cfg.ct or [0.5]
            for ct in ct_list:
                candidate = {
                    "params": (tp, sl, ct),
                    "timeout_bars": timeout_bars,
                    "selected_features_long": features,
                    "selected_features_short": features,
                    "model_hyperparameters": merged_hp,
                    "exit_modifier_params": exit_cfg.exit_modifier_params,
                }

                # hp_ctx carries exit strategy config for evaluate_on_holdout;
                # candidate carries model hyperparameters for the model itself.
                test_result = evaluate_on_holdout(test_df, train_df, candidate, hp_ctx)

                grid_entry = {
                    "fold_id": fold.fold_id,
                    "tp_mult": tp, "sl_mult": sl, "ct": ct,
                    "n_trades": test_result["n_trades"],
                    "pnl": test_result["pnl"],
                    "win_rate": test_result["win_rate"],
                }
                all_grid_results.append(grid_entry)

                if test_result["n_trades"] >= 1 and test_result["pnl"] > best_pnl:
                    best_pnl = test_result["pnl"]
                    best_result = test_result
                    best_config = {
                        "tp": tp, "sl": sl, "ct": ct,
                        "rrr": tp / sl if sl > 0 else 0,
                        "timeout_bars": timeout_bars,
                        "model_hyperparameters": merged_hp,
                        "exit_modifier_params": exit_cfg.exit_modifier_params,
                        "exit_strategy": exit_cfg.name,
                        "exit_params": exit_cfg.params,
                    }

    if not best_result:
        log(2, f"  Fold {fold_idx_1based}: No trades from signal model", sym)
        return None, all_grid_results

    fold_result = {
        "fold_id": fold.fold_id,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_start": str(fold.test_df.index[0]),
        "test_end": str(fold.test_df.index[-1]),
        "inner_val_pnl": best_pnl,
        "test_pnl": best_result["pnl"],
        "test_win_rate": best_result["win_rate"],
        "test_trades": best_result["n_trades"],
        "test_trades_trace": best_result["trades"],
        "test_trades_detail": best_result.get("trades_detailed", []),
        "best_config": best_config,
        "selected_features_long": features,
        "selected_features_short": features,
    }

    log(1, f"  Fold {fold_idx_1based}: WR={best_result['win_rate']:.1%} "
           f"PnL={best_result['pnl']:.1f} Trades={best_result['n_trades']}", sym)

    return fold_result, all_grid_results
