"""Signal-model combos and their selection on inner windows of the training data.

Signal models have no trainable parameters, but the exit strategy x model HP x CT
grid is still a choice. Each combo is scored on the inner windows of the fold's
training window (the same split ML models use) and the best mean inner
validation PnL is chosen; the fold's test set is never looked at here.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fwbg.core.config import ExitStrategyConfig
from fwbg.core.context import SimulationContext

from .nested_cv import evaluate_on_holdout, nested_cv_split

Window = tuple[pd.DataFrame, pd.DataFrame]


@dataclass(frozen=True)
class SignalCombo:
    """One signal-model setting: the candidate, its context and its exit config."""

    candidate: dict[str, Any]
    ctx: SimulationContext
    config: ExitStrategyConfig


def signal_combos(ctx: SimulationContext, features: list[str]) -> list[SignalCombo]:
    """Every exit strategy x model HP variant x CT combo, in config order.

    The combo's ctx carries the exit strategy config for evaluate_on_holdout;
    its candidate carries the model hyperparameters for the model itself.
    """
    combos = []
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

            for ct in exit_cfg.ct or [0.5]:
                candidate = {
                    "params": (tp, sl, ct),
                    "timeout_bars": timeout_bars,
                    "selected_features_long": features,
                    "selected_features_short": features,
                    "model_hyperparameters": merged_hp,
                    "exit_modifier_params": exit_cfg.exit_modifier_params,
                }
                combos.append(SignalCombo(candidate, hp_ctx, exit_cfg))
    return combos


def inner_windows(train_df: pd.DataFrame, ctx: SimulationContext) -> list[Window]:
    """(inner train, validation) windows of the fold's training data, as for ML models."""
    # ponytail: no refit_stateful_features — it dropna()s the inner frames and does not
    # recompute _composed_signal_*; signals keep the fold-level stateful columns.
    split = nested_cv_split(
        train_df,
        holdout_ratio=0.0,
        n_inner_folds=ctx.n_inner_folds,
        embargo_bars=ctx.embargo_bars,
    )
    windows: list[Window] = split["inner_folds"]
    return windows


def score_combo(combo: SignalCombo, windows: list[Window]) -> dict[str, Any]:
    """Grid entry of a combo with its inner validation metrics (a window without trades is 0)."""
    # ponytail: one full evaluate_on_holdout per window (combos x windows per fold);
    # share targets between combos only if the runtime hurts.
    results = [
        evaluate_on_holdout(val_df, inner_train, combo.candidate, combo.ctx)
        for inner_train, val_df in windows
    ]
    pnls = [r["pnl"] for r in results]
    tp, sl, ct = combo.candidate["params"]
    return {
        "exit_strategy": combo.config.name,
        "tp_mult": tp,
        "sl_mult": sl,
        "ct": ct,
        "timeout_bars": combo.candidate["timeout_bars"],
        "exit_params": combo.config.params,
        "exit_modifier_params": combo.config.exit_modifier_params,
        "model_hyperparameters": combo.candidate["model_hyperparameters"],
        "inner_val_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
        "inner_trades": sum(r["n_trades"] for r in results),
        "fold_pnls": pnls,
        "fold_stability": sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0,
    }


def select_combo(
    combos: list[SignalCombo], windows: list[Window]
) -> tuple[tuple[SignalCombo, dict[str, Any]] | None, list[dict[str, Any]]]:
    """Best combo by mean inner validation PnL, and every combo's grid entry.

    Eligible: at least one inner trade. No sign gate; ties keep config order.
    None when no combo is eligible.
    """
    best = None
    entries = []
    for combo in combos:
        entry = score_combo(combo, windows)
        entries.append(entry)
        if entry["inner_trades"] < 1:
            continue
        if best is None or entry["inner_val_pnl"] > best[1]["inner_val_pnl"]:
            best = (combo, entry)
    return best, entries
