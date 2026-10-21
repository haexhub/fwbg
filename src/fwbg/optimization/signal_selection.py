"""Signal-model combos: enumeration over exit strategy x model HP x CT."""

import dataclasses
from dataclasses import dataclass
from typing import Any

from fwbg.core.config import ExitStrategyConfig
from fwbg.core.context import SimulationContext


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
