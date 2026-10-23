"""Causal indicator-variant choice for the walk-forward replay.

The frozen variant (development.json, holdout, deployment, per-fold diagnostics) is
the best over all folds. Folds are anchored, so fold j's inner windows contain the
test periods of the folds before j: a variant chosen over all folds was partly chosen
on fold k's test data. Test fold k is therefore replayed with the variant that is best
over the folds up to k, and with that variant's causal candidate for fold k.
"""

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.utils.logging import log
from .process_fold import precompute_indicators
from .signal_fold import prepare_signal_fold_data
from .unified_simulation import causal_candidates, run_unified_simulation


@dataclass(frozen=True)
class VariantRecord:
    """What the replay needs of an indicator-grid variant; no indicator data."""

    label: str
    strategy: StrategyConfig
    ctx: SimulationContext
    preprocessing_configs: List[Dict[str, Any]]
    fold_results: List[Dict[str, Any]]


_REPLAY_KEYS = (
    "fold_id",
    "inner_val_pnl",
    "best_config",
    "selected_features_long",
    "selected_features_short",
)


def replay_view(fold_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The fold results reduced to what scoring and merging read (no test trades)."""
    return [{key: r[key] for key in _REPLAY_KEYS if key in r} for r in fold_results]


def variant_score(fold_results: Sequence[Dict[str, Any]]) -> Tuple[int, float]:
    """(successful folds, mean inner validation PnL or -inf without any).

    Ranked on inner validation PnL: the test folds must not pick the variant.
    """
    if not fold_results:
        return 0, float("-inf")
    return len(fold_results), float(np.mean([r["inner_val_pnl"] for r in fold_results]))


def prefix_variant_choice(
    wf_folds, variant_fold_results: Sequence[Sequence[Dict[str, Any]]]
) -> List[int]:
    """Per walk-forward fold k, the index of the variant best over its folds up to k.

    Ties go to the lower index, like the strict ``>`` of the all-fold choice. Each
    prefix is scored in fold_id order, so the result does not depend on arrival order.
    """
    ordered = [sorted(results, key=lambda r: r["fold_id"]) for results in variant_fold_results]
    choice = []
    for fold in wf_folds:
        scores = [
            variant_score([r for r in results if r["fold_id"] <= fold.fold_id])
            for results in ordered
        ]
        choice.append(max(range(len(scores)), key=scores.__getitem__))
    return choice


def replay_prefix_variants(
    wf_folds,
    variants: Sequence[VariantRecord],
    frozen_index: int,
    frozen_indicators,
    frozen_raw_df,
    df,
    sym: str,
) -> List[Dict[str, Any]]:
    """Replay each fold once, with the variant best up to it; results in fold_id order.

    The frozen variant replays with its own indicator data. Any other chosen variant
    recomputes its indicators and releases them before the next one, so no more than
    the frozen variant's data plus one other variant's is held at a time.
    """
    choice = prefix_variant_choice(wf_folds, [variant.fold_results for variant in variants])
    # The last prefix holds every fold, so it must pick the frozen variant; anything
    # else means the two rankings diverged and the replay would not end on the frozen one.
    if choice and choice[-1] != frozen_index:
        raise ValueError(
            f"last fold chose variant {choice[-1]}, but the frozen variant is {frozen_index}"
        )
    if len(set(choice)) > 1:
        mapping = " ".join(
            f"{fold.fold_id + 1}:{variants[index].label}" for fold, index in zip(wf_folds, choice)
        )
        log(1, f"  Replay variants per fold: {mapping}", sym)
    results: List[Dict[str, Any]] = []
    for index in sorted(set(choice)):
        variant = variants[index]
        picked = [k for k, chosen in enumerate(choice) if chosen == index]
        all_candidates = causal_candidates(wf_folds, variant.fold_results)
        folds = [wf_folds[k] for k in picked]
        candidates: List[Optional[Dict[str, Any]]] = [all_candidates[k] for k in picked]

        if index == frozen_index:
            indicators, raw_df = frozen_indicators, frozen_raw_df
        elif all(candidate is None for candidate in candidates):
            indicators, raw_df = None, None  # untraded folds prepare no data
        else:
            # ponytail: a non-frozen variant recomputes its indicators once for the
            # replay (memory over CPU, as the variant loop does); caching them would
            # hold every variant's data at once.
            indicators, raw_df, _ = precompute_indicators(df, variant.strategy, sym)

        prepare = prepare_signal_fold_data if variant.ctx.model_type == "signal" else None
        results.extend(
            run_unified_simulation(
                folds,
                candidates,
                indicators,
                raw_df,
                variant.preprocessing_configs,
                variant.ctx,
                sym,
                prepare_data_fn=prepare,
            )
        )
        del indicators, raw_df
        gc.collect()
    return sorted(results, key=lambda r: r["fold_id"])
