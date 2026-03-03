"""Indicator Grid expansion for automated indicator parameter search.

Generates strategy variants from an indicator_grid config, allowing the
optimizer to automatically test different indicator parameter combinations
(e.g., different ORB sessions or range_bars) in a single run.
"""
import copy
import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fwbg.core.config import StrategyConfig


def expand_indicator_grid(
    strategy: "StrategyConfig",
) -> list[tuple[str, "StrategyConfig"]]:
    """Expand indicator_grid config into (label, strategy_variant) tuples.

    Each tuple contains a human-readable label and a StrategyConfig with one
    specific indicator parameter combination applied.

    Without indicator_grid (or empty): returns [("base", strategy)] so the
    optimizer runs exactly once with the original config (backward-compat).

    Config format::

        "optimization": {
            "indicator_grid": {
                "opening_range": {
                    "sessions": [[0], [8], [9]],
                    "range_bars": [[8], [12]]
                }
            }
        }

    This produces 3×2 = 6 variants where the opening_range indicator's
    ``sessions`` and ``range_bars`` params are overridden per variant.
    """
    grid = strategy.optimization.indicator_grid
    if not grid:
        return [("base", strategy)]

    indicators = strategy.pipeline.get("indicators", [])

    # Build per-indicator param combinations
    all_indicator_combos: list[list[tuple[str, str, object]]] = []

    for ind_name, param_grid in grid.items():
        # Find this indicator in the pipeline
        ind_idx = _find_indicator_index(indicators, ind_name)
        if ind_idx is None:
            raise ValueError(
                f"indicator_grid references '{ind_name}' but it's not in "
                f"pipeline.indicators. Available: "
                f"{[i.get('name', '?') for i in indicators]}"
            )

        # Build cartesian product of this indicator's param variants
        param_names = list(param_grid.keys())
        param_value_lists = [param_grid[p] for p in param_names]

        for combo in itertools.product(*param_value_lists):
            # Each combo is one set of overrides for this indicator
            overrides = list(zip([ind_name] * len(param_names), param_names, combo))
            all_indicator_combos.append(overrides)

    if not all_indicator_combos:
        return [("base", strategy)]

    # Generate strategy variants
    variants = []
    for combo in all_indicator_combos:
        variant_strategy = _apply_indicator_overrides(strategy, indicators, combo)
        label = _build_label(combo)
        variants.append((label, variant_strategy))

    return variants


def _find_indicator_index(indicators: list[dict], name: str) -> int | None:
    """Find indicator index by name (supports both short and FQN names)."""
    for i, ind in enumerate(indicators):
        ind_name = ind.get("name", "")
        # Match both "opening_range" and "fwbg-core:opening_range"
        if ind_name == name or ind_name.endswith(f":{name}"):
            return i
    return None


def _apply_indicator_overrides(
    strategy: "StrategyConfig",
    indicators: list[dict],
    overrides: list[tuple[str, str, object]],
) -> "StrategyConfig":
    """Create a deep copy of strategy with indicator params overridden."""
    new_strategy = copy.deepcopy(strategy)
    new_indicators = new_strategy.pipeline.get("indicators", [])

    for ind_name, param_name, param_value in overrides:
        idx = _find_indicator_index(new_indicators, ind_name)
        if idx is not None:
            if "params" not in new_indicators[idx]:
                new_indicators[idx]["params"] = {}
            new_indicators[idx]["params"][param_name] = param_value

    return new_strategy


def _build_label(overrides: list[tuple[str, str, object]]) -> str:
    """Build human-readable label from overrides."""
    parts = []
    current_ind = None
    params = []

    for ind_name, param_name, param_value in overrides:
        if ind_name != current_ind:
            if current_ind is not None:
                parts.append(f"{current_ind}({', '.join(params)})")
            current_ind = ind_name
            params = []
        params.append(f"{param_name}={param_value}")

    if current_ind is not None:
        parts.append(f"{current_ind}({', '.join(params)})")

    return " + ".join(parts)
