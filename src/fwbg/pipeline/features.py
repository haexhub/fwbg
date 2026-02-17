"""Feature utilities for the pipeline system."""
from typing import List, Optional
import numpy as np
import pandas as pd


def normalize_plugin_name(name: str) -> str:
    """Normalize a plugin name to fully qualified form.

    Resolves short names ("trend") dynamically from the plugin registry.
    Already qualified names ("fwbg-core:trend") pass through unchanged.
    """
    if ":" in name:
        return name
    from fwbg.pipeline.registry import get_registry
    registry = get_registry()
    registry.auto_discover()
    return registry.resolve_name(name)


def split_indicators_by_stationarity(
    indicators: list,
    has_preprocessing: bool = True,
) -> tuple:
    """
    Split indicator configs into two groups based on benefits_from_stationary.

    Args:
        indicators: List of indicator configs (dicts with 'name' and 'params')
        has_preprocessing: Whether preprocessing is configured

    Returns:
        (stationary_indicators, raw_indicators)
    """
    if not has_preprocessing:
        return [], list(indicators)

    from fwbg.pipeline import get_registry

    registry = get_registry()
    registry.auto_discover()

    stationary = []
    raw = []

    for ind in indicators:
        name = ind["name"] if isinstance(ind, dict) else ind
        fq_name = normalize_plugin_name(name)

        try:
            plugin_cls = registry.get(fq_name)
            if plugin_cls.benefits_from_stationary:
                stationary.append(ind)
            else:
                raw.append(ind)
        except Exception:
            stationary.append(ind)  # Unknown → safe default: per-fold

    return stationary, raw


# =============================================================================
# Feature Utility Functions
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get all feature columns (excluding internal columns).

    Args:
        df: DataFrame with computed features

    Returns:
        List of feature column names
    """
    exclude = {"O", "H", "L", "C", "V", "Volume"}
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]



def compute_regime_bitmask(
    df: pd.DataFrame,
    regime_params=None
) -> np.ndarray:
    """
    Compute regime bitmask from conditions.

    Bitmask encoding (like Linux file permissions):
        Bit 2 (4) = Long allowed
        Bit 1 (2) = Short allowed
        Bit 0 (1) = Sideways allowed (future use)
        7 = all allowed, 6 = Long+Short, 4 = Long only, 2 = Short only, 0 = blocked

    Each condition produces a per-bar bitmask.
    Final result = AND of all condition bitmasks (intersection).

    Args:
        df: DataFrame with indicators
        regime_params: Optional RegimeFilterConfig with conditions list

    Returns:
        int8 numpy array with bitmask values (0-7)
    """
    if regime_params is None or not regime_params.conditions:
        return np.full(len(df), 7, dtype=np.int8)

    result = np.full(len(df), 7, dtype=np.int8)

    for cond in regime_params.conditions:
        if cond.column not in df.columns:
            continue

        col = df[cond.column].values
        ops = {">=": np.greater_equal, "<=": np.less_equal,
               ">": np.greater, "<": np.less}
        op_fn = ops.get(cond.operator)
        if op_fn is None:
            continue

        mask = op_fn(col, cond.value)
        # Where condition is True → directions, False → else_directions
        cond_bitmask = np.where(mask, cond.directions, cond.else_directions).astype(np.int8)
        result = result & cond_bitmask

    return result


# =============================================================================
# High-Level Compute Functions
# =============================================================================

def compute_indicator_pool(
    df: pd.DataFrame,
    indicators: Optional[List] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Compute indicators using the pipeline system.

    Args:
        df: DataFrame with OHLC data (columns: O, H, L, C, optional V)
        indicators: List of indicator configs (dicts or strings).
                   If None, uses all discovered indicator plugins.
        progress_callback: Optional callback(name, idx, total) for progress updates

    Returns:
        DataFrame with computed features
    """
    from fwbg.pipeline import (
        PipelineRunner, PipelineContext, PipelineConfig, PluginConfig, get_registry
    )
    from fwbg_sdk import PluginPhase

    registry = get_registry()
    registry.auto_discover()

    # Determine which indicators to use
    if indicators is None:
        indicator_names = registry.list_plugins(phase=PluginPhase.INDICATORS)
    else:
        indicator_names = indicators

    # Normalize to list of PluginConfig (with short name resolution)
    indicator_configs = []
    for item in indicator_names:
        if isinstance(item, str):
            fq_name = normalize_plugin_name(item)
            indicator_configs.append(PluginConfig(name=fq_name, params={}))
        elif isinstance(item, dict):
            name = item.get("name", "")
            params = item.get("params", {})
            if name:
                fq_name = normalize_plugin_name(name)
                indicator_configs.append(PluginConfig(name=fq_name, params=params))

    # Create pipeline config
    config = PipelineConfig(indicators=indicator_configs)

    # Create runner with progress callback
    def pipeline_progress(phase_name, current, total):
        if progress_callback and phase_name == "indicators":
            if current <= len(indicator_configs):
                name = indicator_configs[current - 1].name
                progress_callback(name, current, total)

    runner = PipelineRunner(registry, config, progress_callback=pipeline_progress)

    # Run pipeline
    ctx = PipelineContext(df=df.copy(), symbol="", asset_class="FOREX")
    result = runner.run(ctx, phases=["indicators"])

    return result.df


# =============================================================================
# Parameter Plateau Re-exports (for grid search candidate selection)
# =============================================================================
# These are NOT feature selection — they score TP/SL/CT parameter candidates.
# Using lazy loading to avoid circular imports.

_plateau_module = None


def _get_plateau_module():
    """Lazy-load plateau module to avoid circular imports."""
    global _plateau_module
    if _plateau_module is None:
        from fwbg.plugins import import_plugin_module
        _plateau_module = import_plugin_module("fwbg-premium", "feature_selection", "plateau")
    return _plateau_module


def calculate_param_plateau_score(*args, **kwargs):
    """Wrapper for plateau.calculate_param_plateau_score."""
    module = _get_plateau_module()
    if module is None:
        raise ImportError("Plugin 'fwbg-premium:plateau' not installed")
    return module.calculate_param_plateau_score(*args, **kwargs)


def select_best_plateau_candidate(*args, **kwargs):
    """Wrapper for plateau.select_best_plateau_candidate."""
    module = _get_plateau_module()
    if module is None:
        raise ImportError("Plugin 'fwbg-premium:plateau' not installed")
    return module.select_best_plateau_candidate(*args, **kwargs)


__all__ = [
    "get_feature_columns",
    "compute_regime_bitmask",
    "compute_indicator_pool",
    "normalize_plugin_name",
    "split_indicators_by_stationarity",
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
