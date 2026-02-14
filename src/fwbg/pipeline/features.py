"""Feature utilities for the pipeline system."""
from typing import List, Optional
import pandas as pd


# =============================================================================
# Plugin Names (Fully Qualified)
# =============================================================================

# Core package indicators (free)
CORE_INDICATORS = [
    "fwbg-core:trend",
    "fwbg-core:momentum",
    "fwbg-core:volatility",
    "fwbg-core:time_season",
    "fwbg-core:price_action",
]

# Premium package indicators
PREMIUM_INDICATORS = [
    "fwbg-premium:regime",
    "fwbg-premium:structure",
    "fwbg-premium:risk",
    "fwbg-premium:distribution",
    "fwbg-premium:dynamics",
    "fwbg-premium:cross_features",
    "fwbg-premium:ichimoku",
    "fwbg-premium:multi_timeframe",
    "fwbg-premium:macro_surprise",
    "fwbg-premium:microstructure",
]

# All indicators (core + premium)
ALL_INDICATORS = CORE_INDICATORS + PREMIUM_INDICATORS

# Premium preprocessing
PREMIUM_PREPROCESSING = [
    "fwbg-premium:fractional_diff",
]

# Premium feature selection
PREMIUM_FEATURE_SELECTION = [
    "fwbg-premium:boruta",
    "fwbg-premium:plateau",
]

# Mapping from short name to fully qualified name for convenience
_SHORT_TO_FQ_NAME = {}
for fq in ALL_INDICATORS + PREMIUM_PREPROCESSING + PREMIUM_FEATURE_SELECTION:
    parts = fq.split(":")
    if len(parts) == 2:
        _SHORT_TO_FQ_NAME[parts[1]] = fq


def normalize_plugin_name(name: str) -> str:
    """
    Normalize a plugin name to fully qualified form.

    Accepts both short names ("trend") and fully qualified names ("fwbg-core:trend").
    Short names are resolved using the default package mapping.

    Args:
        name: Plugin name (short or fully qualified)

    Returns:
        Fully qualified plugin name
    """
    if ":" in name:
        return name  # Already fully qualified
    return _SHORT_TO_FQ_NAME.get(name, name)  # Fallback to original if not found


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



def compute_regime_filter(
    df: pd.DataFrame,
    regime_params=None
) -> pd.Series:
    """
    Compute regime filter based on generic conditions.

    Each condition checks a DataFrame column against a threshold using
    a comparison operator. Multiple conditions are AND-combined.

    Args:
        df: DataFrame with indicators
        regime_params: Optional RegimeFilterConfig with conditions list

    Returns:
        Boolean Series (True = trading allowed)
    """
    if regime_params is None or not regime_params.conditions:
        return pd.Series(True, index=df.index)

    regime_ok = pd.Series(True, index=df.index)

    for cond in regime_params.conditions:
        if cond.column not in df.columns:
            continue

        col = df[cond.column]
        if cond.operator == ">=":
            regime_ok = regime_ok & (col >= cond.value)
        elif cond.operator == "<=":
            regime_ok = regime_ok & (col <= cond.value)
        elif cond.operator == ">":
            regime_ok = regime_ok & (col > cond.value)
        elif cond.operator == "<":
            regime_ok = regime_ok & (col < cond.value)

    return regime_ok


# =============================================================================
# High-Level Compute Functions
# =============================================================================

def compute_indicator_pool(
    df: pd.DataFrame,
    indicators: Optional[List] = None,
    include_premium: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Compute indicators using the pipeline system.

    Args:
        df: DataFrame with OHLC data (columns: O, H, L, C, optional V)
        indicators: List of fully qualified indicator names (e.g., ["fwbg-core:trend"]).
                   If None, uses ALL_INDICATORS or CORE_INDICATORS based on include_premium.
        include_premium: Include premium indicators when indicators=None
        progress_callback: Optional callback(name, idx, total) for progress updates

    Returns:
        DataFrame with computed features
    """
    from fwbg.pipeline import (
        PipelineRunner, PipelineContext, PipelineConfig, PluginConfig, get_registry
    )

    registry = get_registry()
    registry.auto_discover()

    # Determine which indicators to use
    if indicators is None:
        indicator_names = ALL_INDICATORS if include_premium else CORE_INDICATORS
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
    # Plugin names
    "CORE_INDICATORS",
    "PREMIUM_INDICATORS",
    "ALL_INDICATORS",
    "PREMIUM_PREPROCESSING",
    # Utility functions
    "get_feature_columns",
    "compute_regime_filter",
    "compute_indicator_pool",
    "normalize_plugin_name",
    "split_indicators_by_stationarity",
    # Parameter plateau (grid search)
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
