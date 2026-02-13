"""Feature utilities for the pipeline system."""
from typing import List, Optional
import numpy as np
import pandas as pd
import ta


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
# Feature Groups Configuration
# =============================================================================

FEATURE_GROUPS = {
    "trend": {
        "name": "Trend Indicators",
        "prefixes": ["trend_"],
    },
    "momentum": {
        "name": "Momentum Indicators",
        "prefixes": ["mom_"],
    },
    "volatility": {
        "name": "Volatility Indicators",
        "prefixes": ["vol_"],
    },
    "regime": {
        "name": "Regime Features",
        "prefixes": ["regime_"],
    },
    "structure": {
        "name": "Structure Features",
        "prefixes": ["fft_", "path_", "conv_", "event_", "vwap_"],
    },
    "risk": {
        "name": "Risk Features",
        "prefixes": ["risk_", "dd_", "cvar_", "vov_", "crash_", "corr_"],
    },
    "price_action": {
        "name": "Price Action Features",
        "prefixes": ["pa_"],
    },
    "time_season": {
        "name": "Time & Season Features",
        "prefixes": ["time_", "season_"],
    },
    "distribution": {
        "name": "Distribution Features",
        "prefixes": ["dist_"],
    },
    "dynamics": {
        "name": "Dynamics Features",
        "prefixes": ["dyn_", "lag_", "accel_"],
    },
    "multi_timeframe": {
        "name": "Multi-Timeframe Features",
        "prefixes": ["mtf_"],
    },
    "cross_features": {
        "name": "Cross Features",
        "prefixes": ["cross_"],
    },
    "ichimoku": {
        "name": "Ichimoku Features",
        "prefixes": ["ichi_"],
    },
    "macro_surprise": {
        "name": "Macro Surprise Features",
        "prefixes": [
            "macro_gap", "macro_total", "macro_overnight", "macro_intraday",
            "macro_range", "macro_return", "macro_is_surprise", "macro_vol_ratio",
            "macro_vol_zscore", "macro_surprise_streak"
        ],
    },
    "microstructure": {
        "name": "Microstructure Features",
        "prefixes": ["micro_"],
    },
    "macro": {
        "name": "Macro Features",
        "prefixes": ["macro_", "sent_"],
    },
    # Combined groups
    "trend_momentum": {
        "name": "Trend + Momentum",
        "prefixes": ["trend_", "mom_"],
    },
    "all_core": {
        "name": "All Core Indicators",
        "prefixes": ["trend_", "mom_", "vol_"],
    },
}


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
    exclude = ["O", "H", "L", "C", "V", "Volume", "_atr", "_regime_ok", "_original_close", "_hurst"]
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def filter_features_by_group(all_features: List[str], group_name: str) -> List[str]:
    """
    Filter features by a feature group from FEATURE_GROUPS.

    Args:
        all_features: List of all available features
        group_name: Name of the group (e.g., "trend", "momentum", "trend_momentum", "all")

    Returns:
        List of features belonging to the group
    """
    # "all" returns all features
    if group_name == "all" or group_name not in FEATURE_GROUPS:
        return all_features

    group = FEATURE_GROUPS[group_name]
    prefixes = group["prefixes"]

    filtered = []
    for feat in all_features:
        for prefix in prefixes:
            if feat.startswith(prefix):
                filtered.append(feat)
                break

    return filtered


def compute_regime_filter(
    df: pd.DataFrame,
    regime_params=None
) -> pd.Series:
    """
    Compute regime filter based on configurable conditions.

    Args:
        df: DataFrame with indicators
        regime_params: Optional RegimeFilterConfig with configuration

    Returns:
        Boolean Series (True = trading allowed)
    """
    # Default: No filter active (all trades allowed)
    adx_min = 0
    vix_max = None
    hurst_min = None
    hurst_max = None

    if regime_params is not None:
        adx_min = regime_params.adx_min if regime_params.adx_enabled else 0
        if regime_params.vix_enabled:
            vix_max = regime_params.vix_max
        if regime_params.hurst_enabled:
            hurst_min = regime_params.hurst_min
            hurst_max = regime_params.hurst_max

    # ADX Filter (adx_min=0 means no filter)
    if adx_min > 0:
        adx_14 = df.get("trend_adx_14", ta.trend.adx(df["H"], df["L"], df["C"], window=14))
        regime_ok = adx_14 >= adx_min
    else:
        regime_ok = pd.Series(True, index=df.index)

    # VIX Filter (only if explicitly configured)
    if vix_max is not None and "sent_vix" in df.columns:
        vix_ok = df["sent_vix"] < vix_max
        regime_ok = regime_ok & vix_ok

    # Hurst Filter
    if hurst_min is not None or hurst_max is not None:
        if "_hurst" not in df.columns:
            close_values = (
                df["_original_close"].values
                if "_original_close" in df.columns
                else df["C"].values
            )
            df["_hurst"] = _compute_rolling_hurst(close_values, window=100, step=10)

        if hurst_min is not None:
            regime_ok = regime_ok & (df["_hurst"] >= hurst_min)
        if hurst_max is not None:
            regime_ok = regime_ok & (df["_hurst"] <= hurst_max)

    return regime_ok


def _compute_rolling_hurst(
    series: np.ndarray,
    window: int = 100,
    step: int = 10
) -> np.ndarray:
    """Compute rolling Hurst exponent for regime filter."""
    n = len(series)
    result = np.full(n, np.nan)

    for i in range(window, n, step):
        segment = series[i - window:i]
        try:
            h = _hurst_exponent(segment)
            result[i] = h
        except Exception:
            pass

    # Forward fill
    result = pd.Series(result).ffill().values
    return result


def _hurst_exponent(series: np.ndarray) -> float:
    """Calculate Hurst exponent using R/S analysis."""
    n = len(series)
    if n < 20:
        return 0.5

    # Calculate returns
    returns = np.diff(np.log(series + 1e-10))

    # R/S analysis
    max_k = min(n // 2, 100)
    rs_list = []
    n_list = []

    for k in range(10, max_k, 5):
        rs_values = []
        for start in range(0, len(returns) - k, k):
            segment = returns[start:start + k]
            mean_seg = np.mean(segment)
            cumdev = np.cumsum(segment - mean_seg)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(segment)
            if s > 0:
                rs_values.append(r / s)

        if rs_values:
            rs_list.append(np.mean(rs_values))
            n_list.append(k)

    if len(rs_list) < 2:
        return 0.5

    # Linear regression in log-log space
    log_n = np.log(n_list)
    log_rs = np.log(rs_list)

    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(slope, 0, 1))


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
# Feature Selection Re-exports
# =============================================================================
# These are re-exported from plugins for convenience
# Using lazy loading to avoid circular imports

_boruta_module = None
_plateau_module = None


def _get_boruta_module():
    """Lazy-load boruta module to avoid circular imports."""
    global _boruta_module
    if _boruta_module is None:
        from fwbg.plugins import import_plugin_module
        _boruta_module = import_plugin_module("fwbg-premium", "feature_selection", "boruta")
    return _boruta_module


def _get_plateau_module():
    """Lazy-load plateau module to avoid circular imports."""
    global _plateau_module
    if _plateau_module is None:
        from fwbg.plugins import import_plugin_module
        _plateau_module = import_plugin_module("fwbg-premium", "feature_selection", "plateau")
    return _plateau_module


def select_features_boruta(*args, **kwargs):
    """Wrapper for boruta.select_features_boruta."""
    module = _get_boruta_module()
    if module is None:
        raise ImportError("Plugin 'fwbg-premium:boruta' not installed")
    return module.select_features_boruta(*args, **kwargs)


def select_plateau_features(*args, **kwargs):
    """Wrapper for plateau.select_plateau_features."""
    module = _get_plateau_module()
    if module is None:
        raise ImportError("Plugin 'fwbg-premium:plateau' not installed")
    return module.select_plateau_features(*args, **kwargs)


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
    "PREMIUM_FEATURE_SELECTION",
    # Feature groups
    "FEATURE_GROUPS",
    # Utility functions
    "get_feature_columns",
    "filter_features_by_group",
    "compute_regime_filter",
    "compute_indicator_pool",
    "normalize_plugin_name",
    "split_indicators_by_stationarity",
    # Feature selection
    "select_features_boruta",
    "select_plateau_features",
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
