"""Feature utilities for the pipeline system."""

import inspect
import logging
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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


def _indicator_name_params(indicator) -> tuple:
    """(name, params) of a pipeline indicator entry (plugin name or config dict)."""
    if isinstance(indicator, dict):
        return indicator.get("name", ""), indicator.get("params") or {}
    return indicator, {}


def _column_names(method, params: dict) -> List[str]:
    """Call a plugin's column method with *params* when it accepts them."""
    try:
        takes_params = len(inspect.signature(method).parameters) > 0
    except (TypeError, ValueError):
        takes_params = False
    return list(method(params) if takes_params else method())


def indicator_signal_columns(
    name: str, params: Optional[dict] = None, *, include_calendar: bool = False
) -> List[str]:
    """Unshifted decision columns a configured indicator declares.

    These are the columns of ``get_signal_columns(params)``: row ``i`` describes
    the decision at bar ``i``'s close, so as model features they would run one bar
    ahead of the 1-bar convention. Columns that are known before the bar opens
    (``get_calendar_columns``) are left out unless *include_calendar* is set.
    """
    if not name:
        return []
    from fwbg.pipeline import get_registry

    registry = get_registry()
    registry.auto_discover()
    plugin = registry.get(normalize_plugin_name(name))()
    params = params or {}
    if not hasattr(plugin, "get_signal_columns"):
        return []
    columns = _column_names(plugin.get_signal_columns, params)
    if include_calendar or not hasattr(plugin, "get_calendar_columns"):
        return columns
    known_ahead = set(_column_names(plugin.get_calendar_columns, params))
    return [c for c in columns if c not in known_ahead]


def configured_signal_columns(indicators: Optional[Iterable]) -> List[str]:
    """Unshifted decision columns of every configured indicator, without duplicates."""
    found: dict = {}
    for indicator in indicators or []:
        name, params = _indicator_name_params(indicator)
        for col in indicator_signal_columns(name, params):
            found.setdefault(col, None)
    return list(found)


def ml_feature_pool(
    df: pd.DataFrame, indicators: Optional[Iterable] = None
) -> List[str]:
    """Columns of *df* a model may use as features.

    ``get_feature_columns(df)`` minus the unshifted signal columns of *indicators*
    (the strategy's configured indicators). Those columns carry bar ``i``'s own
    close in row ``i``; they stay in the frame for entry rules and exits but are
    not model inputs. ``signal_columns_as_features`` on an indicator offers their
    shifted ``<col>_prev`` copies instead (see ``add_shifted_signal_columns``).
    """
    signal_columns = set(configured_signal_columns(indicators))
    return [c for c in get_feature_columns(df) if c not in signal_columns]


def add_shifted_signal_columns(df: pd.DataFrame, indicators: Iterable) -> pd.DataFrame:
    """Add ``<col>_prev`` (``col.shift(1)``) for the signal columns of opted-in indicators.

    Only indicators whose entry sets ``signal_columns_as_features: true`` are
    touched. The copy follows the 1-bar convention (row ``i`` uses bars <= i-1),
    so it may enter the model pool; the unshifted original never does.

    Raises:
        ValueError: A ``<col>_prev`` column already exists in *df*.
    """
    from fwbg.core.columns import SIGNAL_PREV_SUFFIX, signal_columns_as_features

    copies = {}
    for indicator in indicators:
        if not signal_columns_as_features(indicator):
            continue
        name, params = _indicator_name_params(indicator)
        columns = [c for c in indicator_signal_columns(name, params) if c in df.columns]
        if not columns:
            log.warning(
                "%s=true on indicator '%s', which has no signal columns to copy",
                "signal_columns_as_features",
                name,
            )
        for col in columns:
            prev = f"{col}{SIGNAL_PREV_SUFFIX}"
            if prev in df.columns or prev in copies:
                raise ValueError(
                    f"cannot add shifted signal column '{prev}': the name is already in use"
                )
            copies[prev] = df[col].shift(1)
    if not copies:
        return df
    result = pd.concat([df, pd.DataFrame(copies, index=df.index)], axis=1)
    result.attrs = df.attrs.copy()
    return result


# Regime bitmask constants (like Linux file permissions)
REGIME_LONG = 4  # Bit 2: Long allowed
REGIME_SHORT = 2  # Bit 1: Short allowed


def compute_regime_bitmask(df: pd.DataFrame, regime_params=None) -> np.ndarray:
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

    Raises:
        ValueError: A condition names a column that is not in ``df`` or uses an
            unsupported operator.
    """
    if regime_params is None or not regime_params.conditions:
        return np.full(len(df), 7, dtype=np.int8)

    result = np.full(len(df), 7, dtype=np.int8)

    for cond in regime_params.conditions:
        if cond.column not in df.columns:
            raise ValueError(
                f"Regime filter column '{cond.column}' is not in the frame; "
                "add the indicator that produces it to the pipeline "
                "(a skipped condition would silently leave the filter open)."
            )

        col = df[cond.column].values
        ops = {
            ">=": np.greater_equal,
            "<=": np.less_equal,
            ">": np.greater,
            "<": np.less,
        }
        op_fn = ops.get(cond.operator)
        if op_fn is None:
            raise ValueError(
                f"Regime filter on '{cond.column}' has unsupported operator "
                f"'{cond.operator}'; use one of {sorted(ops)}."
            )

        mask = op_fn(col, cond.value)
        # Where condition is True → directions, False → else_directions
        cond_bitmask = np.where(mask, cond.directions, cond.else_directions).astype(
            np.int8
        )
        result = result & cond_bitmask

    return result


# =============================================================================
# High-Level Compute Functions
# =============================================================================


def compute_indicator_pool(
    df: pd.DataFrame,
    indicators: Optional[List] = None,
    progress_callback=None,
    fit_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute indicators using the pipeline system.

    Args:
        df: DataFrame with OHLC data (columns: O, H, L, C, optional V)
        indicators: List of indicator configs (dicts or strings).
                   If None, uses all discovered indicator plugins.
        progress_callback: Optional callback(name, idx, total) for progress updates
        fit_df: Training frame used to fit stateful indicators before running.
                Stateful indicators explicitly listed without this frame are
                rejected to prevent accidental full-series fitting.

    Returns:
        DataFrame with computed features
    """
    from fwbg.pipeline import (
        PipelineRunner,
        PipelineContext,
        PipelineConfig,
        PluginConfig,
        get_registry,
    )
    from fwbg_sdk import PluginPhase

    registry = get_registry()
    registry.auto_discover()

    # Determine which indicators to use
    if indicators is None:
        # Stateful indicators need a caller-owned training split.  Omitting
        # them from the convenience "all indicators" path keeps dashboards
        # and exploratory callers from fitting on an entire time series.
        indicator_names = [
            name
            for name in registry.list_plugins(phase=PluginPhase.INDICATORS)
            if not registry.get(name).stateful
        ]
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

    stateful_names = [
        config.name
        for config in indicator_configs
        if registry.get(config.name).stateful
    ]
    if stateful_names and fit_df is None:
        raise ValueError(
            "Stateful indicators require fit_df containing training data: "
            + ", ".join(stateful_names)
        )
    if fit_df is not None and stateful_names:
        fit_ctx = PipelineContext(df=fit_df.copy(), symbol="", asset_class="FOREX")
        runner.fit(fit_ctx)

    # Run pipeline
    ctx = PipelineContext(df=df.copy(), symbol="", asset_class="FOREX")
    result = runner.run(ctx, phases=["indicators"])

    if indicators is None:
        return result.df
    return add_shifted_signal_columns(result.df, indicators)


# =============================================================================
# Parameter Plateau Re-exports (for grid search candidate selection)
# =============================================================================
# These are NOT feature selection — they score TP/SL/CT parameter candidates.
# Using lazy loading to avoid circular imports.
# The plateau plugin ships in the optional, separately installed
# ``fwbg-premium`` plugin package; without it these wrappers raise ImportError.

_plateau_module = None


_PLATEAU_MISSING = (
    "Plugin 'fwbg-premium:plateau' is not installed. It is provided by the optional "
    "'fwbg-premium' plugin package, which must be installed separately."
)


def _get_plateau_module():
    """Lazy-load plateau module to avoid circular imports.

    Raises:
        ImportError: If the plugin package providing the plateau plugin is not installed.
    """
    global _plateau_module
    if _plateau_module is None:
        from fwbg.plugins import import_plugin_module

        module = import_plugin_module("fwbg-premium", "feature_selection", "plateau")
        if module is None:
            raise ImportError(_PLATEAU_MISSING)
        _plateau_module = module
    return _plateau_module


def calculate_param_plateau_score(*args, **kwargs):
    """Wrapper for plateau.calculate_param_plateau_score."""
    module = _get_plateau_module()
    return module.calculate_param_plateau_score(*args, **kwargs)


def select_best_plateau_candidate(*args, **kwargs):
    """Wrapper for plateau.select_best_plateau_candidate."""
    module = _get_plateau_module()
    return module.select_best_plateau_candidate(*args, **kwargs)


__all__ = [
    "REGIME_LONG",
    "REGIME_SHORT",
    "get_feature_columns",
    "ml_feature_pool",
    "indicator_signal_columns",
    "configured_signal_columns",
    "add_shifted_signal_columns",
    "compute_regime_bitmask",
    "compute_indicator_pool",
    "normalize_plugin_name",
    "split_indicators_by_stationarity",
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
