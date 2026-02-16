"""
FWBG Pipeline System

A modular plugin system for building trading strategy pipelines.

Phases (in execution order):
1. data_loading - Load raw market data
2. preprocessing - Transform data (e.g., fractional differentiation)
3. indicators - Compute technical indicators
4. feature_selection - Select relevant features
5. labeling - Generate training labels
6. model - Train/predict with ML models
7. validation - Validate strategy performance

Usage:
    from fwbg.pipeline import PipelineRunner, PipelineConfig, get_registry

    # Load strategy config
    config = parse_pipeline_config(strategy_dict)

    # Create runner
    runner = PipelineRunner(get_registry(), config)

    # Validate plugins
    runner.validate()

    # Fit on training data
    runner.fit(train_ctx)

    # Run pipeline
    result = runner.run(test_ctx)
"""

from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.registry import (
    PluginRegistry,
    PluginNotFoundError,
    PluginValidationError,
    get_registry,
    reset_registry,
    get_user_plugins_dir,
    get_core_plugins_dir,
)
from fwbg.pipeline.config import (
    PluginConfig,
    PipelineConfig,
    parse_pipeline_config,
)
from fwbg.pipeline.runner import PipelineRunner
from fwbg.pipeline.features import (
    get_feature_columns,
    compute_regime_bitmask,
    compute_indicator_pool,
    normalize_plugin_name,
    # Parameter plateau (grid search candidate selection)
    calculate_param_plateau_score,
    select_best_plateau_candidate,
)

__all__ = [
    # Context
    "PipelineContext",
    # Base
    "BasePlugin",
    "PluginPhase",
    # Registry
    "PluginRegistry",
    "PluginNotFoundError",
    "PluginValidationError",
    "get_registry",
    "reset_registry",
    "get_user_plugins_dir",
    "get_core_plugins_dir",
    # Config
    "PluginConfig",
    "PipelineConfig",
    "parse_pipeline_config",
    # Runner
    "PipelineRunner",
    # Features
    "get_feature_columns",
    "compute_regime_bitmask",
    "compute_indicator_pool",
    "normalize_plugin_name",
    # Parameter plateau (grid search candidate selection)
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
