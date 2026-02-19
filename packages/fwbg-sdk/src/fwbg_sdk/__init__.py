"""FWBG SDK - Build plugins for the FWBG trading framework.

Flat namespace: import everything from fwbg_sdk directly.

    from fwbg_sdk import BaseIndicator, shift_features, register_indicator
"""

# Base plugin system
from fwbg_sdk.base import BasePlugin, PluginPhase

# Contexts
from fwbg_sdk.contexts import PipelineContext, AssetInfo

# Enums
from fwbg_sdk.enums import Timeframe, AssetClass, Symbol, Direction, SignalType

# Plugin base classes
from fwbg_sdk.indicators import BaseIndicator, shift_features, safe_divide, EPSILON
from fwbg_sdk.preprocessors import BasePreprocessor
from fwbg_sdk.feature_selectors import BaseFeatureSelector
from fwbg_sdk.exit_strategies import BaseExitStrategy
from fwbg_sdk.risk_managers import BaseRiskManager
from fwbg_sdk.data_loaders import BaseDataLoader

# Model plugin
from fwbg_sdk.models import BaseModel, TrainingContext, ModelProgressReporter

# Documentation validation
from fwbg_sdk.docs import DocsValidationResult, DocsViolation, validate_plugin_docs

# Testing utilities
from fwbg_sdk.testing import (
    create_sample_ohlcv,
    assert_features_shifted,
    assert_no_inf,
    create_sample_asset,
)

# Registration decorators
from fwbg_sdk.registry import (
    register_indicator,
    register_exit_strategy,
    register_feature_selector,
    register_preprocessor,
    register_risk_manager,
    register_data_loader,
    register_model,
    # Registries (for advanced use / fwbg internals)
    INDICATOR_REGISTRY,
    EXIT_STRATEGY_REGISTRY,
    FEATURE_SELECTOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    RISK_MANAGER_REGISTRY,
    DATA_LOADER_REGISTRY,
    MODEL_REGISTRY,
)

__all__ = [
    # Base
    "BasePlugin",
    "PluginPhase",
    # Contexts
    "PipelineContext",
    "AssetInfo",
    # Enums
    "Timeframe",
    "AssetClass",
    "Symbol",
    "Direction",
    "SignalType",
    # Plugin base classes
    "BaseIndicator",
    "BasePreprocessor",
    "BaseFeatureSelector",
    "BaseExitStrategy",
    "BaseRiskManager",
    "BaseDataLoader",
    "BaseModel",
    "TrainingContext",
    "ModelProgressReporter",
    # Indicator helpers
    "shift_features",
    "safe_divide",
    "EPSILON",
    # Registration decorators
    "register_indicator",
    "register_exit_strategy",
    "register_feature_selector",
    "register_preprocessor",
    "register_risk_manager",
    "register_data_loader",
    "register_model",
    # Documentation validation
    "DocsValidationResult",
    "DocsViolation",
    "validate_plugin_docs",
    # Testing utilities
    "create_sample_ohlcv",
    "assert_features_shifted",
    "assert_no_inf",
    "create_sample_asset",
    # Registries
    "INDICATOR_REGISTRY",
    "EXIT_STRATEGY_REGISTRY",
    "FEATURE_SELECTOR_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "RISK_MANAGER_REGISTRY",
    "DATA_LOADER_REGISTRY",
    "MODEL_REGISTRY",
]
