"""Plugin registration decorators and global registries.

The SDK owns the global registry dicts and decorator functions.
fwbg wraps the getters with plugin discovery logic.
"""
import logging
from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from fwbg_sdk.indicators import BaseIndicator
    from fwbg_sdk.exit_strategies import BaseExitStrategy
    from fwbg_sdk.exit_modifiers import BaseExitModifier
    from fwbg_sdk.entry_modifiers import BaseEntryModifier
    from fwbg_sdk.feature_selectors import BaseFeatureSelector
    from fwbg_sdk.preprocessors import BasePreprocessor
    from fwbg_sdk.risk_managers import BaseRiskManager
    from fwbg_sdk.data_loaders import BaseDataLoader
    from fwbg_sdk.models import BaseModel

log = logging.getLogger(__name__)

# Global registries — populated by decorators at import time
INDICATOR_REGISTRY: Dict[str, Type["BaseIndicator"]] = {}
EXIT_STRATEGY_REGISTRY: Dict[str, Type["BaseExitStrategy"]] = {}
EXIT_MODIFIER_REGISTRY: Dict[str, Type["BaseExitModifier"]] = {}
FEATURE_SELECTOR_REGISTRY: Dict[str, Type["BaseFeatureSelector"]] = {}
PREPROCESSOR_REGISTRY: Dict[str, Type["BasePreprocessor"]] = {}
RISK_MANAGER_REGISTRY: Dict[str, Type["BaseRiskManager"]] = {}
DATA_LOADER_REGISTRY: Dict[str, Type["BaseDataLoader"]] = {}
MODEL_REGISTRY: Dict[str, Type["BaseModel"]] = {}
ENTRY_MODIFIER_REGISTRY: Dict[str, Type["BaseEntryModifier"]] = {}


def register_indicator(name: str):
    """Decorator to register an indicator plugin.

    Example:
        @register_indicator("rsi")
        class RSIIndicator(BaseIndicator):
            ...
    """
    def decorator(cls):
        INDICATOR_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered indicator: {name}")
        return cls
    return decorator


def register_exit_strategy(name: str):
    """Decorator to register an exit strategy plugin.

    Example:
        @register_exit_strategy("atr_based")
        class ATRExitStrategy(BaseExitStrategy):
            ...
    """
    def decorator(cls):
        EXIT_STRATEGY_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered exit strategy: {name}")
        return cls
    return decorator


def register_exit_modifier(name: str):
    """Decorator to register an exit modifier plugin.

    Exit modifiers are optional add-ons that replace the simulation kernel
    of a base exit strategy (e.g., atr_based) with additional logic such as
    trailing stops or breakeven protection.

    Example:
        @register_exit_modifier("trailing_stop")
        class TrailingStopModifier(BaseExitModifier):
            ...
    """
    def decorator(cls):
        EXIT_MODIFIER_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered exit modifier: {name}")
        return cls
    return decorator


def register_entry_modifier(name: str):
    """Decorator to register an entry modifier plugin.

    Entry modifiers are optional add-ons that extend the entry behavior
    of a strategy with additional logic such as scale-in at retracement
    levels or pyramiding.

    Example:
        @register_entry_modifier("scale_in")
        class ScaleInModifier(BaseEntryModifier):
            ...
    """
    def decorator(cls):
        ENTRY_MODIFIER_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered entry modifier: {name}")
        return cls
    return decorator


def register_feature_selector(name: str):
    """Decorator to register a feature selector plugin.

    Example:
        @register_feature_selector("boruta")
        class BorutaSelector(BaseFeatureSelector):
            ...
    """
    def decorator(cls):
        FEATURE_SELECTOR_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered feature selector: {name}")
        return cls
    return decorator


def register_preprocessor(name: str):
    """Decorator to register a preprocessor plugin.

    Example:
        @register_preprocessor("fractional_diff")
        class FracDiffPreprocessor(BasePreprocessor):
            ...
    """
    def decorator(cls):
        PREPROCESSOR_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered preprocessor: {name}")
        return cls
    return decorator


def register_risk_manager(name: str):
    """Decorator to register a risk manager plugin.

    Example:
        @register_risk_manager("kelly")
        class KellyRiskManager(BaseRiskManager):
            ...
    """
    def decorator(cls):
        RISK_MANAGER_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered risk manager: {name}")
        return cls
    return decorator


def register_data_loader(name: str):
    """Decorator to register a data loader plugin.

    Example:
        @register_data_loader("macro_data")
        class MacroDataLoader(BaseDataLoader):
            ...
    """
    def decorator(cls):
        DATA_LOADER_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered data loader: {name}")
        return cls
    return decorator


def register_model(name: str):
    """Decorator to register a model plugin.

    Example:
        @register_model("xgboost")
        class XGBoostModel(BaseModel):
            ...
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered model: {name}")
        return cls
    return decorator
