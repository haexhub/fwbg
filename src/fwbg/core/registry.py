"""Plugin registry with entry-point auto-discovery.

Registries and decorators are owned by fwbg_sdk.
This module adds discovery logic and getter functions with auto-loading.
"""
from importlib.metadata import entry_points
from typing import Dict, Type, TYPE_CHECKING
import logging

# Import registries and decorators from SDK (single source of truth)
from fwbg_sdk.registry import (
    INDICATOR_REGISTRY,
    EXIT_STRATEGY_REGISTRY,
    EXIT_MODIFIER_REGISTRY,
    ENTRY_MODIFIER_REGISTRY,
    FEATURE_SELECTOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    RISK_MANAGER_REGISTRY,
    DATA_LOADER_REGISTRY,
    MODEL_REGISTRY,
    register_indicator,  # noqa: F401 - re-exported via core.__init__
    register_exit_strategy,  # noqa: F401
    register_exit_modifier,  # noqa: F401
    register_feature_selector,  # noqa: F401
    register_preprocessor,  # noqa: F401
    register_risk_manager,  # noqa: F401
    register_data_loader,  # noqa: F401
    register_model,  # noqa: F401
)

if TYPE_CHECKING:
    from fwbg_sdk import (
        BaseIndicator,
        BaseExitStrategy,
        BaseExitModifier,
        BaseEntryModifier,
        BaseFeatureSelector,
        BasePreprocessor,
        BaseRiskManager,
        BaseDataLoader,
        BaseModel,
    )
    from ..adapters.broker import BrokerAdapter

log = logging.getLogger(__name__)

# Tracks whether a full auto_discover() has been completed.
# A single plugin loaded via import_plugin_module() does NOT set this flag.
_plugins_fully_loaded = False

# Broker adapter registry (fwbg-internal, not in SDK)
BROKER_ADAPTER_REGISTRY: Dict[str, Type["BrokerAdapter"]] = {}


def register_broker_adapter(name: str):
    """Decorator to register a broker adapter."""
    def decorator(cls):
        BROKER_ADAPTER_REGISTRY[name] = cls
        cls.adapter_type = name
        log.debug(f"Registered broker adapter: {name}")
        return cls
    return decorator


def discover_plugins():
    """Discover and load pip-installed broker adapter plugins via entry points."""
    groups = [
        ("fwbg.broker_adapters", BROKER_ADAPTER_REGISTRY),
    ]

    for group_name, registry in groups:
        try:
            eps = entry_points(group=group_name)
            for ep in eps:
                try:
                    cls = ep.load()
                    registry[ep.name] = cls
                    cls.name = ep.name
                    log.debug(f"Loaded {group_name}: {ep.name}")
                except Exception as e:
                    log.debug(f"Could not load plugin {ep.name}: {e}")
        except Exception as e:
            log.debug(f"No plugins found for {group_name}: {e}")

    total = (
        len(INDICATOR_REGISTRY)
        + len(EXIT_STRATEGY_REGISTRY)
        + len(FEATURE_SELECTOR_REGISTRY)
        + len(PREPROCESSOR_REGISTRY)
        + len(BROKER_ADAPTER_REGISTRY)
        + len(RISK_MANAGER_REGISTRY)
        + len(DATA_LOADER_REGISTRY)
        + len(MODEL_REGISTRY)
    )

    if total > 0:
        log.info(
            f"Plugins loaded: "
            f"{len(INDICATOR_REGISTRY)} indicators, "
            f"{len(EXIT_STRATEGY_REGISTRY)} exit strategies, "
            f"{len(FEATURE_SELECTOR_REGISTRY)} feature selectors, "
            f"{len(PREPROCESSOR_REGISTRY)} preprocessors, "
            f"{len(BROKER_ADAPTER_REGISTRY)} broker adapters, "
            f"{len(RISK_MANAGER_REGISTRY)} risk managers, "
            f"{len(MODEL_REGISTRY)} models"
        )


def _ensure_plugins_loaded():
    """Trigger full plugin discovery if not yet completed."""
    global _plugins_fully_loaded
    if _plugins_fully_loaded:
        return
    try:
        from fwbg.pipeline.registry import get_registry
        registry = get_registry()
        registry.auto_discover()
        _plugins_fully_loaded = True
    except Exception:
        pass


def get_indicator(name: str) -> Type["BaseIndicator"]:
    """Get indicator class by name, auto-discovering if needed."""
    if name not in INDICATOR_REGISTRY:
        _ensure_plugins_loaded()
    if name not in INDICATOR_REGISTRY:
        available = list(INDICATOR_REGISTRY.keys())
        raise ValueError(f"Unknown indicator: '{name}'. Available: {available}")
    return INDICATOR_REGISTRY[name]


def get_exit_strategy(name: str) -> Type["BaseExitStrategy"]:
    """Get exit strategy class by name, auto-discovering if needed."""
    if not EXIT_STRATEGY_REGISTRY:
        _ensure_plugins_loaded()
    if name not in EXIT_STRATEGY_REGISTRY:
        available = list(EXIT_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown exit strategy: '{name}'. Available: {available}")
    return EXIT_STRATEGY_REGISTRY[name]


def get_exit_modifier(name: str) -> Type["BaseExitModifier"]:
    """Get exit modifier class by name, auto-discovering if needed."""
    if not EXIT_MODIFIER_REGISTRY:
        _ensure_plugins_loaded()
    if name not in EXIT_MODIFIER_REGISTRY:
        available = list(EXIT_MODIFIER_REGISTRY.keys())
        raise ValueError(f"Unknown exit modifier: '{name}'. Available: {available}")
    return EXIT_MODIFIER_REGISTRY[name]


def get_entry_modifier(name: str) -> Type["BaseEntryModifier"]:
    """Get entry modifier class by name, auto-discovering if needed."""
    if not ENTRY_MODIFIER_REGISTRY:
        _ensure_plugins_loaded()
    if name not in ENTRY_MODIFIER_REGISTRY:
        available = list(ENTRY_MODIFIER_REGISTRY.keys())
        raise ValueError(f"Unknown entry modifier: '{name}'. Available: {available}")
    return ENTRY_MODIFIER_REGISTRY[name]


def get_feature_selector(name: str) -> Type["BaseFeatureSelector"]:
    """Get feature selector class by name, auto-discovering if needed."""
    if name not in FEATURE_SELECTOR_REGISTRY:
        _ensure_plugins_loaded()
    if name not in FEATURE_SELECTOR_REGISTRY:
        available = list(FEATURE_SELECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown feature selector: '{name}'. Available: {available}")
    return FEATURE_SELECTOR_REGISTRY[name]


def get_preprocessor(name: str) -> Type["BasePreprocessor"]:
    """Get preprocessor class by name, auto-discovering if needed."""
    if not PREPROCESSOR_REGISTRY:
        _ensure_plugins_loaded()
    if name not in PREPROCESSOR_REGISTRY:
        available = list(PREPROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown preprocessor: '{name}'. Available: {available}")
    return PREPROCESSOR_REGISTRY[name]


def list_indicators() -> list:
    """List all registered indicators."""
    return list(INDICATOR_REGISTRY.keys())


def list_exit_strategies() -> list:
    """List all registered exit strategies."""
    return list(EXIT_STRATEGY_REGISTRY.keys())


def list_exit_modifiers() -> list:
    """List all registered exit modifiers."""
    return list(EXIT_MODIFIER_REGISTRY.keys())


def list_feature_selectors() -> list:
    """List all registered feature selectors."""
    return list(FEATURE_SELECTOR_REGISTRY.keys())


def list_preprocessors() -> list:
    """List all registered preprocessors."""
    return list(PREPROCESSOR_REGISTRY.keys())


def get_broker_adapter(name: str) -> Type["BrokerAdapter"]:
    """Get broker adapter class by name."""
    if name not in BROKER_ADAPTER_REGISTRY:
        available = list(BROKER_ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unknown broker adapter: '{name}'. Available: {available}")
    return BROKER_ADAPTER_REGISTRY[name]


def list_broker_adapters() -> list:
    """List all registered broker adapters."""
    return list(BROKER_ADAPTER_REGISTRY.keys())


def get_risk_manager(name: str) -> Type["BaseRiskManager"]:
    """Get risk manager class by name, auto-discovering if needed."""
    if not RISK_MANAGER_REGISTRY:
        _ensure_plugins_loaded()
    if name not in RISK_MANAGER_REGISTRY:
        available = list(RISK_MANAGER_REGISTRY.keys())
        raise ValueError(f"Unknown risk manager: '{name}'. Available: {available}")
    return RISK_MANAGER_REGISTRY[name]


def list_risk_managers() -> list:
    """List all registered risk managers."""
    return list(RISK_MANAGER_REGISTRY.keys())


def get_data_loader(name: str) -> Type["BaseDataLoader"]:
    """Get data loader class by name, auto-discovering if needed."""
    if not DATA_LOADER_REGISTRY:
        _ensure_plugins_loaded()
    if name not in DATA_LOADER_REGISTRY:
        available = list(DATA_LOADER_REGISTRY.keys())
        raise ValueError(f"Unknown data loader: '{name}'. Available: {available}")
    return DATA_LOADER_REGISTRY[name]


def list_data_loaders() -> list:
    """List all registered data loaders."""
    return list(DATA_LOADER_REGISTRY.keys())


def get_model(name: str) -> Type["BaseModel"]:
    """Get model class by name, auto-discovering if needed."""
    if name not in MODEL_REGISTRY:
        _ensure_plugins_loaded()
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]


def list_models() -> list:
    """List all registered models."""
    return list(MODEL_REGISTRY.keys())
