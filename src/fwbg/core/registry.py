"""
Plugin Registry mit Entry-Point Auto-Discovery.

Lädt Plugins automatisch via Python Entry Points (pip install).
"""
from importlib.metadata import entry_points
from typing import Dict, Type, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..plugins.indicator import BaseIndicator
    from ..plugins.exit_strategy import BaseExitStrategy
    from ..plugins.feature_selector import BaseFeatureSelector
    from ..plugins.preprocessor import BasePreprocessor
    from ..plugins.risk_manager import BaseRiskManager
    from ..plugins.data_loader import BaseDataLoader
    from ..adapters.broker import BrokerAdapter

log = logging.getLogger(__name__)

# Globale Registries
INDICATOR_REGISTRY: Dict[str, Type["BaseIndicator"]] = {}
EXIT_STRATEGY_REGISTRY: Dict[str, Type["BaseExitStrategy"]] = {}
FEATURE_SELECTOR_REGISTRY: Dict[str, Type["BaseFeatureSelector"]] = {}
PREPROCESSOR_REGISTRY: Dict[str, Type["BasePreprocessor"]] = {}
BROKER_ADAPTER_REGISTRY: Dict[str, Type["BrokerAdapter"]] = {}
RISK_MANAGER_REGISTRY: Dict[str, Type["BaseRiskManager"]] = {}
DATA_LOADER_REGISTRY: Dict[str, Type["BaseDataLoader"]] = {}


def register_indicator(name: str):
    """
    Decorator zum Registrieren eines Indicators.

    Beispiel:
        ```python
        from fwbg.core import register_indicator
        from fwbg.plugins import BaseIndicator

        @register_indicator("rsi")
        class RSIIndicator(BaseIndicator):
            ...
        ```
    """
    def decorator(cls):
        INDICATOR_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered indicator: {name}")
        return cls
    return decorator


def register_exit_strategy(name: str):
    """
    Decorator zum Registrieren einer Exit-Strategie.

    Beispiel:
        ```python
        from fwbg.core import register_exit_strategy
        from fwbg.plugins import BaseExitStrategy

        @register_exit_strategy("atr_based")
        class ATRExitStrategy(BaseExitStrategy):
            ...
        ```
    """
    def decorator(cls):
        EXIT_STRATEGY_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered exit strategy: {name}")
        return cls
    return decorator


def register_feature_selector(name: str):
    """
    Decorator zum Registrieren eines Feature-Selectors.

    Beispiel:
        ```python
        from fwbg.core import register_feature_selector
        from fwbg.plugins import BaseFeatureSelector

        @register_feature_selector("boruta")
        class BorutaSelector(BaseFeatureSelector):
            ...
        ```
    """
    def decorator(cls):
        FEATURE_SELECTOR_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered feature selector: {name}")
        return cls
    return decorator


def register_preprocessor(name: str):
    """
    Decorator zum Registrieren eines Preprocessors.

    Beispiel:
        ```python
        from fwbg.core import register_preprocessor
        from fwbg.plugins import BasePreprocessor

        @register_preprocessor("fractional_diff")
        class FracDiffPreprocessor(BasePreprocessor):
            ...
        ```
    """
    def decorator(cls):
        PREPROCESSOR_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered preprocessor: {name}")
        return cls
    return decorator


def register_broker_adapter(name: str):
    """
    Decorator zum Registrieren eines BrokerAdapters.

    Beispiel:
        ```python
        from fwbg.core import register_broker_adapter
        from fwbg.adapters import BrokerAdapter

        @register_broker_adapter("ig")
        class IGBrokerAdapter(BrokerAdapter):
            ...
        ```
    """
    def decorator(cls):
        BROKER_ADAPTER_REGISTRY[name] = cls
        cls.adapter_type = name
        log.debug(f"Registered broker adapter: {name}")
        return cls
    return decorator


def register_risk_manager(name: str):
    """
    Decorator zum Registrieren eines Risk Managers.

    Beispiel:
        ```python
        from fwbg.core import register_risk_manager
        from fwbg.plugins import BaseRiskManager

        @register_risk_manager("kelly")
        class KellyRiskManager(BaseRiskManager):
            ...
        ```
    """
    def decorator(cls):
        RISK_MANAGER_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered risk manager: {name}")
        return cls
    return decorator


def discover_plugins():
    """
    Entdeckt und lädt pip-installierte Broker-Adapter Plugins via Entry Points.

    Entry Point Groups:
    - fwbg.broker_adapters
    """
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
                    # Debug statt warning - fehlende Module sind OK
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
    )

    if total > 0:
        log.info(
            f"Plugins loaded: "
            f"{len(INDICATOR_REGISTRY)} indicators, "
            f"{len(EXIT_STRATEGY_REGISTRY)} exit strategies, "
            f"{len(FEATURE_SELECTOR_REGISTRY)} feature selectors, "
            f"{len(PREPROCESSOR_REGISTRY)} preprocessors, "
            f"{len(BROKER_ADAPTER_REGISTRY)} broker adapters, "
            f"{len(RISK_MANAGER_REGISTRY)} risk managers"
        )


def get_indicator(name: str) -> Type["BaseIndicator"]:
    """
    Gibt Indicator-Klasse anhand des Namens zurück.

    Args:
        name: Registrierter Name des Indicators

    Returns:
        Indicator-Klasse

    Raises:
        ValueError: Wenn Indicator nicht gefunden
    """
    if name not in INDICATOR_REGISTRY:
        available = list(INDICATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown indicator: '{name}'. "
            f"Available: {available}"
        )
    return INDICATOR_REGISTRY[name]


def get_exit_strategy(name: str) -> Type["BaseExitStrategy"]:
    """
    Gibt Exit-Strategy-Klasse anhand des Namens zurück.

    Args:
        name: Registrierter Name der Exit-Strategy

    Returns:
        Exit-Strategy-Klasse

    Raises:
        ValueError: Wenn Exit-Strategy nicht gefunden
    """
    # Lazy-load exit strategies from plugins if not yet loaded
    if not EXIT_STRATEGY_REGISTRY:
        try:
            from fwbg.plugins import import_plugin_module
            # Import triggers @register_exit_strategy decorator
            import_plugin_module("fwbg-core", "exit_strategies", "fixed")
            import_plugin_module("fwbg-premium", "exit_strategies", "atr_based")
        except ImportError:
            pass

    if name not in EXIT_STRATEGY_REGISTRY:
        available = list(EXIT_STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown exit strategy: '{name}'. "
            f"Available: {available}"
        )
    return EXIT_STRATEGY_REGISTRY[name]


def get_feature_selector(name: str) -> Type["BaseFeatureSelector"]:
    """
    Gibt Feature-Selector-Klasse anhand des Namens zurück.

    Args:
        name: Registrierter Name des Feature-Selectors

    Returns:
        Feature-Selector-Klasse

    Raises:
        ValueError: Wenn Feature-Selector nicht gefunden
    """
    # Lazy-load feature selection plugins if not yet loaded
    if not FEATURE_SELECTOR_REGISTRY:
        try:
            from fwbg.plugins import import_plugin_module
            import_plugin_module("fwbg-premium", "feature_selection", "boruta")
            import_plugin_module("fwbg-premium", "feature_selection", "plateau")
        except ImportError:
            pass

    if name not in FEATURE_SELECTOR_REGISTRY:
        available = list(FEATURE_SELECTOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown feature selector: '{name}'. "
            f"Available: {available}"
        )
    return FEATURE_SELECTOR_REGISTRY[name]


def get_preprocessor(name: str) -> Type["BasePreprocessor"]:
    """
    Gibt Preprocessor-Klasse anhand des Namens zurück.

    Args:
        name: Registrierter Name des Preprocessors

    Returns:
        Preprocessor-Klasse

    Raises:
        ValueError: Wenn Preprocessor nicht gefunden
    """
    if name not in PREPROCESSOR_REGISTRY:
        available = list(PREPROCESSOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown preprocessor: '{name}'. "
            f"Available: {available}"
        )
    return PREPROCESSOR_REGISTRY[name]


def list_indicators() -> list:
    """Listet alle registrierten Indicators."""
    return list(INDICATOR_REGISTRY.keys())


def list_exit_strategies() -> list:
    """Listet alle registrierten Exit-Strategies."""
    return list(EXIT_STRATEGY_REGISTRY.keys())


def list_feature_selectors() -> list:
    """Listet alle registrierten Feature-Selectors."""
    return list(FEATURE_SELECTOR_REGISTRY.keys())


def list_preprocessors() -> list:
    """Listet alle registrierten Preprocessors."""
    return list(PREPROCESSOR_REGISTRY.keys())


def get_broker_adapter(name: str) -> Type["BrokerAdapter"]:
    """
    Gibt BrokerAdapter-Klasse anhand des Namens zurück.

    Args:
        name: Registrierter Name des BrokerAdapters

    Returns:
        BrokerAdapter-Klasse

    Raises:
        ValueError: Wenn BrokerAdapter nicht gefunden
    """
    if name not in BROKER_ADAPTER_REGISTRY:
        available = list(BROKER_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown broker adapter: '{name}'. "
            f"Available: {available}"
        )
    return BROKER_ADAPTER_REGISTRY[name]


def list_broker_adapters() -> list:
    """Listet alle registrierten BrokerAdapters."""
    return list(BROKER_ADAPTER_REGISTRY.keys())


def get_risk_manager(name: str) -> Type["BaseRiskManager"]:
    """Gibt Risk-Manager-Klasse anhand des Namens zurück."""
    if not RISK_MANAGER_REGISTRY:
        try:
            from fwbg.plugins import import_plugin_module
            import_plugin_module("fwbg-core", "risk_management", "kelly")
        except ImportError:
            pass

    if name not in RISK_MANAGER_REGISTRY:
        available = list(RISK_MANAGER_REGISTRY.keys())
        raise ValueError(
            f"Unknown risk manager: '{name}'. Available: {available}"
        )
    return RISK_MANAGER_REGISTRY[name]


def list_risk_managers() -> list:
    """Listet alle registrierten Risk-Manager."""
    return list(RISK_MANAGER_REGISTRY.keys())


def register_data_loader(name: str):
    """Decorator zum Registrieren eines DataLoaders."""
    def decorator(cls):
        DATA_LOADER_REGISTRY[name] = cls
        cls.name = name
        log.debug(f"Registered data loader: {name}")
        return cls
    return decorator


def get_data_loader(name: str) -> Type["BaseDataLoader"]:
    """Gibt DataLoader-Klasse anhand des Namens zurück."""
    if not DATA_LOADER_REGISTRY:
        try:
            from fwbg.plugins import import_plugin_module
            import_plugin_module("fwbg-premium", "data_loading", "macro_data")
        except ImportError:
            pass

    if name not in DATA_LOADER_REGISTRY:
        available = list(DATA_LOADER_REGISTRY.keys())
        raise ValueError(
            f"Unknown data loader: '{name}'. Available: {available}"
        )
    return DATA_LOADER_REGISTRY[name]


def list_data_loaders() -> list:
    """Listet alle registrierten DataLoader."""
    return list(DATA_LOADER_REGISTRY.keys())
