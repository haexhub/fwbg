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
    from ..adapters.data import DataAdapter
    from ..adapters.execution import ExecutionAdapter

log = logging.getLogger(__name__)

# Globale Registries
INDICATOR_REGISTRY: Dict[str, Type["BaseIndicator"]] = {}
EXIT_STRATEGY_REGISTRY: Dict[str, Type["BaseExitStrategy"]] = {}
FEATURE_SELECTOR_REGISTRY: Dict[str, Type["BaseFeatureSelector"]] = {}
PREPROCESSOR_REGISTRY: Dict[str, Type["BasePreprocessor"]] = {}
DATA_ADAPTER_REGISTRY: Dict[str, Type["DataAdapter"]] = {}
EXECUTION_ADAPTER_REGISTRY: Dict[str, Type["ExecutionAdapter"]] = {}


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


def register_data_adapter(name: str):
    """
    Decorator zum Registrieren eines DataAdapters.

    Beispiel:
        ```python
        from fwbg.core import register_data_adapter
        from fwbg.adapters import DataAdapter

        @register_data_adapter("csv")
        class CSVDataAdapter(DataAdapter):
            ...
        ```
    """
    def decorator(cls):
        DATA_ADAPTER_REGISTRY[name] = cls
        cls.adapter_type = name
        log.debug(f"Registered data adapter: {name}")
        return cls
    return decorator


def register_execution_adapter(name: str):
    """
    Decorator zum Registrieren eines ExecutionAdapters.

    Beispiel:
        ```python
        from fwbg.core import register_execution_adapter
        from fwbg.adapters import ExecutionAdapter

        @register_execution_adapter("ig")
        class IGExecutionAdapter(ExecutionAdapter):
            ...
        ```
    """
    def decorator(cls):
        EXECUTION_ADAPTER_REGISTRY[name] = cls
        cls.adapter_type = name
        log.debug(f"Registered execution adapter: {name}")
        return cls
    return decorator


def discover_plugins():
    """
    Entdeckt und lädt alle installierten Plugins via Entry Points.

    Entry Point Groups:
    - fwbg.indicators
    - fwbg.exit_strategies
    - fwbg.feature_selectors
    - fwbg.preprocessors

    Wird automatisch beim Import von fwbg aufgerufen.
    """
    groups = [
        ("fwbg.indicators", INDICATOR_REGISTRY),
        ("fwbg.exit_strategies", EXIT_STRATEGY_REGISTRY),
        ("fwbg.feature_selectors", FEATURE_SELECTOR_REGISTRY),
        ("fwbg.preprocessors", PREPROCESSOR_REGISTRY),
        ("fwbg.data_adapters", DATA_ADAPTER_REGISTRY),
        ("fwbg.execution_adapters", EXECUTION_ADAPTER_REGISTRY),
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
                    log.warning(f"Failed to load plugin {ep.name}: {e}")
        except Exception as e:
            log.debug(f"No plugins found for {group_name}: {e}")

    total = (
        len(INDICATOR_REGISTRY)
        + len(EXIT_STRATEGY_REGISTRY)
        + len(FEATURE_SELECTOR_REGISTRY)
        + len(PREPROCESSOR_REGISTRY)
        + len(DATA_ADAPTER_REGISTRY)
        + len(EXECUTION_ADAPTER_REGISTRY)
    )

    if total > 0:
        log.info(
            f"Plugins loaded: "
            f"{len(INDICATOR_REGISTRY)} indicators, "
            f"{len(EXIT_STRATEGY_REGISTRY)} exit strategies, "
            f"{len(FEATURE_SELECTOR_REGISTRY)} feature selectors, "
            f"{len(PREPROCESSOR_REGISTRY)} preprocessors, "
            f"{len(DATA_ADAPTER_REGISTRY)} data adapters, "
            f"{len(EXECUTION_ADAPTER_REGISTRY)} execution adapters"
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


def get_data_adapter(name: str) -> Type["DataAdapter"]:
    """
    Gibt DataAdapter-Klasse anhand des Namens zurück.

    Args:
        name: Registrierter Name des DataAdapters

    Returns:
        DataAdapter-Klasse

    Raises:
        ValueError: Wenn DataAdapter nicht gefunden
    """
    if name not in DATA_ADAPTER_REGISTRY:
        available = list(DATA_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown data adapter: '{name}'. "
            f"Available: {available}"
        )
    return DATA_ADAPTER_REGISTRY[name]


def get_execution_adapter(name: str) -> Type["ExecutionAdapter"]:
    """
    Gibt ExecutionAdapter-Klasse anhand des Namens zurück.

    Args:
        name: Registrierter Name des ExecutionAdapters

    Returns:
        ExecutionAdapter-Klasse

    Raises:
        ValueError: Wenn ExecutionAdapter nicht gefunden
    """
    if name not in EXECUTION_ADAPTER_REGISTRY:
        available = list(EXECUTION_ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown execution adapter: '{name}'. "
            f"Available: {available}"
        )
    return EXECUTION_ADAPTER_REGISTRY[name]


def list_data_adapters() -> list:
    """Listet alle registrierten DataAdapters."""
    return list(DATA_ADAPTER_REGISTRY.keys())


def list_execution_adapters() -> list:
    """Listet alle registrierten ExecutionAdapters."""
    return list(EXECUTION_ADAPTER_REGISTRY.keys())
