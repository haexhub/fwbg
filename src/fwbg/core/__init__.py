"""
Core Module - Kern-Infrastruktur für FWBG.
"""

from .grid_params import ExitConfig, GridParams


def _auto_load_exit_strategies():
    """Load exit strategies from plugins to populate EXIT_STRATEGY_REGISTRY."""
    try:
        from fwbg.plugins import import_plugin_module
        # Import triggers the @register_exit_strategy decorators
        import_plugin_module("fwbg-core", "exit_strategies", "fixed")
        import_plugin_module("fwbg-premium", "exit_strategies", "atr_based")
    except ImportError:
        pass  # Plugins may not be installed


# Defer loading to avoid circular imports
import atexit as _atexit
_exit_strategies_loaded = False


def _ensure_exit_strategies_loaded():
    """Ensure exit strategies are loaded (called lazily)."""
    global _exit_strategies_loaded
    if not _exit_strategies_loaded:
        _auto_load_exit_strategies()
        _exit_strategies_loaded = True

from .enums import (
    Timeframe,
    AssetClass,
    Symbol,
    Direction,
    SignalType,
)

from .data_sources import (
    # Types
    SourceType,
    DataSourceConfig,
    CSVSourceConfig,
    RESTSourceConfig,
    WebSocketSourceConfig,
    DataSource,
    # Registration
    register_csv_source,
    register_rest_source,
    register_websocket_source,
    # Getters
    get_data_source,
    list_data_sources,
    get_all_data_sources,
    set_data_root,
    get_data_root,
)

from .registry import (
    # Discovery
    discover_plugins,
    # Plugin Registration
    register_indicator,
    register_exit_strategy,
    register_feature_selector,
    register_preprocessor,
    register_broker_adapter,
    register_risk_manager,
    # Plugin Getters
    get_indicator,
    get_exit_strategy,
    get_feature_selector,
    get_preprocessor,
    get_broker_adapter,
    get_risk_manager,
    # Plugin Listers
    list_indicators,
    list_exit_strategies,
    list_feature_selectors,
    list_preprocessors,
    list_broker_adapters,
    list_risk_managers,
    # Registries
    INDICATOR_REGISTRY,
    EXIT_STRATEGY_REGISTRY,
    FEATURE_SELECTOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    BROKER_ADAPTER_REGISTRY,
    RISK_MANAGER_REGISTRY,
)

__all__ = [
    # Grid Params
    "ExitConfig",
    "GridParams",
    # Enums
    "Timeframe",
    "AssetClass",
    "Symbol",
    "Direction",
    "SignalType",
    # Data Source Types
    "SourceType",
    "DataSourceConfig",
    "CSVSourceConfig",
    "RESTSourceConfig",
    "WebSocketSourceConfig",
    "DataSource",
    # Data Source Registration
    "register_csv_source",
    "register_rest_source",
    "register_websocket_source",
    # Data Source Getters
    "get_data_source",
    "list_data_sources",
    "get_all_data_sources",
    "set_data_root",
    "get_data_root",
    # Discovery
    "discover_plugins",
    # Plugin Registration
    "register_indicator",
    "register_exit_strategy",
    "register_feature_selector",
    "register_preprocessor",
    "register_broker_adapter",
    "register_risk_manager",
    # Plugin Getters
    "get_indicator",
    "get_exit_strategy",
    "get_feature_selector",
    "get_preprocessor",
    "get_broker_adapter",
    "get_risk_manager",
    # Plugin Listers
    "list_indicators",
    "list_exit_strategies",
    "list_feature_selectors",
    "list_preprocessors",
    "list_broker_adapters",
    "list_risk_managers",
    # Registries
    "INDICATOR_REGISTRY",
    "EXIT_STRATEGY_REGISTRY",
    "FEATURE_SELECTOR_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "BROKER_ADAPTER_REGISTRY",
    "RISK_MANAGER_REGISTRY",
]
