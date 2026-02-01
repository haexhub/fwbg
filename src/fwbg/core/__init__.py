"""
Core Module - Kern-Infrastruktur für FWBG.
"""

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
    register_data_source,
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
    register_data_adapter,
    register_execution_adapter,
    # Plugin Getters
    get_indicator,
    get_exit_strategy,
    get_feature_selector,
    get_preprocessor,
    get_data_adapter,
    get_execution_adapter,
    # Plugin Listers
    list_indicators,
    list_exit_strategies,
    list_feature_selectors,
    list_preprocessors,
    list_data_adapters,
    list_execution_adapters,
    # Registries
    INDICATOR_REGISTRY,
    EXIT_STRATEGY_REGISTRY,
    FEATURE_SELECTOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    DATA_ADAPTER_REGISTRY,
    EXECUTION_ADAPTER_REGISTRY,
)

__all__ = [
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
    "register_data_source",
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
    "register_data_adapter",
    "register_execution_adapter",
    # Plugin Getters
    "get_indicator",
    "get_exit_strategy",
    "get_feature_selector",
    "get_preprocessor",
    "get_data_adapter",
    "get_execution_adapter",
    # Plugin Listers
    "list_indicators",
    "list_exit_strategies",
    "list_feature_selectors",
    "list_preprocessors",
    "list_data_adapters",
    "list_execution_adapters",
    # Registries
    "INDICATOR_REGISTRY",
    "EXIT_STRATEGY_REGISTRY",
    "FEATURE_SELECTOR_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "DATA_ADAPTER_REGISTRY",
    "EXECUTION_ADAPTER_REGISTRY",
]
