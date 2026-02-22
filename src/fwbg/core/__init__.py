"""
Core Module - Kern-Infrastruktur für FWBG.
"""

from .grid_params import ExitConfig, GridParams

from fwbg_sdk.enums import (
    Timeframe,
    AssetClass,
    Symbol,
    Direction,
    SignalType,
)

from .data_sources import (
    # Types
    SourceType,
    LoadResult,
    DataSourceConfig,
    CSVSourceConfig,
    RESTSourceConfig,
    WebSocketSourceConfig,
    DBSourceConfig,
    DataSource,
    # Registration
    register_csv_source,
    register_rest_source,
    register_websocket_source,
    register_db_source,
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
    register_exit_modifier,
    register_feature_selector,
    register_preprocessor,
    register_broker_adapter,
    register_risk_manager,
    register_data_loader,
    register_model,
    # Plugin Getters
    get_indicator,
    get_exit_strategy,
    get_exit_modifier,
    get_feature_selector,
    get_preprocessor,
    get_broker_adapter,
    get_risk_manager,
    get_data_loader,
    get_model,
    # Plugin Listers
    list_indicators,
    list_exit_strategies,
    list_exit_modifiers,
    list_feature_selectors,
    list_preprocessors,
    list_broker_adapters,
    list_risk_managers,
    list_data_loaders,
    list_models,
    # Registries
    INDICATOR_REGISTRY,
    EXIT_STRATEGY_REGISTRY,
    EXIT_MODIFIER_REGISTRY,
    FEATURE_SELECTOR_REGISTRY,
    PREPROCESSOR_REGISTRY,
    BROKER_ADAPTER_REGISTRY,
    RISK_MANAGER_REGISTRY,
    DATA_LOADER_REGISTRY,
    MODEL_REGISTRY,
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
    "LoadResult",
    "DataSourceConfig",
    "CSVSourceConfig",
    "RESTSourceConfig",
    "WebSocketSourceConfig",
    "DBSourceConfig",
    "DataSource",
    # Data Source Registration
    "register_csv_source",
    "register_rest_source",
    "register_websocket_source",
    "register_db_source",
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
    "register_exit_modifier",
    "register_feature_selector",
    "register_preprocessor",
    "register_broker_adapter",
    "register_risk_manager",
    "register_data_loader",
    "register_model",
    # Plugin Getters
    "get_indicator",
    "get_exit_strategy",
    "get_exit_modifier",
    "get_feature_selector",
    "get_preprocessor",
    "get_broker_adapter",
    "get_risk_manager",
    "get_data_loader",
    "get_model",
    # Plugin Listers
    "list_indicators",
    "list_exit_strategies",
    "list_exit_modifiers",
    "list_feature_selectors",
    "list_preprocessors",
    "list_broker_adapters",
    "list_risk_managers",
    "list_data_loaders",
    "list_models",
    # Registries
    "INDICATOR_REGISTRY",
    "EXIT_STRATEGY_REGISTRY",
    "EXIT_MODIFIER_REGISTRY",
    "FEATURE_SELECTOR_REGISTRY",
    "PREPROCESSOR_REGISTRY",
    "BROKER_ADAPTER_REGISTRY",
    "RISK_MANAGER_REGISTRY",
    "DATA_LOADER_REGISTRY",
    "MODEL_REGISTRY",
]
