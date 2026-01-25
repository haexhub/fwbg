"""
Optimizer Module für Walk-Forward Strategie-Optimierung
"""
from .config import (
    ACCOUNT_NAME, DATA_PATH, BASE_PATH, EXPORT_FILE, PLOT_PATH,
    TIMEFRAME, WALK_FORWARD_FOLDS, OOS_SIZE, MAX_TRADE_BARS,
    ASSET_CONFIG, CLASS_GRIDS, MIN_TRADES, CORR_THRESHOLD,
    get_asset_config, convert_numpy
)

from .data_loader import (
    load_data_aligned,
    load_macro_indicators,
    load_interest_rates
)

from .indicators import (
    compute_indicator_pool,
    get_feature_columns,
    compute_regime_filter
)

from .simulation import (
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_annual_return,
    check_feature_stability,
    simulate_pro_trade
)

from .process import (
    walk_forward_split,
    process_symbol
)

from .main import (
    filter_correlated_assets,
    run_optimizer,
    show_runs,
    show_comparison,
    main
)

from .results import (
    generate_run_id,
    create_run_directory,
    save_run_results,
    list_runs,
    load_run,
    compare_runs,
    create_strategy_metadata
)

from .resource_manager import (
    AdaptivePoolManager,
    get_resource_info,
    calculate_safe_workers
)

__all__ = [
    # Config
    'ACCOUNT_NAME', 'DATA_PATH', 'BASE_PATH', 'EXPORT_FILE', 'PLOT_PATH',
    'TIMEFRAME', 'WALK_FORWARD_FOLDS', 'OOS_SIZE', 'MAX_TRADE_BARS',
    'ASSET_CONFIG', 'CLASS_GRIDS', 'MIN_TRADES', 'CORR_THRESHOLD',
    'get_asset_config', 'convert_numpy',
    # Data
    'load_data_aligned', 'load_macro_indicators', 'load_interest_rates',
    # Indicators
    'compute_indicator_pool', 'get_feature_columns', 'compute_regime_filter',
    # Simulation
    'calculate_sharpe_ratio', 'calculate_calmar_ratio', 'calculate_max_drawdown',
    'calculate_annual_return', 'check_feature_stability', 'simulate_pro_trade',
    # Process
    'walk_forward_split', 'process_symbol',
    # Main
    'filter_correlated_assets', 'run_optimizer', 'show_runs', 'show_comparison', 'main',
    # Results
    'generate_run_id', 'create_run_directory', 'save_run_results',
    'list_runs', 'load_run', 'compare_runs', 'create_strategy_metadata',
    # Resource Manager
    'AdaptivePoolManager', 'get_resource_info', 'calculate_safe_workers',
]
