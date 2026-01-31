"""
Technische Indikatoren für den Optimizer.

Dieses Paket enthält alle Indikator-Berechnungen, aufgeteilt in Module:
- preprocessing: Fractional Differentiation, Log-Returns, Z-Score
- regime: Hurst-Exponent, Regime-Filter
- structure: FFT, Path Efficiency, Convexity, Event Features, VWAP
- risk: Drawdown, CVaR, Vol-of-Vol, Crash Probability, Correlation
- core: Haupt-Funktion compute_indicator_pool()
"""

# Preprocessing
from .preprocessing import (
    apply_preprocessing,
    apply_frac_diff_preprocessing,
    apply_log_returns_preprocessing,
    apply_normalize_preprocessing,
    frac_diff,
    get_frac_diff_weights,
    find_min_d_for_stationarity,
)

# Regime
from .regime import (
    compute_hurst_exponent,
    compute_rolling_hurst,
    compute_regime_filter,
    compute_regime_features,
)

# Structure
from .structure import (
    compute_fft_features,
    compute_event_features,
    compute_path_efficiency,
    compute_convexity_features,
    compute_vwap_features,
)

# Risk
from .risk import (
    compute_drawdown_features,
    compute_cvar_features,
    compute_vol_of_vol_features,
    compute_crash_probability_features,
    compute_correlation_features,
)

# Core
from .core import (
    compute_indicator_pool,
    get_feature_columns,
    filter_features_by_group,
)

__all__ = [
    # Preprocessing
    "apply_preprocessing",
    "apply_frac_diff_preprocessing",
    "apply_log_returns_preprocessing",
    "apply_normalize_preprocessing",
    "frac_diff",
    "get_frac_diff_weights",
    "find_min_d_for_stationarity",
    # Regime
    "compute_hurst_exponent",
    "compute_rolling_hurst",
    "compute_regime_filter",
    "compute_regime_features",
    # Structure
    "compute_fft_features",
    "compute_event_features",
    "compute_path_efficiency",
    "compute_convexity_features",
    "compute_vwap_features",
    # Risk
    "compute_drawdown_features",
    "compute_cvar_features",
    "compute_vol_of_vol_features",
    "compute_crash_probability_features",
    "compute_correlation_features",
    # Core
    "compute_indicator_pool",
    "get_feature_columns",
    "filter_features_by_group",
]
