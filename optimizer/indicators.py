"""
Technische Indikatoren für den Optimizer.

HINWEIS: Dieses Modul wurde in das Package 'indicators/' aufgeteilt.
Diese Datei existiert nur für Rückwärtskompatibilität und re-exportiert
alle Funktionen aus dem neuen Package.

Module:
- indicators/preprocessing.py: Fractional Differentiation, Log-Returns
- indicators/regime.py: Hurst-Exponent, Regime-Filter
- indicators/structure.py: FFT, Path Efficiency, Convexity, Events
- indicators/risk.py: Drawdown, CVaR, Vol-of-Vol, Crash Probability
- indicators/core.py: Haupt-Funktion compute_indicator_pool()
"""

# Re-export alles aus dem neuen Package
from .indicators import (
    # Preprocessing
    apply_preprocessing,
    apply_frac_diff_preprocessing,
    apply_log_returns_preprocessing,
    apply_normalize_preprocessing,
    frac_diff,
    get_frac_diff_weights,
    find_min_d_for_stationarity,
    # Regime
    compute_hurst_exponent,
    compute_rolling_hurst,
    compute_regime_filter,
    compute_regime_features,
    # Structure
    compute_fft_features,
    compute_event_features,
    compute_path_efficiency,
    compute_convexity_features,
    compute_vwap_features,
    # Risk
    compute_drawdown_features,
    compute_cvar_features,
    compute_vol_of_vol_features,
    compute_crash_probability_features,
    compute_correlation_features,
    # Core
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
