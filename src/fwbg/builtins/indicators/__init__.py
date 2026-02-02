"""
Indicator Plugins.

Verfügbare Indicator-Gruppen:

Core Indicators:
- trend: ADX, EMA, SMA, MACD, CCI, Aroon, Efficiency Ratio
- momentum: RSI, Stochastic, Williams %R, Ultimate Oscillator, ROC
- volatility: ATR, Bollinger Bands, Keltner Channel, Donchian Channel

Advanced Indicators:
- regime: Hurst Exponent, Market Regime Features
- structure: FFT, Path Efficiency, Convexity, Event Features, VWAP
- risk: Drawdown, CVaR, Vol-of-Vol, Crash Probability, Correlations
- price_action: Candle Features, HH/LL, Gaps, Volume Features
- time_season: Intraday Zeit, Saisonalität, Trading Sessions
- distribution: Skewness, Kurtosis, Tail Risk
- dynamics: Indicator Changes, Lags, Acceleration
- multi_timeframe: H4/D1 Features, Trend Alignment
- cross_features: Indicator Combinations, Confluences, Divergences
- ichimoku: Ichimoku Cloud Components
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import ta

# Core Indicators
from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators

# Advanced Indicators
from .regime import RegimeIndicators
from .structure import StructureIndicators
from .risk import RiskIndicators
from .price_action import PriceActionIndicators
from .time_season import TimeSeasonIndicators
from .distribution import DistributionIndicators
from .dynamics import DynamicsIndicators
from .multi_timeframe import MultiTimeframeIndicators
from .cross_features import CrossFeatureIndicators
from .ichimoku import IchimokuIndicators

# Preprocessing
from .preprocessing import apply_preprocessing

# Feature Groups Configuration
FEATURE_GROUPS = {
    "trend": {
        "name": "Trend Indicators",
        "prefixes": ["trend_"],
    },
    "momentum": {
        "name": "Momentum Indicators",
        "prefixes": ["mom_"],
    },
    "volatility": {
        "name": "Volatility Indicators",
        "prefixes": ["vol_"],
    },
    "regime": {
        "name": "Regime Features",
        "prefixes": ["regime_"],
    },
    "structure": {
        "name": "Structure Features",
        "prefixes": ["fft_", "path_", "conv_", "event_", "vwap_"],
    },
    "risk": {
        "name": "Risk Features",
        "prefixes": ["risk_", "dd_", "cvar_", "vov_", "crash_", "corr_"],
    },
    "price_action": {
        "name": "Price Action Features",
        "prefixes": ["pa_"],
    },
    "time_season": {
        "name": "Time & Season Features",
        "prefixes": ["time_", "season_"],
    },
    "distribution": {
        "name": "Distribution Features",
        "prefixes": ["dist_"],
    },
    "dynamics": {
        "name": "Dynamics Features",
        "prefixes": ["dyn_", "lag_", "accel_"],
    },
    "multi_timeframe": {
        "name": "Multi-Timeframe Features",
        "prefixes": ["mtf_"],
    },
    "cross_features": {
        "name": "Cross Features",
        "prefixes": ["cross_"],
    },
    "ichimoku": {
        "name": "Ichimoku Features",
        "prefixes": ["ichi_"],
    },
    "macro": {
        "name": "Macro Features",
        "prefixes": ["macro_", "sent_"],
    },
    # Combined groups
    "trend_momentum": {
        "name": "Trend + Momentum",
        "prefixes": ["trend_", "mom_"],
    },
    "all_core": {
        "name": "All Core Indicators",
        "prefixes": ["trend_", "mom_", "vol_"],
    },
}


# Default indicator instances for quick access
_DEFAULT_INDICATORS = {
    "trend": TrendIndicators(),
    "momentum": MomentumIndicators(),
    "volatility": VolatilityIndicators(),
    "regime": RegimeIndicators(),
    "structure": StructureIndicators(),
    "risk": RiskIndicators(),
    "price_action": PriceActionIndicators(),
    "time_season": TimeSeasonIndicators(),
    "distribution": DistributionIndicators(),
    "dynamics": DynamicsIndicators(),
    "multi_timeframe": MultiTimeframeIndicators(),
    "cross_features": CrossFeatureIndicators(),
    "ichimoku": IchimokuIndicators(),
}


def compute_indicator_pool(
    df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    include_advanced: bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Berechnet alle Indikatoren für einen DataFrame.

    Dies ist die Haupt-Utility-Funktion für den Trading-Bot und andere
    Komponenten, die alle Features auf einmal berechnen wollen.

    Args:
        df: DataFrame mit OHLC-Daten (Spalten: O, H, L, C, optional V)
        indicators: Liste spezifischer Indikatoren (None = alle)
        include_advanced: Auch Advanced-Indikatoren berechnen
        progress_callback: Optionaler Callback(name, idx, total) für Progress-Updates

    Returns:
        DataFrame mit allen berechneten Features
    """
    if indicators is None:
        if include_advanced:
            indicators = list(_DEFAULT_INDICATORS.keys())
        else:
            # Nur Core-Indikatoren
            indicators = ["trend", "momentum", "volatility"]

    total = len(indicators)
    for idx, name in enumerate(indicators):
        if name in _DEFAULT_INDICATORS:
            try:
                if progress_callback:
                    progress_callback(name, idx + 1, total)
                indicator = _DEFAULT_INDICATORS[name]
                df = indicator.compute(df)
            except Exception:
                # Fehler bei einzelnen Indikatoren ignorieren
                pass

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Gibt alle Feature-Spalten zurück (ohne interne Spalten wie _atr).

    Args:
        df: DataFrame mit berechneten Features

    Returns:
        Liste der Feature-Spaltennamen
    """
    exclude = ["O", "H", "L", "C", "V", "Volume", "_atr", "_regime_ok", "_original_close", "_hurst"]
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def filter_features_by_group(all_features: List[str], group_name: str) -> List[str]:
    """
    Filtert Features nach einer Feature-Gruppe aus FEATURE_GROUPS.

    Args:
        all_features: Liste aller verfügbaren Features
        group_name: Name der Gruppe (z.B. "trend", "momentum", "trend_momentum")

    Returns:
        Liste der Features die zur Gruppe gehören
    """
    if group_name not in FEATURE_GROUPS:
        return all_features  # Fallback: alle Features

    group = FEATURE_GROUPS[group_name]
    prefixes = group["prefixes"]

    filtered = []
    for feat in all_features:
        for prefix in prefixes:
            if feat.startswith(prefix):
                filtered.append(feat)
                break

    return filtered


def _compute_rolling_hurst(
    series: np.ndarray,
    window: int = 100,
    step: int = 10
) -> np.ndarray:
    """Berechnet Rolling Hurst-Exponent für Regime-Filter."""
    from .regime import _compute_rolling_hurst as regime_rolling_hurst
    return regime_rolling_hurst(series, window, step)


def compute_regime_filter(
    df: pd.DataFrame,
    regime_params=None
) -> pd.Series:
    """
    Berechnet Regime-Filter basierend auf konfigurierbaren Bedingungen.

    Args:
        df: DataFrame mit Indikatoren
        regime_params: Optional RegimeFilterConfig mit Konfiguration

    Returns:
        Boolean Series (True = Trading erlaubt)
    """
    # Default: Kein Filter aktiv (alle Trades erlaubt)
    adx_min = 0
    vix_max = None
    hurst_min = None
    hurst_max = None

    if regime_params is not None:
        adx_min = regime_params.adx_min if regime_params.adx_enabled else 0
        if regime_params.vix_enabled:
            vix_max = regime_params.vix_max
        if regime_params.hurst_enabled:
            hurst_min = regime_params.hurst_min
            hurst_max = regime_params.hurst_max

    # ADX Filter (adx_min=0 bedeutet kein Filter)
    if adx_min > 0:
        adx_14 = df.get("trend_adx_14", ta.trend.adx(df["H"], df["L"], df["C"], window=14))
        regime_ok = adx_14 >= adx_min
    else:
        # Kein ADX-Filter - alle Bars erlaubt
        regime_ok = pd.Series(True, index=df.index)

    # VIX Filter (nur wenn explizit konfiguriert)
    if vix_max is not None and "sent_vix" in df.columns:
        vix_ok = df["sent_vix"] < vix_max
        regime_ok = regime_ok & vix_ok

    # Hurst Filter
    if hurst_min is not None or hurst_max is not None:
        if "_hurst" not in df.columns:
            close_values = (
                df["_original_close"].values
                if "_original_close" in df.columns
                else df["C"].values
            )
            df["_hurst"] = _compute_rolling_hurst(close_values, window=100, step=10)

        if hurst_min is not None:
            regime_ok = regime_ok & (df["_hurst"] >= hurst_min)
        if hurst_max is not None:
            regime_ok = regime_ok & (df["_hurst"] <= hurst_max)

    return regime_ok


__all__ = [
    # Core
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    # Advanced
    "RegimeIndicators",
    "StructureIndicators",
    "RiskIndicators",
    "PriceActionIndicators",
    "TimeSeasonIndicators",
    "DistributionIndicators",
    "DynamicsIndicators",
    "MultiTimeframeIndicators",
    "CrossFeatureIndicators",
    "IchimokuIndicators",
    # Utility
    "compute_indicator_pool",
    "get_feature_columns",
    "filter_features_by_group",
    "compute_regime_filter",
    "apply_preprocessing",
    "FEATURE_GROUPS",
]
