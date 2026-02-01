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
import pandas as pd

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
) -> pd.DataFrame:
    """
    Berechnet alle Indikatoren für einen DataFrame.

    Dies ist die Haupt-Utility-Funktion für den Trading-Bot und andere
    Komponenten, die alle Features auf einmal berechnen wollen.

    Args:
        df: DataFrame mit OHLC-Daten (Spalten: O, H, L, C, optional V)
        indicators: Liste spezifischer Indikatoren (None = alle)
        include_advanced: Auch Advanced-Indikatoren berechnen

    Returns:
        DataFrame mit allen berechneten Features
    """
    if indicators is None:
        if include_advanced:
            indicators = list(_DEFAULT_INDICATORS.keys())
        else:
            # Nur Core-Indikatoren
            indicators = ["trend", "momentum", "volatility"]

    for name in indicators:
        if name in _DEFAULT_INDICATORS:
            try:
                indicator = _DEFAULT_INDICATORS[name]
                df = indicator.compute(df)
            except Exception:
                # Fehler bei einzelnen Indikatoren ignorieren
                pass

    return df


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
]
