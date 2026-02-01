"""
FWBG Built-in Plugins.

Dieses Package enthält alle Standard-Plugins für FWBG.
Plugins werden automatisch über Entry Points registriert.

Plugin-Kategorien:
- indicators/: Feature-Indikatoren (Trend, Momentum, Volatility, etc.)
- exit_strategies/: TP/SL Strategien (Fixed, ATR-based)
- feature_selection/: ML Feature-Auswahl (Boruta)
- preprocessing/: Daten-Vorverarbeitung (Fractional Differentiation)

Für Plugin-Entwicklung siehe builtins/README.md
"""
# Re-export für einfachen Import
from .indicators import (
    TrendIndicators,
    MomentumIndicators,
    VolatilityIndicators,
    RegimeIndicators,
    StructureIndicators,
    RiskIndicators,
    PriceActionIndicators,
    TimeSeasonIndicators,
    DistributionIndicators,
    DynamicsIndicators,
    MultiTimeframeIndicators,
    CrossFeatureIndicators,
    IchimokuIndicators,
)
from .exit_strategies import FixedExitStrategy, AtrExitStrategy
from .feature_selection import BorutaSelector
from .preprocessing import FractionalDiffPreprocessor

__all__ = [
    # Indicators
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
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
    # Exit Strategies
    "FixedExitStrategy",
    "AtrExitStrategy",
    # Feature Selection
    "BorutaSelector",
    # Preprocessing
    "FractionalDiffPreprocessor",
]
