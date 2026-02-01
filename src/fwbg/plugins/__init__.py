"""
Plugin Base Classes.

Diese Module definieren die abstrakten Basisklassen für alle Plugin-Typen.
Entwickler importieren diese Klassen um eigene Plugins zu erstellen.
"""

from .indicator import BaseIndicator
from .exit_strategy import BaseExitStrategy
from .feature_selector import BaseFeatureSelector
from .preprocessor import BasePreprocessor

__all__ = [
    "BaseIndicator",
    "BaseExitStrategy",
    "BaseFeatureSelector",
    "BasePreprocessor",
]
