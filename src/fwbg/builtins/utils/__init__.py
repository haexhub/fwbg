"""
Utility-Module für Plugins.

Enthält:
- macro_loader: Makro-Daten und Zinsdaten laden
"""
from .macro_loader import load_macro_indicators, load_interest_rates, load_macro_csv

__all__ = [
    "load_macro_indicators",
    "load_interest_rates",
    "load_macro_csv",
]
