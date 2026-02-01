"""
Preprocessing Plugins.

Verfügbare Methoden:
- fractional_diff: Fractional Differentiation (López de Prado)
- normalization: Z-Score Normalisierung
"""
from .fractional_diff import FractionalDiffPreprocessor

__all__ = ["FractionalDiffPreprocessor"]
