"""
Feature Selection Plugins.

Verfügbare Methoden:
- boruta: Boruta All-Relevant Feature Selection
- plateau: Plateau-basierte Feature-Validierung (robustere Auswahl)
"""
from .boruta import BorutaSelector
from .plateau import (
    PlateauSelector,
    find_feature_neighbors,
    calculate_feature_plateau_scores,
    calculate_param_plateau_score,
    select_best_plateau_candidate,
)

__all__ = [
    "BorutaSelector",
    "PlateauSelector",
    "find_feature_neighbors",
    "calculate_feature_plateau_scores",
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
