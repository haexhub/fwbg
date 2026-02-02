"""
Feature Selection Plugins.

Verfügbare Methoden:
- boruta: Boruta All-Relevant Feature Selection
- plateau: Plateau-basierte Feature-Validierung (robustere Auswahl)
"""
from .boruta import BorutaSelector
from .boruta.selector import select_features_boruta
from .plateau import (
    PlateauSelector,
    find_feature_neighbors,
    calculate_feature_plateau_scores,
    calculate_param_plateau_score,
    select_best_plateau_candidate,
)
from .plateau.selector import select_plateau_features

__all__ = [
    "BorutaSelector",
    "select_features_boruta",
    "PlateauSelector",
    "select_plateau_features",
    "find_feature_neighbors",
    "calculate_feature_plateau_scores",
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
