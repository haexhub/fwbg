"""
Plateau Feature Selection Plugin.

Plateau-basierte Feature-Auswahl bevorzugt Features, die:
1. Hohe Importance haben
2. Ähnliche Importance wie ihre "Nachbarn" (z.B. RSI_14 und RSI_12/RSI_16)
3. Nicht isolierte Ausreißer sind (stabiler über Parameter-Variationen)

Vorteile gegenüber reiner Importance-basierter Auswahl:
- Robustere Feature-Auswahl
- Weniger Overfitting auf zufällige Spitzen
- Features sind oft auch mit leicht anderen Parametern performant
"""
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fwbg_sdk import BaseFeatureSelector, register_feature_selector
from .selector import (
    select_plateau_features,
    find_feature_neighbors,
    calculate_feature_plateau_score,
    calculate_param_plateau_score,
    select_best_plateau_candidate,
)


def calculate_feature_plateau_scores(
    feature_importances: Dict[str, float],
    all_features: List[str],
    min_neighbors: int = 1,
) -> Dict[str, Dict]:
    """
    Berechnet Plateau-Scores für Features basierend auf Nachbar-Importance.

    Ein stabiles Feature hat ähnliche Importance wie seine Nachbarn
    (z.B. RSI_14 und RSI_12/RSI_16 sollten ähnlich wichtig sein).

    Args:
        feature_importances: Dict von Feature-Name -> Importance
        all_features: Liste aller verfügbaren Features
        min_neighbors: Mindestanzahl Nachbarn für Plateau-Berechnung

    Returns:
        Dict mit Feature-Name -> {importance, neighbors, stability, plateau_score}
    """
    results = {}

    for feat, importance in feature_importances.items():
        neighbors = find_feature_neighbors(feat, all_features)
        neighbor_importances = [
            feature_importances.get(n, 0) for n in neighbors if n in feature_importances
        ]

        if len(neighbor_importances) >= min_neighbors:
            neighbor_mean = np.mean(neighbor_importances)
            neighbor_std = np.std(neighbor_importances)

            # Stability: Wie konsistent sind die Nachbarn?
            cv = neighbor_std / (neighbor_mean + 1e-10)
            stability = 1.0 / (1.0 + cv)

            # Plateau: Feature sollte nicht viel wichtiger als Nachbarn sein
            relative_diff = abs(importance - neighbor_mean) / (neighbor_mean + 1e-10)
            plateau_factor = 1.0 / (1.0 + relative_diff * 0.5)

            # Kombinierter Score
            plateau_score = importance * (
                0.6 + 0.25 * stability + 0.15 * plateau_factor
            )

            results[feat] = {
                "importance": importance,
                "neighbors": neighbors,
                "neighbor_importances": neighbor_importances,
                "stability": stability,
                "plateau_factor": plateau_factor,
                "plateau_score": plateau_score,
                "is_plateau": stability > 0.5 and plateau_factor > 0.6,
            }
        else:
            # Keine Nachbarn gefunden - leichte Penalty
            results[feat] = {
                "importance": importance,
                "neighbors": neighbors,
                "neighbor_importances": [],
                "stability": 0.5,
                "plateau_factor": 0.5,
                "plateau_score": importance * 0.8,
                "is_plateau": False,
            }

    return results


@register_feature_selector("plateau")
class PlateauSelector(BaseFeatureSelector):
    """
    Plateau-basierte Feature Selection.

    Bevorzugt Features die auf einem "Plateau" liegen, d.h.
    deren Nachbar-Features ähnliche Performance haben.
    Dies führt zu robusteren Feature-Sets.
    """

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_features: int = None,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_importance: float = 0.01,
        min_neighbors: int = 1,
        prefer_plateau: bool = True,
        n_jobs: int = 1,
        **params
    ) -> Tuple[List[str], dict]:
        """
        Wählt Features basierend auf Plateau-Scores.

        Args:
            X: Feature DataFrame
            y: Target Array (0/1)
            max_features: Maximum Features (None = alle relevanten)
            n_estimators: Bäume für XGBoost
            max_depth: Max Tiefe
            min_importance: Minimale Feature Importance
            min_neighbors: Mindest-Nachbarn für Plateau-Bonus
            prefer_plateau: Sortiere nach Plateau-Score statt Importance
            n_jobs: Threads für XGBoost

        Returns:
            (selected_features, metadata)
        """
        if len(X.columns) == 0:
            return [], {}

        # NaN/Inf behandeln
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        all_features = list(X.columns)

        # Feature Importances mit XGBoost
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

        importances = dict(zip(X.columns, model.feature_importances_))

        # Filter Features mit minimaler Importance
        filtered_importances = {
            k: v for k, v in importances.items() if v >= min_importance
        }

        if not filtered_importances:
            # Fallback: Top-Features nach Importance
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            top_n = max_features if max_features else 10
            return [f[0] for f in sorted_features[:top_n]], {
                "importances": importances,
                "plateau_scores": {},
                "method": "importance_fallback",
            }

        # Plateau-Scores berechnen
        plateau_results = calculate_feature_plateau_scores(
            filtered_importances, all_features, min_neighbors
        )

        # Sortieren nach Plateau-Score oder Importance
        if prefer_plateau:
            sorted_features = sorted(
                plateau_results.items(),
                key=lambda x: x[1]["plateau_score"],
                reverse=True
            )
        else:
            sorted_features = sorted(
                plateau_results.items(),
                key=lambda x: x[1]["importance"],
                reverse=True
            )

        # Max Features begrenzen
        if max_features and max_features > 0:
            sorted_features = sorted_features[:max_features]

        selected = [f[0] for f in sorted_features]

        # Metadata
        metadata = {
            "importances": importances,
            "plateau_scores": {k: v["plateau_score"] for k, v in plateau_results.items()},
            "plateau_features": [k for k, v in plateau_results.items() if v["is_plateau"]],
            "n_original": len(all_features),
            "n_with_neighbors": sum(1 for v in plateau_results.values() if len(v["neighbors"]) > 0),
            "n_selected": len(selected),
            "method": "plateau" if prefer_plateau else "importance",
        }

        return selected, metadata

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "n_estimators": 100,
            "max_depth": 5,
            "min_importance": 0.01,
            "min_neighbors": 1,
            "prefer_plateau": True,
            "n_jobs": 1,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "description": "Number of XGBoost trees for computing feature importances",
                "min": 1,
                "max": 10000,
                "step": 10,
            },
            "max_depth": {
                "type": "int",
                "default": 5,
                "description": "Maximum tree depth for the XGBoost importance model",
                "min": 1,
                "max": 50,
                "step": 1,
            },
            "min_importance": {
                "type": "float",
                "default": 0.01,
                "description": "Minimum feature importance threshold; features below this are excluded before plateau scoring",
                "min": 0.0,
                "max": 1.0,
                "step": 0.005,
            },
            "min_neighbors": {
                "type": "int",
                "default": 1,
                "description": "Minimum number of parameter-neighbor features required for plateau bonus (otherwise penalized)",
                "min": 0,
                "max": 100,
                "step": 1,
            },
            "prefer_plateau": {
                "type": "bool",
                "default": True,
                "description": "Sort by plateau score instead of raw importance (recommended for robustness)",
            },
            "n_jobs": {
                "type": "int",
                "default": 1,
                "description": "Number of parallel threads for XGBoost training",
                "min": 1,
                "max": 128,
                "step": 1,
            },
        }


__all__ = [
    "select_plateau_features",
    "PlateauSelector",
    "find_feature_neighbors",
    "calculate_feature_plateau_score",
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
