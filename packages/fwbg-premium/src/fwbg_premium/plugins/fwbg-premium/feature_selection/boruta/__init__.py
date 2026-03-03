"""
Boruta Feature Selection Plugin.

Boruta ist ein All-Relevant Feature Selection Algorithmus:
1. Erstelle Shadow-Features (permutierte Kopien aller Features)
2. Trainiere Random Forest auf Original + Shadow Features
3. Berechne Z-Score jedes Features vs. beste Shadow-Feature
4. Features mit Z-Score signifikant über Shadow = bestätigt relevant

Referenz: Kursa & Rudnicki (2010) "Feature Selection with the Boruta Package"
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fwbg_sdk import BaseFeatureSelector, register_feature_selector
from .selector import (
    select_features_boruta,
    create_shadow_features,
    boruta_iteration,
    boruta_select,
    boruta_select_fast,
)


def _create_shadow_features(X: pd.DataFrame) -> pd.DataFrame:
    """Erstellt Shadow-Features durch Permutation jeder Spalte."""
    X_shadow = X.apply(lambda col: np.random.permutation(col.values))
    X_shadow.columns = [f"shadow_{c}" for c in X.columns]
    return pd.concat([X, X_shadow], axis=1)


def _boruta_iteration(
    X: pd.DataFrame,
    y: np.ndarray,
    original_features: List[str],
    n_estimators: int = 100,
    max_depth: int = 5,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, float]:
    """Führt eine Boruta-Iteration durch."""
    if len(np.unique(y)) < 2:
        return np.zeros(len(original_features)), 0.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=np.random.randint(10000),
        verbosity=0,
    )
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    shadow_cols = [c for c in X.columns if c.startswith("shadow_")]
    shadow_max = importances[shadow_cols].max() if shadow_cols else 0
    original_importances = importances[original_features].values

    return original_importances, shadow_max


@register_feature_selector("boruta")
class BorutaSelector(BaseFeatureSelector):
    """
    Boruta Feature Selection.

    Findet alle statistisch relevanten Features durch Vergleich
    mit Shadow-Features (permutierte Kopien).
    """

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_features: int = None,
        n_iter: int = 10,
        n_estimators: int = 50,
        max_depth: int = 4,
        min_z_score: float = 0.5,
        n_jobs: int = 1,
        **params
    ) -> Tuple[List[str], dict]:
        """
        Wählt Features mit Boruta.

        Args:
            X: Feature DataFrame
            y: Target Array (0/1)
            max_features: Maximum Features (None = kein Limit)
            n_iter: Anzahl Iterationen
            n_estimators: Bäume pro Iteration
            max_depth: Max Tiefe
            min_z_score: Minimum Z-Score für Akzeptanz
            n_jobs: Threads für XGBoost

        Returns:
            (selected_features, metadata)
        """
        if len(X.columns) == 0:
            return [], {}

        if len(np.unique(y)) < 2:
            return [], {}

        # NaN/Inf behandeln
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        original_features = list(X.columns)
        n_features = len(original_features)

        # Z-Scores akkumulieren
        z_scores_sum = np.zeros(n_features)

        for _ in range(n_iter):
            X_with_shadow = _create_shadow_features(X)
            importances, shadow_max = _boruta_iteration(
                X_with_shadow, y, original_features,
                n_estimators, max_depth, n_jobs
            )

            shadow_std = max(shadow_max * 0.1, 1e-10)
            z_scores = (importances - shadow_max) / shadow_std
            z_scores_sum += z_scores

        # Durchschnittlicher Z-Score
        avg_z_scores = z_scores_sum / n_iter

        # Features sortieren nach Z-Score
        feature_scores = list(zip(original_features, avg_z_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        # Features mit positivem Z-Score
        selected = [f for f, z in feature_scores if z >= min_z_score]

        # Max-Features Limit anwenden
        if max_features and max_features > 0 and len(selected) > max_features:
            selected = selected[:max_features]

        # Metadata
        metadata = {
            "z_scores": dict(feature_scores),
            "n_original": n_features,
            "n_selected": len(selected),
        }

        return selected, metadata

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "n_iter": 10,
            "n_estimators": 50,
            "max_depth": 4,
            "min_z_score": 0.5,
            "n_jobs": 1,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "n_iter": {
                "type": "int",
                "default": 10,
                "description": "Number of Boruta iterations (shadow feature comparisons); more iterations = more stable results",
                "min": 1,
                "max": 1000,
                "step": 1,
            },
            "n_estimators": {
                "type": "int",
                "default": 50,
                "description": "Number of XGBoost trees per iteration for importance estimation",
                "min": 1,
                "max": 10000,
                "step": 10,
            },
            "max_depth": {
                "type": "int",
                "default": 4,
                "description": "Maximum tree depth per XGBoost iteration",
                "min": 1,
                "max": 50,
                "step": 1,
            },
            "min_z_score": {
                "type": "float",
                "default": 0.5,
                "description": "Minimum average z-score vs shadow features for a feature to be accepted as relevant",
                "min": -10.0,
                "max": 100.0,
                "step": 0.1,
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
    "BorutaSelector",
    "select_features_boruta",
    "create_shadow_features",
    "boruta_iteration",
    "boruta_select",
    "boruta_select_fast",
]
