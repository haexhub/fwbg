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
import re
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fwbg.plugins import BaseFeatureSelector
from fwbg.core import register_feature_selector
from .selector import (
    select_plateau_features,
    find_feature_neighbors,
    calculate_feature_plateau_score,
    calculate_param_plateau_score,
    select_best_plateau_candidate,
)


def find_feature_neighbors(feature_name: str, all_features: List[str]) -> List[str]:
    """
    Findet ähnliche Features basierend auf Lookback-Perioden.

    Beispiel: 'rsi_14' -> ['rsi_12', 'rsi_16'] oder ['rsi_10', 'rsi_20']
              'macro_vix_chg_24h' -> ['macro_vix_chg_12h', 'macro_vix_chg_48h']

    Args:
        feature_name: Name des Features
        all_features: Liste aller verfügbaren Features

    Returns:
        Liste von Nachbar-Features
    """
    neighbors = []

    # Pattern für Lookback-Zahlen in Feature-Namen
    # Matches: rsi_14, ema_20, atr_14, macro_vix_chg_24h, macro_vix_chg_5d, etc.
    # Reihenfolge wichtig: spezifischere Patterns zuerst!
    patterns = [
        (r"_(\d+)h$", "h"),  # Stunden am Ende: chg_24h
        (r"_(\d+)d$", "d"),  # Tage am Ende: chg_5d
        (r"_(\d+)_", "_"),   # Mittendrin: sma_20_slope
        (r"_(\d+)$", ""),    # Suffix: rsi_14, ema_20
    ]

    for pattern, suffix in patterns:
        match = re.search(pattern, feature_name)
        if match:
            current_value = int(match.group(1))
            prefix = feature_name[: match.start(1)]

            # Definiere Nachbar-Werte basierend auf Größenordnung
            if current_value <= 5:
                deltas = [-1, 1, -2, 2, -3, 3]
            elif current_value <= 20:
                deltas = [-2, 2, -4, 4, -5, 5]
            elif current_value <= 50:
                deltas = [-5, 5, -10, 10, -12, 12]
            else:
                deltas = [-10, 10, -20, 20, -24, 24]

            for delta in deltas:
                new_value = current_value + delta
                if new_value > 0:
                    new_name = prefix + str(new_value) + suffix
                    if new_name in all_features and new_name != feature_name:
                        neighbors.append(new_name)

            break  # Nur ein Pattern matchen

    return neighbors


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


# Utility-Funktionen für Parameter-Plateau (Grid-Search)
def calculate_param_plateau_score(
    candidates: List[Dict],
    grid_tp: List[int],
    grid_sl: List[int],
    grid_ct: List[float]
) -> List[Dict]:
    """
    Berechnet für jeden Kandidaten einen Plateau-Score basierend auf Nachbar-Performance.

    Ein guter Plateau-Kandidat hat:
    - Hohen eigenen Score
    - Ähnliche Scores bei benachbarten TP/SL/CT Werten
    - Niedrige Varianz in der Nachbarschaft

    Args:
        candidates: Liste von Kandidaten mit 'params' (tp, sl, ct) und 'score'
        grid_tp: Liste der TP-Werte im Grid
        grid_sl: Liste der SL-Werte im Grid
        grid_ct: Liste der CT-Werte im Grid

    Returns:
        Kandidaten mit zusätzlichem 'plateau_score' und 'stability_score'
    """
    if not candidates:
        return candidates

    # Index-Lookup für schnellen Zugriff
    param_to_score = {}
    for c in candidates:
        key = (c["params"][0], c["params"][1], c["params"][2])
        param_to_score[key] = c["score"]

    def get_neighbors(tp, sl, ct):
        """Gibt alle direkten Nachbarn im Grid zurück."""
        neighbors = []

        tp_idx = grid_tp.index(tp) if tp in grid_tp else -1
        sl_idx = grid_sl.index(sl) if sl in grid_sl else -1
        ct_idx = grid_ct.index(ct) if ct in grid_ct else -1

        if tp_idx < 0 or sl_idx < 0 or ct_idx < 0:
            return neighbors

        # TP Nachbarn
        for delta in [-1, 1]:
            new_tp_idx = tp_idx + delta
            if 0 <= new_tp_idx < len(grid_tp):
                key = (grid_tp[new_tp_idx], sl, ct)
                if key in param_to_score:
                    neighbors.append(param_to_score[key])

        # SL Nachbarn
        for delta in [-1, 1]:
            new_sl_idx = sl_idx + delta
            if 0 <= new_sl_idx < len(grid_sl):
                key = (tp, grid_sl[new_sl_idx], ct)
                if key in param_to_score:
                    neighbors.append(param_to_score[key])

        # CT Nachbarn
        for delta in [-1, 1]:
            new_ct_idx = ct_idx + delta
            if 0 <= new_ct_idx < len(grid_ct):
                key = (tp, sl, grid_ct[new_ct_idx])
                if key in param_to_score:
                    neighbors.append(param_to_score[key])

        return neighbors

    # Plateau-Score für jeden Kandidaten berechnen
    for c in candidates:
        tp, sl, ct = c["params"]
        own_score = c["score"]
        neighbors = get_neighbors(tp, sl, ct)

        if len(neighbors) >= 2:
            neighbor_mean = np.mean(neighbors)
            neighbor_std = np.std(neighbors)

            # Stability: Wie konsistent sind die Nachbarn?
            cv = neighbor_std / (neighbor_mean + 1e-10)
            stability = 1.0 / (1.0 + cv)

            # Plateau-Check: Eigener Score sollte nahe am Nachbar-Durchschnitt liegen
            relative_diff = abs(own_score - neighbor_mean) / (neighbor_mean + 1e-10)
            plateau_penalty = 1.0 / (1.0 + relative_diff)

            # Kombinierter Plateau-Score
            c["neighbor_mean"] = neighbor_mean
            c["neighbor_count"] = len(neighbors)
            c["stability_score"] = stability
            c["plateau_penalty"] = plateau_penalty
            c["plateau_score"] = own_score * (
                0.5 + 0.3 * stability + 0.2 * plateau_penalty
            )
        else:
            # Zu wenig Nachbarn - Penalty für Rand-Konfigurationen
            c["neighbor_mean"] = 0
            c["neighbor_count"] = len(neighbors)
            c["stability_score"] = 0.5
            c["plateau_penalty"] = 0.5
            c["plateau_score"] = own_score * 0.7

    return candidates


def select_best_plateau_candidate(
    candidates: List[Dict],
    grid_tp: List[int],
    grid_sl: List[int],
    grid_ct: List[float],
    min_neighbors: int = 2,
) -> Optional[Dict]:
    """
    Wählt den besten Kandidaten basierend auf Plateau-Score.

    Args:
        candidates: Liste von Kandidaten
        grid_*: Grid-Werte für TP/SL/CT
        min_neighbors: Mindestanzahl Nachbarn

    Returns:
        Bester Kandidat nach Plateau-Score oder None
    """
    if not candidates:
        return None

    # Plateau-Scores berechnen
    candidates = calculate_param_plateau_score(candidates, grid_tp, grid_sl, grid_ct)

    # Nach Plateau-Score sortieren, nur Kandidaten mit genug Nachbarn
    valid_candidates = [
        c for c in candidates if c.get("neighbor_count", 0) >= min_neighbors
    ]

    if valid_candidates:
        valid_candidates.sort(key=lambda x: x["plateau_score"], reverse=True)
        return valid_candidates[0]

    # Fallback: Bester nach normalem Score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]


__all__ = [
    "select_plateau_features",
    "PlateauSelector",
    "find_feature_neighbors",
    "calculate_feature_plateau_score",
    "calculate_param_plateau_score",
    "select_best_plateau_candidate",
]
