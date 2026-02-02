"""
Boruta Feature Selection.

Boruta ist ein All-Relevant Feature Selection Algorithmus:
1. Erstelle Shadow-Features (permutierte Kopien aller Features)
2. Trainiere Random Forest auf Original + Shadow Features
3. Berechne Z-Score jedes Features vs. beste Shadow-Feature
4. Features mit Z-Score signifikant über Shadow = bestätigt relevant
5. Features mit Z-Score signifikant unter Shadow = bestätigt irrelevant
6. Wiederhole bis alle Features klassifiziert oder max_iter erreicht

Referenz: Kursa & Rudnicki (2010) "Feature Selection with the Boruta Package"
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from xgboost import XGBClassifier
from scipy import stats

from fwbg.utils.xgb_config import get_xgboost_n_jobs


def create_shadow_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt Shadow-Features durch Permutation jeder Spalte.

    Args:
        X: Original Feature-DataFrame

    Returns:
        DataFrame mit Original + Shadow Features
    """
    X_shadow = X.apply(lambda col: np.random.permutation(col.values))
    X_shadow.columns = [f"shadow_{c}" for c in X.columns]
    return pd.concat([X, X_shadow], axis=1)


def boruta_iteration(
    X: pd.DataFrame,
    y: np.ndarray,
    original_features: List[str],
    n_estimators: int = 100,
    max_depth: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Führt eine Boruta-Iteration durch.

    Args:
        X: Feature DataFrame (Original + Shadow)
        y: Target-Array
        original_features: Liste der Original-Feature-Namen
        n_estimators: Anzahl Bäume
        max_depth: Max Tiefe

    Returns:
        (feature_importances, shadow_max) - Importances und max Shadow-Importance
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=get_xgboost_n_jobs(),
        random_state=np.random.randint(10000),
        verbosity=0,
    )
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)

    # Maximale Shadow-Feature Importance
    shadow_cols = [c for c in X.columns if c.startswith("shadow_")]
    shadow_max = importances[shadow_cols].max() if shadow_cols else 0

    # Nur Original-Feature Importances zurückgeben
    original_importances = importances[original_features].values

    return original_importances, shadow_max


def boruta_select(
    X: pd.DataFrame,
    y: np.ndarray,
    max_iter: int = 20,
    alpha: float = 0.05,
    n_estimators: int = 100,
    max_depth: int = 5,
    verbose: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Führt Boruta Feature Selection durch.

    Args:
        X: Feature DataFrame
        y: Target Array
        max_iter: Maximale Iterationen
        alpha: Signifikanz-Level für Entscheidung
        n_estimators: Bäume pro Iteration
        max_depth: Max Tiefe der Bäume
        verbose: Debug-Output

    Returns:
        (confirmed, rejected, tentative) - Listen von Feature-Namen
    """
    original_features = list(X.columns)
    n_features = len(original_features)

    # Tracking: Wie oft schlägt Feature die Shadow-Max?
    hits = np.zeros(n_features)

    for iteration in range(max_iter):
        # Shadow-Features erstellen (jede Iteration neu)
        X_with_shadow = create_shadow_features(X)

        # Iteration durchführen
        importances, shadow_max = boruta_iteration(
            X_with_shadow, y, original_features, n_estimators, max_depth
        )

        # Zähle Hits (Feature > Shadow Max)
        hits += (importances > shadow_max).astype(int)

        if verbose and iteration % 5 == 0:
            confirmed = np.sum(hits > iteration * 0.5 + 3)
            print(f"Iteration {iteration}: {confirmed} features confirmed")

    # Statistische Entscheidung via Binomialtest
    confirmed = []
    rejected = []
    tentative = []

    for i, feat in enumerate(original_features):
        hit_rate = hits[i] / max_iter

        # Binomialtest: Ist hit_rate signifikant über 0.5?
        # H0: Feature ist nicht besser als Zufall (p = 0.5)
        p_value = stats.binom_test(
            int(hits[i]), max_iter, 0.5, alternative='greater'
        ) if hasattr(stats, 'binom_test') else stats.binomtest(
            int(hits[i]), max_iter, 0.5, alternative='greater'
        ).pvalue

        if p_value < alpha:
            confirmed.append(feat)
        elif p_value > 1 - alpha:
            rejected.append(feat)
        else:
            tentative.append(feat)

    return confirmed, rejected, tentative


def boruta_select_fast(
    X: pd.DataFrame,
    y: np.ndarray,
    n_iter: int = 10,
    n_estimators: int = 50,
    max_depth: int = 4,
    min_z_score: float = 0.5,
) -> List[str]:
    """
    Schnelle Boruta-Variante für Grid-Search.

    Weniger Iterationen, aber KEIN hartes Feature-Limit.
    Alle Features mit positivem Z-Score werden akzeptiert.

    Args:
        X: Feature DataFrame
        y: Target Array
        n_iter: Anzahl Iterationen
        n_estimators: Bäume pro Iteration
        max_depth: Max Tiefe
        min_z_score: Minimum Z-Score für Akzeptanz (0.5 = leicht über Shadow)

    Returns:
        Liste aller bestätigten Features (kein hartes Limit!)
    """
    if len(X.columns) == 0:
        return []

    original_features = list(X.columns)
    n_features = len(original_features)

    # Tracking: Summe der Z-Scores über Iterationen
    z_scores_sum = np.zeros(n_features)

    for _ in range(n_iter):
        X_with_shadow = create_shadow_features(X)
        importances, shadow_max = boruta_iteration(
            X_with_shadow, y, original_features, n_estimators, max_depth
        )

        # Z-Score: Wie viel besser als Shadow-Max?
        # Höher = besser
        shadow_std = max(shadow_max * 0.1, 1e-10)  # Approximierte Std
        z_scores = (importances - shadow_max) / shadow_std
        z_scores_sum += z_scores

    # Durchschnittlicher Z-Score
    avg_z_scores = z_scores_sum / n_iter

    # Sortiere nach Z-Score
    feature_scores = list(zip(original_features, avg_z_scores))
    feature_scores.sort(key=lambda x: x[1], reverse=True)

    # KEIN hartes Limit - alle Features mit Z-Score über Threshold
    selected = [f for f, z in feature_scores if z >= min_z_score]

    return selected


def select_features_boruta(
    train_df: pd.DataFrame,
    targets: np.ndarray,
    group_features: List[str],
    min_trades: int = 50,
    min_z_score: float = 0.5,
    max_features: int = 0,
) -> Tuple[Optional[List[str]], dict]:
    """
    Wählt Features mit Boruta (Drop-in Replacement für select_features_from_fold).

    Args:
        train_df: Training DataFrame
        targets: Target Array
        group_features: Features der aktuellen Gruppe
        min_trades: Minimum Trades für Training
        min_z_score: Minimum Z-Score für Feature-Akzeptanz
        max_features: Maximum Features (0 = kein Limit, aber Default 15 als Obergrenze)

    Returns:
        (selected_features, importances_dict) oder (None, {})
    """
    if np.count_nonzero(targets) < min_trades // 2:
        return None, {}

    # Nur verfügbare Features nutzen
    available = [f for f in group_features if f in train_df.columns]
    if not available:
        return None, {}

    X = train_df[available].copy()

    # NaN/Inf behandeln (nutzt konsolidierte Utils)
    from fwbg.utils import clean_dataframe
    X = clean_dataframe(X)

    # Boruta durchführen
    selected = boruta_select_fast(
        X, targets,
        n_iter=10,
        n_estimators=50,
        max_depth=4,
        min_z_score=min_z_score,
    )

    # Max Features Limit anwenden (gegen Overfitting)
    # Default: 15 Features wenn max_features=0
    limit = max_features if max_features > 0 else 15
    if len(selected) > limit:
        # Behalte die Top-N nach Z-Score (bereits sortiert)
        selected = selected[:limit]

    # Importances für Logging (approximiert durch letztes Modell)
    if selected:
        model = XGBClassifier(
            n_estimators=50, max_depth=4, n_jobs=get_xgboost_n_jobs(),
            random_state=42, verbosity=0,
        )
        model.fit(X[selected], targets)
        importances = dict(zip(selected, model.feature_importances_))
    else:
        importances = {}

    if len(selected) >= 2:
        return selected, importances

    return None, importances
