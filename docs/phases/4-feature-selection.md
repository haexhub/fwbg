# Phase 4: Feature Selection

## Zweck

Die Feature-Selection-Phase wählt die relevantesten Features für das ML-Modell aus. Das reduziert Overfitting (weniger irrelevante Features = weniger Rauschen) und beschleunigt das Training.

Feature Selection wird **pro CV-Fold** auf den **Trainingsdaten** ausgeführt — dadurch werden unterschiedliche Features pro Fold selektiert, was eine realistische Evaluation ermöglicht.

---

## BaseFeatureSelector

Basisklasse: `src/fwbg/plugins/feature_selector.py`

```python
class BaseFeatureSelector(BasePlugin, ABC):
    phase = PluginPhase.FEATURE_SELECTION

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                       max_features: int = None, **params) -> Tuple[List[str], dict]:
        """
        Wählt die wichtigsten Features aus.

        Args:
            X: Feature-DataFrame (alle berechneten Features)
            y: Target-Array (0/1 für Loss/Win)
            max_features: Maximale Anzahl Features (None = unbegrenzt)

        Returns:
            (selected_features, metadata)
            - selected_features: Liste der selektierten Feature-Namen
            - metadata: Dict mit Zusatzinfos (z.B. Feature Importances)
        """
```

- Registrierung: `@register_feature_selector("name")`
- Wird im Inner CV Loop aufgerufen — nur auf Train-Daten

---

## Verfügbare Plugins

### boruta (fwbg-premium)

Shadow-Feature-Vergleich nach dem Boruta-Algorithmus: Erstellt für jedes Feature eine randomisierte "Shadow"-Version und prüft ob das Original signifikant besser ist als sein Shadow.

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `max_features` | `None` | Maximale Anzahl Features |
| `n_iter` | `5` | Boruta-Iterationen |
| `n_estimators` | `30` | Bäume pro Iteration |
| `max_depth` | `4` | Maximale Baumtiefe |
| `min_z_score` | `0.5` | Mindest-Z-Score vs Shadow |

### stability (fwbg-premium) — Empfohlen

Bootstrap-basierte Stability Selection. Wrapped einen Inner Selector (z.B. Boruta) und führt ihn auf mehreren Bootstrap-Samples aus. Nur Features die in mehr als `threshold` der Bootstraps selektiert werden, überleben.

**Vorteil:** Deutlich robustere Feature-Selektion als ein einzelner Boruta-Durchlauf. Reduziert Overfitting signifikant.

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `inner_selector` | `"boruta"` | Welcher Selector geWrapped wird |
| `inner_params` | `{}` | Parameter für den Inner Selector |
| `n_bootstrap` | `7` | Anzahl Bootstrap-Samples |
| `threshold` | `0.6` | Mindest-Selektionsrate (60% der Bootstraps) |
| `bootstrap_ratio` | `0.8` | Anteil der Daten pro Bootstrap |
| `max_features` | `20` | Maximale Anzahl Features |

### plateau (fwbg-premium)

Plateau-basierte Selektion — bewertet die Stabilität der Feature Importances über verschiedene Parameterkombinationen.

---

## Feature-Stabilität über Folds

Da Feature Selection pro Fold läuft, können unterschiedliche Features pro Fold selektiert werden. Die **Feature-Stabilität** misst, wie konsistent ein Feature über alle Walk-Forward Folds hinweg ausgewählt wird:

```json
"feature_stability": {
  "stable_count": 12,
  "unstable_count": 3,
  "details": {
    "trend_adx_14": {"count": 8, "stability": 1.0},
    "vol_atr_pct_14_rank": {"count": 6, "stability": 0.75},
    "macro_yield_spread_us_de_chg_5d": {"count": 2, "stability": 0.25}
  }
}
```

| stability | Bedeutung |
|-----------|-----------|
| `>= 0.50` | Stabil — Feature in mindestens 50% der Folds selektiert |
| `< 0.50` | Instabil — deutet auf Noise-Fitting hin |

Hohe Stabilität ist ein gutes Zeichen: Das Modell nutzt konsistent die gleichen Features, unabhängig vom Zeitfenster.

---

## Strategy-JSON Konfiguration

```json
"pipeline": {
  "feature_selection": [
    {
      "name": "stability",
      "params": {
        "inner_selector": "boruta",
        "inner_params": {
          "n_iter": 5,
          "n_estimators": 30,
          "max_depth": 4,
          "min_z_score": 0.5
        },
        "n_bootstrap": 7,
        "threshold": 0.6,
        "bootstrap_ratio": 0.8,
        "max_features": 20
      }
    }
  ]
}
```

---

## Eigenes Feature-Selection-Plugin erstellen

Siehe [Plugin Development Guide](../plugin-development.md) für die vollständige Anleitung.
