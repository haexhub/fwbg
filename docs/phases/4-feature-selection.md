# Phase 4: Feature Selection

## Purpose

The feature selection phase selects the most relevant features for the ML model. This reduces overfitting (fewer irrelevant features = less noise) and speeds up training.

Feature selection is executed **per CV fold** on the **training data** — this means different features may be selected per fold, enabling a realistic evaluation.

---

## BaseFeatureSelector

Base class: `src/fwbg/plugins/feature_selector.py`

```python
class BaseFeatureSelector(BasePlugin, ABC):
    phase = PluginPhase.FEATURE_SELECTION

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                       max_features: int = None, **params) -> Tuple[List[str], dict]:
        """
        Selects the most important features.

        Args:
            X: Feature DataFrame (all computed features)
            y: Target array (0/1 for Loss/Win)
            max_features: Maximum number of features (None = unlimited)

        Returns:
            (selected_features, metadata)
            - selected_features: List of selected feature names
            - metadata: Dict with additional info (e.g., feature importances)
        """
```

- Registration: `@register_feature_selector("name")`
- Called in the inner CV loop — only on training data

---

## Available Plugins

### boruta (fwbg-premium)

Shadow feature comparison using the Boruta algorithm: Creates a randomized "shadow" version for each feature and checks whether the original is significantly better than its shadow.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_features` | `None` | Maximum number of features |
| `n_iter` | `5` | Boruta iterations |
| `n_estimators` | `30` | Trees per iteration |
| `max_depth` | `4` | Maximum tree depth |
| `min_z_score` | `0.5` | Minimum z-score vs shadow |

### stability (fwbg-premium) — Recommended

Bootstrap-based stability selection. Wraps an inner selector (e.g., Boruta) and runs it on multiple bootstrap samples. Only features selected in more than `threshold` of the bootstraps survive.

**Advantage:** Significantly more robust feature selection than a single Boruta run. Reduces overfitting substantially.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `inner_selector` | `"boruta"` | Which selector to wrap |
| `inner_params` | `{}` | Parameters for the inner selector |
| `n_bootstrap` | `7` | Number of bootstrap samples |
| `threshold` | `0.6` | Minimum selection rate (60% of bootstraps) |
| `bootstrap_ratio` | `0.8` | Fraction of data per bootstrap |
| `max_features` | `20` | Maximum number of features |

### plateau (fwbg-premium)

Plateau-based selection — evaluates the stability of feature importances across different parameter combinations.

---

## Feature Stability Across Folds

Since feature selection runs per fold, different features may be selected per fold. **Feature stability** measures how consistently a feature is selected across all walk-forward folds:

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

| stability | Meaning |
|-----------|---------|
| `>= 0.50` | Stable — feature selected in at least 50% of folds |
| `< 0.50` | Unstable — suggests noise fitting |

High stability is a good sign: the model consistently uses the same features regardless of the time window.

---

## Strategy JSON Configuration

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

## Creating a Custom Feature Selection Plugin

See [Plugin Development Guide](../plugin-development.md) for the complete guide.
