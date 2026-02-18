# Phase 2: Preprocessing

## Purpose

The preprocessing phase transforms OHLC data before feature computation. The primary use case is **stationarity transformation** — financial time series are typically non-stationary, but many ML models perform better with stationary input data.

---

## BasePreprocessor

Module: `fwbg_sdk.preprocessors`

```python
class BasePreprocessor(BasePlugin, ABC):
    phase = PluginPhase.PREPROCESSING
    name: str = "base"
    order: int = 100       # Execution order (lower = earlier)
    fitted_: bool = False  # Whether fit() has been called

    def fit(self, df: pd.DataFrame, **params) -> "BasePreprocessor":
        """Learn parameters from training data. NEVER on test/OOS data!"""

    @abstractmethod
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Transform DataFrame using learned parameters."""

    def fit_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Combines fit() and transform() for training data."""

    def inverse_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Optional: reverse transformation."""
```

- Import: `from fwbg_sdk import BasePreprocessor, register_preprocessor`
- Registration: `@register_preprocessor("name")`
- `order` determines execution order when multiple preprocessors are configured (lower = earlier)
- Follows the **sklearn fit/transform pattern**

---

## Lifecycle: fit/transform per CV Fold

Preprocessors are **stateful** — they learn parameters from training data and apply them to any data. The lifecycle per walk-forward fold:

```
┌─ Fold N ──────────────────────────────────────────────┐
│                                                        │
│  1. reset()                     Reset state            │
│  2. fit(train_df)               Learn parameters       │
│  3. transform(train_df)         Transform training data │
│  4. transform(test_df)          Transform test data    │
│     (using the parameters learned in step 2!)          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Why fit() Only on Training Data?

This is essential for **lookahead bias prevention**:

- `fit()` learns statistical parameters (e.g., the optimal differentiation degree d)
- If `fit()` is called on **all** data, future information flows into the transformation → the model "sees" the future
- Therefore: call `fit()` **exclusively** on the training split
- `transform()` can be applied to any data — it only uses the parameters learned in `fit()`

---

## Interaction with Indicators

Preprocessing affects when indicators are computed:

| `benefits_from_stationary` | Computation | Caching |
|----------------------------|-------------|---------|
| `False` (default) | **Once on raw OHLC** before preprocessing | Cached across all folds |
| `True` | **Per fold on preprocessed OHLC** after preprocessing | Not cached |

The split happens automatically via `split_indicators_by_stationarity()` in `src/fwbg/pipeline/features.py`:

```
Raw Indicators (benefits_from_stationary=False):
  → Computed once on original data
  → Cached, fast

Stationary Indicators (benefits_from_stationary=True):
  → Computed per fold on preprocessed data
  → Slower, but correct for stationarity-dependent features
```

---

## Available Plugins

### fractional_diff (fwbg-premium)

Fractional differentiation following López de Prado — makes time series stationary **without losing all memory** (unlike regular differencing).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_d` | `false` | Automatically find optimal d-value (ADF test) |
| `default_d` | `0.4` | Fixed d-value (0=original, 1=full differencing) |
| `columns` | `["O", "H", "L", "C"]` | Which columns to transform |

**Warning:** `auto_d: true` can cause lookahead bias if the ADF test runs on the entire dataset instead of just training data. Recommendation: `auto_d: false` with a fixed `default_d`.

---

## Strategy JSON Configuration

```json
"pipeline": {
  "preprocessing": [
    {
      "name": "fractional_diff",
      "params": {
        "auto_d": false,
        "default_d": 0.4,
        "columns": ["O", "H", "L", "C"]
      }
    }
  ]
}
```

---

## Creating a Custom Preprocessing Plugin

See [Plugin Development Guide](../plugin-development.md) for the full guide.

### Quick Example

```python
from fwbg_sdk import BasePreprocessor, register_preprocessor

@register_preprocessor("my_normalizer")
class MyNormalizer(BasePreprocessor):
    name = "my_normalizer"
    order = 50  # Before fractional_diff (order=100)

    def fit(self, df, **params):
        self.mean_ = df["C"].mean()
        self.std_ = df["C"].std()
        return super().fit(df, **params)

    def transform(self, df, **params):
        super().transform(df, **params)  # Checks that fit() was called
        result = df.copy()
        result["C"] = (result["C"] - self.mean_) / self.std_
        return result
```
