# Plugin Development Guide

Guide for creating custom FWBG plugins. Plugins can be indicators, preprocessors, feature selectors, exit strategies, risk managers, or data loaders.

---

## Quick Start: Custom Indicator

### 1. Directory Structure

```
~/.fwbg/plugins/
└── my-package/
    ├── manifest.json                    # Package manifest
    └── indicators/
        └── my_indicator/
            ├── manifest.json            # Plugin manifest
            ├── __init__.py              # Implementation
            └── tests.py                 # Optional: plugin tests
```

### 2. Package Manifest (`my-package/manifest.json`)

```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "My trading indicators",
  "plugins": {
    "indicators": ["my_indicator"]
  }
}
```

### 3. Plugin Manifest (`indicators/my_indicator/manifest.json`)

```json
{
  "name": "my_indicator",
  "version": "1.0.0",
  "description": "Custom momentum indicator",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

### 4. Implementation (`indicators/my_indicator/__init__.py`)

```python
import pandas as pd
import numpy as np
from fwbg.plugins.indicator import BaseIndicator, shift_features, safe_divide
from fwbg.pipeline.base import PluginPhase
from fwbg.core.registry import register_indicator


@register_indicator("my_indicator")
class MyIndicator(BaseIndicator):
    name = "my_indicator"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS
    group = "custom"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        lookback = params.get("lookback", 14)

        features = {}
        returns = df["C"].pct_change()
        features["my_momentum"] = returns.rolling(lookback).mean()
        features["my_volatility"] = returns.rolling(lookback).std()
        features["my_ratio"] = safe_divide(
            features["my_momentum"], features["my_volatility"]
        )

        # REQUIRED: shift_features() prevents lookahead bias
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> list:
        return ["my_momentum", "my_volatility", "my_ratio"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"lookback": 14}

    def validate(self) -> bool:
        return True
```

### 5. Use in Strategy JSON

```json
{
  "pipeline": {
    "indicators": [
      {"name": "my-package:my_indicator", "params": {"lookback": 21}},
      {"name": "trend", "params": {}}
    ]
  }
}
```

The plugin is automatically discovered and registered from `~/.fwbg/plugins/` on startup. If the name is unambiguous, the short name also works: `"name": "my_indicator"`.

---

## Plugin Type Reference

| Type | Base Class | Decorator | Phase | Executed by PipelineRunner? | Directory |
|------|-----------|-----------|-------|-----------------------------|-----------|
| Indicator | `BaseIndicator` | `@register_indicator` | INDICATORS | Yes | `indicators/` |
| Preprocessor | `BasePreprocessor` | `@register_preprocessor` | PREPROCESSING | Yes | `preprocessing/` |
| Feature Selector | `BaseFeatureSelector` | `@register_feature_selector` | FEATURE_SELECTION | Yes (in inner CV) | `feature_selection/` |
| Exit Strategy | `BaseExitStrategy` | `@register_exit_strategy` | EXIT_STRATEGIES | No (optimization code) | `exit_strategies/` |
| Risk Manager | `BaseRiskManager` | `@register_risk_manager` | RISK_MANAGEMENT | No (optimization code) | `risk_management/` |
| Data Loader | `BaseDataLoader` | `@register_data_loader` | DATA_LOADING | Yes | `data_loading/` |

---

## Plugin Type: Indicator

**File:** `src/fwbg/plugins/indicator.py`

```python
class BaseIndicator(BasePlugin, ABC):
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True
    group: str = "custom"
    benefits_from_stationary: bool = False

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...
    def get_feature_columns(self) -> List[str]: ...
```

**Required:**
- `shift_features()` at the end of `compute()` — 1-bar shift to prevent lookahead bias
- `safe_divide()` for all divisions — NaN instead of inf on division by ~0

**Key attributes:**
- `benefits_from_stationary = False` → compute once on raw OHLC, cached
- `benefits_from_stationary = True` → compute per fold on preprocessed data

Detailed documentation: [Phase 3: Indicators](phases/3-indicators.md)

---

## Plugin Type: Preprocessor

**File:** `src/fwbg/plugins/preprocessor.py`

```python
class BasePreprocessor(BasePlugin, ABC):
    phase = PluginPhase.PREPROCESSING
    order: int = 100

    def fit(self, df: pd.DataFrame, **params) -> "BasePreprocessor": ...

    @abstractmethod
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...

    def fit_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...
    def inverse_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame: ...
```

**Lifecycle per fold:** `reset()` → `fit(train)` → `transform(train)` → `transform(test)`

**Required:**
- Call `fit()` only on training data — lookahead bias prevention
- Set `order` for execution priority when using multiple preprocessors

**Example:**

```python
@register_preprocessor("my_normalizer")
class MyNormalizer(BasePreprocessor):
    name = "my_normalizer"
    order = 50

    def fit(self, df, **params):
        self.mean_ = df["C"].mean()
        self.std_ = df["C"].std()
        return super().fit(df, **params)

    def transform(self, df, **params):
        super().transform(df, **params)
        result = df.copy()
        result["C"] = (result["C"] - self.mean_) / self.std_
        return result
```

Detailed documentation: [Phase 2: Preprocessing](phases/2-preprocessing.md)

---

## Plugin Type: Feature Selector

**File:** `src/fwbg/plugins/feature_selector.py`

```python
class BaseFeatureSelector(BasePlugin, ABC):
    phase = PluginPhase.FEATURE_SELECTION

    @abstractmethod
    def select_features(self, X: pd.DataFrame, y: np.ndarray,
                       max_features: int = None, **params) -> Tuple[List[str], dict]: ...
```

**Return:** `(selected_feature_names, metadata_dict)`

**Example:**

```python
@register_feature_selector("my_selector")
class MySelector(BaseFeatureSelector):
    name = "my_selector"

    def select_features(self, X, y, max_features=None, **params):
        importances = compute_importances(X, y)
        top_features = sorted(importances, key=importances.get, reverse=True)
        if max_features:
            top_features = top_features[:max_features]
        return top_features, {"importances": importances}
```

Detailed documentation: [Phase 4: Feature Selection](phases/4-feature-selection.md)

---

## Plugin Type: Exit Strategy

**File:** `src/fwbg/plugins/exit_strategy.py`

```python
class BaseExitStrategy(BasePlugin, ABC):
    phase = PluginPhase.EXIT_STRATEGIES

    @abstractmethod
    def compute_targets(self, df, ctx, **params) -> Tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def iterate_grid(self, grid_config, ctx) -> Iterator[dict]: ...

    @abstractmethod
    def get_cache_key(self, params) -> str: ...
```

**Three abstract methods:**
1. `compute_targets()` — computes win/loss arrays (1.0/0.0) for long and short
2. `iterate_grid()` — generates parameter combinations from grid config
3. `get_cache_key()` — unique cache key per parameter combination

**Example:**

```python
@register_exit_strategy("my_exit")
class MyExitStrategy(BaseExitStrategy):
    name = "my_exit"

    def compute_targets(self, df, ctx, **params):
        tp = params.get("tp", 30)
        sl = params.get("sl", 20)
        # Simulation via Numba...
        return targets_long, targets_short

    def iterate_grid(self, grid_config, ctx):
        for tp in grid_config.get("tp", [30]):
            for sl in grid_config.get("sl", [20]):
                yield {"tp": tp, "sl": sl}

    def get_cache_key(self, params):
        return f"my_tp{params['tp']}_sl{params['sl']}"
```

Detailed documentation: [Phase 5: Exit Strategies](phases/5-exit-strategies.md)

---

## Plugin Type: Risk Manager

**File:** `src/fwbg/plugins/risk_manager.py`

```python
class BaseRiskManager(BasePlugin, ABC):
    phase = PluginPhase.RISK_MANAGEMENT

    @abstractmethod
    def compute_risk_params(self, trades, win_rate, rrr, **params) -> Dict[str, Any]: ...
```

**Return dict must contain:**
- `risk_per_trade`: float — position size as fraction of capital
- `trade_returns`: List[float] — per-trade returns
- `circuit_breaker`: dict — pause logic after consecutive losses
- `risk_adjustment`: dict — scaling factors

Detailed documentation: [Phase 6: Risk Management](phases/6-risk-management.md)

---

## Plugin Type: Data Loader

**File:** `src/fwbg/plugins/data_loader.py`

```python
class BaseDataLoader(BasePlugin, ABC):
    phase = PluginPhase.DATA_LOADING
    stateful = False

    @abstractmethod
    def execute(self, ctx, **params): ...
```

**Important:** Data loaders do no I/O. Raw data is already in the DataFrame (loaded by the orchestrator). Data loaders only compute derived features.

Detailed documentation: [Phase 1: Data Loading](phases/1-data-loading.md)

---

## Plugin Testing

Every plugin can have a `tests.py` in the plugin directory:

```python
# my_indicator/tests.py
def test_compute_produces_features():
    from . import MyIndicator
    indicator = MyIndicator()
    df = pd.DataFrame({"O": [...], "H": [...], "L": [...], "C": [...], "V": [...]})
    result = indicator.compute(df)
    assert "my_momentum" in result.columns

def test_shift_applied():
    from . import MyIndicator
    indicator = MyIndicator()
    df = pd.DataFrame({"O": [...], "H": [...], "L": [...], "C": [...], "V": [...]})
    result = indicator.compute(df)
    assert pd.isna(result["my_momentum"].iloc[0])  # First row NaN from shift
```

Running tests:
```python
plugin = MyIndicator()
passed, failed, errors = plugin.run_tests()
print(f"{passed} passed, {failed} failed")
```

---

## Entry-Point Registration (pip-installable packages)

For plugin packages installed via `pip install`, an entry point must be defined in `pyproject.toml`:

```toml
[project.entry-points."fwbg.plugin_packages"]
my-package = "my_package:get_plugins_dir"
```

The entry-point function returns the path to the plugin directory:

```python
# my_package/__init__.py
from pathlib import Path

def get_plugins_dir() -> Path:
    return Path(__file__).parent / "plugins" / "my-package"
```

The plugin directory has the same structure as user plugins (manifest.json, subdirectories per plugin type).

---

## Common Mistakes

### 1. Forgetting shift_features()

```python
# WRONG — Lookahead bias!
def compute(self, df, **params):
    features = {"my_rsi": compute_rsi(df["C"])}
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

# CORRECT
def compute(self, df, **params):
    features = {"my_rsi": compute_rsi(df["C"])}
    features_df = shift_features(features, df.index)  # ← REQUIRED
    return pd.concat([df, features_df], axis=1)
```

Without shift_features(), the model sees the indicator value for bar `i` at bar `i` — the current, not yet completed bar. This produces unrealistic backtesting results.

### 2. Fitting preprocessor on all data

```python
# WRONG — Lookahead bias!
preprocessor.fit(all_data)
preprocessor.transform(train_data)
preprocessor.transform(test_data)

# CORRECT
preprocessor.fit(train_data)           # Train only!
preprocessor.transform(train_data)
preprocessor.transform(test_data)      # Same parameters as fit()
```

### 3. Forgetting safe_divide()

```python
# WRONG — can produce inf
ratio = momentum / volatility

# CORRECT — returns NaN when denominator ~0
ratio = safe_divide(momentum, volatility)
```

### 4. Wrong benefits_from_stationary setting

- `True` for indicators that benefit from stationary data (trend, moving averages)
- `False` for indicators that are already normalized (RSI, stochastic) or scale-independent (ATR)

Wrong setting: either unnecessarily slow (False→True) or incorrect results (True→False, when the indicator actually needs stationary input data).

---

## Further Documentation

- [Architecture & Plugin System](architecture.md) — Lifecycle, discovery, naming
- [Phase 1: Data Loading](phases/1-data-loading.md)
- [Phase 2: Preprocessing](phases/2-preprocessing.md)
- [Phase 3: Indicators](phases/3-indicators.md)
- [Phase 4: Feature Selection](phases/4-feature-selection.md)
- [Phase 5: Exit Strategies](phases/5-exit-strategies.md)
- [Phase 6: Risk Management](phases/6-risk-management.md)
