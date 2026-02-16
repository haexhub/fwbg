# Architecture & Plugin System

FWBG is built on a plugin pipeline architecture. Every piece of functionality — from indicators to exit strategies to data sources — is implemented as a plugin. Plugins can be added, swapped, or removed entirely without changing any other code.

---

## Pipeline Phases

The `PluginPhase` enum defines the execution order:

```
1. DATA_LOADING      → Load external data, compute features
2. PREPROCESSING     → Transform OHLC data (stationarity)
3. INDICATORS        → Compute technical indicators
4. FEATURE_SELECTION → Select relevant features
5. EXIT_STRATEGIES   → TP/SL computation
6. RISK_MANAGEMENT   → Position sizing and risk controls
7. LABELING          → Generate training labels (internal)
8. MODEL             → Train / predict ML model (internal)
9. VALIDATION        → Validate strategy performance (internal)
```

**Important:** The `PipelineRunner` executes phases 1–4 automatically. EXIT_STRATEGIES and RISK_MANAGEMENT are called **directly by the optimization code** (not by the runner). LABELING, MODEL, and VALIDATION are internal phases without user-configurable plugins.

Detailed per-phase documentation: [docs/phases/](phases/)

---

## BasePlugin — The Plugin Interface

All plugins inherit from `BasePlugin` (`src/fwbg/pipeline/base.py`):

```python
class BasePlugin(ABC):
    # Required attributes (must be defined by subclasses)
    name: str                    # Unique name (e.g., "trend")
    phase: PluginPhase           # Pipeline phase (e.g., PluginPhase.INDICATORS)

    # Optional attributes with defaults
    version: str = "0.1.0"      # Semantic version
    stateful: bool = False       # Stores state across calls?
    cacheable: bool = True       # Can results be cached?
    depends_on: List[str] = []   # Dependencies on other plugins

    # Methods
    def execute(self, ctx: PipelineContext, **params) -> PipelineContext: ...
    def fit(self, ctx: PipelineContext, **params) -> None: ...
    def reset(self) -> None: ...
    def validate(self) -> bool: ...

    @classmethod
    def get_default_params(cls) -> dict: ...

    def get_feature_columns(self) -> List[str]: ...
```

### Methods in Detail

| Method | Description |
|--------|-------------|
| `execute(ctx, **params)` | Main method — processes PipelineContext and returns it |
| `fit(ctx, **params)` | Learns parameters from training data (only for `stateful=True` plugins) |
| `reset()` | Resets learned state (called between CV folds) |
| `validate()` | Checks whether the plugin is correctly configured |
| `get_default_params()` | Returns default parameters (classmethod) |
| `get_feature_columns()` | Returns the generated feature column names |

---

## Plugin Lifecycle: stateful / cacheable / benefits_from_stationary

These three attributes determine when and how often a plugin is executed.

### `stateful` (bool, default: False)

Determines whether the plugin stores state learned from training data across calls.

- **`False` (default): Stateless.** The plugin produces the same result for the same input on every call. There is no `fit()` step. Example: Most indicators — `trend` computes ADX/EMA the same way regardless of which fold.

- **`True`: Stateful.** The plugin has a `fit()` step that learns parameters from training data. These learned parameters are then reused in `execute()`/`transform()`. `fit()` is called per CV fold **only on training data** (lookahead bias protection). Between folds, `reset()` is called. Example: `fractional_diff` — learns the optimal d-value on training data, then applies it to train/test/OOS.

### `cacheable` (bool, default: True)

Determines whether results can be cached to avoid redundant computation across folds.

- **`True` (default): Cacheable.** The `PipelineRunner` may cache and reuse results when input data and parameters haven't changed. Example: `volatility` — ATR on original data is identical for all folds, needs to be computed only once.

- **`False`: Not cacheable.** Results depend on state that changes between folds (e.g., fitted parameters). Each fold must be recomputed.

### `benefits_from_stationary` (bool, indicators only, default: False)

Determines whether an indicator is computed on preprocessed (stationary) or raw OHLC data. Only relevant when preprocessing is configured.

- **`False` (default):** Indicator is computed **once on original data** (before preprocessing). Results are reused across all folds. Example: `volatility`, `momentum`, `price_action`.

- **`True`:** Indicator is computed **per fold on preprocessed data**. Preprocessing (e.g., fractional differentiation) produces different transformed data per fold because `fit()` only learns on training data per fold. Example: `trend` — ADX on differentiated data yields different values than on raw data.

### Combination Table

| stateful | cacheable | Behavior | Example |
|----------|-----------|----------|---------|
| False | True | Compute once, cached across folds | `momentum`, `volatility` |
| False | False | Recompute every time | — |
| True | True | Fit per fold, cached within fold | — |
| True | False | Fit per fold, never cached | `fractional_diff` |

### Decision Guide for Plugin Developers

> **Question 1:** Does my plugin need to learn parameters from training data?
> - Yes → `stateful = True`, `cacheable = False`
> - No → `stateful = False`
>
> **Question 2:** Does my plugin always produce the same result for the same input?
> - Yes → `cacheable = True`
> - No → `cacheable = False`
>
> **Question 3 (indicators only):** Does my indicator benefit from stationary input data?
> - Yes → `benefits_from_stationary = True`
> - No → `benefits_from_stationary = False`

---

## Lifecycle Diagram (per Outer Fold)

```
┌─ Fold Start ──────────────────────────────────────────────────────────┐
│                                                                       │
│  Preprocessors:  reset() → fit(train) → transform(train)             │
│                                         → transform(test)             │
│                                                                       │
│  Stationary Indicators:  compute(preprocessed_data)  [per fold]       │
│  Raw Indicators:         (already precomputed once + cached)          │
│                                                                       │
│  Feature Selection:  select_features(X_train, y_train)                │
│                                                                       │
│  ── Inner CV Loop ──                                                  │
│  │  Exit Strategy:   compute_targets(df, ctx)  [cached per params]    │
│  │  Model:           train(X_train, y_train) → predict(X_test)        │
│  │  Validation:      evaluate fold results                            │
│  └──────────────────                                                  │
│                                                                       │
│  Risk Management:  compute_risk_params(trades, win_rate, rrr)         │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## PipelineContext

The context is passed through all phases (`src/fwbg/pipeline/context.py`):

```python
@dataclass
class PipelineContext:
    df: pd.DataFrame                  # Main DataFrame with OHLCV + features
    symbol: str                       # Asset symbol (e.g., "EURUSD")
    asset_class: str                  # Asset class (e.g., "FOREX", "CRYPTO")
    metadata: Dict[str, Any]          # Inter-plugin communication
    fold_info: Optional[Dict] = None  # Walk-forward fold info
```

- `df` is extended by each plugin (new columns for features)
- `metadata` enables inter-plugin communication (e.g., a data loader stores loaded supplementary data here)
- `clone()` creates a copy (for parallel processing)

---

## PipelineRunner

The `PipelineRunner` (`src/fwbg/pipeline/runner.py`) orchestrates plugin execution:

### Initialization

```python
runner = PipelineRunner(registry=registry, config=pipeline_config)
```

The runner:
1. Creates plugin instances from the config
2. Topologically sorts plugins within each phase by `depends_on`
3. Validates all dependencies

### Execution

```python
# Fit stateful plugins (per fold)
runner.fit(ctx)

# Run pipeline
ctx = runner.run(ctx)

# Reset stateful plugins (between folds)
runner.reset()
```

### Topological Sort (depends_on)

Plugins can declare dependencies within the same phase:

```python
class MyPlugin(BaseIndicator):
    name = "my_indicator"
    depends_on = ["trend", "momentum"]  # Short names suffice
```

The runner:
- Resolves short names to FQNs ("trend" → "fwbg-core:trend")
- Validates all dependencies exist and are in the same phase
- Sorts using Kahn's algorithm (dependencies execute first)
- Detects circular dependencies → ValueError

### Parameter Hierarchy

Parameters are merged (higher priority overrides):

```
1. Plugin.get_default_params()    → lowest priority
2. Strategy JSON config params    → medium priority
3. Global runtime params (CLI)    → highest priority
```

---

## Plugin Discovery

Plugins are automatically discovered from three sources — in this order:

### 1. Core Packages (Builtin)

Directory: `src/fwbg/plugins/fwbg-core/`

The bundled plugins (trend, momentum, volatility, fixed exit strategy, kelly, etc.).

### 2. Entry-Point Packages (pip-installed)

Installed Python packages with `fwbg.plugin_packages` entry point:

```toml
# In the plugin package's pyproject.toml:
[project.entry-points."fwbg.plugin_packages"]
fwbg-premium = "fwbg_premium:get_plugins_dir"
```

The entry-point function returns the path to the plugin directory. Example: `fwbg-premium` (`packages/fwbg-premium/`).

### 3. User Packages

Directory: `~/.fwbg/plugins/`

Custom plugins in the same directory format as core packages. Discovered last.

### Package Structure

Every plugin package has a standardized directory structure:

```
my-package/
├── manifest.json              # Package manifest
├── indicators/
│   └── my_indicator/
│       ├── manifest.json      # Plugin manifest
│       ├── __init__.py        # Plugin class
│       └── tests.py           # Optional: plugin tests
├── preprocessing/
│   └── ...
├── exit_strategies/
│   └── ...
├── feature_selection/
│   └── ...
├── risk_management/
│   └── ...
└── data_loading/
    └── ...
```

**Package Manifest** (`manifest.json` in root):
```json
{
  "name": "my-package",
  "version": "1.0.0",
  "description": "My plugin package",
  "plugins": {
    "indicators": ["my_indicator"],
    "preprocessing": []
  }
}
```

**Plugin Manifest** (`manifest.json` in plugin directory):
```json
{
  "name": "my_indicator",
  "version": "1.0.0",
  "description": "My custom indicator",
  "phase": "indicators",
  "benefits_from_stationary": false
}
```

Manifest attributes like `benefits_from_stationary` are automatically propagated to the plugin class during discovery.

---

## Name Resolution: FQN vs Short Names

### Fully Qualified Names (FQN)

Format: `"package:plugin"` — e.g., `"fwbg-core:trend"`, `"fwbg-premium:regime"`

FQNs are always unambiguous and are used internally by the registry.

### Short Names

Format: just `"plugin"` — e.g., `"trend"`, `"regime"`, `"fractional_diff"`

Short names are automatically resolved via `PluginRegistry.resolve_name()`. The method searches **all registered packages** (not just fwbg-core!) for a plugin with that name.

**Important:** Short names work for **all packages** — core, premium, and user packages alike. This is not a privilege of fwbg-core.

### Ambiguity

If two packages register a plugin with the same name, `resolve_name()` raises a `ValueError`:

```
ValueError: Ambiguous plugin name 'trend' — found in: fwbg-core, my-package.
Use fully qualified name: 'fwbg-core:trend' or 'my-package:trend'
```

### Recommendation

- **Short names** in strategy JSONs for readability: `{"name": "trend", "params": {}}`
- **FQN** only for name conflicts: `{"name": "fwbg-premium:regime", "params": {}}`

---

## Registration Decorators

Plugins are automatically registered during discovery. Additionally, there are decorators for explicit registration (`src/fwbg/core/registry.py`):

| Decorator | Plugin Type |
|-----------|------------|
| `@register_indicator("name")` | Indicator |
| `@register_preprocessor("name")` | Preprocessor |
| `@register_feature_selector("name")` | Feature Selector |
| `@register_exit_strategy("name")` | Exit Strategy |
| `@register_risk_manager("name")` | Risk Manager |
| `@register_data_loader("name")` | Data Loader |
| `@register_broker_adapter("name")` | Broker Adapter |

The decorators set `plugin.name` and register the class in the global registry.

---

## Plugin Tests

Every plugin can have its own `tests.py` in the plugin directory:

```python
# my_indicator/tests.py
def test_compute_basic():
    indicator = MyIndicator()
    df = create_sample_ohlcv()
    result = indicator.compute(df)
    assert "my_feature" in result.columns

def test_shift_applied():
    indicator = MyIndicator()
    df = create_sample_ohlcv()
    result = indicator.compute(df)
    assert pd.isna(result["my_feature"].iloc[0])  # NaN from shift
```

Tests are executed via:
- `plugin.run_tests()` → (passed, failed, errors)
- `plugin.has_tests()` → True/False

---

## Further Documentation

- [Plugin Development Guide](plugin-development.md) — Creating custom plugins
- [Phase 1: Data Loading](phases/1-data-loading.md)
- [Phase 2: Preprocessing](phases/2-preprocessing.md)
- [Phase 3: Indicators](phases/3-indicators.md)
- [Phase 4: Feature Selection](phases/4-feature-selection.md)
- [Phase 5: Exit Strategies](phases/5-exit-strategies.md)
- [Phase 6: Risk Management](phases/6-risk-management.md)
- [Phase 7: Validation](phases/7-validation.md)
