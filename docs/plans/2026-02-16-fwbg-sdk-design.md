# fwbg-sdk Design

## Goal

A lightweight, pip-installable SDK package that allows external developers to build plugins (indicators, preprocessors, feature selectors, exit strategies, risk managers, data loaders) for the FWBG framework — without pulling in the full optimizer.

## Key Decisions

| Decision | Choice |
|----------|--------|
| Location | Monorepo: `packages/fwbg-sdk/` |
| Dependency direction | fwbg depends on fwbg-sdk (SDK is source of truth) |
| Import style | Flat namespace: `from fwbg_sdk import ...` |
| Testing utilities | Included (create_sample_ohlcv, assert_*) |
| CLI scaffolding | From day 1: `fwbg-sdk init` |
| SimulationContext | Simplified: `AssetInfo` dataclass instead of full context |
| Versioning | SemVer with stable public API |

---

## Package Structure

```
packages/fwbg-sdk/
├── pyproject.toml                    # deps: pandas, numpy, click
├── src/fwbg_sdk/
│   ├── __init__.py                   # Flat re-export of everything
│   ├── base.py                       # BasePlugin, PluginPhase
│   ├── indicators.py                 # BaseIndicator, shift_features, safe_divide
│   ├── preprocessors.py              # BasePreprocessor
│   ├── feature_selectors.py          # BaseFeatureSelector
│   ├── exit_strategies.py            # BaseExitStrategy
│   ├── risk_managers.py              # BaseRiskManager
│   ├── data_loaders.py               # BaseDataLoader
│   ├── contexts.py                   # PipelineContext, AssetInfo
│   ├── enums.py                      # Timeframe, AssetClass, Symbol, Direction
│   ├── registry.py                   # @register_indicator, etc. + global registries
│   ├── testing.py                    # Test helpers for plugin developers
│   └── cli.py                        # fwbg-sdk init / add
└── tests/
    ├── test_base_classes.py
    ├── test_helpers.py
    ├── test_registry.py
    ├── test_testing_utils.py
    └── test_cli.py
```

### Dependencies (pyproject.toml)

```toml
[project]
name = "fwbg-sdk"
dependencies = ["pandas", "numpy", "click"]

[project.scripts]
fwbg-sdk = "fwbg_sdk.cli:main"
```

No numba, sklearn, xgboost, or other heavy dependencies.

---

## Public API (Flat Namespace)

Everything accessible via `from fwbg_sdk import ...`:

```python
from fwbg_sdk import (
    # Base classes
    BasePlugin, BaseIndicator, BasePreprocessor,
    BaseFeatureSelector, BaseExitStrategy, BaseRiskManager, BaseDataLoader,

    # Helpers (mandatory for indicators)
    shift_features, safe_divide,

    # Registration decorators
    register_indicator, register_preprocessor, register_feature_selector,
    register_exit_strategy, register_risk_manager, register_data_loader,

    # Types & Contexts
    PluginPhase, PipelineContext, AssetInfo,

    # Enums
    Timeframe, AssetClass, Symbol, Direction,

    # Testing utilities
    create_sample_ohlcv, assert_features_shifted, assert_no_lookahead,
    assert_no_inf, create_sample_context, create_sample_asset,
)
```

---

## AssetInfo (Simplified Exit Strategy Interface)

Replaces the heavy `SimulationContext` for plugin-facing APIs:

```python
@dataclass
class AssetInfo:
    symbol: str         # e.g. "EURUSD"
    asset_class: str    # e.g. "FOREX"
    spread: float       # Asset spread
    point: float        # Minimum price movement (pip value)
```

Exit strategy interface becomes:
```python
class BaseExitStrategy(BasePlugin, ABC):
    @abstractmethod
    def compute_targets(self, df: pd.DataFrame, asset: AssetInfo,
                       **params) -> Tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def iterate_grid(self, grid_config: dict, asset: AssetInfo) -> Iterator[dict]: ...

    @abstractmethod
    def get_cache_key(self, params: dict) -> str: ...
```

The orchestrator in fwbg builds `AssetInfo` from `SimulationContext` — a 3-line bridge.

---

## Testing Utilities

| Function | Purpose |
|----------|---------|
| `create_sample_ohlcv(bars, seed)` | Realistic OHLCV data (random walk + trends) |
| `assert_features_shifted(df, cols)` | First row NaN, no off-by-one |
| `assert_no_lookahead(df, col, original)` | Feature at bar i only uses data <= i-1 |
| `assert_no_inf(df, cols)` | No inf values (safe_divide used correctly) |
| `create_sample_context(symbol)` | PipelineContext with sample data |
| `create_sample_asset(symbol)` | AssetInfo instance for exit strategy tests |

---

## CLI Scaffolding

### Create new plugin package

```bash
fwbg-sdk init my-indicators --plugin indicator:my_rsi
```

Generates:

```
my-indicators/
├── pyproject.toml                    # pip-installable, entry-point registered
├── README.md
├── src/my_indicators/
│   ├── __init__.py                   # get_plugins_dir() for entry-point
│   └── plugins/
│       └── my-indicators/
│           ├── manifest.json
│           └── indicators/
│               └── my_rsi/
│                   ├── manifest.json
│                   ├── __init__.py   # Working boilerplate with TODOs
│                   └── tests.py      # 3 green smoke tests
└── tests/
    └── test_my_rsi.py                # pytest-compatible
```

Generated `pyproject.toml`:
```toml
[project]
name = "fwbg-my-indicators"
version = "0.1.0"
dependencies = ["fwbg-sdk"]

[project.entry-points."fwbg.plugin_packages"]
my-indicators = "my_indicators:get_plugins_dir"
```

### Add plugin to existing package

```bash
cd my-indicators/
fwbg-sdk add indicator my_macd
```

Adds `indicators/my_macd/` and updates `manifest.json`. No existing files overwritten.

### Distribution

```bash
# Developer builds:
pip install build && python -m build

# Users install:
pip install fwbg-my-indicators
# or from git:
pip install git+https://github.com/user/fwbg-my-indicators.git

# Plugin auto-discovered by FWBG via entry-point
```

---

## Migration: fwbg Internal Changes

### Step 1: Move code from fwbg to fwbg-sdk

| Source (fwbg) | Destination (fwbg-sdk) |
|---------------|----------------------|
| `src/fwbg/pipeline/base.py` | `base.py` (BasePlugin, PluginPhase) |
| `src/fwbg/plugins/indicator.py` | `indicators.py` (BaseIndicator, helpers) |
| `src/fwbg/plugins/preprocessor.py` | `preprocessors.py` |
| `src/fwbg/plugins/feature_selector.py` | `feature_selectors.py` |
| `src/fwbg/plugins/exit_strategy.py` | `exit_strategies.py` |
| `src/fwbg/plugins/risk_manager.py` | `risk_managers.py` |
| `src/fwbg/plugins/data_loader.py` | `data_loaders.py` |
| `src/fwbg/pipeline/context.py` | `contexts.py` (PipelineContext) |
| `src/fwbg/core/enums.py` | `enums.py` |
| Registration decorators from `src/fwbg/core/registry.py` | `registry.py` |

### Step 2: Delete old files in fwbg

No backward-compatibility re-exports. Delete the moved files.

### Step 3: Update all imports in fwbg (~50-80 sites)

```python
# Before:
from fwbg.plugins.indicator import BaseIndicator, shift_features
from fwbg.pipeline.base import PluginPhase

# After:
from fwbg_sdk import BaseIndicator, shift_features, PluginPhase
```

### Step 4: Update fwbg-core and fwbg-premium plugins

Same mechanical import change.

### What stays in fwbg:

- `SimulationContext` (orchestrator-internal, builds AssetInfo for plugins)
- `PipelineRunner` (orchestration)
- `PluginRegistry` discovery logic
- Optimization, simulation, CLI, results

---

## API Stability (SemVer)

### Stable (breaking change = major version bump):

- Base class method signatures
- Helper function behavior and parameters
- Required fields on AssetInfo / PipelineContext
- Registration decorator interface
- Enum values

### Non-breaking (minor/patch):

- New optional fields on contexts
- New test helpers
- New CLI commands
- New optional base class methods with default implementation

Plugin developers pin `fwbg-sdk>=1.0,<2.0` for guaranteed compatibility.

---

## Community Package

The same mechanism supports a curated community package:

```
fwbg-community/
├── pyproject.toml
├── src/fwbg_community/
│   ├── __init__.py
│   └── plugins/
│       └── fwbg-community/
│           ├── manifest.json
│           ├── indicators/
│           │   ├── community_rsi_divergence/
│           │   ├── community_order_flow/
│           │   └── ...
│           └── exit_strategies/
│               └── community_trailing_stop/
```

Users install: `pip install fwbg-community` — all plugins auto-discovered.
