# fwbg-sdk Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract plugin base classes, helpers, types, and registry into a standalone `fwbg-sdk` pip package under `packages/fwbg-sdk/`, then update all imports in fwbg, fwbg-premium, and tests.

**Architecture:** fwbg-sdk is the source of truth for all plugin interfaces. fwbg depends on fwbg-sdk. All `from fwbg.plugins.indicator import BaseIndicator` become `from fwbg_sdk import BaseIndicator`. No backward-compatibility re-exports.

**Tech Stack:** Python 3.11+, pandas, numpy, click (CLI only)

---

## Task 1: SDK Package Skeleton

**Files:**
- Create: `packages/fwbg-sdk/pyproject.toml`
- Create: `packages/fwbg-sdk/src/fwbg_sdk/__init__.py` (empty placeholder)

**Step 1: Create directory structure**

```bash
mkdir -p packages/fwbg-sdk/src/fwbg_sdk
mkdir -p packages/fwbg-sdk/tests
```

**Step 2: Write pyproject.toml**

Create `packages/fwbg-sdk/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fwbg-sdk"
version = "1.0.0"
description = "SDK for building FWBG plugins - indicators, preprocessors, exit strategies & more"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "FWBG Team"}
]
keywords = ["trading", "plugins", "sdk", "fwbg"]

dependencies = [
    "numpy>=2.2",
    "pandas>=3.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=9.0",
]

[project.scripts]
fwbg-sdk = "fwbg_sdk.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

**Step 3: Write empty __init__.py**

Create `packages/fwbg-sdk/src/fwbg_sdk/__init__.py`:
```python
"""FWBG SDK - Build plugins for the FWBG trading framework."""
```

**Step 4: Install SDK in editable mode**

```bash
pip install -e packages/fwbg-sdk
```

**Step 5: Verify import works**

```bash
python -c "import fwbg_sdk; print('OK')"
```

**Step 6: Commit**

```bash
git add packages/fwbg-sdk/
git commit -m "feat(sdk): add fwbg-sdk package skeleton"
```

---

## Task 2: Move Base Types (PluginPhase, BasePlugin, PipelineContext, AssetInfo, Enums)

**Files:**
- Create: `packages/fwbg-sdk/src/fwbg_sdk/base.py` (from `src/fwbg/pipeline/base.py`)
- Create: `packages/fwbg-sdk/src/fwbg_sdk/contexts.py` (from `src/fwbg/pipeline/context.py` + new `AssetInfo`)
- Create: `packages/fwbg-sdk/src/fwbg_sdk/enums.py` (from `src/fwbg/core/enums.py`)

**Step 1: Copy `base.py`**

Copy `src/fwbg/pipeline/base.py` → `packages/fwbg-sdk/src/fwbg_sdk/base.py`

Change the TYPE_CHECKING import:
```python
# Old:
if TYPE_CHECKING:
    from fwbg.pipeline.context import PipelineContext

# New:
if TYPE_CHECKING:
    from fwbg_sdk.contexts import PipelineContext
```

Everything else stays identical (BasePlugin, PluginPhase — no other internal imports).

**Step 2: Copy `enums.py`**

Copy `src/fwbg/core/enums.py` → `packages/fwbg-sdk/src/fwbg_sdk/enums.py` verbatim. This file has zero internal imports.

**Step 3: Create `contexts.py`**

Copy `src/fwbg/pipeline/context.py` → `packages/fwbg-sdk/src/fwbg_sdk/contexts.py`.

Then add `AssetInfo` dataclass at the top:
```python
@dataclass
class AssetInfo:
    """Lightweight asset info for plugin-facing APIs (exit strategies)."""
    symbol: str
    asset_class: str
    spread: float
    point: float
    max_trade_bars: Optional[int] = None
```

This file has zero internal imports (only stdlib + pandas).

**Step 4: Verify imports work**

```bash
python -c "from fwbg_sdk.base import BasePlugin, PluginPhase; print('OK')"
python -c "from fwbg_sdk.contexts import PipelineContext, AssetInfo; print('OK')"
python -c "from fwbg_sdk.enums import Timeframe, Symbol, AssetClass; print('OK')"
```

**Step 5: Commit**

```bash
git add packages/fwbg-sdk/src/fwbg_sdk/
git commit -m "feat(sdk): add base types, contexts, enums"
```

---

## Task 3: Move Plugin Base Classes

**Files:**
- Create: `packages/fwbg-sdk/src/fwbg_sdk/indicators.py`
- Create: `packages/fwbg-sdk/src/fwbg_sdk/preprocessors.py`
- Create: `packages/fwbg-sdk/src/fwbg_sdk/feature_selectors.py`
- Create: `packages/fwbg-sdk/src/fwbg_sdk/exit_strategies.py`
- Create: `packages/fwbg-sdk/src/fwbg_sdk/risk_managers.py`
- Create: `packages/fwbg-sdk/src/fwbg_sdk/data_loaders.py`

**Step 1: Copy `indicators.py`**

Copy `src/fwbg/plugins/indicator.py` → `packages/fwbg-sdk/src/fwbg_sdk/indicators.py`

Update internal imports:
```python
# Old:
from fwbg.pipeline.base import BasePlugin, PluginPhase
if TYPE_CHECKING:
    from fwbg.pipeline.context import PipelineContext

# New:
from fwbg_sdk.base import BasePlugin, PluginPhase
if TYPE_CHECKING:
    from fwbg_sdk.contexts import PipelineContext
```

This file exports: `EPSILON`, `safe_divide`, `shift_features`, `BaseIndicator`.

**Step 2: Copy remaining base classes**

For each of these files, copy and update the single import line:

| Source | Destination | Import change |
|--------|------------|---------------|
| `src/fwbg/plugins/preprocessor.py` | `fwbg_sdk/preprocessors.py` | `from fwbg.pipeline.base` → `from fwbg_sdk.base` |
| `src/fwbg/plugins/feature_selector.py` | `fwbg_sdk/feature_selectors.py` | `from fwbg.pipeline.base` → `from fwbg_sdk.base` |
| `src/fwbg/plugins/risk_manager.py` | `fwbg_sdk/risk_managers.py` | `from fwbg.pipeline.base` → `from fwbg_sdk.base` |
| `src/fwbg/plugins/data_loader.py` | `fwbg_sdk/data_loaders.py` | `from fwbg.pipeline.base` → `from fwbg_sdk.base` |

**Step 3: Create `exit_strategies.py` with AssetInfo**

Copy `src/fwbg/plugins/exit_strategy.py` → `packages/fwbg-sdk/src/fwbg_sdk/exit_strategies.py`

Update imports AND change the `SimulationContext` type hint to `AssetInfo`:
```python
# Old:
from fwbg.pipeline.base import BasePlugin, PluginPhase
if TYPE_CHECKING:
    from ..core.context import SimulationContext

# New:
from fwbg_sdk.base import BasePlugin, PluginPhase
from fwbg_sdk.contexts import AssetInfo
```

In the abstract method signatures, change `ctx: "SimulationContext"` to `ctx: "AssetInfo"`:
```python
@abstractmethod
def compute_targets(self, df: pd.DataFrame, ctx: "AssetInfo", **params) -> ...:
    ...

@abstractmethod
def iterate_grid(self, grid_config: dict, ctx: "AssetInfo") -> ...:
    ...
```

**Note:** The `ctx` parameter name stays as `ctx` (not `asset`) to minimize changes in implementations. Exit strategy plugins already use `ctx.spread` — with AssetInfo they'll still use `ctx.spread`. Only the type changes.

**Step 4: Verify all imports**

```bash
python -c "from fwbg_sdk.indicators import BaseIndicator, shift_features, safe_divide, EPSILON; print('OK')"
python -c "from fwbg_sdk.preprocessors import BasePreprocessor; print('OK')"
python -c "from fwbg_sdk.feature_selectors import BaseFeatureSelector; print('OK')"
python -c "from fwbg_sdk.exit_strategies import BaseExitStrategy; print('OK')"
python -c "from fwbg_sdk.risk_managers import BaseRiskManager; print('OK')"
python -c "from fwbg_sdk.data_loaders import BaseDataLoader; print('OK')"
```

**Step 5: Commit**

```bash
git add packages/fwbg-sdk/src/fwbg_sdk/
git commit -m "feat(sdk): add all plugin base classes"
```

---

## Task 4: Move Registry (Decorators + Global Dicts)

**Files:**
- Create: `packages/fwbg-sdk/src/fwbg_sdk/registry.py`
- Modify: `src/fwbg/core/registry.py` (keep discovery, import registries from SDK)

**Step 1: Create SDK registry**

Create `packages/fwbg-sdk/src/fwbg_sdk/registry.py` containing:
- All 7 global registry dicts (`INDICATOR_REGISTRY`, etc.)
- All 7 decorator functions (`register_indicator`, etc.)
- All 7 getter functions (`get_indicator`, etc.) — simple dict lookups, NO discovery
- All 7 list functions (`list_indicators`, etc.)
- `BROKER_ADAPTER_REGISTRY` + `register_broker_adapter` + `get_broker_adapter` + `list_broker_adapters`

The decorator functions are extracted from `src/fwbg/core/registry.py`. They currently do:
1. Add class to registry dict
2. Set `cls.name`
3. Log registration

The getter functions become simple lookups:
```python
def get_indicator(name: str):
    if name not in INDICATOR_REGISTRY:
        raise KeyError(f"Indicator '{name}' not found. Available: {list(INDICATOR_REGISTRY.keys())}")
    return INDICATOR_REGISTRY[name]
```

No `_ensure_plugins_loaded()` call — that stays in fwbg.

**Step 2: Update `src/fwbg/core/registry.py`**

Replace the registry dicts, decorators, getters, and listers with imports from SDK:
```python
from fwbg_sdk.registry import (
    INDICATOR_REGISTRY, EXIT_STRATEGY_REGISTRY, FEATURE_SELECTOR_REGISTRY,
    PREPROCESSOR_REGISTRY, BROKER_ADAPTER_REGISTRY, RISK_MANAGER_REGISTRY,
    DATA_LOADER_REGISTRY,
    register_indicator, register_exit_strategy, register_feature_selector,
    register_preprocessor, register_broker_adapter, register_risk_manager,
    register_data_loader,
)
```

Keep in `src/fwbg/core/registry.py`:
- `discover_plugins()` function
- `_ensure_plugins_loaded()` function
- Wrapped getters that call `_ensure_plugins_loaded()` first (override the SDK imports)

```python
# Override SDK getters with discovery-aware versions
def get_indicator(name: str):
    _ensure_plugins_loaded()
    return _sdk_get_indicator(name)  # imported as _sdk_get_indicator from SDK
```

Actually simpler: just call `_ensure_plugins_loaded()` then look up in the SDK registry dict directly:
```python
from fwbg_sdk.registry import INDICATOR_REGISTRY

def get_indicator(name: str):
    _ensure_plugins_loaded()
    if name not in INDICATOR_REGISTRY:
        raise KeyError(...)
    return INDICATOR_REGISTRY[name]
```

**Step 3: Verify**

```bash
python -c "from fwbg_sdk.registry import register_indicator, INDICATOR_REGISTRY; print('OK')"
python -c "from fwbg.core.registry import discover_plugins; print('OK')"
```

**Step 4: Commit**

```bash
git add packages/fwbg-sdk/src/fwbg_sdk/registry.py src/fwbg/core/registry.py
git commit -m "feat(sdk): extract registry decorators and dicts"
```

---

## Task 5: SDK Flat Namespace (__init__.py)

**Files:**
- Modify: `packages/fwbg-sdk/src/fwbg_sdk/__init__.py`

**Step 1: Write the flat namespace**

```python
"""FWBG SDK - Build plugins for the FWBG trading framework."""

from fwbg_sdk.base import BasePlugin, PluginPhase
from fwbg_sdk.contexts import PipelineContext, AssetInfo
from fwbg_sdk.enums import Timeframe, AssetClass, Symbol, Direction, SignalType
from fwbg_sdk.indicators import BaseIndicator, shift_features, safe_divide, EPSILON
from fwbg_sdk.preprocessors import BasePreprocessor
from fwbg_sdk.feature_selectors import BaseFeatureSelector
from fwbg_sdk.exit_strategies import BaseExitStrategy
from fwbg_sdk.risk_managers import BaseRiskManager
from fwbg_sdk.data_loaders import BaseDataLoader
from fwbg_sdk.registry import (
    register_indicator, register_preprocessor, register_feature_selector,
    register_exit_strategy, register_risk_manager, register_data_loader,
    register_broker_adapter,
)

__all__ = [
    # Base
    "BasePlugin", "PluginPhase",
    # Plugin base classes
    "BaseIndicator", "BasePreprocessor", "BaseFeatureSelector",
    "BaseExitStrategy", "BaseRiskManager", "BaseDataLoader",
    # Helpers
    "shift_features", "safe_divide", "EPSILON",
    # Contexts
    "PipelineContext", "AssetInfo",
    # Enums
    "Timeframe", "AssetClass", "Symbol", "Direction", "SignalType",
    # Registration
    "register_indicator", "register_preprocessor", "register_feature_selector",
    "register_exit_strategy", "register_risk_manager", "register_data_loader",
    "register_broker_adapter",
]
```

**Step 2: Verify flat import works**

```bash
python -c "from fwbg_sdk import BaseIndicator, shift_features, register_indicator, PipelineContext, AssetInfo, PluginPhase; print('OK')"
```

**Step 3: Commit**

```bash
git add packages/fwbg-sdk/src/fwbg_sdk/__init__.py
git commit -m "feat(sdk): flat namespace re-exports"
```

---

## Task 6: Testing Utilities

**Files:**
- Create: `packages/fwbg-sdk/src/fwbg_sdk/testing.py`
- Create: `packages/fwbg-sdk/tests/test_testing.py`

**Step 1: Write failing tests for test utilities**

Create `packages/fwbg-sdk/tests/test_testing.py`:
```python
import pandas as pd
import numpy as np
from fwbg_sdk.testing import (
    create_sample_ohlcv, assert_features_shifted,
    assert_no_inf, create_sample_asset,
)


def test_create_sample_ohlcv_returns_dataframe():
    df = create_sample_ohlcv(bars=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert set(df.columns) >= {"O", "H", "L", "C", "V"}


def test_create_sample_ohlcv_is_deterministic_with_seed():
    df1 = create_sample_ohlcv(bars=50, seed=42)
    df2 = create_sample_ohlcv(bars=50, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_create_sample_ohlcv_hlc_valid():
    df = create_sample_ohlcv(bars=200)
    assert (df["H"] >= df["L"]).all()
    assert (df["H"] >= df["O"]).all()
    assert (df["H"] >= df["C"]).all()
    assert (df["L"] <= df["O"]).all()
    assert (df["L"] <= df["C"]).all()


def test_assert_features_shifted_passes_on_shifted():
    from fwbg_sdk import shift_features
    features = {"feat_a": pd.Series([1.0, 2.0, 3.0, 4.0])}
    idx = pd.RangeIndex(4)
    shifted = shift_features(features, idx)
    df = pd.DataFrame({"O": [1, 2, 3, 4]}, index=idx)
    result = pd.concat([df, shifted], axis=1)
    assert_features_shifted(result, ["feat_a"])  # should not raise


def test_assert_features_shifted_fails_on_unshifted():
    df = pd.DataFrame({"O": [1, 2, 3, 4], "feat_a": [1.0, 2.0, 3.0, 4.0]})
    try:
        assert_features_shifted(df, ["feat_a"])
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_assert_no_inf_passes_clean():
    df = pd.DataFrame({"feat": [1.0, 2.0, float("nan"), 4.0]})
    assert_no_inf(df, ["feat"])


def test_assert_no_inf_fails_on_inf():
    df = pd.DataFrame({"feat": [1.0, float("inf"), 3.0]})
    try:
        assert_no_inf(df, ["feat"])
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_create_sample_asset():
    asset = create_sample_asset("EURUSD")
    assert asset.symbol == "EURUSD"
    assert asset.asset_class == "FOREX"
    assert asset.spread > 0
    assert asset.point > 0
```

**Step 2: Run tests — verify they fail**

```bash
python -m pytest packages/fwbg-sdk/tests/test_testing.py -x -q
```
Expected: FAIL (module not found)

**Step 3: Implement testing.py**

Create `packages/fwbg-sdk/src/fwbg_sdk/testing.py`:
```python
"""Testing utilities for FWBG plugin developers."""
import numpy as np
import pandas as pd
from fwbg_sdk.contexts import AssetInfo


def create_sample_ohlcv(bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV data for testing."""
    rng = np.random.default_rng(seed)

    # Random walk for close prices
    returns = rng.normal(0, 0.001, bars)
    close = 1.1000 + np.cumsum(returns)

    # Generate OHLC from close
    high_offset = np.abs(rng.normal(0, 0.0005, bars))
    low_offset = np.abs(rng.normal(0, 0.0005, bars))
    open_offset = rng.normal(0, 0.0003, bars)

    opens = close + open_offset
    highs = np.maximum(close, opens) + high_offset
    lows = np.minimum(close, opens) - low_offset
    volume = rng.integers(100, 10000, bars).astype(float)

    idx = pd.date_range("2020-01-01", periods=bars, freq="h")
    return pd.DataFrame(
        {"O": opens, "H": highs, "L": lows, "C": close, "V": volume},
        index=idx,
    )


def assert_features_shifted(df: pd.DataFrame, feature_cols: list) -> None:
    """Assert that feature columns are properly shifted (first row NaN)."""
    for col in feature_cols:
        assert col in df.columns, f"Column '{col}' not in DataFrame"
        assert pd.isna(df[col].iloc[0]), (
            f"Feature '{col}' first row is {df[col].iloc[0]}, expected NaN. "
            f"Did you forget shift_features()?"
        )


def assert_no_inf(df: pd.DataFrame, feature_cols: list) -> None:
    """Assert that no inf values exist in feature columns."""
    for col in feature_cols:
        assert col in df.columns, f"Column '{col}' not in DataFrame"
        inf_count = np.isinf(df[col].dropna()).sum()
        assert inf_count == 0, (
            f"Feature '{col}' has {inf_count} inf values. "
            f"Did you forget safe_divide()?"
        )


def create_sample_asset(symbol: str = "EURUSD") -> AssetInfo:
    """Create a sample AssetInfo for testing exit strategies."""
    defaults = {
        "EURUSD": ("FOREX", 0.00010, 0.00001),
        "GBPUSD": ("FOREX", 0.00015, 0.00001),
        "USDJPY": ("FOREX", 0.015, 0.001),
        "XAUUSD": ("COMMODITY", 0.30, 0.01),
        "BTCUSD": ("CRYPTO", 10.0, 0.01),
    }
    asset_class, spread, point = defaults.get(symbol, ("FOREX", 0.00010, 0.00001))
    return AssetInfo(
        symbol=symbol,
        asset_class=asset_class,
        spread=spread,
        point=point,
    )
```

**Step 4: Add testing to flat namespace**

Update `packages/fwbg-sdk/src/fwbg_sdk/__init__.py` — add to imports and `__all__`:
```python
from fwbg_sdk.testing import (
    create_sample_ohlcv, assert_features_shifted,
    assert_no_inf, create_sample_asset,
)
```

**Step 5: Run tests — verify they pass**

```bash
python -m pytest packages/fwbg-sdk/tests/test_testing.py -x -q
```
Expected: All PASS

**Step 6: Commit**

```bash
git add packages/fwbg-sdk/
git commit -m "feat(sdk): add testing utilities"
```

---

## Task 7: CLI Scaffolding

**Files:**
- Create: `packages/fwbg-sdk/src/fwbg_sdk/cli.py`
- Create: `packages/fwbg-sdk/tests/test_cli.py`

**Step 1: Write failing tests for CLI**

Create `packages/fwbg-sdk/tests/test_cli.py`:
```python
import json
from pathlib import Path
from click.testing import CliRunner
from fwbg_sdk.cli import main


def test_init_creates_package(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "init", "my-indicators",
        "--plugin", "indicator:my_rsi",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0

    pkg_dir = tmp_path / "my-indicators"
    assert (pkg_dir / "pyproject.toml").exists()
    assert (pkg_dir / "src" / "my_indicators" / "__init__.py").exists()

    # Check plugin files
    plugin_dir = pkg_dir / "src" / "my_indicators" / "plugins" / "my-indicators" / "indicators" / "my_rsi"
    assert (plugin_dir / "__init__.py").exists()
    assert (plugin_dir / "manifest.json").exists()
    assert (plugin_dir / "tests.py").exists()

    # Check package manifest
    manifest = json.loads((pkg_dir / "src" / "my_indicators" / "plugins" / "my-indicators" / "manifest.json").read_text())
    assert manifest["name"] == "my-indicators"
    assert "my_rsi" in manifest["plugins"]["indicators"]


def test_init_pyproject_has_entry_point(tmp_path):
    runner = CliRunner()
    runner.invoke(main, [
        "init", "my-indicators",
        "--plugin", "indicator:my_rsi",
        "--output-dir", str(tmp_path),
    ])
    content = (tmp_path / "my-indicators" / "pyproject.toml").read_text()
    assert "fwbg.plugin_packages" in content
    assert "fwbg-sdk" in content


def test_add_plugin_to_existing(tmp_path):
    runner = CliRunner()
    # First create package
    runner.invoke(main, [
        "init", "my-indicators",
        "--plugin", "indicator:my_rsi",
        "--output-dir", str(tmp_path),
    ])
    # Then add another plugin
    result = runner.invoke(main, [
        "add", "indicator", "my_macd",
        "--package-dir", str(tmp_path / "my-indicators"),
    ])
    assert result.exit_code == 0

    plugin_dir = tmp_path / "my-indicators" / "src" / "my_indicators" / "plugins" / "my-indicators" / "indicators" / "my_macd"
    assert (plugin_dir / "__init__.py").exists()
```

**Step 2: Run tests — verify they fail**

```bash
python -m pytest packages/fwbg-sdk/tests/test_cli.py -x -q
```

**Step 3: Implement cli.py**

Create `packages/fwbg-sdk/src/fwbg_sdk/cli.py` with:
- `fwbg-sdk init <name> --plugin <type:name> [--output-dir <dir>]`
- `fwbg-sdk add <type> <name> [--package-dir <dir>]`

The `init` command generates:
1. `pyproject.toml` with entry-point
2. `src/<module>/__init__.py` with `get_plugins_dir()`
3. `src/<module>/plugins/<package>/manifest.json`
4. Plugin directory with `__init__.py`, `manifest.json`, `tests.py`

The `add` command:
1. Finds existing package manifest
2. Creates new plugin directory
3. Updates manifest

Plugin `__init__.py` templates use the correct base class per type:
- `indicator` → `BaseIndicator` with `compute()`, `shift_features`, `safe_divide`
- `preprocessor` → `BasePreprocessor` with `fit()`, `transform()`
- `feature_selector` → `BaseFeatureSelector` with `select_features()`
- `exit_strategy` → `BaseExitStrategy` with `compute_targets()`, `iterate_grid()`, `get_cache_key()`
- `risk_manager` → `BaseRiskManager` with `compute_risk_params()`
- `data_loader` → `BaseDataLoader` with `execute()`

**Step 4: Re-install SDK (for CLI entry point)**

```bash
pip install -e packages/fwbg-sdk
```

**Step 5: Run tests — verify they pass**

```bash
python -m pytest packages/fwbg-sdk/tests/test_cli.py -x -q
```

**Step 6: Verify CLI works end-to-end**

```bash
cd /tmp && fwbg-sdk init test-pkg --plugin indicator:test_ind && ls -la test-pkg/
```

**Step 7: Commit**

```bash
git add packages/fwbg-sdk/
git commit -m "feat(sdk): add CLI scaffolding (init, add)"
```

---

## Task 8: Update fwbg Imports — Delete Old Files, Rewire Everything

This is the largest task. All `from fwbg.plugins.indicator import ...`, `from fwbg.pipeline.base import ...`, etc. become `from fwbg_sdk import ...`.

**Files to delete (source of truth moved to SDK):**
- `src/fwbg/pipeline/base.py` → deleted (moved to `fwbg_sdk/base.py`)
- `src/fwbg/pipeline/context.py` → deleted (moved to `fwbg_sdk/contexts.py`)
- `src/fwbg/plugins/indicator.py` → deleted (moved to `fwbg_sdk/indicators.py`)
- `src/fwbg/plugins/preprocessor.py` → deleted (moved to `fwbg_sdk/preprocessors.py`)
- `src/fwbg/plugins/feature_selector.py` → deleted (moved to `fwbg_sdk/feature_selectors.py`)
- `src/fwbg/plugins/exit_strategy.py` → deleted (moved to `fwbg_sdk/exit_strategies.py`)
- `src/fwbg/plugins/risk_manager.py` → deleted (moved to `fwbg_sdk/risk_managers.py`)
- `src/fwbg/plugins/data_loader.py` → deleted (moved to `fwbg_sdk/data_loaders.py`)
- `src/fwbg/core/enums.py` → deleted (moved to `fwbg_sdk/enums.py`)

**Files to update in `src/fwbg/`:**

**Step 1: Add fwbg-sdk dependency to fwbg**

In `pyproject.toml`, add to dependencies:
```toml
"fwbg-sdk>=1.0.0",
```

**Step 2: Delete moved files**

```bash
rm src/fwbg/pipeline/base.py
rm src/fwbg/pipeline/context.py
rm src/fwbg/plugins/indicator.py
rm src/fwbg/plugins/preprocessor.py
rm src/fwbg/plugins/feature_selector.py
rm src/fwbg/plugins/exit_strategy.py
rm src/fwbg/plugins/risk_manager.py
rm src/fwbg/plugins/data_loader.py
rm src/fwbg/core/enums.py
```

**Step 3: Update `src/fwbg/plugins/__init__.py`**

Replace base class imports with SDK imports:
```python
# Old:
from .indicator import BaseIndicator
from .exit_strategy import BaseExitStrategy
# etc.

# New:
from fwbg_sdk import (
    BaseIndicator, BaseExitStrategy, BaseFeatureSelector,
    BasePreprocessor, BaseRiskManager, BaseDataLoader,
)
```

Keep the discovery functions (`get_plugins_dir`, `import_plugin_module`, `run_plugin_tests`) — they stay in fwbg.

**Step 4: Update `src/fwbg/pipeline/__init__.py`**

```python
# Old:
from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.base import BasePlugin, PluginPhase

# New:
from fwbg_sdk import PipelineContext, BasePlugin, PluginPhase
```

**Step 5: Update `src/fwbg/core/__init__.py`**

Replace enum imports:
```python
# Old:
from .enums import Timeframe, AssetClass, Symbol, Direction, SignalType

# New:
from fwbg_sdk import Timeframe, AssetClass, Symbol, Direction, SignalType
```

Registry imports already handled in Task 4.

**Step 6: Update `src/fwbg/pipeline/runner.py`**

```python
# Old:
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext

# New:
from fwbg_sdk import BasePlugin, PluginPhase, PipelineContext
```

**Step 7: Update `src/fwbg/pipeline/registry.py`**

```python
# Old:
from fwbg.pipeline.base import BasePlugin, PluginPhase

# New:
from fwbg_sdk import BasePlugin, PluginPhase
```

**Step 8: Update remaining src/fwbg/ files**

Search for all remaining old imports and update. Key files:
- `src/fwbg/optimization/process_fold.py` — PipelineContext import
- `src/fwbg/data/loader.py` — PipelineContext import
- `src/fwbg/adapters/broker/**` — enums imports
- `src/fwbg/core/data_sources.py` — registry imports

Pattern: `from fwbg.pipeline.base import` → `from fwbg_sdk import`
Pattern: `from fwbg.pipeline.context import` → `from fwbg_sdk import`
Pattern: `from fwbg.core.enums import` → `from fwbg_sdk import`

**Step 9: Update exit strategy callers for AssetInfo**

In `src/fwbg/optimization/targets.py`, the `compute_targets_cached` function calls `strategy.compute_targets(full_df, ctx, ...)` where `ctx` is `SimulationContext`.

Add a bridge:
```python
from fwbg_sdk import AssetInfo

def _make_asset_info(ctx: SimulationContext) -> AssetInfo:
    return AssetInfo(
        symbol=ctx.symbol,
        asset_class=ctx.asset_class,
        spread=ctx.spread,
        point=ctx.point,
        max_trade_bars=ctx.max_trade_bars,
    )
```

Then change `strategy.compute_targets(full_df, ctx, ...)` to `strategy.compute_targets(full_df, _make_asset_info(ctx), ...)`.

Same in `src/fwbg/optimization/grid_search.py` where `iterate_grid(grid_config, ctx)` is called.

**Step 10: Update core plugin files (fwbg-core)**

All plugins in `src/fwbg/plugins/fwbg-core/`:
```python
# Old:
from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features, safe_divide
from fwbg.core import register_indicator

# New:
from fwbg_sdk import BaseIndicator, shift_features, safe_divide, register_indicator
```

Files to update:
- `indicators/trend/__init__.py`
- `indicators/momentum/__init__.py`
- `indicators/volatility/__init__.py`
- `indicators/price_action/__init__.py`
- `indicators/time_season/__init__.py`
- `exit_strategies/fixed/__init__.py` (also change `ctx: SimulationContext` to `ctx: AssetInfo` type hints)
- `risk_management/kelly/__init__.py`
- `risk_management/vol_targeted_kelly/__init__.py`

**Step 11: Verify fwbg still loads**

```bash
python -c "import fwbg; print('OK')"
python -c "from fwbg.core import register_indicator, get_indicator; print('OK')"
python -c "from fwbg.pipeline import PipelineRunner, PipelineContext; print('OK')"
```

**Step 12: Commit**

```bash
git add -A
git commit -m "refactor: rewire fwbg imports to fwbg-sdk"
```

---

## Task 9: Update fwbg-premium Imports

**Files:** All plugin files in `packages/fwbg-premium/`

**Step 1: Add fwbg-sdk dependency**

In `packages/fwbg-premium/pyproject.toml`:
```toml
dependencies = ["fwbg-sdk>=1.0.0"]
```

(Remove or keep `fwbg>=2.2.1` — premium may still need fwbg for numba simulation imports)

**Step 2: Update all indicator plugins**

For all 14 indicator plugins in `packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/`:

```python
# Old:
from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features, safe_divide
from fwbg.core import register_indicator

# New:
from fwbg_sdk import BaseIndicator, shift_features, safe_divide, register_indicator
```

Some also import `EPSILON` — add to the import.

Files: `regime/`, `structure/`, `risk/`, `distribution/`, `dynamics/`, `multi_timeframe/`, `cross_features/`, `ichimoku/`, `macro_surprise/`, `microstructure/`, `market_regime/`, `regime_cluster/`

**Step 3: Update feature selection plugins**

```python
# Old:
from fwbg.plugins import BaseFeatureSelector
from fwbg.core import register_feature_selector

# New:
from fwbg_sdk import BaseFeatureSelector, register_feature_selector
```

Files: `boruta/`, `plateau/`, `stability/` (stability also imports `get_feature_selector` — keep from `fwbg.core` since it needs discovery)

**Step 4: Update preprocessor**

```python
# Old:
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.core import register_preprocessor

# New:
from fwbg_sdk import BasePlugin, PluginPhase, PipelineContext, register_preprocessor
```

File: `preprocessing/fractional_diff/__init__.py`

**Step 5: Update exit strategy**

```python
# Old:
from fwbg.plugins import BaseExitStrategy
from fwbg.core import register_exit_strategy

# New:
from fwbg_sdk import BaseExitStrategy, register_exit_strategy
```

File: `exit_strategies/atr_based/__init__.py`
Also update `ctx: SimulationContext` type hints to `ctx: AssetInfo` (import `AssetInfo` from fwbg_sdk).

**Step 6: Update data loaders**

```python
# Old:
from fwbg.plugins.data_loader import BaseDataLoader
from fwbg.core import register_data_loader

# New:
from fwbg_sdk import BaseDataLoader, register_data_loader
```

Files: `data_loading/macro_data/`, `data_loading/cot_positioning/`

**Step 7: Verify premium loads**

```bash
python -c "import fwbg_premium; print('OK')"
```

**Step 8: Commit**

```bash
git add packages/fwbg-premium/
git commit -m "refactor: rewire fwbg-premium imports to fwbg-sdk"
```

---

## Task 10: Update Test Imports

**Files:** All test files in `tests/`

**Step 1: Bulk update test imports**

Pattern replacements across all test files:

```python
# from fwbg.pipeline.base import BasePlugin, PluginPhase
→ from fwbg_sdk import BasePlugin, PluginPhase

# from fwbg.pipeline.context import PipelineContext
→ from fwbg_sdk import PipelineContext

# from fwbg.plugins.indicator import BaseIndicator, shift_features, safe_divide, EPSILON
→ from fwbg_sdk import BaseIndicator, shift_features, safe_divide, EPSILON

# from fwbg.plugins.indicator import shift_features
→ from fwbg_sdk import shift_features

# from fwbg.plugins.data_loader import BaseDataLoader
→ from fwbg_sdk import BaseDataLoader

# from fwbg.plugins import BaseRiskManager
→ from fwbg_sdk import BaseRiskManager

# from fwbg.plugins import BaseDataLoader
→ from fwbg_sdk import BaseDataLoader
```

Key test files (~20 files):
- `tests/pipeline/test_base.py`
- `tests/pipeline/test_runner.py`
- `tests/pipeline/test_registry.py`
- `tests/pipeline/test_context.py`
- `tests/pipeline/test_entry_point_discovery.py`
- `tests/pipeline/test_user_plugins.py`
- `tests/test_indicator_utils.py`
- `tests/test_indicator_stationarity.py`
- `tests/test_no_bias_in_system.py`
- `tests/test_dependency_sort.py`
- `tests/test_data_loading.py`
- `tests/test_cot_positioning.py`
- `tests/test_macro_vol_features.py`
- `tests/test_risk_management.py`
- `tests/test_computation_correctness.py`
- `tests/test_regime.py`
- `tests/test_market_regime.py`
- `tests/test_new_features.py`
- `tests/test_registry.py`

**Note:** Many test files have imports inside functions. Search for ALL patterns including inside functions.

Also update `examples/plugins/custom_indicator/__init__.py`.
Also update `src/fwbg/plugins/fwbg-core/indicators/trend/tests.py`.

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -x -q
```

Expected: All ~521 tests pass (minus the known machine-dependent `test_max_workers_is_minimum_of_limits`).

**Step 3: Commit**

```bash
git add tests/ examples/
git commit -m "refactor: update test imports to fwbg-sdk"
```

---

## Task 11: Final Verification & Cleanup

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -x -q
```

All tests must pass.

**Step 2: Run SDK tests**

```bash
python -m pytest packages/fwbg-sdk/tests/ -x -q
```

**Step 3: Verify CLI works**

```bash
fwbg-sdk init test-pkg --plugin indicator:test_ind --output-dir /tmp/sdk-test
cd /tmp/sdk-test/test-pkg && pip install -e . && python -c "import test_pkg; print('OK')"
```

**Step 4: Verify no old imports remain**

```bash
# Should return no results:
grep -rn "from fwbg.pipeline.base import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.plugins.indicator import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.plugins.preprocessor import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.plugins.exit_strategy import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.plugins.feature_selector import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.plugins.risk_manager import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.plugins.data_loader import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.pipeline.context import" src/fwbg/ packages/fwbg-premium/ tests/
grep -rn "from fwbg.core.enums import" src/fwbg/ packages/fwbg-premium/ tests/
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: cleanup after fwbg-sdk migration"
```
