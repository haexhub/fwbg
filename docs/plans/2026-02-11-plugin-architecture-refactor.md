# Plugin Architecture Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move all components (indicators, preprocessing, feature_selection, exit_strategies) from `builtins/` to a unified `plugins/` directory structure, enabling modular deployment and future premium/free separation.

**Architecture:** All plugins live in `src/fwbg/plugins/` with subdirectories for each category. Each plugin has a `manifest.json` for metadata. The registry discovers plugins via directory scanning. `builtins/` will be deleted entirely - no legacy code remains.

**Tech Stack:** Python 3.10+, existing BasePlugin from pipeline.base, manifest.json for plugin metadata

---

## Overview

### Current Structure (to be removed):
```
src/fwbg/
├── builtins/           # DELETE THIS
│   ├── indicators/     # 15 indicator plugins
│   ├── preprocessing/  # fractional_diff, normalization
│   ├── feature_selection/  # boruta, plateau
│   └── exit_strategies/    # atr_based, fixed
└── plugins/            # Only has base classes currently
```

### Target Structure:
```
src/fwbg/
└── plugins/
    ├── __init__.py     # Plugin API exports
    ├── indicators/
    │   ├── trend/
    │   │   ├── __init__.py
    │   │   └── manifest.json
    │   ├── momentum/
    │   ├── volatility/
    │   └── ... (15 total)
    ├── preprocessing/
    │   └── fractional_diff/
    ├── feature_selection/
    │   ├── boruta/
    │   └── plateau/
    └── exit_strategies/
        ├── atr_based/
        └── fixed/
```

---

## Phase 1: Move Indicator Plugins

### Task 1: Create plugins/indicators directory structure

**Files:**
- Create: `src/fwbg/plugins/indicators/__init__.py`

**Step 1: Create directory and __init__.py**

```bash
mkdir -p src/fwbg/plugins/indicators
```

```python
# src/fwbg/plugins/indicators/__init__.py
"""
Indicator Plugins.

All indicator plugins are discovered automatically via manifest.json.
Import the registry and call auto_discover() to load them.
"""
```

**Step 2: Verify directory exists**

```bash
ls src/fwbg/plugins/indicators/
```
Expected: `__init__.py`

**Step 3: Commit**

```bash
git add src/fwbg/plugins/indicators/
git commit -m "feat(plugins): create indicators plugin directory"
```

---

### Task 2: Move trend indicator to plugins

**Files:**
- Move: `src/fwbg/builtins/indicators/trend/__init__.py` → `src/fwbg/plugins/indicators/trend/__init__.py`
- Create: `src/fwbg/plugins/indicators/trend/manifest.json`

**Step 1: Create directory and copy files**

```bash
mkdir -p src/fwbg/plugins/indicators/trend
cp src/fwbg/builtins/indicators/trend/__init__.py src/fwbg/plugins/indicators/trend/
```

**Step 2: Create manifest.json**

```json
{
  "name": "trend",
  "version": "2.0.0",
  "description": "Trend indicators: ADX, EMA, SMA, MACD, CCI, Aroon, Efficiency Ratio",
  "author": "fwbg-team",
  "phase": "indicators",
  "dependencies": {
    "ta": ">=0.10.0",
    "pandas": ">=1.5.0"
  }
}
```

Save to: `src/fwbg/plugins/indicators/trend/manifest.json`

**Step 3: Update imports in __init__.py**

Remove the old registry decorator import if present, ensure it uses `BasePlugin` from pipeline:

```python
# At top of src/fwbg/plugins/indicators/trend/__init__.py
# Remove: from fwbg.core.registry import register_indicator
# Keep: from fwbg.pipeline.base import BasePlugin, PluginPhase
```

**Step 4: Verify plugin loads**

```bash
PYTHONPATH=src python3 -c "
from fwbg.pipeline.registry import PluginRegistry
from pathlib import Path

r = PluginRegistry()
discovered = r.discover_from_directory(Path('src/fwbg/plugins/indicators'))
print(f'Discovered: {discovered}')
assert 'trend' in discovered, 'trend plugin not found'
print('OK')
"
```
Expected: `Discovered: ['trend']` then `OK`

**Step 5: Commit**

```bash
git add src/fwbg/plugins/indicators/trend/
git commit -m "feat(plugins): move trend indicator to plugins directory"
```

---

### Task 3: Move momentum indicator to plugins

**Files:**
- Move: `src/fwbg/builtins/indicators/momentum/__init__.py` → `src/fwbg/plugins/indicators/momentum/__init__.py`
- Create: `src/fwbg/plugins/indicators/momentum/manifest.json`

**Step 1: Create directory and copy**

```bash
mkdir -p src/fwbg/plugins/indicators/momentum
cp src/fwbg/builtins/indicators/momentum/__init__.py src/fwbg/plugins/indicators/momentum/
```

**Step 2: Create manifest.json**

```json
{
  "name": "momentum",
  "version": "2.0.0",
  "description": "Momentum indicators: RSI, Stochastic, Williams %R, Ultimate Oscillator, ROC",
  "author": "fwbg-team",
  "phase": "indicators",
  "dependencies": {
    "ta": ">=0.10.0",
    "pandas": ">=1.5.0"
  }
}
```

**Step 3: Commit**

```bash
git add src/fwbg/plugins/indicators/momentum/
git commit -m "feat(plugins): move momentum indicator to plugins directory"
```

---

### Task 4: Move volatility indicator to plugins

**Files:**
- Move: `src/fwbg/builtins/indicators/volatility/__init__.py` → `src/fwbg/plugins/indicators/volatility/__init__.py`
- Create: `src/fwbg/plugins/indicators/volatility/manifest.json`

**Step 1: Create directory and copy**

```bash
mkdir -p src/fwbg/plugins/indicators/volatility
cp src/fwbg/builtins/indicators/volatility/__init__.py src/fwbg/plugins/indicators/volatility/
```

**Step 2: Create manifest.json**

```json
{
  "name": "volatility",
  "version": "2.0.0",
  "description": "Volatility indicators: ATR, Bollinger Bands, Keltner Channel, Donchian",
  "author": "fwbg-team",
  "phase": "indicators",
  "dependencies": {
    "ta": ">=0.10.0",
    "pandas": ">=1.5.0"
  }
}
```

**Step 3: Commit**

```bash
git add src/fwbg/plugins/indicators/volatility/
git commit -m "feat(plugins): move volatility indicator to plugins directory"
```

---

### Task 5: Move remaining 12 indicators (batch)

Repeat the same pattern for each:
- regime
- structure
- risk
- price_action
- time_season
- distribution
- dynamics
- multi_timeframe
- cross_features
- ichimoku
- macro_surprise
- microstructure

**For each indicator:**

```bash
# Template (replace INDICATOR_NAME)
mkdir -p src/fwbg/plugins/indicators/INDICATOR_NAME
cp src/fwbg/builtins/indicators/INDICATOR_NAME/__init__.py src/fwbg/plugins/indicators/INDICATOR_NAME/
```

Create manifest.json for each (adjust description):

```json
{
  "name": "INDICATOR_NAME",
  "version": "2.0.0",
  "description": "Description here",
  "author": "fwbg-team",
  "phase": "indicators",
  "dependencies": {
    "pandas": ">=1.5.0",
    "numpy": ">=1.20.0"
  }
}
```

**Commit after all are moved:**

```bash
git add src/fwbg/plugins/indicators/
git commit -m "feat(plugins): move all indicators to plugins directory"
```

---

## Phase 2: Move Preprocessing Plugins

### Task 6: Create preprocessing directory and move fractional_diff

**Files:**
- Create: `src/fwbg/plugins/preprocessing/__init__.py`
- Move: `src/fwbg/builtins/preprocessing/fractional_diff/__init__.py` → `src/fwbg/plugins/preprocessing/fractional_diff/__init__.py`
- Create: `src/fwbg/plugins/preprocessing/fractional_diff/manifest.json`

**Step 1: Create directories**

```bash
mkdir -p src/fwbg/plugins/preprocessing/fractional_diff
```

**Step 2: Create preprocessing __init__.py**

```python
# src/fwbg/plugins/preprocessing/__init__.py
"""Preprocessing plugins for data transformation."""
```

**Step 3: Copy fractional_diff**

```bash
cp src/fwbg/builtins/preprocessing/fractional_diff/__init__.py src/fwbg/plugins/preprocessing/fractional_diff/
```

**Step 4: Create manifest.json**

```json
{
  "name": "fractional_diff",
  "version": "2.0.0",
  "description": "Fractional differentiation for stationarity while preserving memory",
  "author": "fwbg-team",
  "phase": "preprocessing",
  "stateful": true,
  "dependencies": {
    "statsmodels": ">=0.13.0",
    "pandas": ">=1.5.0",
    "numpy": ">=1.20.0"
  }
}
```

**Step 5: Commit**

```bash
git add src/fwbg/plugins/preprocessing/
git commit -m "feat(plugins): move fractional_diff to plugins directory"
```

---

## Phase 3: Move Feature Selection Plugins

### Task 7: Move boruta feature selector

**Files:**
- Create: `src/fwbg/plugins/feature_selection/__init__.py`
- Move: `src/fwbg/builtins/feature_selection/boruta/` → `src/fwbg/plugins/feature_selection/boruta/`
- Create: `src/fwbg/plugins/feature_selection/boruta/manifest.json`

**Step 1: Create directories and copy**

```bash
mkdir -p src/fwbg/plugins/feature_selection/boruta
cp src/fwbg/builtins/feature_selection/boruta/__init__.py src/fwbg/plugins/feature_selection/boruta/
cp src/fwbg/builtins/feature_selection/boruta/selector.py src/fwbg/plugins/feature_selection/boruta/
```

**Step 2: Create __init__.py for feature_selection**

```python
# src/fwbg/plugins/feature_selection/__init__.py
"""Feature selection plugins."""
```

**Step 3: Create manifest.json**

```json
{
  "name": "boruta",
  "version": "2.0.0",
  "description": "Boruta feature selection using random forest",
  "author": "fwbg-team",
  "phase": "feature_selection",
  "dependencies": {
    "scikit-learn": ">=1.0.0",
    "pandas": ">=1.5.0"
  }
}
```

**Step 4: Migrate boruta to BasePlugin interface**

The boruta selector needs to be updated to inherit from BasePlugin. See existing pattern from fractional_diff.

**Step 5: Commit**

```bash
git add src/fwbg/plugins/feature_selection/
git commit -m "feat(plugins): move boruta to plugins directory"
```

---

### Task 8: Move plateau feature selector

Similar pattern to boruta.

---

## Phase 4: Move Exit Strategy Plugins

### Task 9: Move exit strategies

**Files:**
- Create: `src/fwbg/plugins/exit_strategies/__init__.py`
- Move: `src/fwbg/builtins/exit_strategies/atr_based/` → `src/fwbg/plugins/exit_strategies/atr_based/`
- Move: `src/fwbg/builtins/exit_strategies/fixed/` → `src/fwbg/plugins/exit_strategies/fixed/`
- Move: `src/fwbg/builtins/exit_strategies/base.py` → `src/fwbg/plugins/exit_strategies/base.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/fwbg/plugins/exit_strategies/atr_based
mkdir -p src/fwbg/plugins/exit_strategies/fixed
```

**Step 2: Copy base and strategies**

```bash
cp src/fwbg/builtins/exit_strategies/base.py src/fwbg/plugins/exit_strategies/
cp -r src/fwbg/builtins/exit_strategies/atr_based/* src/fwbg/plugins/exit_strategies/atr_based/
cp -r src/fwbg/builtins/exit_strategies/fixed/* src/fwbg/plugins/exit_strategies/fixed/
```

**Step 3: Create manifests for each**

`atr_based/manifest.json`:
```json
{
  "name": "atr_based",
  "version": "2.0.0",
  "description": "ATR-based stop loss and take profit exit strategy",
  "author": "fwbg-team",
  "phase": "exit_strategy",
  "dependencies": {}
}
```

`fixed/manifest.json`:
```json
{
  "name": "fixed",
  "version": "2.0.0",
  "description": "Fixed pip-based stop loss and take profit",
  "author": "fwbg-team",
  "phase": "exit_strategy",
  "dependencies": {}
}
```

**Step 4: Commit**

```bash
git add src/fwbg/plugins/exit_strategies/
git commit -m "feat(plugins): move exit strategies to plugins directory"
```

---

## Phase 5: Update Registry Discovery

### Task 10: Update registry to scan new plugin locations

**Files:**
- Modify: `src/fwbg/pipeline/registry.py`

**Step 1: Update get_core_plugins_dir() and auto_discover()**

```python
def get_core_plugins_dir() -> Path:
    """Get core plugins directory."""
    return Path(__file__).parent.parent / "plugins"


def auto_discover(self) -> List[str]:
    """Discover all plugins from core and user directories."""
    discovered: List[str] = []
    core_dir = get_core_plugins_dir()

    # Scan all plugin categories
    for category in ["indicators", "preprocessing", "feature_selection", "exit_strategies"]:
        category_dir = core_dir / category
        if category_dir.exists():
            discovered.extend(self.discover_from_directory(category_dir))

    # User plugins
    user_dir = get_user_plugins_dir()
    if user_dir.exists():
        discovered.extend(self.discover_from_directory(user_dir))

    return discovered
```

**Step 2: Run tests**

```bash
PYTHONPATH=src python3 -m pytest tests/pipeline/ -v
```

**Step 3: Commit**

```bash
git add src/fwbg/pipeline/registry.py
git commit -m "refactor(registry): update discovery paths for new plugin structure"
```

---

## Phase 6: Update All Import References

### Task 11: Update process.py imports

**Files:**
- Modify: `src/fwbg/optimization/process.py`

Remove all `from fwbg.builtins.*` imports. The pipeline system handles everything.

```python
# Remove these imports:
# from fwbg.builtins.indicators import (...)
# from fwbg.builtins.feature_selection.plateau import (...)

# Keep only pipeline imports:
from fwbg.pipeline import (
    PipelineRunner, PipelineContext, PipelineConfig, PluginConfig,
    parse_pipeline_config, get_registry, PluginPhase,
)
```

**Step 1: Update imports**

**Step 2: Run tests**

```bash
PYTHONPATH=src python3 -m pytest tests/optimization/ -v
```

**Step 3: Commit**

```bash
git add src/fwbg/optimization/process.py
git commit -m "refactor(process): remove builtins imports, use pipeline only"
```

---

### Task 12: Update nested_cv.py imports

**Files:**
- Modify: `src/fwbg/optimization/nested_cv.py`

Remove imports from builtins:
- `from fwbg.builtins.feature_selection.plateau import ...`
- `from fwbg.builtins.feature_selection.boruta import ...`
- `from fwbg.builtins.exit_strategies import ...`

Replace with registry lookups.

---

### Task 13: Update grid_search.py imports

**Files:**
- Modify: `src/fwbg/optimization/grid_search.py`

Remove `from fwbg.builtins.indicators import filter_features_by_group`.

Move `filter_features_by_group` and `FEATURE_GROUPS` to `fwbg.pipeline.features` or similar utility module.

---

### Task 14: Update bot.py imports

**Files:**
- Modify: `src/fwbg/bot.py`

Remove:
- `from fwbg.builtins.indicators import compute_indicator_pool`
- `from fwbg.builtins.utils import ...`

Use pipeline instead.

---

### Task 15: Update cli/main.py imports

**Files:**
- Modify: `src/fwbg/cli/main.py`

Remove `from fwbg.builtins.indicators import FEATURE_GROUPS`.

---

## Phase 7: Move Utility Functions

### Task 16: Create fwbg.pipeline.features module

**Files:**
- Create: `src/fwbg/pipeline/features.py`

Move these from `builtins/indicators/__init__.py`:
- `FEATURE_GROUPS`
- `filter_features_by_group()`
- `get_feature_columns()`
- `compute_regime_filter()`

```python
# src/fwbg/pipeline/features.py
"""Feature utilities for the pipeline system."""
from typing import List
import pandas as pd

FEATURE_GROUPS = {
    "trend": {"name": "Trend Indicators", "prefixes": ["trend_"]},
    # ... rest of groups
}

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all feature columns (excluding internal columns)."""
    exclude = ["O", "H", "L", "C", "V", "Volume", "_atr", "_regime_ok", "_original_close", "_hurst"]
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]

def filter_features_by_group(all_features: List[str], group_name: str) -> List[str]:
    """Filter features by group."""
    if group_name == "all" or group_name not in FEATURE_GROUPS:
        return all_features
    prefixes = FEATURE_GROUPS[group_name]["prefixes"]
    return [f for f in all_features if any(f.startswith(p) for p in prefixes)]
```

**Step 1: Create the module**

**Step 2: Update exports in `fwbg.pipeline.__init__.py`**

**Step 3: Update all imports to use new location**

**Step 4: Commit**

```bash
git add src/fwbg/pipeline/features.py
git commit -m "refactor(pipeline): move feature utilities to pipeline.features"
```

---

## Phase 8: Delete builtins Directory

### Task 17: Remove builtins directory

**Files:**
- Delete: `src/fwbg/builtins/` (entire directory)

**Step 1: Verify all tests pass with new structure**

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

**Step 2: Delete builtins**

```bash
rm -rf src/fwbg/builtins/
```

**Step 3: Verify imports still work**

```bash
PYTHONPATH=src python3 -c "
from fwbg.pipeline import get_registry
r = get_registry()
r.auto_discover()
print(f'Plugins: {len(r.list_plugins())}')
assert len(r.list_plugins()) >= 15, 'Not enough plugins discovered'
print('OK')
"
```

**Step 4: Run full test suite**

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove builtins directory, all plugins in plugins/"
```

---

## Phase 9: Update Tests

### Task 18: Update test imports

All tests that import from `fwbg.builtins.*` need to be updated to use:
1. Registry discovery for plugin access
2. `fwbg.pipeline.features` for utility functions

**Files to update:**
- `tests/test_cli.py`
- `tests/test_grid_search_feature_groups.py`
- `tests/test_indicator_inf_prevention.py`
- `tests/test_core_indicators.py`
- `tests/test_advanced_indicators.py`
- `tests/test_new_indicators.py`
- And others...

**Pattern for tests:**

```python
# Old:
from fwbg.builtins.indicators import compute_indicator_pool, get_feature_columns

# New:
from fwbg.pipeline import get_registry, PipelineRunner, PipelineConfig, PluginConfig, PipelineContext
from fwbg.pipeline.features import get_feature_columns, FEATURE_GROUPS

def compute_indicators(df, indicators):
    """Helper to compute indicators in tests."""
    registry = get_registry()
    registry.auto_discover()
    configs = [PluginConfig(name=name, params={}) for name in indicators]
    config = PipelineConfig(indicators=configs)
    runner = PipelineRunner(registry, config)
    ctx = PipelineContext(df=df.copy(), symbol="TEST", asset_class="FOREX")
    return runner.run(ctx, phases=["indicators"]).df
```

---

## Summary

**Total Tasks:** 18 main tasks

**Key Changes:**
1. All plugins moved to `src/fwbg/plugins/`
2. Each plugin has `manifest.json`
3. Registry discovers from `plugins/` not `builtins/`
4. Utility functions in `fwbg.pipeline.features`
5. `builtins/` deleted entirely

**Benefits:**
- Clean plugin architecture
- Easy to add/remove plugins
- Premium plugins can be in separate repo/package
- No more hardcoded imports
