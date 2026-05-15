# Moving Average Base Class — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate ~140 lines of duplicated boilerplate between the EMA and SMA indicator plugins by introducing a shared `BaseMovingAverageIndicator` class.

**Architecture:** Both plugins currently implement identical parsing (`_line_key`, `_parse_lines`), crossing-feature logic, column resolution, and feature-column listing. The only difference is the underlying call to `ta.trend.ema_indicator` vs `ta.trend.sma_indicator`. Introduce a base class in the existing fwbg-sdk plugin namespace (`src/fwbg/plugins/fwbg-core/indicators/_base/`) with a hook method `_compute_ma(series, window)` that subclasses override.

**Tech Stack:** Python, pandas, ta-lib (`ta.trend`), pytest.

---

## Background

Side-by-side diff of [ema/__init__.py](../../src/fwbg/plugins/fwbg-core/indicators/ema/__init__.py) and [sma/__init__.py](../../src/fwbg/plugins/fwbg-core/indicators/sma/__init__.py) shows ~140 LOC are line-for-line identical:

- `_line_key(lookback)` helper
- `_parse_lines(spec)` helper
- `_resolve_lines(df, lines)` method
- `get_feature_columns(params)` method
- crossing-detection logic

Only the actual MA function call differs:

```python
# ema:
result = ta.trend.ema_indicator(series, window=window)
# sma:
result = ta.trend.sma_indicator(series, window=window)
```

---

## Success Criteria

1. New `BaseMovingAverageIndicator` class with abstract `_compute_ma`.
2. EMA and SMA each shrink to <50 LOC (mostly: subclass + override + manifest).
3. All existing tests in `src/fwbg/plugins/fwbg-core/indicators/ema/tests.py` and `.../sma/tests.py` still pass.
4. Cross-asset tests (`tests/test_core_indicators.py`) still pass.
5. Feature columns and crossing features produced by EMA/SMA before and after are byte-identical on a snapshot input.

---

## Out of Scope

- Adding new moving averages (WMA, HMA, KAMA) — separate plan.
- Refactoring other indicators (ADX, MACD, etc.) — separate plan.
- Changing the manifest format.

---

## Task 1: Snapshot current EMA/SMA output to lock in behaviour

**Files:**
- Test: `tests/test_ma_snapshot.py` (create)

**Step 1: Write the snapshot test**

```python
"""Lock current EMA/SMA outputs so the refactor cannot silently change them."""
import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline.registry import get_registry


def _synth_df(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "O": np.linspace(1.0, 1.1, n),
        "H": np.linspace(1.0, 1.1, n) + 0.001,
        "L": np.linspace(1.0, 1.1, n) - 0.001,
        "C": np.linspace(1.0, 1.1, n) + np.sin(np.arange(n) / 10) * 0.005,
    }, index=idx)


@pytest.mark.parametrize("plugin_fqn", ["fwbg-core:ema", "fwbg-core:sma"])
def test_ma_plugin_output_stable(plugin_fqn):
    registry = get_registry()
    registry.auto_discover()
    cls = registry.get(plugin_fqn)
    plugin = cls()
    df = _synth_df()
    params = {"lines": [{"lookback": 20}, {"lookback": 50}], "crossings": True}
    features = plugin.compute(df, params)
    # Snapshot a handful of values per produced column
    snapshot = {col: features[col].iloc[[50, 100, 150]].round(6).tolist()
                for col in features.columns}
    # Print on failure so you can manually verify / update.
    print(plugin_fqn, snapshot)
    assert all(features.columns), "no columns produced"
```

**Step 2: Run and capture the printed snapshot — pin it explicitly**

```bash
pytest tests/test_ma_snapshot.py -v -s
```

Copy the printed dict and turn the test into a hard equality check:

```python
EXPECTED = {
    "fwbg-core:ema": {
        "ema_20": [...],
        "ema_50": [...],
        ...
    },
    "fwbg-core:sma": { ... },
}

assert snapshot == EXPECTED[plugin_fqn]
```

**Step 3: Commit**

```bash
git add tests/test_ma_snapshot.py
git commit -m "test: snapshot EMA/SMA outputs before base-class refactor"
```

---

## Task 2: Create the base class

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/indicators/_base/__init__.py`
- Create: `src/fwbg/plugins/fwbg-core/indicators/_base/moving_average.py`

NOTE: Directories starting with `_` should NOT be auto-discovered as plugins. Verify by checking the discovery code or by adding the directory to a skip list.

**Step 1: Write the base class**

```python
"""Shared scaffolding for moving-average indicator plugins (EMA, SMA, ...)."""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

import pandas as pd

from fwbg_sdk import BaseIndicator
from fwbg.utils.indicators import shift_features  # already exists


class BaseMovingAverageIndicator(BaseIndicator):
    """Common scaffolding for moving-average indicators.

    Subclasses must override `_compute_ma`. Everything else (parameter
    parsing, crossing detection, feature column listing) is provided here.
    """

    # ---- Hook methods ---------------------------------------------------

    @abstractmethod
    def _compute_ma(self, series: pd.Series, window: int) -> pd.Series:
        """Return the moving average series for *series* with *window* bars."""

    @property
    @abstractmethod
    def _prefix(self) -> str:
        """Column-name prefix, e.g. 'ema' or 'sma'."""

    # ---- Shared helpers --------------------------------------------------

    @staticmethod
    def _line_key(lookback: int) -> str:
        return f"_{lookback}"

    def _parse_lines(self, raw: Any) -> List[Dict[str, int]]:
        """Normalize the `lines` param to a list of {'lookback': int}."""
        if raw is None:
            return []
        out: List[Dict[str, int]] = []
        for entry in raw:
            if isinstance(entry, int):
                out.append({"lookback": entry})
            elif isinstance(entry, dict) and "lookback" in entry:
                out.append({"lookback": int(entry["lookback"])})
            else:
                raise ValueError(f"Invalid line spec: {entry!r}")
        return out

    def get_feature_columns(self, params: dict) -> list[str]:
        lines = self._parse_lines(params.get("lines"))
        cols = [f"{self._prefix}{self._line_key(l['lookback'])}" for l in lines]
        if params.get("crossings") and len(lines) >= 2:
            cols.extend(
                f"{self._prefix}_cross_{a['lookback']}_{b['lookback']}"
                for a, b in zip(lines, lines[1:])
            )
        return cols

    def compute(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        lines = self._parse_lines(params.get("lines"))
        if not lines:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)
        series = df["C"]
        for line in lines:
            col = f"{self._prefix}{self._line_key(line['lookback'])}"
            features[col] = self._compute_ma(series, line["lookback"])

        if params.get("crossings") and len(lines) >= 2:
            for a, b in zip(lines, lines[1:]):
                fast = features[f"{self._prefix}{self._line_key(a['lookback'])}"]
                slow = features[f"{self._prefix}{self._line_key(b['lookback'])}"]
                cross_col = f"{self._prefix}_cross_{a['lookback']}_{b['lookback']}"
                features[cross_col] = (fast > slow).astype(int).diff().fillna(0)

        return shift_features(features, df.index)
```

**Step 2: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/indicators/_base/
git commit -m "feat: add BaseMovingAverageIndicator scaffolding"
```

---

## Task 3: Migrate EMA

**Files:**
- Modify: `src/fwbg/plugins/fwbg-core/indicators/ema/__init__.py`

**Step 1: Rewrite EMA as a thin subclass**

```python
"""EMA indicator plugin."""
import pandas as pd
import ta

from fwbg_sdk import register_indicator
from fwbg.plugins._base.moving_average import BaseMovingAverageIndicator


@register_indicator("ema")
class EMAIndicator(BaseMovingAverageIndicator):
    @property
    def _prefix(self) -> str:
        return "ema"

    def _compute_ma(self, series: pd.Series, window: int) -> pd.Series:
        return ta.trend.ema_indicator(series, window=window)
```

NOTE: The `fwbg.plugins._base.moving_average` import path needs to match wherever the base class lives. Verify by running `python -c "from fwbg.plugins._base.moving_average import BaseMovingAverageIndicator"` after writing.

**Step 2: Run the snapshot test from Task 1**

```bash
pytest tests/test_ma_snapshot.py::test_ma_plugin_output_stable -v -s
```

Expected: only EMA passes; SMA still uses old code so also passes.

**Step 3: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/indicators/ema/__init__.py
git commit -m "refactor: migrate EMA to BaseMovingAverageIndicator"
```

---

## Task 4: Migrate SMA

**Files:**
- Modify: `src/fwbg/plugins/fwbg-core/indicators/sma/__init__.py`

**Step 1: Mirror the EMA pattern**

```python
"""SMA indicator plugin."""
import pandas as pd
import ta

from fwbg_sdk import register_indicator
from fwbg.plugins._base.moving_average import BaseMovingAverageIndicator


@register_indicator("sma")
class SMAIndicator(BaseMovingAverageIndicator):
    @property
    def _prefix(self) -> str:
        return "sma"

    def _compute_ma(self, series: pd.Series, window: int) -> pd.Series:
        return ta.trend.sma_indicator(series, window=window)
```

**Step 2: Run snapshot test for both plugins**

```bash
pytest tests/test_ma_snapshot.py -v -s
```
Expected: both pass.

**Step 3: Run all related test suites**

```bash
pytest src/fwbg/plugins/fwbg-core/indicators/ema/tests.py \
       src/fwbg/plugins/fwbg-core/indicators/sma/tests.py \
       tests/test_core_indicators.py -x
```
Expected: all pass.

**Step 4: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/indicators/sma/__init__.py
git commit -m "refactor: migrate SMA to BaseMovingAverageIndicator"
```

---

## Task 5: Cleanup + LOC accounting

**Step 1: Verify size reduction**

```bash
wc -l src/fwbg/plugins/fwbg-core/indicators/{ema,sma}/__init__.py
```
Expected: each file < 50 LOC.

**Step 2: Remove the snapshot test if you prefer a leaner test surface**

The snapshot was a refactor scaffold; if the dedicated `tests.py` files for EMA/SMA cover the same ground, delete `tests/test_ma_snapshot.py`.

```bash
git rm tests/test_ma_snapshot.py  # optional
git commit -m "test: remove ma snapshot test (covered by per-plugin tests)"
```
