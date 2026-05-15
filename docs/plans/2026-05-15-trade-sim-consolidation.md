# Trade Simulation Helpers Consolidation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge the three near-duplicate trade-simulation wrappers (`simulate_trades_sequential`, `simulate_trades_sequential_separate_ct`, `_simulate_single_direction`) into a single function with explicit parameters.

**Architecture:** All three wrappers ultimately call the same Numba-compiled core (`_simulate_trades_core`) with slightly different argument plumbing. Introduce one canonical entry point with parameters that cover all three call-sites' needs (direction filter, separate-CT mode, etc.). The Numba core stays unchanged.

**Tech Stack:** Python, numpy, numba, pytest.

---

## Background

[src/fwbg/simulation/trade.py](../../src/fwbg/simulation/trade.py) contains three wrappers:

- `simulate_trades_sequential(...)` — both directions, joint CT (confidence-threshold) sweep
- `simulate_trades_sequential_separate_ct(...)` — both directions, but CT chosen per direction
- `_simulate_single_direction(direction, ...)` — single direction only

The body of each differs only in a few branches before calling the Numba core. The risk of bug-by-divergence is real: a fix to the TP/SL hit check in the core stays correct, but a fix to the wrapper-level argument prep can be applied to only one of three sites.

---

## Success Criteria

1. Single canonical entry point `simulate_trades(...)` with explicit `direction_filter: Optional[int] = None` and `separate_ct: bool = False` parameters.
2. Three old wrappers either deleted or kept as thin compatibility shims that delegate to the canonical version.
3. All callers migrated. `grep -rn "simulate_trades_sequential\|_simulate_single_direction"` returns zero hits in non-test code.
4. Existing tests in `tests/simulation/`, `tests/test_exit_strategies.py`, `tests/test_exit_strategy_dispatch.py`, `tests/test_per_trade_params_simulation.py` all pass.
5. End-to-end strategy run produces bit-identical output (md5 of `trades.json`) before vs after.

---

## Out of Scope

- Changing the Numba core's logic.
- Removing scale-in support.
- Adding new exit strategies.

---

## Task 1: Snapshot current wrapper outputs

**Files:**
- Test: `tests/simulation/test_simulate_trades_consolidation.py` (create)

**Step 1: Write a snapshot test that exercises all three wrappers**

```python
"""Lock current behaviour of the three trade-simulation wrappers before merging."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from fwbg.simulation.trade import (
    simulate_trades_sequential,
    simulate_trades_sequential_separate_ct,
    _simulate_single_direction,
)
from fwbg.core.context import SimulationContext


def _fixture():
    np.random.seed(7)
    n = 300
    df = pd.DataFrame({
        "O": 1.0 + np.cumsum(np.random.randn(n) * 0.0005),
        "H": None, "L": None, "C": None,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
    df["C"] = df["O"] + np.random.randn(n) * 0.0002
    df["H"] = np.maximum(df["O"], df["C"]) + 0.0003
    df["L"] = np.minimum(df["O"], df["C"]) - 0.0003
    ctx = SimulationContext(spread=0.0001, max_trade_bars=20)
    signals_long = np.random.rand(n) > 0.7
    signals_short = np.random.rand(n) > 0.7
    return df, ctx, signals_long, signals_short


def _hash_trades(trades):
    """Stable hash of a trade list."""
    canonical = json.dumps(trades, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()


def test_snapshot_both_directions_joint_ct():
    df, ctx, sl, ss = _fixture()
    trades = simulate_trades_sequential(df, ctx, sl, ss, tp=30, sl=20, ct=0.5)
    print("joint_ct:", _hash_trades(trades))


def test_snapshot_both_directions_separate_ct():
    df, ctx, sl, ss = _fixture()
    trades = simulate_trades_sequential_separate_ct(df, ctx, sl, ss, tp=30, sl=20, ct_long=0.5, ct_short=0.5)
    print("separate_ct:", _hash_trades(trades))


def test_snapshot_long_only():
    df, ctx, sl, _ = _fixture()
    trades = _simulate_single_direction(1, df, ctx, sl, tp=30, sl=20, ct=0.5)
    print("long_only:", _hash_trades(trades))
```

NOTE: the signatures above are illustrative — match them to the *current* code before running.

**Step 2: Run, capture the three hashes, harden into assertions**

```bash
pytest tests/simulation/test_simulate_trades_consolidation.py -v -s
```

Replace the `print(...)` lines with `assert _hash_trades(trades) == "<captured-hash>"`.

**Step 3: Commit**

```bash
git add tests/simulation/test_simulate_trades_consolidation.py
git commit -m "test: snapshot three trade-sim wrappers before consolidation"
```

---

## Task 2: Design the canonical entry point

**Files:**
- Modify: `src/fwbg/simulation/trade.py`

**Step 1: Design the unified signature**

```python
def simulate_trades(
    df: pd.DataFrame,
    ctx: SimulationContext,
    signals_long: np.ndarray,
    signals_short: np.ndarray,
    *,
    tp: float,
    sl: float,
    ct: Optional[float] = None,
    ct_long: Optional[float] = None,
    ct_short: Optional[float] = None,
    direction_filter: Optional[int] = None,  # +1, -1, or None
    timeout_bars: Optional[int] = None,
    scale_levels: Optional[Sequence[float]] = None,
) -> list[dict]:
    """Unified trade-simulation entry point.

    - direction_filter=None: both directions.
    - direction_filter=1 or -1: skip the other direction.
    - ct vs ct_long+ct_short: ct is the joint threshold; ct_long/ct_short
      override per direction. Mutually exclusive with ct.
    """
    ...
```

**Step 2: Validate parameter coherence**

```python
if ct is not None and (ct_long is not None or ct_short is not None):
    raise ValueError("Use either `ct` or `ct_long`/`ct_short`, not both")
if direction_filter not in (None, 1, -1):
    raise ValueError(f"direction_filter must be None/1/-1, got {direction_filter!r}")
```

**Step 3: Body — delegate to the existing Numba core**

Port whatever per-wrapper logic was unique into the unified function. Make sure every branch covered by the three snapshot tests is reachable.

**Step 4: Make the three old wrappers thin shims that delegate**

```python
def simulate_trades_sequential(df, ctx, sl_, ss, tp, sl, ct):
    return simulate_trades(df, ctx, sl_, ss, tp=tp, sl=sl, ct=ct)


def simulate_trades_sequential_separate_ct(df, ctx, sl_, ss, tp, sl, ct_long, ct_short):
    return simulate_trades(df, ctx, sl_, ss, tp=tp, sl=sl, ct_long=ct_long, ct_short=ct_short)


def _simulate_single_direction(direction, df, ctx, signals, tp, sl, ct):
    if direction == 1:
        return simulate_trades(df, ctx, signals, np.zeros_like(signals), tp=tp, sl=sl, ct=ct, direction_filter=1)
    return simulate_trades(df, ctx, np.zeros_like(signals), signals, tp=tp, sl=sl, ct=ct, direction_filter=-1)
```

**Step 5: Run snapshot tests**

```bash
pytest tests/simulation/test_simulate_trades_consolidation.py -v
```
Expected: all three pass — hashes match exactly.

**Step 6: Commit**

```bash
git add src/fwbg/simulation/trade.py
git commit -m "refactor: introduce unified simulate_trades entry point"
```

---

## Task 3: Migrate callers off the shims

**Files:**
- Modify: `src/fwbg/optimization/process.py`, `process_fold.py`, `unified_simulation.py`, `grid_search.py`, anywhere else.

**Step 1: Locate callers**

```bash
grep -rn "simulate_trades_sequential\|_simulate_single_direction" src/
```

**Step 2: Replace each call with the canonical entry point**

Be explicit: pass `ct=...` or `ct_long=...,ct_short=...` per the caller's intent. No keyword shuffling.

**Step 3: Re-run full simulation + optimization tests**

```bash
pytest tests/simulation/ tests/optimization/ tests/test_exit_strategies.py tests/test_exit_strategy_dispatch.py tests/test_per_trade_params_simulation.py -x
```

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: migrate callers to unified simulate_trades"
```

---

## Task 4: Drop the shims

**Files:**
- Modify: `src/fwbg/simulation/trade.py`

**Step 1: Verify no callers remain**

```bash
grep -rn "simulate_trades_sequential\|_simulate_single_direction" src/ tests/
```

The only matches that may remain are in tests — update those tests to call `simulate_trades` directly.

**Step 2: Delete the three shim functions**

**Step 3: Run the full test suite one more time**

```bash
pytest tests/ -x
```

**Step 4: Commit**

```bash
git add src/fwbg/simulation/trade.py tests/
git commit -m "refactor: drop trade-sim wrapper shims"
```

---

## Task 5: End-to-end byte-identity check

**Step 1: Run the same strategy on the same data at the parent commit and the head**

```bash
git checkout <pre-refactor-commit> -- :
fwbg --assets EURUSD --strategy-file strategies/configs/<deterministic-strategy>.json --run-id pre
git checkout -
fwbg --assets EURUSD --strategy-file strategies/configs/<deterministic-strategy>.json --run-id post

md5sum test_results/pre/grid_details/EURUSD/trades.json \
       test_results/post/grid_details/EURUSD/trades.json
```

Expected: identical hashes. If they differ, the refactor is not yet behaviour-preserving — investigate before merging.
