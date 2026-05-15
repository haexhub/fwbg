# Trade Metrics Utility Module — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate four-plus scattered implementations of drawdown, PnL-to-returns, win-rate, and profit-factor into a single, tested utility module at `src/fwbg/utils/metrics.py`.

**Architecture:** Each duplicated metric becomes a single, well-tested function with explicit input contract. All current call sites (simulation/trade.py, optimization/process.py, optimization/nested_cv.py, etc.) import from the new module. Behaviour is preserved exactly — this is a pure refactor.

**Tech Stack:** Python, numpy, pandas, pytest.

---

## Background

The code review found the following metrics duplicated:

| Metric | Implementations | Locations |
|---|---|---|
| Drawdown | ≥4 | [trade.py:763-784](../../src/fwbg/simulation/trade.py#L763), [trade.py:196-207](../../src/fwbg/simulation/trade.py#L196), nested_cv analog, results aggregation |
| PnL → returns | ≥2 | [trade.py:811-829](../../src/fwbg/simulation/trade.py#L811), [process.py:709](../../src/fwbg/optimization/process.py#L709) |
| Win rate | ≥10 occurrences | scattered as `sum(1 for t in trades if t["result"] > 0) / len(trades)` |
| Equity simulation | 2 variants | `monte_carlo_equity_simulation` vs `monte_carlo_equity_from_returns` — only input format differs |

Risk of bug-by-divergence: a fix applied in one location silently leaves the other implementations buggy. Already happened once with the Sharpe annualization formula (see `tests/test_metric_bugs.py`).

---

## Success Criteria

1. New module `src/fwbg/utils/metrics.py` with at minimum: `compute_drawdown`, `pnl_to_returns`, `win_rate`, `profit_factor`.
2. Each function has a unit test verifying behaviour against a small known-correct fixture.
3. Existing call sites import from the new module; duplicated inline implementations are deleted.
4. `tests/test_metrics.py`, `tests/test_metric_bugs.py`, `tests/simulation/`, `tests/optimization/` all pass.
5. Numerical outputs of an end-to-end optimization match the pre-refactor baseline to within 1e-9.

---

## Out of Scope

- Sharpe / Sortino / Calmar (already centralised — leave alone).
- Monte Carlo equity simulators (covered by separate plan if the maintainer wants them merged).
- Adding new metrics.

---

## Task 1: Define the utility module + tests

**Files:**
- Create: `src/fwbg/utils/metrics.py`
- Create: `tests/utils/test_metrics.py`

**Step 1: Write the tests first**

```python
"""Tests for the consolidated trade-metrics utility module."""
import numpy as np
import pytest

from fwbg.utils.metrics import (
    compute_drawdown,
    pnl_to_returns,
    win_rate,
    profit_factor,
)


def test_compute_drawdown_simple_equity_curve():
    equity = np.array([100.0, 110.0, 105.0, 120.0, 90.0, 100.0])
    max_dd, dd_series = compute_drawdown(equity)
    # peak at 120, trough at 90 → drawdown 30/120 = 0.25
    assert max_dd == pytest.approx(0.25)
    assert dd_series[0] == 0.0
    assert dd_series[-2] == pytest.approx(0.25)


def test_compute_drawdown_no_drawdown():
    equity = np.array([100.0, 110.0, 120.0])
    max_dd, _ = compute_drawdown(equity)
    assert max_dd == 0.0


def test_compute_drawdown_empty():
    max_dd, dd = compute_drawdown(np.array([]))
    assert max_dd == 0.0
    assert len(dd) == 0


def test_pnl_to_returns_from_starting_balance():
    pnls = np.array([10.0, -5.0, 20.0])
    returns = pnl_to_returns(pnls, starting_balance=100.0)
    # equity: 100 → 110 → 105 → 125; returns: 10/100, -5/110, 20/105
    assert returns == pytest.approx([0.1, -5 / 110, 20 / 105])


def test_pnl_to_returns_zero_balance_handles_gracefully():
    with pytest.raises(ValueError):
        pnl_to_returns(np.array([1.0]), starting_balance=0.0)


def test_win_rate_basic():
    trades = [{"result": 1.0}, {"result": -1.0}, {"result": 1.0}, {"result": 0.0}]
    assert win_rate(trades) == pytest.approx(2 / 4)


def test_win_rate_empty_trades_returns_zero():
    assert win_rate([]) == 0.0


def test_profit_factor():
    trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}, {"pnl": -10.0}]
    # gains = 30, losses = 15 → PF = 2
    assert profit_factor(trades) == pytest.approx(2.0)


def test_profit_factor_no_losses_returns_inf():
    trades = [{"pnl": 10.0}, {"pnl": 20.0}]
    assert profit_factor(trades) == float("inf")
```

**Step 2: Run, expect ImportError**

```bash
pytest tests/utils/test_metrics.py -v
```

**Step 3: Implement the module**

```python
"""Consolidated trade-metrics utility.

Single source of truth for drawdown, PnL→returns, win-rate, profit-factor.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


def compute_drawdown(equity: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return (max_drawdown, drawdown_series).

    Drawdown is computed relative to the running peak. Empty input → (0, []).
    """
    arr = np.asarray(equity, dtype=np.float64)
    if arr.size == 0:
        return 0.0, np.array([], dtype=np.float64)
    peaks = np.maximum.accumulate(arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - arr) / peaks, 0.0)
    return float(dd.max()), dd


def pnl_to_returns(pnls: np.ndarray, starting_balance: float) -> np.ndarray:
    """Convert a sequence of trade PnLs (currency) into return fractions.

    return_i = pnl_i / equity_before_trade_i
    """
    if starting_balance <= 0:
        raise ValueError("starting_balance must be > 0")
    arr = np.asarray(pnls, dtype=np.float64)
    equity = np.concatenate(([starting_balance], starting_balance + np.cumsum(arr)))
    return arr / equity[:-1]


def win_rate(trades: Sequence[Mapping]) -> float:
    """Fraction of trades with result > 0."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("result", 0) > 0)
    return wins / len(trades)


def profit_factor(trades: Iterable[Mapping]) -> float:
    """gross profit / gross loss (positive)."""
    gross_win = 0.0
    gross_loss = 0.0
    for t in trades:
        pnl = t.get("pnl", 0.0)
        if pnl > 0:
            gross_win += pnl
        elif pnl < 0:
            gross_loss += -pnl
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss
```

**Step 4: Re-run tests**

```bash
pytest tests/utils/test_metrics.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/fwbg/utils/metrics.py tests/utils/test_metrics.py
git commit -m "feat: add consolidated trade-metrics utility module"
```

---

## Task 2: Migrate trade.py drawdown call sites

**Files:**
- Modify: `src/fwbg/simulation/trade.py`

**Step 1: Find all drawdown implementations**

```bash
grep -n "drawdown\|max_dd\|maximum.accumulate" src/fwbg/simulation/trade.py
```

Each match is a candidate to replace. The two known ones live around lines 763-784 and 196-207.

**Step 2: Replace each with a call to `compute_drawdown`**

Before:
```python
peaks = np.maximum.accumulate(equity)
dd = (peaks - equity) / peaks
max_dd = dd.max()
```

After:
```python
from fwbg.utils.metrics import compute_drawdown
max_dd, dd = compute_drawdown(equity)
```

**Step 3: Run trade + simulation tests**

```bash
pytest tests/simulation/ tests/test_metrics.py -x
```

**Step 4: Commit**

```bash
git add src/fwbg/simulation/trade.py
git commit -m "refactor: use compute_drawdown in trade.py"
```

---

## Task 3: Migrate pnl_to_returns call sites

**Files:**
- Modify: `src/fwbg/simulation/trade.py:811-829`
- Modify: `src/fwbg/optimization/process.py:709`

**Step 1: Find all sites**

```bash
grep -rn "pnl.*returns\|cumsum.*starting\|/ equity_before" src/fwbg/
```

**Step 2: Replace with `pnl_to_returns`**

Match the existing semantics carefully — some sites use "return relative to starting balance" (constant denominator), not "return relative to equity-before-trade". If you find the former, add it as a `pnl_to_returns_constant_base(pnls, base)` helper rather than forcing the existing semantics into the new function.

**Step 3: Run optimization + simulation tests**

```bash
pytest tests/test_pnl_correctness.py tests/test_computation_correctness.py tests/optimization/ tests/simulation/ -x
```

**Step 4: Commit**

```bash
git add src/fwbg/simulation/trade.py src/fwbg/optimization/process.py
git commit -m "refactor: use pnl_to_returns helper"
```

---

## Task 4: Migrate win_rate + profit_factor

**Files:**
- Multiple — search:

```bash
grep -rn 'result"\] > 0\|t\["pnl"\] > 0\|gross_profit\|gross_loss' src/fwbg/
```

**Step 1: For each match, replace the inline expression with `win_rate(trades)` / `profit_factor(trades)`**

Watch for subtle differences:
- some sites filter to "non-zero result trades" first
- some include break-even trades as wins
- preserve original semantics — add a `include_breakeven=False` flag if needed

**Step 2: Run full suite**

```bash
pytest tests/test_metrics.py tests/test_metric_bugs.py tests/simulation/ tests/optimization/ -x
```

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: use win_rate / profit_factor helpers"
```

---

## Task 5: Verify end-to-end numerical stability

**Step 1: Pick a deterministic strategy and run before + after**

```bash
git stash  # temporarily revert refactor
fwbg --assets EURUSD --strategy-file strategies/configs/<a-strategy>.json --run-id baseline_pre
git stash pop
fwbg --assets EURUSD --strategy-file strategies/configs/<a-strategy>.json --run-id baseline_post
```

**Step 2: Diff the produced metrics**

```bash
diff <(jq -S . test_results/baseline_pre/grid_details/EURUSD/unified_metrics.json) \
     <(jq -S . test_results/baseline_post/grid_details/EURUSD/unified_metrics.json)
```
Expected: zero diff, or differences below 1e-9.

**Step 3: If non-zero diff: investigate before merging**

A truly pure refactor must not change numbers. Any diff means a semantic mismatch — fix.
