# Nested CV Data Leakage — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate two known nested-CV data-leakage paths so backtest results are not biased by future information bleeding into inner-fold training data.

**Architecture:** Both issues share the same root cause: a quantity (target labels, regime classification) is computed once on the outer fold's full data and then sliced for inner folds. The slice respects fold boundaries but ignores the embargo, and any non-causal computation contaminates inner-train rows with inner-val statistics. Fix by either (a) recomputing per inner fold, or (b) verifying the computation is strictly causal (rolling with `.shift(1)` and no fillna(method='bfill') etc.).

**Tech Stack:** Python, pandas, numpy, pytest.

---

## Background

The code review identified two specific leak paths:

### Leak 1: Targets sliced without embargo
[src/fwbg/optimization/targets.py:631-666](../../src/fwbg/optimization/targets.py#L631) (`slice_targets_for_fold`)

```python
# Targets are precomputed on the WHOLE outer-fold inner_df
full_targets_long, full_targets_short = compute_targets_cached(inner_df, ...)
# Then sliced by index for each inner fold:
fold_targets_long = full_targets_long[fold_df.index]
```

Problem: the target for bar `t` is the outcome over the *next* N bars (`timeout_bars`). If bar `t` is the last train bar and `t+1` is in val, the target at `t` is computed using val data — exactly the embargo violation walk-forward CV is supposed to prevent.

### Leak 2: Regime bitmask computed on full outer fold
[src/fwbg/optimization/process_fold.py:363-369](../../src/fwbg/optimization/process_fold.py#L363)

```python
# Regime computed once on (train + test) combined:
regime_df = compute_regime(combined_df)
# Then copied to inner-fold train/test slices by index:
train_df_fold["_regime"] = regime_df.loc[train_df_fold.index, "_regime"]
```

Problem: if `compute_regime` uses any non-causal operation (rolling without `.shift(1)`, ATR(window), etc.) the regime label at bar `t` depends on bars after `t`. When we then assert "this is a train-only feature", that's false.

---

## Success Criteria

1. New tests prove targets slice for fold N does *not* see bars from fold N+1 within `timeout_bars + embargo_bars`.
2. New tests prove regime values for inner-train bars do not change when inner-val bars are altered.
3. All existing tests in `tests/test_lookahead_bias.py`, `tests/test_no_bias_in_system.py`, `tests/optimization/` still pass.
4. Walk-forward optimization end-to-end run still completes (no crashes, no regressions in metrics structure).

---

## Out of Scope

- COT/macro data lookahead (separate plan: `2026-05-15-cot-macro-release-lag.md`).
- General indicator causality audit (covered by older plan `2026-02-06-lookahead-bias-fixes.md`).
- Performance optimization of per-fold recomputation (only do it if measurable regression).

---

## Task 1: Reproduce targets-embargo leak with a failing test

**Files:**
- Test: `tests/optimization/test_targets_embargo.py` (create)

**Step 1: Write failing test**

```python
"""Embargo regression test for slice_targets_for_fold.

The target at bar t encodes the trade outcome over the next `timeout_bars`
bars. If t is the last bar of a train fold and t+1..t+timeout_bars overlap
the val fold, the train target leaks val information.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import compute_targets_cached, slice_targets_for_fold


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "O": np.linspace(1.0, 1.1, n),
        "H": np.linspace(1.0, 1.1, n) + 0.001,
        "L": np.linspace(1.0, 1.1, n) - 0.001,
        "C": np.linspace(1.0, 1.1, n),
    }, index=idx)


def test_target_for_last_train_bar_does_not_use_val_data():
    ctx = SimulationContext(spread=0.0001, max_trade_bars=10, embargo_bars=10)
    full_df = _make_synthetic_df(200)
    # First 100 bars are "train", next 100 are "val". Embargo = 10.
    train_df = full_df.iloc[:100]
    # Pre-compute targets on the full set (current behaviour)
    long_full, short_full, *_ = compute_targets_cached(full_df, ctx, tp=20, sl=10)
    # Now mutate the val region — the train target slice must NOT change.
    mutated = full_df.copy()
    mutated.loc[mutated.index[100]:, "H"] = 999.0  # massive favourable spike in val
    long_mut, short_mut, *_ = compute_targets_cached(mutated, ctx, tp=20, sl=10)

    slice_train, _, *_ = slice_targets_for_fold(long_full, short_full, full_df, train_df, ctx)
    slice_train_mut, _, *_ = slice_targets_for_fold(long_mut, short_mut, mutated, train_df, ctx)

    # The last `timeout_bars + embargo_bars` train targets MUST be embargoed
    # (either masked / NaN / recomputed without val data). Compare per-bar.
    assert np.array_equal(slice_train, slice_train_mut), (
        "Train targets changed when val data was mutated — embargo leak."
    )
```

**Step 2: Run test to confirm it fails**

```bash
pytest tests/optimization/test_targets_embargo.py -v
```
Expected: FAIL (train slice changes when val data is mutated).

**Step 3: Commit the failing test**

```bash
git add tests/optimization/test_targets_embargo.py
git commit -m "test: add failing regression test for targets embargo leak"
```

---

## Task 2: Fix targets embargo leak

**Files:**
- Modify: `src/fwbg/optimization/targets.py` (function `slice_targets_for_fold` and/or `compute_targets_cached`)

**Decide between two fixes — pick the one the maintainer prefers:**

- **Option A (preferred — purity):** recompute targets per inner fold over `train_df + embargo_buffer` only, never touching val/test data.
- **Option B (cheaper — masking):** keep the precompute path but mask out the last `timeout_bars + embargo_bars` targets in any train slice (set to NaN, downstream code must already handle NaN labels).

Recommendation: **Option B** is safer to ship as a hotfix; **Option A** is the right long-term fix and what the test in Task 1 implicitly requires (mutated val → unchanged train slice).

**Step 1: Implement chosen fix**

For Option A:
```python
def slice_targets_for_fold(full_long, full_short, full_df, fold_df, ctx):
    # Recompute targets only over fold_df's own bars; the slicing path through
    # full_long/full_short is dropped. The full arrays are still useful as a
    # cache key but must not be sliced into the output.
    long_fold, short_fold, ok_long, ok_short = compute_targets_cached(
        fold_df, ctx, tp=ctx.tp, sl=ctx.sl,
    )
    return long_fold, short_fold, ok_long, ok_short
```

For Option B:
```python
def slice_targets_for_fold(full_long, full_short, full_df, fold_df, ctx):
    mask = full_df.index.get_indexer(fold_df.index)
    long = full_long[mask].copy()
    short = full_short[mask].copy()
    # Last `timeout + embargo` rows of any contiguous fold see future data
    cutoff = ctx.max_trade_bars + ctx.embargo_bars
    long[-cutoff:] = np.nan
    short[-cutoff:] = np.nan
    return long, short, ...
```

**Step 2: Run the new test**

```bash
pytest tests/optimization/test_targets_embargo.py -v
```
Expected: PASS.

**Step 3: Run the broader optimization suite to catch regressions**

```bash
pytest tests/optimization/ tests/test_lookahead_bias.py -x
```
Expected: PASS.

**Step 4: Commit**

```bash
git add src/fwbg/optimization/targets.py
git commit -m "fix: embargo train targets in slice_targets_for_fold"
```

---

## Task 3: Reproduce regime-leak with a failing test

**Files:**
- Test: `tests/optimization/test_regime_causality.py` (create)

**Step 1: Write failing test**

```python
"""Regression test: regime label for bar t must not depend on bars > t.

The regime bitmask is computed once on the outer fold's combined train+test
data and then re-attached to inner folds. If the underlying regime function
uses any non-causal operation, mutating bars after t will change the regime
at t.
"""
import numpy as np
import pandas as pd

from fwbg.optimization.process_fold import _attach_regime_to_fold  # see below
from tests.optimization.test_targets_embargo import _make_synthetic_df  # reuse


def test_regime_at_t_independent_of_future_bars():
    df = _make_synthetic_df(500)
    # Split at index 200; "train" is [0:200], "test" is [200:500]
    train_df = df.iloc[:200]

    regime_baseline = _attach_regime_to_fold(df.copy())
    mutated = df.copy()
    mutated.loc[mutated.index[200]:, "H"] = 999.0
    mutated.loc[mutated.index[200]:, "L"] = -1.0
    regime_mut = _attach_regime_to_fold(mutated)

    # The regime label on train bars must be identical.
    np.testing.assert_array_equal(
        regime_baseline.loc[train_df.index, "_regime"].values,
        regime_mut.loc[train_df.index, "_regime"].values,
        err_msg="Regime on train bars changed when test bars were mutated.",
    )
```

NOTE: `_attach_regime_to_fold` is the function we will extract in Task 4; the test imports it preemptively so the test will fail with ImportError first, then with a value mismatch after the function is extracted.

**Step 2: Run, expect ImportError (then later a value mismatch)**

```bash
pytest tests/optimization/test_regime_causality.py -v
```

**Step 3: Commit**

```bash
git add tests/optimization/test_regime_causality.py
git commit -m "test: add failing regression test for regime causality"
```

---

## Task 4: Make regime causal

**Files:**
- Modify: `src/fwbg/optimization/process_fold.py:363-369`
- Modify: whichever module owns `compute_regime` — find with `grep -rn "def compute_regime\|class.*Regime" src/`.

**Step 1: Locate every non-causal operation in `compute_regime`**

```bash
grep -n "rolling\|ewm\|shift\|fillna" $(grep -lrn "def compute_regime\|class.*Regime" src/)
```

For each match, verify the operation either (a) shifts the result by 1, or (b) uses `min_periods` such that early NaN is honest, or (c) only reads past data by construction.

**Step 2: Refactor compute_regime to be causal**

Typical fix pattern:

```python
# Before — uses centered or zero-shift rolling:
df["vol"] = df["C"].rolling(20).std()

# After — explicitly causal:
df["vol"] = df["C"].rolling(20, min_periods=20).std().shift(1)
```

Apply consistently. Add a docstring stating "regime at bar t uses bars [..t-1] only".

**Step 3: Extract `_attach_regime_to_fold` helper in process_fold.py**

Move the inline regime computation/assignment block into a named function so the test can import it. Signature:

```python
def _attach_regime_to_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regime causally and return df with a `_regime` column."""
    ...
```

Update the existing call site in `process_fold.py:363-369` to use the new helper.

**Step 4: Run the regime causality test**

```bash
pytest tests/optimization/test_regime_causality.py -v
```
Expected: PASS.

**Step 5: Run broader optimization + bias suite**

```bash
pytest tests/optimization/ tests/test_lookahead_bias.py tests/test_no_bias_in_system.py tests/test_regime.py tests/test_generic_regime_filter.py -x
```
Expected: PASS.

**Step 6: Commit**

```bash
git add src/fwbg/optimization/process_fold.py src/fwbg/<regime-file>.py
git commit -m "fix: ensure regime computation is strictly causal"
```

---

## Task 5: End-to-end smoke test

**Step 1: Run a small full optimization**

```bash
fwbg --assets EURUSD --strategy-file strategies/configs/<some-existing-strategy>.json
```
Expected: completes without error; metrics still produced.

**Step 2: Compare key aggregate metrics (mean sharpe, total trades) with a pre-fix baseline run if available.** A small but non-zero change in metrics is expected — that's the bias being removed. Document the delta in the commit message.

**Step 3: Final commit (if anything to commit)**

```bash
git commit -m "docs: record metric delta from nested-CV leak fixes"
```
