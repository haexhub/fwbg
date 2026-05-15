# Progress Tracking Unification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the responsibilities of `src/fwbg/utils/progress.py` and `src/fwbg/utils/run_progress.py` so they no longer duplicate state — `run_progress.py` becomes the single source of truth for progress state; `progress.py` becomes the terminal/UI renderer that consumes that state.

**Architecture:** Today, both modules independently track phases, stages, fold positions, and worker status. Race conditions are documented in `progress.py:81` (parallel mode suppresses updates inconsistently). The fix: `RunProgressWriter` in `run_progress.py` is authoritative; `progress.py` no longer holds state but renders snapshots produced by the writer. The terminal "live progress bar" subscribes to writer events / re-reads `progress.json`.

**Tech Stack:** Python, threading, JSON, terminal rendering (rich/plain).

---

## Background

The code review highlighted:

- [progress.py:81](../../src/fwbg/utils/progress.py#L81) — `report_progress()` reads/writes `worker_status` dict without holding a lock in parallel mode.
- [progress.py:692](../../src/fwbg/utils/progress.py#L692) — `fold_progress = max(0, (fold - 1)) / total_folds` papers over an unclear semantics for `fold == 0`.
- [run_progress.py:254](../../src/fwbg/utils/run_progress.py#L254) — overall progress fraction (recently capped at 1.0, but still computed redundantly).
- Both modules carry their own concept of "current phase", "current stage", and "current asset".

Risk: any change to progress semantics has to be made in both places; the two drift over time.

---

## Success Criteria

1. `progress.py` owns no mutable state beyond pure display/render concerns (formatting, ANSI, throttling).
2. `run_progress.py` (specifically `RunProgressWriter` / `RunProgress`) is the single authoritative state container.
3. Public functions in `progress.py` (e.g. `report_progress`, `set_parallel_mode`) become thin wrappers that forward to the writer.
4. The dashboard `/runs/{id}/progress` endpoint and the terminal renderer produce numerically identical progress fractions for the same run.
5. All tests in `tests/test_progress_display.py` and any other progress-touching tests still pass.

---

## Out of Scope

- Adding new progress UIs.
- Changing the on-disk `progress.json` schema (compatibility with the dashboard).
- Removing rich/colour rendering.

---

## Task 1: Map the current contract

**Step 1: Document every public function in `progress.py`**

```bash
grep -n "^def \|^class " src/fwbg/utils/progress.py
grep -n "^def \|^class " src/fwbg/utils/run_progress.py
```

**Step 2: Write `docs/architecture/progress-tracking-current.md` (one-pager)**

For each public function, capture:
- callers (`grep -rn` for usage)
- what state it reads/writes
- which module is "right" if both have a version

This is the bridge document; without it, the refactor will move ad-hoc.

**Step 3: Commit**

```bash
git add docs/architecture/progress-tracking-current.md
git commit -m "docs: map current progress-tracking contract"
```

---

## Task 2: Lock current rendering behaviour

**Files:**
- Test: `tests/test_progress_render_snapshot.py` (create)

**Step 1: Snapshot terminal renderer output for fixed inputs**

The existing `tests/test_progress_display.py` covers some renderer paths but mostly via internal state. Add a snapshot test that:

- builds a known `RunProgress` instance (or the equivalent state in `progress.py`)
- renders a frame
- asserts on the rendered string (or its hash)

```python
from fwbg.utils.progress import render_overall_bar, render_asset_lines

def test_render_overall_bar_50pct():
    line = render_overall_bar(0.5, eta_seconds=120)
    assert "[█████" in line or "50%" in line  # adapt to the actual renderer
    print(repr(line))
```

**Step 2: Capture + harden into hard equality.**

**Step 3: Commit**

```bash
git add tests/test_progress_render_snapshot.py
git commit -m "test: snapshot progress renderer output"
```

---

## Task 3: Promote `RunProgress` to single source of truth

**Files:**
- Modify: `src/fwbg/utils/run_progress.py`
- Modify: `src/fwbg/utils/progress.py`

**Step 1: Add a read-only accessor on `RunProgressWriter`**

```python
def snapshot(self) -> RunProgress:
    """Return a deep-copy snapshot of the current progress state."""
    with self._lock:
        return copy.deepcopy(self._progress)
```

**Step 2: Replace `progress.py` internal state dicts with calls into the writer**

For example, `worker_status` becomes:

```python
def get_worker_status(writer: RunProgressWriter) -> dict:
    snap = writer.snapshot()
    return {sym: a.status for sym, a in snap.assets.items()}
```

Anywhere `progress.py` used to mutate `worker_status` directly, route the mutation through the writer (or refuse it — `progress.py` should not mutate state at all).

**Step 3: Run progress tests**

```bash
pytest tests/test_progress_display.py tests/test_progress_render_snapshot.py -x
```

**Step 4: Commit**

```bash
git add src/fwbg/utils/progress.py src/fwbg/utils/run_progress.py
git commit -m "refactor: route progress state through RunProgressWriter"
```

---

## Task 4: Migrate callers to use the writer

**Step 1: Find all callers**

```bash
grep -rn "from fwbg.utils.progress\|from fwbg.utils.run_progress" src/ tests/
```

**Step 2: For each caller, decide which side it belongs to**

- "I update state" → use `RunProgressWriter` methods (`update_asset`, `update_stage`, …)
- "I render to terminal" → use `progress.py` rendering functions (passing in a snapshot)
- "I read from disk for the dashboard" → use `safe_load_json` on `progress.json`

**Step 3: Migrate each, one commit per file**

```bash
git add src/fwbg/<file>.py
git commit -m "refactor: migrate <file> to RunProgressWriter"
```

---

## Task 5: Fix the `fold == 0` ambiguity

**Files:**
- Modify: `src/fwbg/utils/progress.py:692` (or its new home in `run_progress.py`)

**Step 1: Replace the `max(0, fold - 1)` papering-over with explicit semantics**

```python
# Before:
fold_progress = max(0, (fold - 1)) / total_folds

# After — explicit:
if fold <= 0:
    fold_progress = 0.0  # not started yet
else:
    fold_progress = (fold - 1) / max(total_folds, 1)
```

Document the contract in the docstring: "fold is 1-based; fold=0 means 'not yet started'".

**Step 2: Add a unit test**

```python
def test_fold_progress_zero_means_not_started():
    assert compute_fold_progress(fold=0, total_folds=5) == 0.0


def test_fold_progress_first_fold_in_progress():
    # fold=1 means "fold 1 currently executing, none completed yet"
    assert compute_fold_progress(fold=1, total_folds=5) == 0.0


def test_fold_progress_last_fold_completed():
    assert compute_fold_progress(fold=5, total_folds=5) == 0.8  # 4/5 completed
```

**Step 3: Commit**

```bash
git add src/fwbg/utils/progress.py tests/test_progress_display.py
git commit -m "fix: clarify fold_progress semantics for fold=0"
```

---

## Task 6: End-to-end verification

**Step 1: Run a short optimization**

```bash
fwbg --assets EURUSD --strategy-file strategies/configs/<small-strategy>.json
```

**Step 2: Watch the terminal output for any rendering regressions**

- Progress bar advances monotonically
- Asset count is correct
- ETA decreases
- No double-render or torn lines in parallel mode

**Step 3: Read the dashboard progress endpoint**

```bash
curl http://localhost:8420/api/runs/<id>/progress | jq .overall_progress_fraction
```

This number must match what the terminal showed (give or take a poll interval).
