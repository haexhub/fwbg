# Parallel Walk-Forward Folds

## Problem

Walk-forward outer folds run sequentially. A single-asset run with 8 folds
spends ~85% of its time in grid search (XGBoost training). With 24 CPU cores,
only one XGBoost fit runs at a time — the rest of the cores sit idle between
tree-building rounds.

## Solution

Run outer folds in parallel using `ThreadPoolExecutor`. XGBoost and Numba
release the GIL, so multiple folds can train concurrently within one process.

## Changes

### 1. `ResourceConfig` (config.py)

Add `max_parallel_folds: int = 1` — default preserves current behavior.

### 2. `_process_single_variant` (process.py)

Replace the sequential fold loop with a `ThreadPoolExecutor`:

- Submit all folds as tasks
- Collect results via `as_completed`
- Aggregate progress across folds before reporting (ProgressTracker sees
  a single monotonically increasing counter per asset — no changes needed
  in the progress system or dashboard)

### 3. XGBoost thread budget (xgb_config.py)

Before starting the executor, set:

```
n_jobs = max(1, cpu_count // max_parallel_folds)
```

This prevents K parallel folds × N OpenMP threads from over-subscribing cores.
The existing `set_xgboost_n_jobs()` setter is already available.

## What stays unchanged

- `process_fold.py` — each fold already creates its own DataFrames
- `grid_search.py` / `nested_cv.py` — no shared mutable state between folds
- Progress system (progress.py, run_progress.py) — aggregation happens at source
- Dashboard API — reads the same progress.json format

## Trade-offs

- **Memory**: K parallel folds hold K copies of per-fold DataFrames (~50-100 MB
  each for indicator pools). With `max_parallel_folds=4`, expect ~200-400 MB
  extra. The heavy `precomputed_raw_df` is shared read-only (no copy).
- **GIL contention**: Pandas operations (indicator computation, feature
  selection) hold the GIL but account for only ~10-15% of fold time. The
  dominant cost (XGBoost `.fit()`) is GIL-free.
- **Diminishing returns**: Beyond `cpu_count / 2` parallel folds, each XGBoost
  fit gets fewer threads and may slow down. Sweet spot is likely 2-4 folds.
