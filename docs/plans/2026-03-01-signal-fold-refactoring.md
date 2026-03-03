# Signal/ML Fold Processing Separation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all `if model_type == "signal"` branches by cleanly separating ML and signal fold processing into distinct modules.

**Architecture:** Extract shared data preparation (preprocessing, indicator computation, feature pool extraction) into `_prepare_fold_common()`. Create a new `signal_fold.py` module with its own `prepare_signal_fold_data()` and `process_signal_fold()`. Keep `process_fold.py` for ML-only logic. `process_single_fold()` dispatches to the right module at the top — no interleaved branches.

**Tech Stack:** Python, pandas, numpy, dataclasses

**Key Bug Fix Included:** The `orb_based` exit strategy needs auxiliary columns (`*_range`, `*_sl_dist`) that the ML path's column-dropping logic removes. The signal path must preserve them. Additionally, `unified_simulation.py` must propagate `exit_strategy`/`exit_params` from fold results into the holdout context.

---

### Task 1: Extract `_prepare_fold_common` from `prepare_fold_data`

**Files:**
- Modify: `src/fwbg/optimization/process_fold.py`

**Step 1: Create `_prepare_fold_common` that returns raw data before cleaning**

Extract lines 73–133 (preprocessing + indicator computation + precomputed concat) plus the feature pool extraction (lines 135–164) into a shared helper. This function does NOT handle dropna/fillna/column-dropping — that's model-type-specific.

```python
# In process_fold.py — add above prepare_fold_data

def _prepare_fold_common(fold, fold_indicators, precomputed_raw_df,
                         preprocessing_configs, ctx, sym,
                         indicator_progress_callback=None):
    """Shared fold preparation: preprocessing, indicators, feature pool.

    Returns:
        dict with keys: train_df, test_df, full_pool, drop_cols,
        orig_train_ohlc, orig_test_ohlc, excluded_inf, excluded_nan.
        Returns None if no features remain.
    """
    pp_train_raw = fold.train_df
    pp_test_raw = fold.test_df
    orig_train_ohlc = None
    orig_test_ohlc = None
    ohlc_cols = ['O', 'H', 'L', 'C']

    if preprocessing_configs:
        from fwbg.core import get_preprocessor
        from fwbg_sdk import PipelineContext

        orig_train_ohlc = {col: fold.train_df[col].copy() for col in ohlc_cols}
        orig_test_ohlc = {col: fold.test_df[col].copy() for col in ohlc_cols}

        for pp_config in preprocessing_configs:
            pp_name = pp_config.get("name", "")
            pp_params = pp_config.get("params", {})
            try:
                pp_cls = get_preprocessor(pp_name)
                pp = pp_cls()

                train_ctx = PipelineContext(
                    df=pp_train_raw.copy(), symbol=sym, asset_class=ctx.asset_class
                )
                pp.fit(train_ctx, **pp_params)

                train_ctx = PipelineContext(
                    df=pp_train_raw.copy(), symbol=sym, asset_class=ctx.asset_class
                )
                train_ctx = pp.execute(train_ctx, **pp_params)
                pp_train_raw = train_ctx.df

                test_ctx = PipelineContext(
                    df=pp_test_raw.copy(), symbol=sym, asset_class=ctx.asset_class
                )
                test_ctx = pp.execute(test_ctx, **pp_params)
                pp_test_raw = test_ctx.df
            except Exception as e:
                log(1, f"  Preprocessing {pp_name} failed: {e}", sym)

        log(2, f"  Preprocessing: Train {len(fold.train_df)}→{len(pp_train_raw)}, "
               f"Test {len(fold.test_df)}→{len(pp_test_raw)}", sym)

    if fold_indicators:
        train_df = compute_indicator_pool(
            pp_train_raw, indicators=fold_indicators,
            progress_callback=indicator_progress_callback,
        )
        test_df = compute_indicator_pool(
            pp_test_raw, indicators=fold_indicators, progress_callback=None
        )
    else:
        train_df = pp_train_raw.copy()
        test_df = pp_test_raw.copy()

    if precomputed_raw_df is not None:
        train_df = pd.concat(
            [train_df, precomputed_raw_df.reindex(train_df.index)], axis=1
        )
        test_df = pd.concat(
            [test_df, precomputed_raw_df.reindex(test_df.index)], axis=1
        )

    # Feature pool cleaning: identify columns to drop (inf, >10% NaN)
    full_pool = get_feature_columns(train_df)

    protected_cols = set(ctx.required_features) if ctx.required_features else set()
    clean_pool = []
    excluded_inf = 0
    excluded_nan = 0
    drop_cols = []
    for col in full_pool:
        if col in train_df.columns:
            if col in protected_cols:
                clean_pool.append(col)
                continue
            has_inf = np.isinf(train_df[col]).any()
            nan_ratio = train_df[col].isna().sum() / len(train_df)
            if has_inf:
                excluded_inf += 1
                drop_cols.append(col)
            elif nan_ratio >= 0.1:
                excluded_nan += 1
                drop_cols.append(col)
            else:
                clean_pool.append(col)
    full_pool = clean_pool

    for col in list(full_pool):
        if col in test_df.columns and test_df[col].isna().all():
            full_pool.remove(col)
            drop_cols.append(col)
            excluded_nan += 1

    return {
        "train_df": train_df,
        "test_df": test_df,
        "full_pool": full_pool,
        "drop_cols": drop_cols,
        "orig_train_ohlc": orig_train_ohlc,
        "orig_test_ohlc": orig_test_ohlc,
        "excluded_inf": excluded_inf,
        "excluded_nan": excluded_nan,
    }
```

**Step 2: Create `_finalize_fold_data` for OHLC restoration + validation**

```python
def _finalize_fold_data(train_df, test_df, full_pool,
                        orig_train_ohlc, orig_test_ohlc, fold, sym):
    """Restore original OHLC after preprocessing and validate data sizes.

    Returns:
        (train_df, test_df, full_pool) or None if insufficient data.
    """
    ohlc_cols = ['O', 'H', 'L', 'C']
    if orig_train_ohlc:
        for col in ohlc_cols:
            train_df[col] = orig_train_ohlc[col].reindex(train_df.index)
            test_df[col] = orig_test_ohlc[col].reindex(test_df.index)

    log(2, f"  Fold {fold.fold_id + 1}: Train={train_df.shape} Test={test_df.shape}", sym)

    if len(train_df) < MIN_TRADES * 2:
        log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Train-Daten ({len(train_df)})", sym)
        return None

    if len(test_df) < MIN_TRADES:
        log(1, f"  Fold {fold.fold_id + 1}: SKIP - Zu wenig Test-Daten ({len(test_df)})", sym)
        return None

    if len(full_pool) < 1:
        log(1, f"  Fold {fold.fold_id + 1}: SKIP - Keine Features", sym)
        return None

    return train_df, test_df, full_pool
```

**Step 3: Rewrite `prepare_fold_data` as ML-only (no signal checks)**

```python
def prepare_fold_data(fold, fold_indicators, precomputed_raw_df,
                      preprocessing_configs, ctx, sym,
                      indicator_progress_callback=None):
    """Prepare train/test DataFrames for ML models.

    Drops columns with >10% NaN or inf, then drops NaN rows.
    """
    common = _prepare_fold_common(
        fold, fold_indicators, precomputed_raw_df,
        preprocessing_configs, ctx, sym, indicator_progress_callback,
    )
    if common is None:
        return None

    train_df = common["train_df"]
    test_df = common["test_df"]
    full_pool = common["full_pool"]
    drop_cols = common["drop_cols"]

    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        test_df = test_df.drop(columns=drop_cols, errors="ignore")

    log(2, f"  Fold {fold.fold_id + 1}: {len(full_pool)} clean features "
           f"(excl: {common['excluded_inf']} inf, {common['excluded_nan']} nan)", sym)

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    return _finalize_fold_data(
        train_df, test_df, full_pool,
        common["orig_train_ohlc"], common["orig_test_ohlc"], fold, sym,
    )
```

**Step 4: Run existing tests to ensure ML path is unchanged**

Run: `cd /home/haex/Projekte/fwbg && .venv/bin/python -m pytest tests/optimization/test_process_fold.py -v`
Expected: All existing ML tests pass (the signal tests may need updating — that's Task 3).

**Step 5: Commit**

```bash
git add src/fwbg/optimization/process_fold.py
git commit -m "refactor: extract _prepare_fold_common and _finalize_fold_data"
```

---

### Task 2: Create `signal_fold.py` with dedicated signal path

**Files:**
- Create: `src/fwbg/optimization/signal_fold.py`
- Modify: `src/fwbg/optimization/process_fold.py` (remove `_process_signal_fold`, update dispatch)

**Step 1: Create `signal_fold.py`**

```python
"""Signal model fold processing — no training, no grid search.

Signal models read pre-computed indicator columns as entry signals.
This module handles their specific data preparation (keep all rows,
fill NaN with 0, preserve auxiliary columns for exit strategies)
and direct evaluation (no inner CV needed).
"""
import dataclasses
import time

import numpy as np
import pandas as pd

from fwbg.utils.logging import log
from fwbg.utils.progress import report_phase
from .nested_cv import evaluate_on_holdout
from .process_fold import _prepare_fold_common, _finalize_fold_data


def prepare_signal_fold_data(fold, fold_indicators, precomputed_raw_df,
                             preprocessing_configs, ctx, sym,
                             indicator_progress_callback=None):
    """Prepare train/test DataFrames for signal models.

    Unlike ML models, signal models:
    - Keep ALL columns (exit strategies need auxiliary cols like *_range, *_sl_dist)
    - Fill feature NaN with 0 (NaN = "no signal") instead of dropping rows
    - Restrict feature pool to required_features only
    """
    common = _prepare_fold_common(
        fold, fold_indicators, precomputed_raw_df,
        preprocessing_configs, ctx, sym, indicator_progress_callback,
    )
    if common is None:
        return None

    train_df = common["train_df"]
    test_df = common["test_df"]
    full_pool = common["full_pool"]

    # Don't drop columns — exit strategies need auxiliary columns
    # (e.g. *_range, *_sl_dist) that have high NaN outside session windows.

    log(2, f"  Fold {fold.fold_id + 1}: {len(full_pool)} clean features "
           f"(excl: {common['excluded_inf']} inf, {common['excluded_nan']} nan)", sym)

    # Fill feature NaN with 0 (NaN = "no signal") instead of dropping rows.
    # Dropping bars creates gaps that break sequential trade simulation
    # (a trade opened during session may hit TP/SL outside session).
    for col in full_pool:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(0)

    # Only use required_features (signal columns) as the feature pool
    full_pool = [c for c in ctx.required_features if c in train_df.columns]

    return _finalize_fold_data(
        train_df, test_df, full_pool,
        common["orig_train_ohlc"], common["orig_test_ohlc"], fold, sym,
    )


def process_signal_fold(
    fold, fold_idx, n_folds,
    train_df, test_df, full_pool,
    ctx, sym, total_indicators,
):
    """Process a walk-forward fold for signal models.

    Signal models have no trainable parameters — they read pre-computed
    signal columns directly.  Skips feature selection, inner CV, and
    grid search.  Evaluates each exit strategy combo directly on the
    out-of-sample test set.

    Returns:
        (fold_result dict or None, grid_results list)
    """
    fold_idx_1based = fold.fold_id + 1
    features = list(ctx.required_features)

    best_result = None
    best_pnl = float("-inf")
    best_config = None
    all_grid_results = []

    for exit_cfg in ctx.exit_strategies:
        tp = exit_cfg.params.get("tp_mult", 1.0)
        sl = exit_cfg.params.get("sl_mult", 1.0)
        timeout_bars = exit_cfg.params.get("timeout_bars", None)

        combo_ctx = dataclasses.replace(
            ctx,
            exit_strategy=exit_cfg.name,
            exit_params=exit_cfg.params,
            exit_modifier=exit_cfg.exit_modifier,
            exit_modifier_params=exit_cfg.exit_modifier_params,
            separate_long_short=bool(exit_cfg.long_ct or exit_cfg.short_ct),
        )

        for model_hp_variant in ctx.grid_model_hyperparameters:
            hp_ctx = combo_ctx
            merged_hp = combo_ctx.model_hyperparameters
            if model_hp_variant is not None:
                merged_hp = {**combo_ctx.model_hyperparameters, **model_hp_variant}
                hp_ctx = dataclasses.replace(combo_ctx, model_hyperparameters=merged_hp)

            ct_list = exit_cfg.ct or [0.5]
            for ct in ct_list:
                candidate = {
                    "params": (tp, sl, ct),
                    "timeout_bars": timeout_bars,
                    "selected_features_long": features,
                    "selected_features_short": features,
                    "model_hyperparameters": merged_hp,
                    "exit_modifier_params": exit_cfg.exit_modifier_params,
                }

                test_result = evaluate_on_holdout(test_df, train_df, candidate, hp_ctx)

                grid_entry = {
                    "fold_id": fold.fold_id,
                    "tp_mult": tp, "sl_mult": sl, "ct": ct,
                    "n_trades": test_result["n_trades"],
                    "pnl": test_result["pnl"],
                    "win_rate": test_result["win_rate"],
                }
                all_grid_results.append(grid_entry)

                if test_result["n_trades"] >= 1 and test_result["pnl"] > best_pnl:
                    best_pnl = test_result["pnl"]
                    best_result = test_result
                    best_config = {
                        "tp": tp, "sl": sl, "ct": ct,
                        "rrr": tp / sl if sl > 0 else 0,
                        "timeout_bars": timeout_bars,
                        "model_hyperparameters": merged_hp,
                        "exit_modifier_params": exit_cfg.exit_modifier_params,
                        "exit_strategy": exit_cfg.name,
                        "exit_params": exit_cfg.params,
                    }

    if not best_result or best_result["n_trades"] < 1:
        log(2, f"  Fold {fold_idx_1based}: No trades from signal model", sym)
        return None, all_grid_results

    fold_result = {
        "fold_id": fold.fold_id,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_start": str(fold.test_df.index[0]),
        "test_end": str(fold.test_df.index[-1]),
        "inner_val_pnl": best_pnl,
        "test_pnl": best_result["pnl"],
        "test_win_rate": best_result["win_rate"],
        "test_trades": best_result["n_trades"],
        "test_trades_trace": best_result["trades"],
        "test_trades_detail": best_result.get("trades_detailed", []),
        "best_config": best_config,
        "selected_features_long": features,
        "selected_features_short": features,
    }

    log(1, f"  Fold {fold_idx_1based}: WR={best_result['win_rate']:.1%} "
           f"PnL={best_result['pnl']:.1f} Trades={best_result['n_trades']}", sym)

    return fold_result, all_grid_results
```

**Step 2: Update `process_single_fold` dispatch in `process_fold.py`**

Remove `_process_signal_fold` from `process_fold.py`. Update `process_single_fold` to import and dispatch:

```python
def process_single_fold(
    fold, fold_idx, n_folds,
    fold_indicators, precomputed_raw_df, preprocessing_configs,
    ctx, sym, total_indicators,
):
    """Process a single walk-forward fold.

    Dispatches to signal_fold or ML path based on model type.
    """
    log(1, f"=== Processing Fold {fold.fold_id + 1}/{n_folds} ===", sym)
    report_phase(sym, f"Fold {fold.fold_id + 1}/{n_folds}: Computing indicators...")

    def indicator_progress(name, idx, total):
        report_phase(sym, f"Fold {fold.fold_id + 1}: Indicators {name} ({idx}/{total})")

    t0 = time.time()

    # --- Signal models: separate data prep + fold processing ---
    if ctx.model_type == "signal":
        from .signal_fold import prepare_signal_fold_data, process_signal_fold
        fold_data = prepare_signal_fold_data(
            fold, fold_indicators, precomputed_raw_df,
            preprocessing_configs, ctx, sym,
            indicator_progress_callback=indicator_progress,
        )
        if fold_data is None:
            return None, []
        train_df, test_df, full_pool = fold_data
        log(2, f"  Fold {fold.fold_id + 1}: Data prepared ({time.time()-t0:.1f}s)", sym)
        return process_signal_fold(
            fold, fold_idx, n_folds,
            train_df, test_df, full_pool,
            ctx, sym, total_indicators,
        )

    # --- ML models: standard data prep + grid search ---
    fold_data = prepare_fold_data(
        fold, fold_indicators, precomputed_raw_df,
        preprocessing_configs, ctx, sym,
        indicator_progress_callback=indicator_progress,
    )
    if fold_data is None:
        return None, []
    train_df, test_df, full_pool = fold_data

    log(2, f"  Fold {fold.fold_id + 1}: Data prepared ({time.time()-t0:.1f}s)", sym)

    # ... rest of ML fold processing (unchanged from line 362 onwards) ...
```

Note: The dispatch is ONE `if/return` at the top — the rest of the function is purely ML code, no interleaved branches.

**Step 3: Run tests**

Run: `cd /home/haex/Projekte/fwbg && .venv/bin/python -m pytest tests/optimization/test_process_fold.py -v`
Expected: ML tests pass. Signal tests may need updating (Task 3).

**Step 4: Commit**

```bash
git add src/fwbg/optimization/signal_fold.py src/fwbg/optimization/process_fold.py
git commit -m "refactor: move signal fold processing to signal_fold.py"
```

---

### Task 3: Update tests for signal fold path

**Files:**
- Modify: `tests/optimization/test_process_fold.py`

**Step 1: Update `TestSignalModelSkipsDropna` to import from signal_fold**

The test `test_signal_model_preserves_all_rows` currently imports `prepare_fold_data` — it should now import `prepare_signal_fold_data` from `signal_fold.py`.

```python
class TestSignalModelSkipsDropna:
    """Signal models must keep all bars (fillna instead of dropna)."""

    def test_signal_model_preserves_all_rows(self):
        """Signal path must not drop rows and must preserve auxiliary columns."""
        from fwbg.optimization.signal_fold import prepare_signal_fold_data
        from fwbg.optimization.robust_validation import WalkForwardFold

        n = 400
        rng = np.random.default_rng(42)
        price = 1.1 + np.cumsum(rng.normal(0, 0.0005, n))

        signal_long = np.full(n, np.nan)
        signal_short = np.full(n, np.nan)
        range_col = np.full(n, np.nan)
        session_mask = rng.random(n) < 0.2
        signal_long[session_mask] = rng.choice([0.0, 1.0], size=session_mask.sum(), p=[0.9, 0.1])
        signal_short[session_mask] = rng.choice([0.0, 1.0], size=session_mask.sum(), p=[0.9, 0.1])
        range_col[session_mask] = rng.uniform(10, 50, size=session_mask.sum())

        df = pd.DataFrame({
            "O": price, "H": price + 0.001, "L": price - 0.001, "C": price,
            "_atr": np.full(n, 0.001),
            "_regime": np.full(n, 7, dtype=np.int8),
            "orb_signal_long": signal_long,
            "orb_signal_short": signal_short,
            "orb_range": range_col,
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        split = int(n * 0.7)
        fold = WalkForwardFold(
            fold_id=0, train_start=0, train_end=split,
            test_start=split, test_end=n,
            train_df=df.iloc[:split].copy(), test_df=df.iloc[split:].copy(),
        )
        ctx = _make_ctx(
            model_type="signal",
            required_features=["orb_signal_long", "orb_signal_short"],
            min_trades=5,
        )

        result = prepare_signal_fold_data(
            fold, fold_indicators=[], precomputed_raw_df=None,
            preprocessing_configs=None, ctx=ctx, sym="TEST",
        )
        assert result is not None
        train_df, test_df, full_pool = result

        # All rows preserved (no dropna)
        assert len(train_df) == split
        assert len(test_df) == n - split

        # Signal columns filled with 0
        assert train_df["orb_signal_long"].isna().sum() == 0
        assert test_df["orb_signal_short"].isna().sum() == 0

        # Feature pool restricted to required_features
        assert set(full_pool) == {"orb_signal_long", "orb_signal_short"}

        # Auxiliary columns preserved (not dropped)
        assert "orb_range" in train_df.columns
        assert "orb_range" in test_df.columns

    def test_xgboost_model_still_drops_na(self):
        """ML path: dropna still applies, unchanged."""
        from fwbg.optimization.process_fold import prepare_fold_data
        # ... (keep existing test body unchanged)
```

**Step 2: Add test for auxiliary column preservation (the actual bug fix)**

```python
    def test_signal_model_preserves_auxiliary_columns_for_exit_strategy(self):
        """Exit strategy auxiliary columns (range, sl_dist) must survive
        even though they have >10% NaN. ML path drops them, signal keeps them."""
        from fwbg.optimization.signal_fold import prepare_signal_fold_data
        from fwbg.optimization.process_fold import prepare_fold_data
        from fwbg.optimization.robust_validation import WalkForwardFold

        n = 400
        rng = np.random.default_rng(42)
        price = 1.1 + np.cumsum(rng.normal(0, 0.0005, n))

        # 80% NaN auxiliary columns (typical for session-based indicators)
        range_col = np.full(n, np.nan)
        sl_dist_col = np.full(n, np.nan)
        session_mask = rng.random(n) < 0.2
        range_col[session_mask] = rng.uniform(10, 50, size=session_mask.sum())
        sl_dist_col[session_mask] = rng.uniform(5, 25, size=session_mask.sum())

        df = pd.DataFrame({
            "O": price, "H": price + 0.001, "L": price - 0.001, "C": price,
            "_atr": np.full(n, 0.001),
            "_regime": np.full(n, 7, dtype=np.int8),
            "orb_signal_long": np.where(session_mask, 1.0, np.nan),
            "orb_range": range_col,
            "orb_sl_dist": sl_dist_col,
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        split = int(n * 0.7)
        fold = WalkForwardFold(
            fold_id=0, train_start=0, train_end=split,
            test_start=split, test_end=n,
            train_df=df.iloc[:split].copy(), test_df=df.iloc[split:].copy(),
        )

        # Signal path: auxiliary columns preserved
        ctx_signal = _make_ctx(
            model_type="signal",
            required_features=["orb_signal_long"],
            min_trades=5,
        )
        result = prepare_signal_fold_data(
            fold, fold_indicators=[], precomputed_raw_df=None,
            preprocessing_configs=None, ctx=ctx_signal, sym="TEST",
        )
        assert result is not None
        train_df, _, _ = result
        assert "orb_range" in train_df.columns
        assert "orb_sl_dist" in train_df.columns

        # ML path: auxiliary columns dropped (>10% NaN)
        ctx_ml = _make_ctx(model_type="xgboost", min_trades=5)
        result_ml = prepare_fold_data(
            _make_fold(df.copy(), fold_id=0), fold_indicators=[],
            precomputed_raw_df=None,
            preprocessing_configs=None, ctx=ctx_ml, sym="TEST",
        )
        assert result_ml is not None
        ml_train, _, _ = result_ml
        assert "orb_range" not in ml_train.columns
        assert "orb_sl_dist" not in ml_train.columns
```

**Step 3: Run tests**

Run: `cd /home/haex/Projekte/fwbg && .venv/bin/python -m pytest tests/optimization/test_process_fold.py -v`
Expected: ALL tests pass.

**Step 4: Commit**

```bash
git add tests/optimization/test_process_fold.py
git commit -m "test: update signal fold tests for refactored module structure"
```

---

### Task 4: Fix unified simulation exit strategy propagation

**Files:**
- Modify: `src/fwbg/optimization/unified_simulation.py`

**Step 1: Propagate exit_strategy/exit_params from unified candidate to holdout context**

The `merge_unified_settings` already extracts `exit_strategy`/`exit_params` from fold configs (added earlier). Now `run_unified_simulation` needs to apply them.

In `run_unified_simulation`, after building `holdout_context`, add:

```python
    # Build holdout context with unified params
    holdout_context = ctx
    if model_hp and model_hp != ctx.model_hyperparameters:
        holdout_context = dataclasses.replace(
            holdout_context, model_hyperparameters=model_hp,
        )
    if exit_mod and exit_mod != ctx.exit_modifier_params:
        holdout_context = dataclasses.replace(
            holdout_context, exit_modifier_params=exit_mod,
        )
    # Propagate exit strategy from unified candidate (critical for non-fixed
    # strategies like orb_based that need specific exit_params for TP/SL).
    exit_strategy_name = unified_candidate.get("exit_strategy")
    exit_params = unified_candidate.get("exit_params")
    if exit_strategy_name and exit_strategy_name != ctx.exit_strategy:
        holdout_context = dataclasses.replace(
            holdout_context,
            exit_strategy=exit_strategy_name,
            exit_params=exit_params or {},
        )
```

**Step 2: Use signal data prep in unified simulation for signal models**

The unified simulation also calls `prepare_fold_data` — for signal models it must use `prepare_signal_fold_data` instead:

```python
    # Import signal data prep conditionally
    if ctx.model_type == "signal":
        from .signal_fold import prepare_signal_fold_data as _prepare_data
    else:
        _prepare_data = prepare_fold_data
```

Wait — the user doesn't want `if model_type` checks. Better approach: pass the prepare function as a parameter from `process.py`.

Update the signature:

```python
def run_unified_simulation(
    wf_folds,
    unified_candidate,
    fold_indicators,
    precomputed_raw_df,
    preprocessing_configs,
    ctx,
    sym,
    prepare_data_fn=None,  # NEW: caller passes the right prepare function
):
```

And in the loop body:

```python
        _prepare = prepare_data_fn or prepare_fold_data
        fold_data = _prepare(
            fold, fold_indicators, precomputed_raw_df,
            preprocessing_configs, ctx, sym,
        )
```

In `process.py`, the caller passes the right function:

```python
        # In process.py, before calling run_unified_simulation:
        if ctx.model_type == "signal":
            from .signal_fold import prepare_signal_fold_data
            prepare_fn = prepare_signal_fold_data
        else:
            prepare_fn = None  # uses default (prepare_fold_data)

        unified_fold_results = run_unified_simulation(
            wf_folds, unified_candidate,
            fold_indicators, precomputed_raw_df, preprocessing_configs,
            ctx, sym, prepare_data_fn=prepare_fn,
        )
```

**Step 3: Run tests**

Run: `cd /home/haex/Projekte/fwbg && .venv/bin/python -m pytest tests/optimization/test_process_fold.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/fwbg/optimization/unified_simulation.py src/fwbg/optimization/process.py
git commit -m "fix: propagate exit strategy through unified simulation"
```

---

### Task 5: Integration test — run orb-asx200

**Step 1: Run the strategy**

```bash
cd /home/haex/Projekte/fwbg
.venv/bin/python -m fwbg.cli optimize --strategy orb-asx200 2>&1 | head -100
```

Expected: Folds should show trades > 0. The range column should be preserved (orb_based exit strategy finds `*_range` column for TP calculation).

**Step 2: Verify no regressions — run full test suite**

```bash
cd /home/haex/Projekte/fwbg
.venv/bin/python -m pytest tests/optimization/ -v
```

Expected: All tests pass.

**Step 3: Final commit**

```bash
git add -A
git commit -m "refactor: cleanly separate signal and ML fold processing"
```
