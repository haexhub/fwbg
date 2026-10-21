# Signal Models: Inner Selection Within Each Fold's Training Window — Implementation Plan

Execution: task by task. Each task gets a review for spec compliance and code quality.

**Goal:** Signal (rule-based) models choose their exit/HP/CT combination per walk-forward fold
on inner validation windows inside that fold's training data — the same split and criterion
family as ML — and evaluate only the chosen combination, once, on the fold's test set.
Indicator-grid variants (signal and ML) are ranked on inner validation PnL instead of test PnL.
Runs record a selection-protocol marker; the research register ranks earlier signal-model
evaluations no longer (they stay counted in N).

**Architecture:**
- New module `src/fwbg/optimization/signal_selection.py` (~150 LoC): `SignalCombo` (frozen
  dataclass: `candidate`, `ctx`, `config`), `signal_combos(ctx)` (moved from
  `signal_fold.py:96-128`, order unchanged), `inner_windows(train_df, ctx)` (`nested_cv_split`
  with the ML arguments, no stateful refit — `ponytail:`), `score_combo(combo, windows)`,
  `select_combo(combos, windows)`, `SELECTION_PROTOCOL = 2`.
- `signal_fold.process_signal_fold`: windows → select → no pick: `None` (test fold untouched) →
  one `evaluate_on_holdout(test_df, train_df, …)` for the chosen combo → fold result.
- `process.py`: variant score (`:316-324`) and the marker in `_build_walk_forward_summary`.
- `research/legacy.py` / `research/stats.py`: marker extraction and ranking.
- Untouched: `unified_simulation.py`, `holdout.py`.

## Findings that shaped the plan (verified on 4e444bb)

1. Signal models evaluate every combo on the fold's TEST set and pick the best by test PnL
   (`signal_fold.py:132-154`); `inner_val_pnl` is set to that test PnL (`:166`), so bias ratios
   are always 1.0 and `check_asset_bias` never fires; grid entries lack `inner_val_pnl`, so PBO
   is silently n/a for signals and the `no_successful_folds` diagnostic is blind.
2. Indicator-grid variants are ranked by mean test PnL (`process.py:316-324, 351-363`) — for
   signal and ML models.
3. `fold_stability` for signals is the share of folds whose best-of-grid TEST PnL is positive —
   the register's `min_fold_stability` criterion checks exactly this number.
4. Stateful indicators: `refit_stateful_features` `dropna()`s inner frames and does not recompute
   `_composed_signal_*`; not reusable for signals. Only `fractional_diff` is stateful today.
5. Inner windows on low timeframes are small (4000-bar cap per window); selection is honest but
   noisy; early folds may have no eligible combo → no fold result (fail closed).
6. Runtime of the fold stage grows ~3-5× (C × K inner evaluations + 1 test evaluation).
7. `grid_results.json` changes schema for signal runs (inner metrics instead of per-combo test
   metrics); no consumer found in this repo.

## Operator decisions (2026-10-21, binding)

- **D1 Criterion:** mean inner validation PnL over all windows; a window without trades counts as
  0; eligible = at least one pooled inner trade; no sign gate (a negative best is chosen and
  reported honestly); ties keep config order (strictly greater wins).
- **D2 Chosen combo with 0 test trades:** the fold is kept (`test_pnl` 0, `test_trades` 0) —
  dropping it would let the test set filter folds.
- **D3 Geometry:** exactly the ML call (`validation.n_inner_folds`, `validation.embargo_bars`,
  built-in cap); no new config keys (`n_inner_folds: 1` gives a single validation tail);
  no stateful refit for signals (`ponytail:`).
- **D4 Indicator variants:** ranked by mean `inner_val_pnl` in this step, for signal AND ML models.
- **D5 Register:** runs record `walk_forward.selection_protocol = 2`; the ledger copies it per
  symbol. Earlier SIGNAL-model development evaluations (no marker, `model.type == "signal"` in the
  stored config `configs/<fingerprint>.json`) stay counted in N but are excluded from
  `best_result`, `return_samples` and `deflated_best`; the verdict and `show` state how many were
  left out. Earlier ML evaluations and evaluations without a stored config stay rankable.
  Hand-curated constant, no data-derived threshold.
- **D6 Unified-merge leak** (settings merged from later folds re-simulate earlier test folds):
  separate next step (causal unified simulation, `selection_protocol` 3). Documented as a known
  limitation meanwhile.

## Rules for every task

- Work only in the step's worktree (branch `fix/signal-inner-selection`). Never touch the main
  checkout or `~/fwbg`. No bare `git stash`.
- TDD: failing test first, then the code. Tests deterministic on small synthetic frames (fixed
  seed; exit `fixed`, `spread=1.0`, `min_trades=2`, `n_inner_folds=2`, `embargo_bars=0` where
  practical).
- 500-LoC limit for new modules; `process.py` grows by a few lines only. `ponytail:` comments for
  deliberate shortcuts. Imports at the top.
- Do not black-reformat `process.py`, `cli/main.py`, `core/config.py`, `results/storage.py`; black
  only new modules and `src/fwbg/research/`.
- Commits: short English messages, no AI/model references or trailers. Stage only touched files.
- The local venv's editable fwbg-premium points at the main checkout: run tests with
  `PYTHONPATH=packages/fwbg-premium/src:src`.
- Before reporting a task:
  ```
  ../../.venv/bin/black src/fwbg/optimization/signal_selection.py <touched research files>
  ../../.venv/bin/ruff check src/ packages/
  ../../.venv/bin/mypy --follow-imports=silent src/fwbg/research
  PYTHONPATH=packages/fwbg-premium/src:src ../../.venv/bin/python -m pytest -p no:cacheprovider tests/optimization tests/strategy tests/research -q
  ```

**Fixtures to reuse:** `tests/strategy/test_grid_search_signal_model.py` (`_ctx`,
`_make_exit_strategies`, `_make_multi_day_bull_data`, `_compute_indicator`, `_train_signal_model`);
`tests/optimization/test_process_fold.py` (`_make_fold`, `_make_ctx` pattern);
`tests/optimization/symbol_harness.py` (`run_inline`, `patch_development`, `fold_result`);
`tests/research/factories.py` (`make_trial`, `make_run`), `tests/research/conftest.py`
(`gate_register`); PBO tests `tests/optimization/test_overfitting.py`.

---

### Task 1: Extract combo enumeration (pure refactor)

Files: create `src/fwbg/optimization/signal_selection.py`; modify `signal_fold.py`; test
`tests/optimization/test_signal_selection.py`.

Tests first:
- `test_combos_cover_exits_x_hp_x_ct_in_config_order`: 2 exits (one with `ct=[0.5, 0.6]`),
  `grid_model_hyperparameters=[None, {"signal_column_long": "x"}]` → 6 combos in order,
  `candidate["params"] == (tp, sl, ct)`, merged `model_hyperparameters`, `combo.ctx` carries
  `exit_strategy`, `exit_params`, modifiers, `separate_long_short`.
- Characterisation (green on the current code before the move): `process_signal_fold` grid
  entries keep the same `(tp_mult, sl_mult, ct)` sequence.

Implementation: move the loop body into `signal_combos`; `process_signal_fold` iterates over it,
still selecting on test (behaviour identical).

Commit: `refactor(signal): enumerate signal combos in their own module`

### Task 2: Inner windows, scoring and selection

Files: `signal_selection.py`, `tests/optimization/test_signal_selection.py`.

Tests first (monkeypatch `signal_selection.evaluate_on_holdout` with canned results keyed by
combo/window):
- `test_windows_match_the_ml_split` (same frames as
  `nested_cv_split(train, 0.0, K, embargo)["inner_folds"]`; every window ⊆ train).
- `test_picks_highest_mean_inner_pnl`; `test_window_without_trades_counts_as_zero`
  ([+3, 0-trade] vs [+2, +2] → the latter); `test_combo_without_inner_trades_is_ineligible`;
  `test_no_eligible_combo_returns_none`; `test_tie_keeps_config_order`.
- `test_grid_entries_carry_inner_metrics`: `exit_strategy`, `tp_mult`, `sl_mult`, `ct`,
  `timeout_bars`, `model_hyperparameters`, `inner_val_pnl`, `inner_trades`, `fold_pnls`,
  `fold_stability`; no per-combo test `pnl`.
- `test_scoring_only_sees_train_rows` (spy on frame indices).

Implementation: `inner_windows` = `nested_cv_split(train_df, holdout_ratio=0.0,
n_inner_folds=ctx.n_inner_folds, embargo_bars=ctx.embargo_bars)["inner_folds"]` (`ponytail:` no
stateful refit); `score_combo` = one `evaluate_on_holdout(val, inner_train, combo.candidate,
combo.ctx)` per window (`ponytail:` on the cost); `select_combo` strictly greater.

Commit: `feat(signal): score signal combos on inner windows of the training data`

### Task 3: Wire selection into `process_signal_fold` (the fix), with leakage tests

Files: `signal_fold.py` (logic + module docstring), new `tests/optimization/test_signal_fold.py`.

Synthetic data: hourly bars, `_composed_signal_long = 1` every 24th bar. In train each signal is
followed by +3 then −5 (TP 2/SL 4 wins, TP 6 loses); in test by +8 (TP 6 wins). Exits `fixed`
`{tp_mult: 2, sl_mult: 4}` and `{tp_mult: 6, sl_mult: 4}`, `spread 1.0`; each test asserts these
preconditions with `evaluate_on_holdout`.

Tests first:
- `test_choice_ignores_the_test_fold`: same train, two different test frames → identical
  `best_config` and `inner_val_pnl`.
- `test_train_best_combo_is_chosen_even_if_another_wins_on_test` → TP 2 chosen;
  `test_pnl == evaluate_on_holdout(test, train, tp2_candidate, ctx)["pnl"]`.
- `test_test_fold_is_evaluated_once` (spies on both modules' `evaluate_on_holdout`): exactly one
  call touches a test index, for the chosen combo.
- `test_no_inner_trades_returns_no_fold_and_never_touches_test`.
- `test_chosen_combo_without_test_trades_keeps_the_fold` (D2).
- `test_fold_result_reports_inner_metrics`: `inner_val_pnl` = chosen inner mean (not test PnL);
  `inner_fold_stability`, `n_candidates` present.

Also run `tests/optimization/test_orb_column_ambiguity.py`, `tests/test_hash_seed_determinism.py`,
`tests/strategy` (premium path). If an ORB fixture loses all folds, enlarge the fixture's frame —
do not loosen the rule.

Commit: `fix(signal): select signal combos on the fold's training window, not its test set`

### Task 4: Downstream consumers and indicator-variant ranking (D4)

Files: `process.py` (`:316-324` + log at `:599-604`), `tests/optimization/test_overfitting.py`,
new `tests/optimization/test_indicator_variant_score.py`.

Tests first: `test_performance_matrix_reads_signal_grid_entries` (Task 2 entry shape over 2 folds
→ non-empty PBO matrix); `test_variants_rank_by_inner_validation_pnl` (variant A inner 10/test 1
vs B inner 1/test 10 → A); `test_failed_variant_still_ranks_last`.

Implementation: `mean_pnl = mean(r["inner_val_pnl"] for r in fold results)`; log label
`mean_inner_pnl`. Applies to ML too.

Commit: `fix(optimization): rank indicator variants by inner validation PnL`

### Task 5: Selection-protocol marker in results and ledger

Files: `signal_selection.py` (constant; or a neutral module if ML imports it), `process.py`
(`_build_walk_forward_summary`, 1 line), `src/fwbg/research/legacy.py` (`_symbol_metrics`).

Tests first: harness run → `fold_results.json` `walk_forward.selection_protocol == 2`;
`run_outcome` metrics carry `selection_protocol` (2, or None for old runs); a holdout's
`development.walk_forward` keeps it. The register reads JSON only (no optimizer import).

Commit: `feat(results): record the selection protocol per symbol`

### Task 6: Register ranks no earlier signal-model evaluations (D5)

Files: `src/fwbg/research/stats.py`, `commands.py` (verdict line), `views.py` (show), tests.

Tests first: unmarked signal rows (stored config `strategy.model.type == "signal"`) count in N
but are excluded from `best_result`, `return_samples`, `deflated_best`; unmarked ML rows and rows
without a stored config stay rankable; marked rows always rankable; the verdict prints
"k signal-model evaluations predate selection protocol 2 (counted in N, not ranked)"; a card with
only such rows → no best line and "too few trials with return statistics".

Implementation: `CURRENT_PROTOCOL = 2` (stdlib-only, hand-curated); a predicate reading the
model type from `configs/<fingerprint>.json` (cache per call); filter inside `best_result` and
`return_samples`; `evaluations` unchanged.

Commit: `feat(research): stop ranking signal evaluations from before inner selection`

### Task 7: Docs and end-to-end check

Files: `docs/plans/2026-10-19-research-register-design.md` (Known limitations: replace the signal
item with the unified-merge item; verdict statistics; a short "Selection protocol" note), the
`signal_fold.py` docstring.

Verification: full suite (premium path); time the hash-seed ORB backtest before/after for a rough
runtime factor. No gated real backtests (the operator decides on reruns).

Commit: `docs: signal models select on inner windows`

**Follow-up step (D6):** causal unified simulation, `SELECTION_PROTOCOL` 3.
