# Causal Unified Replay (Selection Protocol 3) — Implementation Plan

Execution: task by task. Each task gets a review for spec compliance and code quality.

**Goal:** The reported development metrics of a walk-forward run no longer depend on the test
folds through the unified merge or through ML fold filtering. Test fold k is replayed with the
unified setting merged from the successful folds 0..k only; ML folds are no longer dropped for
having few test trades. Runs record `selection_protocol` 3; the research register ranks no
optimizer evaluation below protocol 3 (they stay counted in N).

**Architecture:**
- `src/fwbg/optimization/unified_simulation.py`: the HP-consistency rule moves here from
  `process.py` (`hp_key`, `consistent_folds`); new `causal_candidates(wf_folds, fold_results)`;
  `run_unified_simulation` takes one candidate per fold (`None` = fold not traded).
- `process.py`: full merge stays the frozen/deployment setting (`unified_candidate`,
  `development.json`, `best_config`, holdout); the replay behind the reported metrics uses the
  causal candidates.
- `holdout.py`: passes `[development["unified_candidate"]]` (behaviour unchanged).
- `process_fold.py`: the `< 5` test-trade return goes.
- `selection_protocol.py`: `SELECTION_PROTOCOL = 3`.
- `research/stats.py`: `RANKED_PROTOCOL = 3` (renamed from `INNER_SELECTION_PROTOCOL`), new
  reason `UNIFIED_MERGE`.
- Untouched: `merge_unified_settings`, `causal_reporting.py`, the per-fold selection.

## Findings that shaped the plan (verified on f51aaea)

1. `merge_unified_settings(consistent_folds, all_fold_results)` (`unified_simulation.py:14`)
   merges all folds; `run_unified_simulation` replays every fold with it
   (`unified_simulation.py:122-226`). The reported trades, WR, Sharpe, DSR inputs and causal
   sizing come from that replay (`process.py:783-900`). Fold j > i trains on fold i's test
   period and chooses its setting there, so fold i's replay depends on its own test data.
2. The HP-consistency majority vote (`process.py:715-748`) runs over all folds, so it leaks the
   same way.
3. `process_fold.py:522-524` drops an ML fold whose chosen setting makes < 5 test trades: the
   test set filters which settings enter the merge and which folds enter PnL, bias and PBO.
   Signal folds were kept since protocol 2.
4. `compute_causal_reporting` (`causal_reporting.py:122`) treats leading folds without trades
   as having no prior data; an untraded fold (empty result) needs no change there.
5. The register still ranks unmarked optimizer evaluations of ML models without a multi-variant
   grid. Stored optimizer configs carry a `strategy.model` mapping. Backfilled study evaluations
   have no stored config and stay ranked.

## Decisions (operator, 2026-10-22)

- D1 prefix merge: fold k uses `merge_unified_settings` over the successful folds with
  `fold_id <= k`, with the HP-consistency rule applied to that prefix. Each of those folds chose
  its setting on its own training data (protocol 2), which ends before fold k's test data.
- D2 no setting yet (no successful fold up to k): the fold is not traded, gets an empty result
  (0 trades, PnL 0) and still counts as a fold and in the test period. A later failed fold uses
  the merge of the earlier ones (the deployed setting stays).
- D3 the full merge over all folds stays the frozen setting (development.json, holdout,
  `best_config`, deployment): it is what one would deploy after the last fold.
- D4 ML folds with few test trades are kept, like signal folds; the `min_trades` warning stays.
- D5 register: every optimizer evaluation (stored config with a `strategy.model` mapping)
  without `selection_protocol >= 3` counts in N but is not ranked; reasons signal-model,
  indicator-grid, unified-merge. A protocol-2 row is reported as unified-merge. A study script
  that passes a config with a `strategy.model` mapping is treated as an optimizer row
  (conservative; documented).

---

### Task 1: Causal candidates in `unified_simulation.py`

**Files:**
- Modify: `src/fwbg/optimization/unified_simulation.py`
- Modify: `src/fwbg/optimization/holdout.py:153-161` (pass `[development["unified_candidate"]]`)
- Test: `tests/optimization/test_unified_simulation.py`

**Step 1: failing tests** (append to `test_unified_simulation.py`, reuse `_make_fold_result`):

- `consistent_folds`:
  - all folds share HP → returns all, in order;
  - HP {a:1} ×2, {a:2} ×1 → returns the two {a:1} folds;
  - tie → the key seen first wins (`max` over insertion-ordered counts, as today);
  - `hp_key` ignores `None` values and turns lists into tuples (`{"a": [1, 2], "b": None}` →
    `(("a", (1, 2)),)`); empty/None HP → `()`.
- `causal_candidates(wf_folds, fold_results)` with `wf_folds` as
  `types.SimpleNamespace(fold_id=i)` for i in 0..3:
  - fold results for ids 0,1,2,3 with TP 1,3,5,7 → candidate TP per fold 1, 2, 3, 4 (medians
    of the prefix);
  - fold results only for ids 1 and 3 → `[None, c1, c1, c13]` (fold 0 has no setting; fold 2
    uses folds ≤ 2 = only fold 1);
  - feature stability is computed on the prefix: feature `f` in fold 0 only → in fold 0's and
    fold 1's candidate (1/1, 1/2 ≥ 0.5), not in fold 2's (1/3);
  - HP majority on the prefix: HP {a:1} in fold 0, {a:2} in folds 1 and 2 → fold 0 candidate
    has {a:1}; fold 2 candidate has {a:2};
  - the last candidate equals `merge_unified_settings(consistent_folds(all), all)`.
- `run_unified_simulation` with a `None` candidate: monkeypatch nothing heavy — pass
  `prepare_data_fn` that fails the test if called for that fold; the returned list has an entry
  for that fold `{"trades": [], "trades_detailed": [], "pnl": 0.0, "win_rate": 0.0,
  "n_trades": 0, "dropped_features": []}`. With a candidate list whose length differs from
  `wf_folds` → `ValueError`.
- `run_unified_simulation` with two different candidates: patch
  `fwbg.optimization.unified_simulation.evaluate_on_holdout` to record `(candidate, ctx)` and
  return a minimal result; assert fold 0 got candidate 0 and fold 1 candidate 1, and that a
  candidate's `model_hyperparameters` / `exit_modifier_params` / `exit_strategy` reach the ctx
  of its own fold only.

Run: `PYTHONPATH=packages/fwbg-premium/src:src ../../.venv/bin/python -m pytest -p no:cacheprovider -q tests/optimization/test_unified_simulation.py`
Expected: FAIL (names not defined).

**Step 2: implementation**

```python
def hp_key(hp: Optional[Dict[str, Any]]) -> tuple:
    """Hashable key for model_hyperparameters (ignoring None values)."""
    # body moved verbatim from process.py's _hp_key


def consistent_folds(fold_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The fold results that share the majority model_hyperparameters.

    For rule-based signal strategies the HP define the strategy; folds that disagree
    are left out of the merge. Ties go to the HP seen first.
    """


def causal_candidates(wf_folds, fold_results) -> List[Optional[Dict[str, Any]]]:
    """Per walk-forward fold, the unified setting known before its test period.

    Fold k gets merge_unified_settings over the successful folds with fold_id <= k: each
    chose its setting on its own training data, which ends before fold k's test data.
    None while no fold up to k has succeeded (that fold is not traded).
    """
```

`run_unified_simulation(wf_folds, candidates, fold_indicators, precomputed_raw_df,
preprocessing_configs, ctx, sym, prepare_data_fn=None)`:
- `len(candidates) != len(wf_folds)` → `ValueError`.
- Per fold: `None` → log level 2 `"  Unified fold k: not traded (no setting before this fold)"`,
  append the empty result above, continue (no data preparation).
- Otherwise build the fold's context with a private helper `_candidate_context(ctx, candidate)`
  (the existing model_hp / exit_mod / exit_strategy replacement logic, moved unchanged) and
  log the fold's settings at level 2 (TP/SL/CT/timeout, number of features).
- Keep the existing per-fold feature filtering, `dropped_features`, totals and logs. The
  header log at level 1 stays `"=== Unified Simulation ==="`; drop the single-setting lines
  that no longer apply (they move to process.py for the frozen setting, Task 2).
- Update the module docstring: replay is causal; the frozen setting is the full merge.

`holdout.py`: pass `[development["unified_candidate"]]`; adjust the ponytail comment only if it
becomes wrong.

**Step 3:** tests pass, plus `tests/optimization/test_holdout.py` and
`tests/optimization/test_holdout_files.py`.

**Step 4: commit** `feat(optimization): replay each fold with the setting known before it`

---

### Task 2: `process.py`, ML fold filter, protocol 3

**Files:**
- Modify: `src/fwbg/optimization/process.py:705-780` (HP block, candidates, replay call, logs)
- Modify: `src/fwbg/optimization/process_fold.py:522-524`
- Modify: `src/fwbg/optimization/selection_protocol.py`
- Test: `tests/optimization/test_process_fold.py`, `tests/optimization/test_symbol_metrics.py`,
  `tests/optimization/test_holdout_files.py`, a new test in
  `tests/optimization/test_fold_aggregation.py` or `test_symbol_metrics.py` (whichever already
  drives `process_symbol` through `symbol_harness`)

**Step 1: failing tests**
- `test_process_fold.py`: copy `test_fold_result_stores_winning_model_hyperparameters`'s setup
  with `evaluate_on_holdout` returning `n_trades: 2` (and `win_rate 0.5, pnl -3.0`) → the fold
  result is not `None` and `test_trades == 2`. Same with `n_trades: 0`.
- Marker tests asserting `SELECTION_PROTOCOL == 2` now assert `== 3`.
- Through `symbol_harness`: monkeypatch `unified_simulation.run_unified_simulation` (the harness
  already does) with a recorder; assert it received
  `causal_candidates(wf_folds, all_fold_results)` (compare TP per fold), and that
  `development.json`'s `unified_candidate` is still the full merge. If the harness makes all
  folds identical, give the recorder test fold results with different TP (extend the harness
  minimally, e.g. a `fold_tp` parameter; keep the harness deterministic).

**Step 2: implementation**
- Replace `_hp_key` and the inline majority logic with `hp_key` / `consistent_folds` from
  `unified_simulation.py`; keep the existing per-fold ✓/✗ and "Model HP inconsistent" logs
  (computed with `hp_key`).
- `unified_candidate = merge_unified_settings(consistent, all_fold_results)` stays (frozen).
  Log it at level 1 as the frozen setting (`"  Frozen setting: TP=… SL=… CT=… timeout=…"`,
  plus features count), replacing what `run_unified_simulation` logged before.
- `replay_candidates = causal_candidates(wf_folds, all_fold_results)`; pass it to
  `run_unified_simulation`. Do not name it `causal` (that name is taken by the causal reporting
  below).
- Update the "UNIFIED SIMULATION" comment: causal replay, frozen setting.
- `process_fold.py`: delete the `< 5` return; comment like `signal_fold.py:108` that a fold is
  kept whatever its test trade count (dropping it would let the test set filter folds).
- `selection_protocol.py`: `SELECTION_PROTOCOL = 3`; comment: protocol 3 replays each test fold
  with the setting merged from folds up to it and keeps ML folds regardless of test trades; no
  known test-set influence on the reported development metrics remains. Protocol 2 described in
  one line.

**Step 3:** `tests/optimization` green.

**Step 4: commit** `fix(optimization): keep ML folds with few test trades; protocol 3`

---

### Task 3: Register ranks only protocol 3

**Files:**
- Modify: `src/fwbg/research/stats.py`
- Modify: `docs/plans/2026-10-19-research-register-design.md` (Known limitations: remove the two
  items; Selection protocol: protocol 3 paragraph and Register paragraph)
- Test: `tests/research/test_ranking.py`

**Step 1: failing tests**
- `RANKED_PROTOCOL == 3` (replaces the `INNER_SELECTION_PROTOCOL == 2` test).
- `unrankable_configs`: an ML config without grid → `UNIFIED_MERGE`; signal → `SIGNAL_MODEL`;
  ML with a multi-variant grid → `INDICATOR_GRID`; a config without a `strategy.model` mapping
  (e.g. `{"strategy": {"timeframe": "HOUR_1"}, "symbols": ["A"]}`) → not in the map; unstored →
  not in the map.
- Ranking: an ML row marked 2 and an unmarked ML row are not ranked; marked 3 is ranked. A
  signal row marked 2 is not ranked and is counted as unified-merge in
  `unranked_evaluations`.
- `unranked_note`: `"… pre-protocol-3 evaluations not ranked (counted in N): 1 signal-model,
  1 indicator-grid, 2 unified-merge"` (order signal-model, indicator-grid, unified-merge).
- Existing tests that relied on the ML row being ranked (best result, return samples, DSR,
  `list_lines`, `show_text`, `card_summary`, verdict) mark that row with
  `selection_protocol: 3`; update the expected texts accordingly.

**Step 2: implementation**
- Rename `INNER_SELECTION_PROTOCOL` → `RANKED_PROTOCOL = 3`; rewrite its comment (hand-curated;
  protocol 2 inner selection, protocol 3 causal replay; rows below count in N, never ranked).
- `UNIFIED_MERGE = "unified-merge"`.
- `unrankable_configs`: after the signal / grid checks,
  `elif isinstance(strategy.get("model"), dict): found[fingerprint] = UNIFIED_MERGE`. Docstring:
  every optimizer config (one with a `strategy.model` mapping); a study script passing such a
  config counts as one.
- `_ranked`: `protocol >= RANKED_PROTOCOL`.
- `unranked_evaluations`: reason `UNIFIED_MERGE` when the row is marked with a protocol ≥ 2
  (inner selection was already in place), else `unrankable[fingerprint]`.
- `unranked_note`: reasons in the order above; noun uses `RANKED_PROTOCOL`.
- Grep the repo for `INNER_SELECTION_PROTOCOL` and update every reference.
- Design doc: no new numbers from the private register.

**Step 3:** `tests/research` green.

**Step 4: commit** `feat(research): rank only protocol-3 optimizer evaluations`

---

### Task 4: Causal indicator-variant choice in the replay (operator decision 2026-10-22)

**Finding (final review):** with a multi-variant `indicator_grid`, `_run_indicator_variants`
(`process.py:345`) keeps the variant with the best `(n_successful, mean inner_val_pnl)` over ALL
folds (`process.py:324-333`). Folds are anchored, so fold j's inner windows contain the test
periods of folds k < j: fold k's replay used a variant chosen partly on its own test data.

**Decision D6:** replay fold k with the variant whose score over the folds with `fold_id <= k`
is best (same score tuple, same tie rule: the earlier variant wins), and with that variant's
causal candidate for fold k. The frozen variant (development.json, holdout, deployment,
per-fold diagnostics: fold_stability, bias ratios, PBO, grid results) stays the best over all
folds; at the last fold both coincide.

**Files:**
- Create: `src/fwbg/optimization/causal_variants.py` (≤ 150 LoC)
- Modify: `src/fwbg/optimization/process.py` (`_process_single_variant` score, `_run_indicator_variants`
  return value, replay call)
- Modify: `src/fwbg/optimization/unified_simulation.py` (each result carries `fold_id`)
- Test: `tests/optimization/test_causal_variants.py`, `tests/optimization/test_indicator_variant_score.py`,
  `tests/optimization/test_unified_simulation.py`

**Design:**
- `variant_score(fold_results) -> (n_successful, mean inner_val_pnl or -inf)`: the score
  `_process_single_variant` computes today, moved here and used by both.
- `prefix_variant_choice(wf_folds, variant_fold_results) -> list[int]`: per walk-forward fold k,
  the index of the variant with the best `variant_score` over its results with
  `fold_id <= k`; ties → lowest index (matches the strict `>` in `_run_indicator_variants`).
- `_run_indicator_variants` additionally returns, for multi-variant grids, a light record per
  variant (label, strategy, ctx, preprocessing_configs, all_fold_results sorted by fold_id) — no
  heavy data. Single variant: unchanged.
- Replay (`process.py`): one variant → as now. Several → `choice = prefix_variant_choice(...)`;
  for each distinct chosen variant (in index order) take its causal candidates
  (`causal_candidates(wf_folds, its fold results)`), keep only the folds assigned to it (other
  folds get no call), prepare its indicators (`precompute_indicators`; reuse the frozen variant's
  data when it is the frozen one), call `run_unified_simulation` on that fold subset, release
  the data. Merge all results and sort by `fold_id`. `ponytail:` comment: non-frozen variants
  recompute their indicators once for the replay (memory over CPU, as the variant loop does).
- `run_unified_simulation` tags every result (untraded included) with the fold's `fold_id`.
- Log at level 1 which variant replays which folds when they differ from the frozen variant.
- The heavy-data `del` after the replay still runs.

**Tests (TDD):**
- `variant_score`: none successful → `(0, -inf)`; mean of inner_val_pnl otherwise.
- `prefix_variant_choice`: variant A best on folds 0-1, B best overall → `[A, A, …, B]` at the
  folds where B's prefix score overtakes; tie → lower index; a variant with more successful folds
  in the prefix wins over a better mean (tuple order); fold results given unsorted give the same
  result.
- Replay through the process path with two fake variants (patch `_process_single_variant` /
  `precompute_indicators` / `run_unified_simulation` as the existing variant-score tests do):
  each fold is simulated exactly once, with its chosen variant's ctx and candidate; results come
  back in fold order; the frozen variant's data is not recomputed; development.json still holds
  the frozen variant's full merge.
- `run_unified_simulation` results carry `fold_id`.

**Register:** protocol 3 then covers grid configs too; `INDICATOR_GRID` stays only a reason for
rows below protocol 3. Update the design doc (Selection protocol: protocol 3 also chooses the
variant per prefix; per-fold diagnostics remain those of the frozen variant).

**Commit:** `fix(optimization): choose the replay variant on folds up to each test fold`

---

### Task 5: Review follow-ups

- `process.py:~942-943` stale comment "re-simulates every fold with the merged config" → replays
  every fold with its causal candidate; untraded leading folds still count in the test period.
- `stats.py` and design doc: a protocol-2 row was kept out by the merge AND the ML test-trade
  filter (label stays unified-merge).
- Design doc limitations: (a) if fold 0 fails, the bars of fold 1 still count in the annualised
  test period although its trades are excluded from causal reporting (conservative: annualised
  Sharpe and return read low; per-trade SR and DSR unaffected); (b) an unmarked run of a strategy
  file without `model` is stored without a `strategy.model` mapping and stays ranked (workspace
  configs set `model`).
- `merge_unified_settings` parameter `consistent_folds` → `majority_folds` (shadows the function).
- `commands.py:262`, `views.py:155` docstrings: "the fingerprints whose runs below
  RANKED_PROTOCOL are not ranked".
- `selection_protocol.py` / `stats.py` comments name only the remaining hindsight (no_edge,
  sizing fallback) once Task 4 lands.

**Commit:** `chore: protocol 3 review follow-ups`

---

### Final

- Whole-branch review.
- Full suite:
  `PYTHONPATH=packages/fwbg-premium/src:src:packages/fwbg-mcp/src ../../.venv/bin/python -m pytest -p no:cacheprovider -q`
  (known env failure: `tests/test_broker_ig_contract.py::test_package_import_path_is_the_core_implementation`).
- Read-only check against a scratch copy of the register: number of ranked optimizer rows is 0,
  study rows unchanged. Never write under `~/fwbg`.
