# Research Register

## Status: Design approved (2026-10-19) — steps 1-5 (core, backfill, CLI gate, API, holdout + DSR) implemented

## Problem

Strategy research in fwbg has no defined workflow for keeping what was
tested. Today:

- Results live in `$FWBG_WORKSPACE/test_results/<run_id>/`, partly
  duplicated into an untracked `fwbg/test_results/` in the repo.
- Run IDs are either bare timestamps or ad-hoc names; nothing links a run
  to the idea it was testing.
- Conclusions ("why did this fail?") are scattered across plan docs,
  commit messages and agent memory. Script-based studies (`scripts/…`)
  bypass the run system entirely and use their own output formats.
- Most runs record `git.dirty: true`, so the stored commit hash does not
  reproduce the code that produced them.
- Nobody knows how many configurations were tried. Without that number,
  any "best" result cannot be corrected for multiple testing — after a
  hundred-plus trials, a p-value around 0.03 is roughly what noise
  produces.

Runs have no holdout of their own. The nested CV inside each walk-forward
fold is called with `holdout_ratio=0.0`
(`src/fwbg/optimization/process_fold.py:418`); the strategy's
`validation.holdout_ratio` is never read (nor are `validation.method` and
`validation.oos_size`). A tail reserved by hand with `end_date` stops being a
holdout once many iterations of one strategy are tuned while looking at it.
The register's holdout look (step 5, see Holdout look) replaces both.

## Goals

1. A strategy idea is registered once, with its success criteria frozen
   *before* the first iteration (pre-registration).
2. A strategy is iterated many times with different configurations. A bad
   configuration never rejects the strategy; only a deliberate verdict
   does.
3. An identical configuration is never re-run by accident.
4. For every strategy one can see which parameters, assets, timeframes and
   exits were already explored.
5. Every evaluation is counted, so verdicts can correct for the number of
   trials (Deflated Sharpe Ratio).
6. Each strategy has a frozen final holdout that iterations cannot see.
7. The same register serves three actors: the operator (CLI), AI coding
   sessions (files + CLI), and fwbg-agents (HTTP API).

## Non-goals

- No automatic rejection of strategies.
- No similarity search beyond the per-strategy coverage view.
- No database; no dashboard page in phase 1.
- No hashing of market data snapshots (data identity is out of scope).

## Concepts

| Term | Meaning |
|---|---|
| Research strategy | Long-lived idea with hypothesis, mechanism, frozen success criteria, holdout cutoff and status. Identified as `S-NNNN`. Named "research" to avoid clashing with fwbg's strategy config JSON and fwbg-agents' `Strategy` row. |
| Trial | One evaluation recorded in the ledger. A normal backtest run is an `iteration`; the final look at the holdout is `holdout`. |
| Fingerprint | SHA-256 over the canonical, resolved run configuration. Used to block identical re-runs. |
| Holdout cutoff | Date fixed at registration and frozen once trials ran. Iterations only see data before it. |
| Verdict | Deliberate status change with reasoning, auto-computed statistics and a reopen condition. |

Status values: `active`, `parked`, `rejected`, `candidate`.
`active → parked | rejected | candidate` via verdict;
`parked | rejected → active` via reopen (requires a reason).

## Storage

```
$FWBG_RESEARCH_DIR/                 # default: $FWBG_WORKSPACE/research
  .git/                             # private git repo, separate from fwbg
  strategies/S-0007-<slug>.md       # one card per research strategy
  trials.jsonl                      # append-only ledger
  configs/<fingerprint>.json        # content-addressed canonical configs
  templates/criteria-default.yaml   # hand-curated conservative defaults
  reports/                          # study write-ups linked from cards
$FWBG_WORKSPACE/test_results/<run_id>/   # run artefacts stay here, untracked
```

Why a separate private repo in the workspace:

- `haexhub/fwbg` is public; hypotheses, criteria and results should not be.
- The API container cannot write into the source repo; it reaches the
  register through a bind mount (see Container).
- Research results are data, not code.

`configs/` keeps the coverage view and reproducibility intact even when
`test_results/` is pruned.

### Card format

Markdown with YAML front matter:

```yaml
id: S-0007
title: CUSUM event continuation
status: active
tags: [event, trend-continuation, daily]
created: 2026-10-19
holdout_cutoff: 2025-04-30
holdout_contaminated: false
success_criteria:            # hand-curated, from templates/criteria-default.yaml
  min_sharpe: …
  min_trades: …
  max_drawdown: …
registered_criteria_hash: sha256:…  # written by the tool at registration
criteria_amendments:         # written by `fwbg research amend`
  - date: 2026-11-02
    reason: …
    criteria_hash: sha256:…  # hash of the amended success_criteria
```

The criteria hash is computed from `success_criteria`; the tool writes the
registration value to `registered_criteria_hash` and never accepts a
hand-edited value. Only the five known criteria are accepted
(`min_sharpe`, `min_trades`, `max_drawdown`, `min_fold_stability`,
`max_p_value`); an unknown name makes the card unreadable. At registration,
that persisted hash is the frozen
baseline for the first trial; it does not need a separate
`criteria_amendments` entry. A later criteria change is valid only when
`fwbg research amend` records the new hash in `criteria_amendments` together
with its reason. Body sections:
`## Hypothesis`, `## Mechanism` (why should this work?), `## History`
(reopenings and other notes in prose), `## Verdict`.

Default `holdout_cutoff` is 18 months before registration, overridable.
Default criteria come from the template; values are hand-curated, never
derived from past results.

### Ledger format

One JSON object per line:

```json
{"trial": "T-000231", "strategy": "S-0007", "run_id": "…",
 "kind": "iteration", "status": "ok", "ts": "2026-10-19T10:12:00",
 "fingerprint": "sha256:…", "fingerprint_version": 1,
 "criteria_hash": "sha256:…", "holdout_cutoff": "2025-04-30",
 "code": {"commit": "…", "dirty": true, "patch": "code.patch"},
 "window": {"start": null, "end": "2025-04-30"},
 "note": "threshold 1.5 → 2.0", "reason": null,
 "metrics": {"<SYMBOL>": {"sharpe": …, "trades": …, "max_dd": …, "p_value": …}}}
```

- Every key shown is required on a stored line (`reason` may be `null`,
  `metrics` may be `{}`); unknown keys are refused.
- `holdout_cutoff`: the card's cutoff at the time of the trial (also for
  legacy and re-indexed trials); the gate freezes the cutoff against it.
- `kind`: `iteration | holdout | legacy | smoke`.
- `status`: `ok | failed`. Failed and smoke trials are recorded but not
  counted towards N.
- Metrics are stored **per symbol**. A run over seven assets is seven
  evaluations; N for the Deflated Sharpe Ratio counts (run, symbol) pairs.
  A holdout look's symbol counts iff its holdout was looked at
  (`holdout.evaluated`).
- Since step 5 the metric keys include the per-trade return statistics
  `sr_per_trade`, `skew`, `kurtosis` (plain, not excess) and `n_returns`,
  the verdict DSR's inputs, plus `test_period_years` (all from
  `unified_metrics.json`).
- `no_edge` writes `sharpe: null` since step 5: without an edge nothing is
  sized, and a 0 would enter statistics. Its rows still carry the naive
  `max_drawdown` and `annual_return`.
- The ledger stores no per-symbol status, so a legacy `no_edge` zero cannot
  be told apart from a real zero. No heuristic: legacy rows of runs before
  step 5 carry no return statistics, so they count towards N only and never
  reach V[SR]. Ungated development runs after step 5 do carry them; once
  backfilled as legacy trials they enter V[SR] like iterations (they are
  development data). Side effect: the "Best development result" line
  (ranked by Sharpe) can pick a legacy zero, but only when every other
  development Sharpe is negative.
- A run whose `strategy.json` carries `holdout_cutoff` but that did not pass
  the gate is never read as a legacy run: the scan raises, so the backfill
  refuses it (its metrics are holdout metrics, not development data).
  `run_optimizer` refuses such an ungated holdout look in the first place.

### Fingerprint

`sha256(canonical_json({"strategy": config, "symbols": symbols}))` where
`config` is the strategy metadata as the CLI passes it to `run_optimizer` —
every CLI/API override (`timeframe`, `start_date`, `end_date` after cutoff
clamping, `cost_multiplier`, `_cli_data_path`) is already merged into it —
**after preset resolution** (`fwbg.core.presets.resolve_presets`, the same
resolution `StrategyConfig.from_dict` applies), and `symbols` is the sorted
list of resolved symbols:

- included: everything that determines the result (datasource, timeframe,
  window, costs, pipeline, model, signal rules, exits, risk, validation,
  filters — preset content, never preset names);
- excluded: descriptive keys (`name`, `description`, `category`, `tags`,
  `hypothesis`, `expected_outcome`, `baseline_run`, `changes`),
  `resources`, the file location `_strategy_dir` and the dashboard
  bookkeeping `_refs`.

Unresolved preset names are refused rather than hashed. Canonical JSON:
sorted keys, no whitespace, integral floats written as integers; a golden
test pins the digest, so any serialisation change forces a
`fingerprint_version` bump. Known ceiling: values are hashed as written, not
as the run interprets them — a spelled-out default vs. an omitted one, or a
timeframe alias (`DAY` vs. `DAY_1`, or an omitted timeframe falling back to
the `TIMEFRAME` environment variable), hash differently. The wiring
canonicalises the timeframe before gating (see the plan's notes).

The code state is deliberately not part of the fingerprint: re-running the
same config after a bugfix is legitimate and only needs `--reason`.

A holdout run's effective metadata carries `holdout_cutoff` (the card's
cutoff, written by the gate), so it is fingerprinted and a holdout look never
matches an iteration over the same window. A strategy or study config may not
set it (`INVALID_CONFIG`, 422): only the register decides what is a look.

### Code state

When the working tree is dirty, the run writes `code.patch` into its run
directory: `git diff HEAD` plus the contents of untracked files under
`src/` and `packages/` (new modules are often untracked). The existing
git capture lives in `src/fwbg/results/storage.py:123`.

## Gate

`fwbg.research.gate.check(card, request, trials) -> Decision` is a pure
function. `request` is a `RunRequest` (research id, preset-resolved strategy
metadata with overrides, canonical symbols, note, reason, holdout flag);
`trials` is the whole ledger in append order. The `Decision` carries every
denial (with its HTTP status), the trial kind, the `effective_metadata` the
run must use (e.g. with the clamped `end_date`) and the fingerprint.

Enforcement points:

- **CLI run path** (`run_optimizer`, `src/fwbg/cli/main.py`) — authoritative.
  Every run passes through here, because `/api/runs/start` spawns the CLI
  (`start_run` in `src/fwbg/api/runs.py`).
- **`POST /api/runs/start`** — preflight before spawning, so callers such
  as fwbg-agents get an immediate error instead of a process that dies. The
  preflight runs the CLI once with `--check-json` (`src/fwbg/api/run_preflight.py`),
  so the API refuses with the gate's own denials and identical fingerprints
  without duplicating asset resolution. `RunStartRequest` gains
  `research_id`, `note`, `reason` and (step 5) `holdout`, which the
  preflight and the spawned run pass on as `--holdout`.

Before either gate call, the CLI and API resolve `--assets` and
`--asset-classes` through the same `AssetRegistry`, canonicalise the result
to an uppercase, sorted symbol tuple, and pass that resolved tuple as
`request.symbols`. The gate never fingerprints unresolved class names or a
caller-supplied symbol list; an empty or failed resolution is rejected before
the run starts. This keeps CLI and API fingerprints identical for the same
asset set.

Checks (a missing research id or card stops immediately; after that every
denial is collected, so one refusal lists all problems):

Data split: `end_date` is exclusive when fwbg slices data, so iterations
see `[.., cutoff)`. A holdout look develops on `[start, cutoff)` and is
evaluated once on `[cutoff, end)` (see Holdout look). `trials` are passed in
ledger (append) order.

1. `--research-id` given, card exists and declares that id → else deny (422).
2. Card parses and `status == active` → else deny (409), quoting the status.
3. Criteria: the hash of the card's `success_criteria` must equal the
   **latest** `criteria_amendments` entry's hash, or `registered_criteria_hash`
   when there are no amendments; without amendments it must also equal the
   hash recorded by the card's last trial; and every earlier trial of the
   card must have run under the registration hash or an amendment hash
   → else deny (409, `CRITERIA_CHANGED`). Criteria edits go through
   `fwbg research amend --reason`; a hand-written amendment is only caught
   by the research repo's git history.
4. The card's `holdout_cutoff` equals the cutoff recorded by every earlier
   trial of the card — it is **frozen once trials ran**, in both directions
   (earlier would put seen data into the holdout; later after a holdout look
   would tune on seen holdout results) → else deny (409).
5. Window: `start_date`/`end_date` must be strict `YYYY-MM-DD` with
   start < end → else deny (422). Iteration: missing end → set to the
   cutoff; end after the cutoff → deny (409). Holdout run: its start must
   lie before the cutoff, since it develops on that data (422); end may be
   open; an end on or before the cutoff → deny (422). The gate then writes
   the card's `holdout_cutoff` into the effective metadata; a caller's
   `holdout_cutoff` → deny (422, `INVALID_CONFIG`).
6. `--holdout`: denied on a card with `holdout_contaminated: true` (409). The
   first holdout trial per strategy is free; any earlier holdout trial
   (whatever its status) makes a further look require `--reason`.
7. Fingerprint already recorded by a successful trial (any card) → deny
   (409), listing the earlier trials (trial, card, run, date), unless
   `--reason` is given. A config that cannot be fingerprinted (unresolved
   preset, NaN, non-JSON value) → deny (422).
8. `--note` (exactly one line: what changed, why) is required (422). The
   diff to the previous iteration is computed automatically.

The CLI wiring adds run-level checks around the pure function: a run id
must be a valid name (`INVALID_RUN_ID`, 422) and is used once — a run id
whose directory already holds `config.json` or `grid_details/`, that is in
the ledger or that has a pending marker is refused (`RUN_ID_IN_USE`, 409; a
bare directory is allowed, the API pre-creates it for logs). An unreadable
card or ledger denies (`REGISTER_UNREADABLE`, 409). A live pending marker
with the same fingerprint counts as a duplicate like a recorded successful
trial. Any pending holdout marker of the card, live or dead, counts as an
earlier look (`HOLDOUT_REPEAT` without `--reason`): a dead look may already
have shown results.

`--no-save` is refused while the gate is on: a run whose results were seen
is a trial, and the staging run directory it would need is not built. (If
`--no-save` is needed again, add a staging run directory instead of lifting
the refusal.)

`--check` (`python -m fwbg.cli --check …`) runs the same gate with the run
command's overrides and asset selection and exits 0 (allowed) or 2
(refused). Neither `--check` nor a refused run writes anything of its own;
both may record and commit *other* runs' stale pending markers, because
recovery runs first so that a duplicate from a killed run is seen.
`--check-json` does the same and prints its result as one
`RESEARCH_CHECK {…}` stdout line (allowed: kind, fingerprint, symbols,
window, `holdout_cutoff` (`null` for an iteration); refused: every denial
with code, HTTP status and message). `--check` prints a holdout look's
development/holdout split.
`parse_check_output(stdout, returncode)` accepts that line only when it
matches the exit code, so a line from a CLI that failed before its gate (or
one a plugin printed) never counts as a gate result.

`FWBG_RESEARCH_GATE=off` disables the gate for the test suite only. It
prints a warning on every run and is never set in `docker-compose.yml`.
A missing research directory does **not** disable the gate.

A gated run is on record from its start. When the gate allows it, the
CLI writes a pending marker `$FWBG_RESEARCH_DIR/.staging/<run_id>.json`
(ignored by the register's git) under the register lock; it holds the whole
trial except its outcome, the canonical config, the run directory, host and
pid. After the run, the CLI decides the outcome from the run directory
(`ok` iff at least one symbol was evaluated), appends the trial to the
ledger, writes `configs/<fingerprint>.json`, adds a `research` block (id,
trial, fingerprint, note, reason, kind) to the run's `config.json`, commits
in the research repo and deletes the marker. A run that never reaches its
own recording — Ctrl-C (the worker pool's handler ends the process through
`os._exit`), SIGKILL, OOM — leaves its marker behind; once its process is
gone, the next gated run, `--check` or `verdict` records it. `append_trial`
locks `trials.jsonl` itself; append plus commit are serialised by a separate
`.register.lock` (locking `trials.jsonl` again from the same process would
deadlock).

### Scripts outside the run system

Study scripts call `fwbg.research.study.register_trial(...)` for every
evaluated variant, failed and discarded ones too. It is not re-exported from
`fwbg.research`, to avoid import cycles. The config must state its
`timeframe` (a CLI run always fingerprints one, so without it duplicates never
match) and the evaluated `end_date`: a study has already run, so the gate
cannot clamp its window. Metrics may only name the trial's symbols, and a run
id, when given, follows the CLI's rules (the `study:` prefix belongs to the
backfill).
Presets are resolved and the timeframe canonicalised as for a CLI run, so
fingerprints match; the same gate applies (a refusal raises `GateDenied`).
`register_trial` records iterations only: it has no holdout parameter, and a
config carrying `holdout_cutoff` is refused (`INVALID_CONFIG`). A holdout
look runs through `python -m fwbg.cli --holdout`, whose split the register
controls.
Calling it is enforced by convention (rule in `CLAUDE.md`/`AGENTS.md`), not
by code.

## Holdout look (step 5)

`python -m fwbg.cli --holdout` (API: `RunStartRequest.holdout`) looks at the
card's holdout once; use it when iterating is done. The per-symbol worker
(`src/fwbg/optimization/holdout.py`) runs it when the gate set
`holdout_cutoff` (operator decision 2026-10-21, "Option A"):

1. **Development** is the normal pipeline on `[start, cutoff)`: walk-forward,
   selection, unified simulation and risk calibration see nothing after the
   cutoff.
2. **Evaluation:** the frozen unified candidate (with the indicator variant
   development chose) is retrained once on the development bars minus the
   embargo, `[0, n_dev - embargo_bars)`, and evaluated once on
   `[cutoff, end)` through a one-fold `run_unified_simulation`
   (`evaluate_on_holdout`). The embargo keeps labels of the last training
   bars from reaching into the holdout. Retraining on all development data
   is what a deployment would do; nothing is selected after the cutoff.
   Selected features missing from the training frame are dropped and
   recorded (`dropped_features`).
3. **Sizing without hindsight:** holdout trades are sized with the
   development's fk (`risk_per_trade`), circuit breaker and loss scale; no
   parameter is fitted on holdout trades. The development calibrates them on
   the trades of its causal replay over all folds (protocol 3, operator
   decision 2026-10-22): the out-of-sample track record a trader would have
   had, the same sizing a deployment exports.
4. **Metrics come from holdout trades only.** The Monte Carlo test on them
   decides `ok` or `not_significant`. `win_rate` is over all holdout trades;
   `pnl`, `sharpe`, `calmar` and the return statistics are over the trades
   the circuit breaker sized. `pnl` is a sum (a development `pnl` is a
   per-fold mean). A holdout result carries no `fold_stability`
   (`min_fold_stability` reads n/a).

Files per symbol: `config.json` gets a `holdout` block (cutoff, `evaluated`,
and either the reason it was not evaluated or the train and test windows,
bars, `embargo_bars`, trades, sized trades and `dropped_features`);
`development.json` keeps the development outcome apart (without per-fold
details, with the frozen candidate; the API's grid detail nests it under
`development`); `unified_metrics.json` holds the holdout metrics and return
statistics. A holdout look exports nothing to `assets.json`: it is an
evaluation, not a deployment.

Statuses: the holdout phase assigns `HOLDOUT_STATUSES` (`ok`,
`not_significant`, `no_unified_trades`, `insufficient_data` for a holdout
window too short to evaluate, `no_data` for no data after the cutoff,
`error`). A development that ends in anything but `ok` (`no_edge`,
`not_significant`, `no_successful_folds`, …) passes its status through with
`holdout.evaluated: false`; the symbol is not evaluated and does not count.
An error before any holdout trade exists is not a look; one after is
(`evaluated: true`). A look in which no symbol was evaluated is recorded
`failed` and still counts as a look (gate rule 6), so the next one needs
`--reason`. Conservative: the run may already have shown something.

Exit code and ledger agree: a holdout run fails (exit 1, "no symbol reached
its holdout") iff no symbol's holdout was looked at, the rule the ledger
counts by; a look that errors after its holdout trades exist completes
(exit 0). The run's elite table labels holdout results `HOLDOUT`, and
`unified_metrics.json` counts `n_long`/`n_short` from the holdout trades'
directions.

Causality: raw indicators and the data-loading pipeline run on the whole
window `[start, end)` so the first holdout bars have their warm-up. The
development rows then equal the development run's only because registered
indicators and data loaders are prefix-stable: computed on `[start, end)`
they equal their values on `[start, cutoff)`. `tests/test_prefix_stability.py`
is that guarantee; `tests/test_lookahead_bias.py` (future scramble) checks
causality but cannot see statistics over the frame's length. Preprocessors
and stateful indicators are fit on the training frame only. Deliberate
shortcut: data is loaded and indicators precomputed twice per symbol.

Known limitations:

- The `no_edge` decision (whether a symbol has an edge; without one its
  Sharpe is null) still uses the deployment sizing, calibrated on all
  folds' replayed trades with the frozen setting's RRR (the risk management
  call in `src/fwbg/optimization/process.py`, before causal reporting).
  That is hindsight by design, for deployment. It can only remove a row from
  the ranking or let it through; it does not change the reported causal
  metrics.
- The deployment sizing (`fk`, circuit breaker, loss scale; also the
  holdout's) takes its win rate and loss sizes from the causal replay's
  trades, which were traded with the prefix settings, but uses the frozen
  setting's RRR. While the prefix settings differ from the frozen one, Kelly
  mixes the two.
- With fewer than 10 causal trades the reported returns fall back to the
  deployment sizing (`fk` calibrated on all folds, `causal_ok` in
  `src/fwbg/optimization/process.py`). `pnl_to_returns` multiplies every
  PnL by one positive scalar (`fk` over the mean loss), so Sharpe,
  per-trade SR, skew and kurtosis, which the register ranks and deflates,
  are unaffected; drawdown, Calmar and annual return in that case still use
  hindsight sizing.
- If fold 0 fails, fold 1 is the first traded fold and has no prior trades,
  so causal reporting excludes its trades, yet its bars still count in the
  annualised test period (`reporting_test_bars` in
  `src/fwbg/optimization/process.py` leaves out fold 0 only). Conservative:
  annualised Sharpe and return read low; per-trade SR and DSR are
  unaffected.
- An unmarked run of a strategy file without `model` is stored without a
  `strategy.model` mapping, so the register does not see it as an optimizer
  evaluation and keeps it ranked (workspace configs set `model`).
- Externally cached signal data (e.g. provider probabilities that
  `jev_signal` reads) trained past the cutoff contaminates development and
  holdout alike; the pipeline cannot detect it.
- Resampling runs on the whole CSV; harmless while a date cutoff (midnight)
  falls on a bar boundary.
- An open-ended look records `window.end: null` and fingerprints no
  `end_date`, so looks months apart share fingerprint and window although
  they see different data. A second look needs `--reason` anyway, and the
  run directory's `holdout` block records the window actually touched.
- Cosmetic: the run summary labels a holdout result "Walk-Forward (0
  folds)", and progress shows a symbol done before its holdout phase.
- Pre-existing, not of this step: the `jev_signal` data-loading plugin
  crashes in `run_data_loading` without `params.indicators`
  (`get_default_params` is an instance method, the orchestrator calls it on
  the class).

## Selection protocol

Every symbol result records how its walk-forward chose settings as
`walk_forward.selection_protocol`
(`src/fwbg/optimization/selection_protocol.py`); the ledger copies it into
the symbol's metrics. Protocol 2: in every fold the setting (signal and ML
models) is chosen on inner windows of the fold's training window, never on
its test set, and indicator variants are ranked by mean inner validation PnL
instead of mean test PnL. Runs without the marker predate it.

Before protocol 2, signal models evaluated every exit strategy × model HP ×
CT combo on the test fold and kept the best: biased development metrics,
bias ratios of always 1, PBO silently n/a, and a `fold_stability` that was
the share of folds whose best-of-grid test PnL was positive. Now
(`src/fwbg/optimization/signal_selection.py`, `signal_fold.py`):

- **Windows:** the inner windows ML uses (`nested_cv_split` with
  `validation.n_inner_folds` and `validation.embargo_bars`); no new config
  keys. Stateful indicators keep their fold-level values (no per-window
  refit; deliberate shortcut, only `fractional_diff` is stateful today).
- **Criterion:** mean inner validation PnL over all windows, a window
  without trades counting 0. Eligible: at least one inner trade. Ties keep
  config order; no sign gate, a negative best is chosen and reported.
- **Test fold:** only the chosen combo is evaluated, once. A fold without an
  eligible combo yields no result, its test set untouched (fail closed). A
  fold whose chosen combo makes no test trade is kept with PnL 0: dropping
  it would let the test set filter folds.
- **Cost:** C combos × K inner windows plus one test evaluation per fold,
  instead of C test evaluations.

Protocol 3 adds a causal unified replay
(`src/fwbg/optimization/unified_simulation.py`, `causal_candidates`). Test
fold k is replayed with `merge_unified_settings` over the successful folds
with `fold_id ≤ k`, the model-HP majority rule applied to that prefix
(folds ordered by `fold_id`): each of them chose its setting on its own
training data, which ends before fold k's test data. With a multi-variant
`optimization.indicator_grid`, protocol 3 also chooses the indicator
variant per prefix (`src/fwbg/optimization/causal_variants.py`): fold k is
replayed with the variant whose score (successful folds, then mean inner
validation PnL; ties go to the earlier variant) is best over the folds with
`fold_id ≤ k`, and with that variant's causal candidate for fold k. Folds
are anchored, so a later fold's inner windows contain fold k's test period;
a variant ranked over all folds was chosen partly on it. A variant that
replays some folds without being the frozen one recomputes its indicators
once for the replay, unless all its assigned folds are untraded. A fold
before the first successful one is not traded (empty result); it still
counts as a fold and in the test period. That differs from causal
reporting's `untraded_folds`: folds that were replayed but get no causal
size because Kelly on the trades of the earlier folds is ≤ 0. Each
replayed fold is sized for causal Kelly reporting with the RRR of its own
replay setting. The frozen setting (`development.json`, holdout,
`best_config`, deployment) stays the merge over all folds of the frozen
variant, the one best over all folds: what one would deploy after the last
fold, where both variant choices coincide. `fold_stability` (a card
criterion) and `walk_forward.profitable_folds` come from the replay: the
share of replayed folds with PnL > 0, an untraded fold counting as not
profitable and a fold without enough data for its replay not counting at
all. A fold whose own selection failed is still replayed with the setting
of the folds before it, so `profitable_folds` counts replayed folds and can
exceed `n_folds` (the successful folds). The per-fold diagnostics (bias
ratios, PBO, the per-fold PnL statistics, grid results) remain those of
the frozen variant. ML folds are kept whatever their test trade count,
like signal folds. Before protocol 3 the replay traded every test fold with
the frozen variant's merge over all folds, so an earlier fold was traded with a
variant and settings chosen in later folds, whose training windows contain
it, and ML folds with fewer than 5 test trades were dropped: the test set
shaped the reported development metrics of signal and ML models alike.

**Register.** An optimizer evaluation (its stored config
`configs/<fingerprint>.json` has a `strategy.model` mapping) without
`selection_protocol ≥ RANKED_PROTOCOL` (`src/fwbg/research/stats.py`,
hand-curated, 3) has development metrics that depended on its test folds.
Such an evaluation counts in N but is never ranked: not the best
development result (also in `list`, CLI and API), not a V[SR] sample, not
the DSR's best. The verdict and `show` state how many were left out per
reason, e.g. "4 pre-protocol-3 evaluations not ranked (counted in N): 1
signal-model, 1 indicator-grid, 2 unified-merge". The reason of an evaluation
without the marker comes from its config: signal-model for
`model.type == "signal"` (the setting itself was chosen on test folds),
else indicator-grid for an `optimization.indicator_grid` with more than one
variant (the variant was ranked by mean test PnL, for ML models too; the
register counts variants as `expand_indicator_grid` does: per indicator the
product of its value lists, summed over indicators), else unified-merge. A
config with several reasons counts once, in that order. An evaluation
marked 2 chose its setting on inner windows; the merge over all folds, the
variant choice over all folds (grid configs) and the ML test-trade filter
kept it out. It is reported as unified-merge. Rows below protocol 3 also
carry the `fold_stability` of the frozen variant's per-fold test PnL, not
the replay's. Evaluations without metrics are not counted there (nothing
to leave out). A study script that passes a config
with a `strategy.model` mapping counts as an optimizer evaluation
(conservative). Evaluations without a stored config (studies, legacy runs)
stay ranked; those with an unreadable config are ranked as unknown.
Rerunning is the operator's call.

ML results changed with protocol 2 too, so old and new runs differ:

- Variants ranked by mean inner validation PnL (before: mean test PnL).
- `mean_bias_ratio` is Σ test PnL / Σ inner PnL over the folds with a
  positive inner PnL (before: mean of per-fold ratios, 0 for a fold with
  inner PnL ≤ 0); no such fold → `None`, which counts as warned. A new
  warning states "k/n folds without bias ratio".
- PBO rows are keyed by the full combo (exit strategy, params, HP, regime),
  so combos that shared a row before are now split and PBO values change.
- Mixed `None`/integer `timeout_bars` across the grid now get DSR and PBO
  instead of falling back to `dsr = 0`.
- Folds with no test trade are left out of win-rate statistics (they still
  count as folds and in the per-fold PnL statistics; from protocol 3 on,
  `fold_stability` comes from the causal replay).
- `selection_protocol` is stamped on every result, ML included.

## Tooling

### CLI `fwbg research`

| Command | Purpose |
|---|---|
| `new "<title>" --tags … [--cutoff DATE]` | Create card from template, status `active`. |
| `list [--status S] [--tag T]` | Id, title, status, iterations, best development result, holdout used. |
| `show S-NNNN` | Card, iteration timeline, coverage. |
| `python -m fwbg.cli --check --strategy-file … [overrides]` | Dry run of the gate (a flag of the run command, not a `research` subcommand: it needs the run's overrides and asset selection). |
| `amend S-NNNN --set NAME=VALUE … --reason …` | Change success criteria; appends a `criteria_amendments` entry (date, reason, new hash) to the front matter. Only on `active` cards; refused when the card's criteria do not match the recorded registration/amendments. |
| `verdict S-NNNN <status> --reason … --reopen-if …` | Write verdict, set status. Refused while the card has pending (running) runs. |
| `reopen S-NNNN --reason …` | `parked`/`rejected` → `active`. |
| `init` | Create the register (idempotent). |
| `backfill scan --out MAPPING` / `backfill apply MAPPING` | Legacy import (see Migration). |
| `reconcile` | Validate the register, record stale pending markers, commit uncommitted register paths (`reports/` included) and report workspace runs missing from the ledger; idempotent. It does not index them: assigning a run to a card needs the operator (`backfill scan/apply`). |

**Coverage view** (`show`): from the canonical configs of all trials,
list every config path whose value varied, with the set or range of
values seen, plus assets, timeframes and date windows. List items are
keyed by their `name` (`strategy.exit_strategies[atr_based].params.tp_mult`), or
by `[*]` when they have none.

### Verdict

Auto-computed (`src/fwbg/research/commands.py`, `stats.py`): N (counted
trials and symbol evaluations of the card, plus the register's for
context), the distinct `criteria_hash` values used, holdout state, the best
development result (by Sharpe) against the current criteria, the Deflated
Sharpe Ratio, one line per holdout look and how many optimizer
evaluations predate `RANKED_PROTOCOL` (counted in N, not ranked; see
Selection protocol). Manual: reason and **reopen condition** (what new
evidence or capability would justify another round).

Deflated Sharpe Ratio (step 5, `src/fwbg/research/dsr.py`, the full formula
of Bailey & López de Prado 2014, stdlib only):

- **N:** every counted symbol evaluation of the card, holdout looks
  included. Holdout looks are not part of the selection the DSR deflates;
  counting them anyway is the conservative choice (a larger N only raises
  E[max SR]).
- **V[SR]:** the unbiased sample variance (n − 1) of the per-trade SRs of
  the card's development evaluations that carry return statistics. Legacy
  evaluations of runs before step 5 and `no_edge` evaluations have none;
  they count in N only, as do optimizer evaluations below
  `RANKED_PROTOCOL` (their SRs depended on the test folds).
- **V[SR] floor** (operator decision 2026-10-21): V[SR] is at least the mean
  over those evaluations of 1/(T − 1), T being each one's return count.
  Under the null of zero true SR, a per-trade SR estimated from T returns
  has a sampling variance of about 1/(T − 1). Without the floor, two or
  a few near-identical samples would give a V[SR] near 0, collapse E[max SR]
  towards 0 and leave the DSR no deflation at all. The floor is theory, not
  a threshold fitted to data. The verdict line states V[SR], whether the
  floor or the sample variance was used, and the other one.
- **Best:** the development evaluation with the highest `sr_per_trade`, the
  statistic being deflated (per trade, never annualised, so trials with
  different trade frequencies share one scale); its own skew, kurtosis and
  return count enter the PSR against E[max SR] of N trials. It can differ
  from the "Best development result" line, which ranks by annualised Sharpe.
- The DSR is printed with four decimals; a value that would round to 1 reads
  "> 0.9999".
- Fewer than 2 evaluations with return statistics → "too few trials with
  return statistics". Skew and kurtosis that leave no positive SR variance
  → "not defined". Ranked optimizer evaluations (protocol 3) take their
  per-trade returns from the causal replay; which of them are `no_edge`,
  and so left without return statistics, is still decided with hindsight
  (see Holdout look, Known limitations).

Holdout result: one line per holdout trial with the evaluated period
(`holdout <cutoff>..<end or open>`), each symbol checked against the current
criteria, labelled like the best-result line (registration or amendment N).
"(criteria changed since this look)" flags a look recorded under a different
`criteria_hash` than the card's current one. A symbol without numbers reads
"no holdout metrics"; a failed look reads "failed, no symbol evaluated".
Symbols whose holdout was not looked at (development ended before
evaluating) are absent from the line: the ledger records only evaluated
symbols of a holdout trial.

The run-level `overfitting` block in each symbol's `config.json`
(`src/fwbg/optimization/overfitting.py`, written since step 5) is advisory
and **not** the verdict DSR: it assumes unit SR variance and N = grid
combinations, so it reads near 0 for per-trade SRs. A `no_edge` config's
`overfitting.dsr.observed_sr` was computed without sizing (fk ≤ 0); never use
it as a trial SR.

### HTTP API (fwbg-agents, dashboard)

Router `src/fwbg/api/research.py` wraps the CLI's card commands and views.

- `GET /api/research[?status=…&tag=…]` → `{"cards": [summary…],
  "pending_error": str|null}`; a card that does not load is listed as
  `{id, error}`.
- `GET /api/research/{id}` → `{card, trials, pending, pending_error,
  coverage, criteria_problems}`.
- `POST /api/research` (title, tags, hypothesis, mechanism, optional cutoff
  and criteria) → 201 `{card}`.
- `POST /api/research/{id}/status` (status, reason, `actor`, `reopen_if` for
  a verdict) → `{card}`; the actor is recorded with the verdict or reopen.
- Write endpoints add `"warning"` when the write landed but was not
  committed; the caller must not retry then. 404 unknown card, 422 input
  errors, 409 conflicts and an unreadable register.
- `POST /api/runs/start` refusals: `detail = {"message", "denials"}` (409
  if any denial is a 409, else 422); a preflight that failed before the gate
  `{"message", "output"}` (422); 504 on preflight timeout (120 s); 409
  "run was cancelled before it started" when the run was cancelled during
  its preflight.

fwbg-agents integration (separate plan in the agents repo): the
Researcher queries `GET /api/research?tag=…` before proposing; its
`Strategy` row stores `research_id`; its `abandon` + post-mortem maps to a
research verdict.

### Container

The Docker stack runs on the operator's machine; compose bind-mounts the
host register (`${FWBG_RESEARCH_HOST_DIR:-$HOME/fwbg/research}`) at
`/root/fwbg/research`, so CLI and API record into one register.

- The register must be created on the host first (`python -m fwbg.cli
  research init`); `create_host_path: false` makes compose refuse a missing
  host path instead of starting an empty second register.
- The image has `git`. The container commits as `fwbg-api
  <fwbg-api@localhost>` (env in compose); it has no signing key and no
  global git config, so its commits are unsigned. The operator's own commits stay signed.
- The container runs as root. On exit, `register_lock` hands every
  root-owned file in the register back to the register directory's owner
  (`restore_owner`); otherwise the host user could no longer write
  `.git/objects`. It chowns only root-owned entries, never through a
  symlink, and relies on `fs.protected_hardlinks` (the kernel default)
  against hard links planted in the register. The lock file is opened
  `O_RDONLY|O_CREAT`, so one root created stays usable (flock needs no
  write access).
- Every register git call passes `-c safe.directory=<register>` (git
  refuses a repository owned by another user) and `gc.auto=0`,
  `maintenance.auto=false` (git detaches those, so they would create
  root-owned files after ownership was restored). Trusting the directory
  runs the register's git config and hooks as root; acceptable, as the
  register's owner is the operator, who controls Docker anyway.
- `hostname: fwbg-api`: pending markers carry the host name, and a marker
  from another host always looks alive. A stable name lets a restarted
  container record its own interrupted runs: `create_app` records stale
  markers at startup (best effort, only with the gate on and an existing
  register; a register error is logged, never fails startup). If that did
  not happen, run `docker exec fwbg-api python -m fwbg.cli research
  reconcile`; a reconcile on the host cannot clear markers written by the
  container, because they name the host `fwbg-api`.
- A container killed while it holds the register lock can leave root-owned
  files (e.g. `.git/objects/xx/`) until its next register write. Fix at once
  with `sudo chown -R "$USER": ~/fwbg/research`.

### AI sessions

`CLAUDE.md` and `AGENTS.md` in fwbg carry a rule (section "Research
register", since step 3): before proposing a test, run `fwbg research
list/show`; a new idea gets a card first; every run uses `--research-id`
and `--note`; never set `FWBG_RESEARCH_GATE`; only a deliberate verdict
closes a strategy. Since step 4 the rule also requires study scripts to
call `fwbg.research.study.register_trial()`; since step 5 it says when to
take a holdout look (`--holdout`, once iterating is done; a further look
needs `--reason`). The rule contains no results and can live in the public
repo.

## Migration

Full backfill of all existing runs and studies:

1. **Index:** `fwbg research backfill scan --out mapping.yaml` scans one
   results directory (the workspace `test_results/` by default; runs that
   exist only in the repo-local copy are consolidated into it by hand
   first), computes fingerprints from the stored `strategy.json` (presets
   resolved, timeframe canonicalised) and the run's symbols, decides each
   run's status from its symbols' own status fields (all three historical
   run formats; see the backfill plan, decision 2) and marks agent smoke
   runs as `smoke`. Stored `strategy.json` files keep preset *names*, so legacy
   fingerprints are an approximation: presets resolve against today's
   files. Many legacy runs point their `_strategy_dir` at a directory whose
   preset folders no longer exist; the backfill must handle these
   deliberately (remap the old base directory to the workspace, or record
   the run as unresolvable) and must never fall back to hashing names.
2. **Grouping:** an AI session groups runs into proposed cards (by name
   family and config similarity) and fills `mapping.yaml` (run → S-ID).
   Studies outside the run system get one card each, with trials counted as
   documented. The operator reviews the mapping before anything is written.
   No LLM runs inside fwbg.
3. **Apply:** `fwbg research backfill apply mapping.yaml` validates the whole
   mapping, then writes cards and legacy trials in one register commit.
   Strategies with a documented conclusion get that verdict; strategies
   without one become `parked` ("no conclusion documented"), never
   `rejected`.
4. **Legacy holdout (forward holdout, operator decision 2026-10-19):**
   cutoff = the day after the latest run (or study) date of the card
   (`end_date` is exclusive, so the last day a legacy run saw stays out of
   the holdout); `holdout_contaminated: false`. No run has seen data after
   the cutoff, so that data is a clean holdout that grows over time. Every
   legacy trial records the **card's** cutoff — the gate freezes the cutoff
   at whatever the card's trials recorded.
5. **Repo cleanup:** move runs that exist only in the repo-local
   `test_results/` into the workspace; remove byte-identical duplicates
   after a diff check and operator confirmation; keep `test_results/`
   ignored.
6. **Reports:** study write-ups move to `research/reports/` and are linked
   from cards. Code plans stay in fwbg.

## Error handling

- Ledger append or research-repo commit fails after a successful run →
  warning, the run is not failed. The pending marker stays and the next
  gated run, `--check` or `verdict` records it; every register commit stages
  all register paths, so it also picks up an earlier failed commit. `reconcile`
  validates the ledger and every card under the register lock (and changes
  nothing if one does not parse), records stale pending markers, then stages
  only the register paths (`strategies/`, `trials.jsonl`, `configs/` and
  `reports/`) and creates a recovery commit, leaving unrelated research-repo
  changes untouched. It then reports workspace runs missing from the ledger
  (runs in progress left out) but does not index them: assigning a run to a
  card needs the operator (`research backfill scan/apply`).
  If validation or the recovery commit fails, it leaves the data in place,
  reports the failure, and remains safe to retry.
- Unparseable card or ledger → the gate denies (fail closed).
- Concurrent runs → serialized by the file lock.

## Testing

- Fingerprint: same config with different name/tags → same hash; changed
  parameter, asset, window or cost → different hash; key order irrelevant;
  CLI overrides included.
- Gate: one test per deny path and per allow path; automatic `end_date`
  clamping; holdout look counting.
- Ledger: two processes appending concurrently lose or tear no lines;
  failed runs recorded.
- Backfill: fixture with legacy runs (with/without `results.json`, smoke)
  → correct kinds; second run adds nothing.
- DSR: matches the numerical example of Bailey & López de Prado (2014)
  (`tests/research/test_dsr.py`) — done.
- Holdout look: prefix stability of indicators and data loaders
  (`tests/test_prefix_stability.py`) is its causality guarantee.
- API: `start_run` returns 409/422 before spawning a process.
- Existing suite: `conftest.py` sets `FWBG_RESEARCH_GATE=off`; no existing
  test writes to a ledger.

## Rollout

Each step is usable on its own.

1. **Core:** `fwbg.research` (card parser, ledger, fingerprint, gate
   function), `code.patch` capture, research repo initialisation.
2. **Backfill + cleanup**, including operator review of the mapping —
   before the gate, so duplicate detection knows the legacy configs from
   day one.
3. **CLI gate** and `fwbg research new/list/show/amend/verdict/reopen`,
   `--check`, `CLAUDE.md`/`AGENTS.md` rule — done (plan
   `docs/plans/2026-10-20-research-register-gate.md`). Until step 4, API,
   dashboard and fwbg-agents runs are refused (no `--research-id`).
4. **API** gate (`--check-json` preflight) and endpoints,
   `fwbg.research.study.register_trial()` (and its line in the
   `CLAUDE.md`/`AGENTS.md` rule), `reconcile`, register mount in the API
   container — done (plan `docs/plans/2026-10-20-research-register-api.md`).
   For later steps: fwbg-agents (step 6) must call `/api/runs/start` with a
   timeout of at least 130 s — the preflight runs the CLI (up to 120 s),
   httpx defaults to 5 s, and a client timeout would orphan a started run.
   The dashboard sends no `research_id`/`note` yet, so dashboard starts are
   refused with 422 until step 7.
5. **Holdout mode** (`--holdout`, API `holdout`, see Holdout look),
   per-trade return statistics in run files and ledger, `no_edge` without
   Sharpe, and verdict statistics (DSR, holdout results) — done (plan
   `docs/plans/2026-10-21-research-register-holdout.md`).
   `/api/runs/compare` still counts a `null` Sharpe as 0 in `avg_sharpe`
   (old behaviour kept). For step 6:
   - `no_edge` symbols now carry `sharpe: null`. fwbg-agents'
     `_best_symbol_metrics` returns `{}` when every symbol is `no_edge`, so
     its lifecycle check reports a missing metric (fails closed); check every
     other place that reads `sharpe` for `null`.
   - A holdout look is started with `holdout: true`; its run directory's
     `config.json` `holdout` block and `development.json` tell holdout and
     development metrics apart. A further look needs a `reason`.
6. **fwbg-agents integration** (separate plan, agents repo).
7. **Dashboard page** (phase 2).

## Open questions

- **Holdout evaluation mechanics** — closed in step 5: develop on
  `[start, cutoff)`, retrain the frozen candidate once on the development
  data minus the embargo, evaluate once on `[cutoff, end)` (see Holdout look).
- **`/api/runs/preview`** — closed in step 4: not a trial. It shows one
  symbol's signals with fixed TP/SL and reports no aggregated metrics;
  fwbg-agents does not call it.

## Alternatives considered

- **SQLite catalog in the workspace.** Better ad-hoc queries and
  concurrency, but reasoning stored in rows is not diffable or reviewable,
  AI sessions could only reach it through the CLI, and it needs its own
  backup. Rejected.
- **Extend fwbg-agents' strategy lifecycle.** Lifecycle and post-mortem
  already exist there, but manual and script-based research would depend
  on the agents service running, the rigorous core would be coupled to the
  experimental LLM repo, and an agents `Strategy` is one concrete config,
  not an idea iterated many times. Rejected; agents link to the register
  instead.
- **Register inside the fwbg repo.** Public visibility of results, and the
  API container cannot write to it. Rejected.
- **Hypothesis-level rejection after one failed config.** Rejected by the
  operator: strategies are iterated many times; only a deliberate verdict
  closes one.
