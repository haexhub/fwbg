# Research Register

## Status: Design approved (2026-10-19) — not implemented

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

Each run also reports a nested-CV holdout (the last `holdout_ratio` of the
data, `src/fwbg/optimization/nested_cv.py:62`). When many iterations of
one strategy are tuned while looking at that number, it stops being a
holdout.

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
| Holdout cutoff | Date fixed at registration. Iterations only see data up to it. |
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
- The API container already persists only the workspace volume
  (`fwbg-workspace:/root/fwbg`); it cannot write into the source repo.
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
hand-edited value. At registration, that persisted hash is the frozen
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
 "criteria_hash": "sha256:…",
 "code": {"commit": "…", "dirty": true, "patch": "code.patch"},
 "window": {"start": null, "end": "2025-04-30"},
 "note": "threshold 1.5 → 2.0", "reason": null,
 "metrics": {"<SYMBOL>": {"sharpe": …, "trades": …, "max_dd": …, "p_value": …}}}
```

- `kind`: `iteration | holdout | legacy | smoke`.
- `status`: `ok | failed`. Failed and smoke trials are recorded but not
  counted towards N.
- Metrics are stored **per symbol**. A run over seven assets is seven
  evaluations; N for the Deflated Sharpe Ratio counts (run, symbol) pairs.

### Fingerprint

`sha256(canonical_json(config))` where `config` is the strategy metadata
exactly as the CLI passes it to `run_optimizer` and stores it as
`strategy.json` — every CLI/API override (`timeframe`, `start_date`,
`end_date` after cutoff clamping, `cost_multiplier`, `_cli_data_path`) is
already merged into it — plus the resolved symbol list:

- included: everything that determines the result (datasource, timeframe,
  window, costs, pipeline, model, signal rules, exits, risk, validation,
  filters) and the symbols (asset classes expanded);
- excluded: descriptive keys (`name`, `description`, `category`, `tags`,
  `hypothesis`, `expected_outcome`, `baseline_run`, `changes`) and
  `resources`.

Canonical JSON: sorted keys, no whitespace, integral floats written as
integers. Because stored `strategy.json` files use the same shape, legacy
runs are fingerprinted with the same definition (`fingerprint_version: 1`).
Known ceiling: defaults are not resolved, so a config that spells out a
default and one that omits it hash differently.

The code state is deliberately not part of the fingerprint: re-running the
same config after a bugfix is legitimate and only needs `--reason`.

### Code state

When the working tree is dirty, the run writes `code.patch` into its run
directory: `git diff HEAD` plus the contents of untracked files under
`src/` and `packages/` (new modules are often untracked). The existing
git capture lives in `src/fwbg/results/storage.py:123`.

## Gate

`fwbg.research.gate.check(card, resolved_config, flags) -> Decision` is a
pure function returning allow/deny, reasons and the effective overrides
(e.g. the clamped `end_date`).

Enforcement points:

- **CLI run path** (`run_optimizer`, `src/fwbg/cli/main.py`) — authoritative.
  Every run passes through here, because `/api/runs/start` spawns the CLI
  (`src/fwbg/api/runs.py:183`).
- **`POST /api/runs/start`** — preflight with the same function before
  spawning, so callers such as fwbg-agents get an immediate error instead
  of a process that dies. `RunStartRequest` gains `research_id`, `note`,
  `reason`, `holdout`.

Before either gate call, the CLI and API resolve `--assets` and
`--asset-classes` through the same `AssetRegistry`, canonicalise the result
to an uppercase, sorted symbol tuple, and pass that resolved tuple as
`request.symbols`. The gate never fingerprints unresolved class names or a
caller-supplied symbol list; an empty or failed resolution is rejected before
the run starts. This keeps CLI and API fingerprints identical for the same
asset set.

Checks, in order:

1. `--research-id` given and card exists → else deny (422).
2. Card parses and `status == active` → else deny (409), quoting status,
   date and verdict reason.
3. Before the first trial, require the hash of the card's
   `success_criteria` to equal `registered_criteria_hash` or a hash recorded
   by a `criteria_amendments` entry. After the first trial, require it to
   equal the one recorded by the last trial or to be carried by a
   `criteria_amendments` entry → else deny (409). Criteria edits go through
   `fwbg research amend --reason`.
4. Effective `end_date`: missing → set to `holdout_cutoff`; later than the
   cutoff → deny unless `--holdout`.
5. `--holdout`: the first holdout trial per strategy is free; every further
   one requires `--reason` and is counted.
6. Fingerprint already in the ledger → deny (409), listing the earlier
   trials (run id, date, key metrics), unless `--reason` is given.
7. `--note` (one line: what changed, why) is required (422). The diff to
   the previous iteration is computed automatically.

`--no-save` runs are gated and recorded like any other run: the operator
still saw the result. They skip normal `test_results` output, but the
research integration creates a staging directory under
`$FWBG_RESEARCH_DIR/.staging/<run_id>` for the trial's `config.json` and
`code.patch`; that staging data is retained until the ledger and research-repo
commit succeed, then cleaned up. `reconcile` repairs abandoned staging data.

`FWBG_RESEARCH_GATE=off` disables the gate for the test suite only. It
prints a warning on every run and is never set in `docker-compose.yml`.
A missing research directory does **not** disable the gate.

After the run, the CLI appends the trial to the ledger, writes
`configs/<fingerprint>.json`, adds a `research` block (id, trial,
fingerprint, note, reason, kind) to the run's `config.json`, and commits
in the research repo. Ledger append and commit share one `fcntl` file lock.

### Scripts outside the run system

Study scripts call `fwbg.research.register_trial(...)` with the same
fields. This is enforced by convention (rule in `CLAUDE.md`/`AGENTS.md`),
not by code.

## Tooling

### CLI `fwbg research`

| Command | Purpose |
|---|---|
| `new "<title>" --tags … [--cutoff DATE]` | Create card from template, status `active`. |
| `list [--status S] [--tag T]` | Id, title, status, iterations, best development result, holdout used. |
| `show S-NNNN` | Card, iteration timeline, coverage. |
| `check <strategy-file> [overrides]` | Dry run of the gate. |
| `amend S-NNNN --criteria … --reason …` | Change success criteria, recorded in `History`. |
| `verdict S-NNNN <status> --reason … --reopen-if …` | Write verdict, set status. |
| `reopen S-NNNN --reason …` | `parked`/`rejected` → `active`. |
| `backfill --plan OUT.yaml` / `--apply MAPPING.yaml` | Legacy import (see Migration). |
| `reconcile` | Repair uncommitted register data and index runs missing from the ledger; idempotent. |

**Coverage view** (`show`): from the canonical configs of all trials,
list every config path whose value varied, with the set or range of
values seen, plus assets, timeframes and date windows.

### Verdict

Auto-computed: N (counted trials for this strategy, plus the global N for
context), Deflated Sharpe Ratio of the best iteration given N, holdout
result, coverage summary, list of distinct `criteria_hash` values used.
Manual: reason and **reopen condition** (what new evidence or capability
would justify another round).

### HTTP API (fwbg-agents, dashboard)

- `GET /api/research`, `GET /api/research/{id}` — card, trials, coverage.
- `POST /api/research` — create card.
- `POST /api/research/{id}/status` — verdict or reopen; requires `actor`.

fwbg-agents integration (separate plan in the agents repo): the
Researcher queries `GET /api/research?tag=…` before proposing; its
`Strategy` row stores `research_id`; its `abandon` + post-mortem maps to a
research verdict.

### AI sessions

`CLAUDE.md` and `AGENTS.md` in fwbg get a rule: before proposing a test,
run `fwbg research list/show`; every run uses `--research-id` and
`--note`; study scripts call `register_trial()`. The rule contains no
results and can live in the public repo.

## Migration

Full backfill of all existing runs and studies:

1. **Index:** `fwbg research backfill --plan mapping.yaml` scans the
   workspace and the repo-local `test_results/`, computes fingerprints from
   the stored `strategy.json` and the run's symbols (same definition as new
   runs), and classifies runs without `results.json` as `failed` and agent
   smoke runs as `smoke`.
2. **Grouping:** an AI session groups runs into proposed cards (by name
   family and config similarity) and fills `mapping.yaml` (run → S-ID).
   Studies outside the run system get one card each, with trials counted as
   documented. The operator reviews the mapping before anything is written.
   No LLM runs inside fwbg.
3. **Apply:** `--apply mapping.yaml` writes cards and legacy trials.
   Strategies with a documented conclusion get that verdict; strategies
   without one become `parked` ("no conclusion documented"), never
   `rejected`.
4. **Legacy holdout:** cutoff = latest data end of the strategy's runs;
   data after it is a clean forward holdout. Strategies whose runs reach
   the present get `holdout_contaminated: true` — confirmation then needs
   forward data or paper trading.
5. **Repo cleanup:** move runs that exist only in the repo-local
   `test_results/` into the workspace; remove byte-identical duplicates
   after a diff check and operator confirmation; keep `test_results/`
   ignored.
6. **Reports:** study write-ups move to `research/reports/` and are linked
   from cards. Code plans stay in fwbg.

## Error handling

- Ledger append or research-repo commit fails after a successful run →
  warning, the run is not failed. `reconcile` first validates the ledger and
  detects register files changed since the last research-repo commit. It
  stages only the register paths (`strategies/`, `trials.jsonl`, `configs/`
  and `reports/`) and creates a recovery commit for that already-appended
  trial, leaving unrelated research-repo changes untouched. It then indexes
  any workspace runs missing from the ledger and commits those changes too.
  If validation or the recovery commit fails again, it leaves the data in
  place, reports the failure, and remains safe to retry.
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
- DSR: matches reference values from Bailey & López de Prado (2014).
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
3. **CLI gate** and `fwbg research new/list/show/check/amend/verdict/reopen`.
4. **API** gate and endpoints, `CLAUDE.md`/`AGENTS.md` rule,
   `register_trial()`.
5. **Holdout mode** and verdict statistics (DSR).
6. **fwbg-agents integration** (separate plan, agents repo).
7. **Dashboard page** (phase 2).

## Open questions

- **Holdout evaluation mechanics.** fwbg's walk-forward and nested CV
  split the requested window internally. A holdout trial must train on
  data up to the cutoff and report metrics only for the period after it.
  Needs investigation before step 5.
- **`/api/runs/preview`.** Counts as a trial only if it reports
  performance metrics; decide during step 4.
- **Legacy symbols.** Old runs record their symbols only indirectly
  (`grid_details/<SYMBOL>/`, `results.json`); the backfill plan must pick
  the reliable source.

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
