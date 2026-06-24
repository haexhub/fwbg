# fwbg-agents M5b — PluginAuthor + PluginEvaluator + API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status**: ✓ done 2026-06-24. Baseline → M5b: 187 → **234 tests green**. Final commits:

| Task | Commit | New tests |
|------|--------|-----------|
| 1 — Migration 0004 + PluginKind + verification_run | `35d41c4` | 7 |
| 2 — PluginAuthor agent + prompts + sidecar consumer | `aaed198` | 8 |
| 3 — PluginEvaluator + scenario_generators | `4cdc1bd` | 23 (17 generator + 6 evaluator) |
| 4 — plugin_flow + API + m5 smoke | `36aca54` | 9 |

`scripts/m5_smoke.py` was run against the dev DB and ended `[m5_smoke] PASSED` —
parent strategy seeded → POST author → POST evaluate → plugin VERIFIED →
2 parquets written, no error_log.json.

Deviations from the as-written plan:
- The `tests/scripts/test_m5_smoke.py` self-test was dropped — the existing
  M3 / M4 smokes have no pytest wrapper either; manual-only is the repo
  convention. The 9 API tests + the live `scripts/m5_smoke.py` run cover the
  same surface.
- Added `numpy>=2.1`, `pandas>=2.2`, `pyarrow>=18` to `pyproject.toml`
  (mid-Task-3 dependency gap surfaced — pandas is the natural DataFrame
  shape the plugins operate on, parquet is the locked-decision file format,
  pyarrow is pandas's parquet engine).

**Goal**: Close the plugin lifecycle from M5a's `add_indicator_request.json` sidecar to a `PluginState.VERIFIED` plugin on disk + DB — an LLM PluginAuthor writes `plugin.py + contract.yaml + spec.md` under `data/plugins/<slug>/v1/`, a deterministic PluginEvaluator runs the contract invariants against hand-curated synthetic scenarios, and HTTP endpoints expose both as background jobs.

**Architecture**:
1. Migration 0004 — extend `PluginKind` to all 8 fwbg categories (string column, additive) and add table `verification_run`.
2. PluginAuthor (LLM, pydantic-ai) — reads sidecar + parent strategy, fetches `n=3` (max 5) fwbg-plugin examples from the catalog, emits Python code + `PluginContract`, persists everything to `data/plugins/<slug>/v1/`, transitions Plugin `SPECIFIED → AUTHORED`.
3. PluginEvaluator (deterministic, no LLM) — loads `contract.yaml`, asks `scenario_generators.py` for hand-curated np-seeded OHLCV frames per scenario, parquet-writes them to `data/plugins/<slug>/v1/test_scenarios/`, dynamic-imports `plugin.py`, runs each scenario, asserts contract invariants. Full pass → `AUTHORED → VERIFIED` + `verification_run(status=passed)`. Any fail → stay `AUTHORED`, write structured JSON `error_log.json`, `verification_run(status=failed)`.
4. API: `POST /strategies/{id}/author-plugin`, `POST /plugins/{id}/evaluate`, `GET /plugins/{id}/verification-runs`. M3/M4 pattern: 202 + BackgroundTasks + AgentRun envelope.
5. `scripts/m5_smoke.py` — ASGI-transport end-to-end: pre-seed strategy with `add_indicator_request.json`, drive author → evaluate → asserts `VERIFIED` + all expected files.

**Locked decisions** (closed in session before plan-write):
- **error_log format**: structured JSON `{scenario_name, invariant_violated, traceback, ts}` — Dashboard-parseable later.
- **Author example budget**: default `n=3`, hard cap `n=5` (validator clamps). Typical fwbg-plugin ≈50–150 LOC → ~3×100=300 LOC of style context.
- **Scenarios layout**: per-plugin `data/plugins/<slug>/v1/test_scenarios/<name>.parquet`. Reproducible, inspectable post-mortem, no shared mutable state.
- **Failed verify policy**: plugin stays `AUTHORED`, manual retry. No counter, no auto-abandon (deferred to M8+).
- **Carries over from M5a**: PluginEvaluator deterministic-only in M5b; LLM-interpretation later. Hand-curated np-seeded scenarios — NO data-derived thresholds. Author writes ONLY to `data/plugins/...`, never to fwbg-repo. `v1/` fixed; revision flow deferred. PluginCatalog cache process-lifetime.

**Tech Stack**: pydantic v2, pydantic-ai (`FunctionModel` stubs in tests), SQLAlchemy 2.x async + Alembic, FastAPI + BackgroundTasks, numpy/pandas/pyarrow (synthetic OHLCV → parquet), pytest-asyncio, ASGI-transport (`httpx.AsyncClient`).

---

## Task 1 — Migration 0004 + PluginKind extension + verification_run

**Files:**
- Create: `src/fwbg_agents/persistence/migrations/versions/0004_plugin_kinds_and_verification.py`
- Modify: `src/fwbg_agents/persistence/models.py` (extend `PluginKind` enum, add `VerificationRun` model)
- Modify: `src/fwbg_agents/orchestrator/plugin_contract.py` (`PluginKindLit` becomes the canonical literal; refresh module docstring — the M5a "Literal duplicated" comment is now stale)
- Test: `tests/persistence/test_migration_0004.py` (alembic up/down)
- Test: `tests/persistence/test_verification_run_orm.py` (round-trip)
- Test: `tests/persistence/test_plugin_kind_enum.py` (8 values, backwards-compatible string values)

**Behaviour:**
- Migration 0004 is **additive only** — `kind` on `plugin` stays a 32-char string column; no DB enum constraint. The Python `PluginKind` enum extends to:
  ```python
  class PluginKind(str, enum.Enum):
      INDICATOR = "indicator"
      MODEL = "model"
      EXIT_STRATEGY = "exit_strategy"
      RISK_MANAGEMENT = "risk_management"
      ENTRY_MODIFIER = "entry_modifier"
      PREPROCESSING = "preprocessing"
      FEATURE_SELECTION = "feature_selection"
      DATA_LOADING = "data_loading"
  ```
  Old M3 rows that wrote `"indicator"` or `"exit"` remain valid; new code must NOT write `"exit"` anymore — use `"exit_strategy"`. Add a one-line `__post_init__`-style data-migration in 0004 that UPDATEs any existing `plugin.kind = 'exit'` rows to `'exit_strategy'` (safe; no production data yet, but the data-migration is correct hygiene).
- New table `verification_run`:
  ```python
  class VerificationRun(Base):
      __tablename__ = "verification_run"
      id: int (PK, autoinc)
      plugin_id: int (FK plugin.id, NOT NULL, indexed)
      status: str  # "running" | "passed" | "failed"
      scenarios_run: int (default 0)
      scenarios_passed: int (default 0)
      error_log_path: str | None
      started_at: datetime (NOT NULL)
      ended_at: datetime | None
      created_at: datetime (NOT NULL, indexed)
  ```
  No JSON payload column — the structured per-scenario error log lives on disk at `error_log_path`. The row itself is the index.
- `plugin_contract.py` keeps `PluginKindLit` as the single source of truth for the Literal type; the module docstring loses the "duplicated" hint. The `PluginKind` enum's `.value`s MUST match the Literal entries exactly — pin this with one test.

**Test sketch:**
1. `test_upgrade_0003_to_0004_creates_verification_run` — alembic upgrade head from 0003, then introspect `verification_run` columns + indices.
2. `test_downgrade_0004_drops_verification_run` — round-trip; drop_table verified.
3. `test_existing_exit_row_migrated_to_exit_strategy` — INSERT a `plugin(kind="exit")` at rev 0003, upgrade to 0004, assert `kind == "exit_strategy"`.
4. `test_verification_run_round_trip` — INSERT + SELECT; FK to plugin.id enforced; status accepts the 3 string values.
5. `test_plugin_kind_enum_values_match_plugin_contract_literal` — assert `{k.value for k in PluginKind} == set(get_args(PluginKindLit))`.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run alembic upgrade head
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/persistence/ -v
```
Commit: `feat(M5b): migration 0004 — PluginKind extension + verification_run table`

---

## Task 2 — PluginAuthor agent + sidecar consumer

**Files:**
- Create: `src/fwbg_agents/agents/plugin_author.py`
- Create: `src/fwbg_agents/agents/prompts/plugin_author.md`
- Modify: `src/fwbg_agents/orchestrator/lifecycle.py` (only if a `plugin_dir(slug)` helper is missing — graph shows it exists at L74; reuse, do NOT duplicate)
- Test: `tests/agents/test_plugin_author.py`

**Behaviour:**
- Agent class `PluginAuthor` mirrors the M4 Translator shape (pydantic-ai `Agent` instance + `.run_fresh()` async entry). Output type is `PluginAuthorResult`:
  ```python
  class PluginAuthorResult(BaseModel):
      slug: str                       # kebab-case; derived from capability if not LLM-provided
      python_code: str                # full plugin.py contents
      contract: PluginContract        # M5a schema, validated
      spec_md: str                    # short rationale + I/O note (≥80 chars)
  ```
- Tools exposed to the LLM:
  - `get_fwbg_plugin_examples(category: PluginKindLit, n: int = 3) -> list[FwbgPluginExample]` — reads up to `min(n, 5)` plugin source files from the existing `PluginCatalog` two-root discovery (fwbg-core + fwbg-premium). Returns `[FwbgPluginExample(slug, path, source: str (truncated at 4000 chars))]`. Hard-clamp at 5; values >5 are silently clamped (log a warning). NOT exposed as a tool in tests — tests use a `FunctionModel` stub that already knows the final answer, so the tool need not be called.
  - `validate_python_syntax(code: str) -> SyntaxCheck` — deterministic: `ast.parse(code)` → `SyntaxCheck(ok=True)` on success; on `SyntaxError`, return `ok=False, line, msg`. Pure Python, no LLM.
- `PluginAuthor.run_fresh(session, *, sidecar_path: Path, parent_strategy: Strategy, model: Model) -> int` (returns the new plugin `id`):
  1. Load sidecar JSON (capability, category, phase, reasoning, requested_at, strategy_slug, strategy_id).
  2. Load parent strategy.json from `strategy_dir(parent_strategy.slug) / "iteration_NNN" / "strategy.json"` (latest iteration). Pass excerpts into prompt.
  3. Run the LLM agent with that context + tools. Constrain output via the `PluginAuthorResult` pydantic-ai output schema.
  4. **Slug guard**: assert the returned `slug` does NOT already exist in `PluginCatalog.merge_with_db(session)`. If collision: raise `PluginAuthorFailed("slug already taken")`. Do NOT auto-suffix — the LLM picks again on retry (M8+ concern).
  5. Persist:
     - `data/plugins/<slug>/v1/plugin.py` — `python_code`
     - `data/plugins/<slug>/v1/contract.yaml` — via `dump_contract(contract, path)`
     - `data/plugins/<slug>/v1/spec.md` — `spec_md`
  6. DB: `INSERT Plugin(slug, kind=contract.kind, current_state=SPECIFIED, spec_path, contract_path, created_at, updated_at)`, flush to get `id`.
  7. `transition_plugin(plugin, PluginState.AUTHORED, reason="plugin_author", payload={"request_path": str(sidecar_path), "request_strategy_id": parent_strategy.id, "examples_count": n_examples_used}, created_by="plugin_author")`.
  8. Return `plugin.id`.
- `PluginAuthorFailed(RunnerFailed)` for slug collisions / contract validation / syntax check failures. No retry loop in M5b — single attempt. The caller surfaces the failure via the AgentRun envelope.
- The prompt (`prompts/plugin_author.md`) MUST:
  - Show the parent strategy's pipeline excerpt.
  - Show the `add_indicator_request` JSON.
  - Instruct: "use `get_fwbg_plugin_examples(category={category}, n=3)` to see the existing fwbg style; copy patterns, do not invent novel imports".
  - Instruct: "call `validate_python_syntax(code)` once before returning your result; if it fails, fix and retry".
  - Output contract: emit a single `PluginAuthorResult` via the final tool-call (pydantic-ai handles it).

**Test sketch** (FunctionModel-stub pattern — same as M4/M5a):
1. `test_author_writes_three_files_and_transitions_to_authored` — stub returns a valid `PluginAuthorResult`; assert `plugin.py`, `contract.yaml`, `spec.md` exist with the right content; plugin row at `AUTHORED`; transition row present with `from_state=SPECIFIED, to_state=AUTHORED, created_by="plugin_author"`.
2. `test_author_slug_collision_raises_and_no_files_written` — pre-seed a plugin with the same slug; stub returns colliding slug; assert `PluginAuthorFailed` AND `data/plugins/<slug>/v1/` does NOT exist.
3. `test_author_contract_kind_indicator_with_no_invariants_rejected` — stub returns a `PluginContract` with `kind="indicator"` and `invariants=[]`; pydantic-ai output validator should already reject it before we see the result. Assert the agent run raises and no DB row was created.
4. `test_author_spec_md_too_short_rejected` — pydantic constraint `min_length=80` on `spec_md`; tiny stub output rejected.
5. `test_get_fwbg_plugin_examples_clamps_above_5` — direct unit test of the tool: ask for `n=99`, get back ≤5 entries; assert the warning logged.
6. `test_validate_python_syntax_returns_line_on_error` — pure deterministic; `code="def x(:\n"` → `ok=False, line=1`.
7. `test_author_persists_payload_with_request_paths` — after run, the transition.payload JSON includes `request_path` ending in `add_indicator_request.json` AND `request_strategy_id == parent_strategy.id`.
8. `test_author_idempotent_per_attempt_is_NOT_a_test` — explicitly skipped: M5b is single-attempt. Document the design choice in a `# NOTE` comment in the agent module.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/agents/test_plugin_author.py -v
```
Commit: `feat(M5b): PluginAuthor agent writes plugin.py + contract + spec and transitions to AUTHORED`

---

## Task 3 — PluginEvaluator + scenario_generators

**Files:**
- Create: `src/fwbg_agents/orchestrator/scenario_generators.py`
- Create: `src/fwbg_agents/agents/plugin_evaluator.py` (despite the `agents/` location it is deterministic — same convention as `agents/runner.py`)
- Test: `tests/orchestrator/test_scenario_generators.py`
- Test: `tests/agents/test_plugin_evaluator.py`

**Behaviour:**
- `scenario_generators.py` exposes one registry:
  ```python
  SCENARIO_GENERATORS: dict[str, Callable[[int], pd.DataFrame]] = {
      "trending_up": gen_trending_up,      # 500 bars, linear drift + low vola
      "trending_down": gen_trending_down,  # 500 bars, mirror of trending_up
      "sideways": gen_sideways,            # 500 bars, mean-reverting noise
      "high_vola": gen_high_vola,          # 500 bars, σ ~3x the others
      "sparse_data": gen_sparse_data,      # 80 bars, gaps every ~10 bars
  }
  ```
  Each generator takes a `seed: int` (default fixed per name — e.g. trending_up uses `seed=20260624`) and returns a 6-column OHLCV DataFrame: `["timestamp", "open", "high", "low", "close", "volume"]`. `timestamp` is monotonic UTC `pd.Timestamp`, 1-minute spacing. Internal randomness via `np.random.default_rng(seed)` — NEVER `np.random.seed` (process-global).
  HARD RULE (from [[feedback-no-data-derived-thresholds]]): no parameter inside any generator is derived from real market data. Every constant is hand-curated and commented.
- `PluginEvaluator.run(session, plugin) -> int` (returns `verification_run.id`):
  1. INSERT `VerificationRun(plugin_id, status="running", scenarios_run=0, scenarios_passed=0, started_at=now)`, flush, capture `vr.id`.
  2. Load `contract = load_contract(plugin.contract_path)`.
  3. For each entry in `contract.test_scenarios`:
     - Look up generator by `scenario.name`. If unknown name → record as failed (this scenario, but continue the loop with the rest? — decision: **abort the whole evaluation** with `verification_run.status="failed"` and a single `error_log` entry `{scenario_name, invariant_violated: "unknown_scenario", traceback: None}`. Rationale: contract.yaml declared a scenario the system can't materialise — that's a contract bug, not a plugin bug; stay strict).
     - Materialise data → write to `data/plugins/<slug>/v1/test_scenarios/<scenario.name>.parquet` (overwrites if present — generation is deterministic per seed).
     - Dynamic-import the plugin: `spec = importlib.util.spec_from_file_location(...); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)`. Look up callable `mod.compute` (M5b convention — document in prompt & contract example). If absent → record as failed (this plugin instance, see below).
     - Call `mod.compute(df, **{p.name: p.default for p in contract.params})` with the scenario DataFrame.
     - Invariant evaluation (M5b minimal set — explicit, no DSL):
       - `"outputs[*].length == inputs[0].length"` — check each output column length matches the input row count, gated by `length_invariant="same_as_input"`.
       - `"outputs[*].dtype matches contract.outputs[i].dtype"` — series → numeric; boolean_series → bool; scalar → 0-D.
       - `"no NaN beyond first warmup window"` — warmup heuristic: largest `params.*.default` that's an int >0 (window-length proxy). After that, NaN is forbidden.
       Each invariant violation produces one structured JSON entry. `contract.invariants` strings are free-text and NOT evaluated as code in M5b — they're advisory documentation. The three above are the hard-coded, evaluated set.
     - Increment `vr.scenarios_run`; if all 3 invariants pass: `vr.scenarios_passed += 1`. Otherwise append entry to `errors: list[dict]`.
  4. Outcomes:
     - `vr.scenarios_passed == vr.scenarios_run AND vr.scenarios_run > 0` →
       `vr.status = "passed"`, `vr.ended_at = now`, then `await transition_plugin(plugin, PluginState.VERIFIED, reason="plugin_evaluator", payload={"verification_run_id": vr.id, "scenarios_passed": vr.scenarios_passed}, created_by="plugin_evaluator")`.
     - Anything else → `vr.status = "failed"`, `vr.ended_at = now`. Write `data/plugins/<slug>/v1/error_log.json` with `{"verification_run_id": vr.id, "errors": [...]}` (overwrites prior; only the latest run's log is kept on disk — older runs are referenced by `verification_run.error_log_path` snapshot, but we accept the lossy overwrite in M5b for simplicity). `vr.error_log_path = str(path)`. **No transition** — plugin stays AUTHORED.
  5. Commit. Return `vr.id`.
- Edge-case explicit: `contract.test_scenarios == []` is permitted by M5a's contract schema (`test_empty_scenarios_allowed`). For M5b: empty scenarios → `vr.status = "failed"` with one error `{scenario_name: "", invariant_violated: "no_scenarios_declared", ...}`. The plugin stays AUTHORED. Future revisions can opt-in to "zero-scenarios = trust" once we have a tagging story.

**Test sketch:**
1. `test_each_generator_returns_deterministic_frame` — call each generator twice with the same seed; assert frames `equals` row-for-row; assert column set; assert dtype('timestamp') is datetime64.
2. `test_sparse_data_has_gaps` — `sparse_data` generator: assert at least 5 timestamps with diff > 60s.
3. `test_evaluator_happy_path_passes_and_transitions_to_verified` — seed a Plugin@AUTHORED with a trivial contract `kind="indicator"` (e.g. moving-average wrapper) + 2 scenarios; run evaluator; assert `vr.status=="passed"`, `plugin.current_state=="verified"`, transition row exists, 2 parquet files written.
4. `test_evaluator_length_mismatch_stays_authored` — plugin returns output of wrong length; assert `vr.status=="failed"`, `plugin.current_state=="authored"` unchanged, `error_log.json` exists with one `invariant_violated="length_mismatch"` entry.
5. `test_evaluator_unknown_scenario_name_fails_whole_run` — contract references `"made_up_scenario"`; assert `vr.status=="failed"`, `errors[0].invariant_violated == "unknown_scenario"`.
6. `test_evaluator_empty_scenarios_fails` — contract with `test_scenarios=[]`; assert `vr.status=="failed"`, `errors[0].invariant_violated == "no_scenarios_declared"`.
7. `test_evaluator_no_compute_callable_fails` — plugin.py without `compute()`; assert `vr.status=="failed"`, error entry mentions `compute`.
8. `test_evaluator_overwrites_previous_error_log` — run evaluator twice on the same failed plugin; assert only the second run's errors are in `error_log.json`.
9. `test_evaluator_records_started_and_ended_at` — assert both timestamps populated, `ended_at >= started_at`.
10. `test_evaluator_zero_scenarios_passed_zero_scenarios_run` — degenerate case (contract has 1 scenario, generator throws on materialisation) — assert `scenarios_run=0` is possible and the run still ends with `status=failed` (NOT crashed).

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_scenario_generators.py tests/agents/test_plugin_evaluator.py -v
```
Commit: `feat(M5b): PluginEvaluator + deterministic scenario_generators with structured error_log`

---

## Task 4 — API + plugin_flow.py + smoke

**Files:**
- Create: `src/fwbg_agents/orchestrator/plugin_flow.py`
- Modify: `src/fwbg_agents/api/plugins.py` (existing M3 file — extend with three new endpoints; keep existing detail/transitions endpoints intact)
- Create: `scripts/m5_smoke.py`
- Test: `tests/api/test_plugin_flow_api.py`
- Test: `tests/scripts/test_m5_smoke.py` (self-test that imports the smoke script and runs it through ASGI-transport)

**Behaviour:**
- `plugin_flow.py` orchestrators (mirrors M4 `research_flow.research_and_translate` shape):
  - `async def author_plugin_from_strategy(session, strategy_id: int, *, model: Model) -> int` —
    - SELECT strategy. If `current_state != BACKTESTED` → raise `ReiteratePreconditionError`-style 4xx. (Use a fresh exception class `AuthorPluginPreconditionError` mirroring M4 conventions.)
    - Locate latest sidecar at `strategy_dir(slug) / "iteration_NNN" / "add_indicator_request.json"`. Highest NNN. None found → raise.
    - Load parent strategy, instantiate `PluginAuthor`, call `.run_fresh(session, sidecar_path=..., parent_strategy=..., model=model)`. Return `plugin_id`.
  - `async def evaluate_plugin(session, plugin_id: int) -> int` —
    - SELECT plugin. If `current_state != AUTHORED` → raise `EvaluatePluginPreconditionError`.
    - Instantiate `PluginEvaluator`, call `.run(session, plugin)`. Return `verification_run_id`.
- API endpoints (all M3/M4 envelope: 202 Accepted + `AgentRun` row + BackgroundTasks job that finalises the run):
  - `POST /strategies/{strategy_id}/author-plugin` —
    - 404 if strategy missing.
    - 422 if precondition fails (wrong state, no sidecar).
    - 202: INSERT `AgentRun(agent_name="plugin_author", strategy_id, input_artifact_path=<sidecar>, status="running")`, schedule `_run_author_background(agent_run_id, strategy_id, model_id)`, return the `AgentRun` envelope.
  - `POST /plugins/{plugin_id}/evaluate` —
    - 404 if plugin missing.
    - 422 if `current_state != "authored"`.
    - 202: INSERT `AgentRun(agent_name="plugin_evaluator", plugin_id, status="running")`, schedule `_run_evaluator_background(agent_run_id, plugin_id)`, return envelope.
  - `GET /plugins/{plugin_id}/verification-runs` — list (newest first) all rows; serialize id, status, scenarios_run, scenarios_passed, error_log_path, started_at, ended_at, created_at. Synchronous (no background).
- Background helpers `_run_author_background` / `_run_evaluator_background` share the same M3/M4 finalise-pattern: try → call orchestrator → set `agent_run.status="done"`, `output_artifact_path` = contract or error_log path, `ended_at=now`. Except → `status="failed"`, `error=str(exc)`. Always commit.
- `scripts/m5_smoke.py` — exact mirror of `scripts/m4_smoke.py` style:
  1. Spin up app via `httpx.AsyncClient(transport=ASGITransport(app=app))`.
  2. `POST /strategies` for a BACKTESTED-eligible parent + fast-forward it through the M2 transitions (or use a fixture helper that does so).
  3. Pre-seed `data/strategies/<slug>/iteration_001/add_indicator_request.json` with a deterministic synthetic body (capability="14-period RSI wrapper", category="indicator", phase="indicators", confidence=0.8, reasoning="smoke synthetic").
  4. `POST /strategies/{id}/author-plugin` → assert 202, poll the AgentRun until done, capture `plugin_id` from final transition payload (or via `GET /plugins?slug=...`).
  5. `POST /plugins/{plugin_id}/evaluate` → assert 202, poll until done.
  6. `GET /plugins/{plugin_id}/verification-runs` → assert one row, `status=="passed"`.
  7. Final assertions: `plugin.current_state == "verified"`, files exist (`plugin.py`, `contract.yaml`, `spec.md`, ≥1 parquet under `test_scenarios/`), NO `error_log.json` (passed runs don't write one).
  - The smoke uses a `FunctionModel` stub for PluginAuthor (no haex-claude-proxy dependency for the smoke; the smoke is for control-flow, not LLM-quality). A second "live" smoke variant with the real model is an M5b-stretch that we deliberately do NOT add — gated by a `--with-llm` flag the script ignores in M5b.

**Test sketch:**
1. `test_post_author_plugin_returns_202_with_agent_run` — strategy in BACKTESTED + sidecar present → 202; response has `agent_run.id, status="running"`. Background task finishes → poll → `status=="done"`, `plugin.current_state=="authored"`.
2. `test_post_author_plugin_422_when_no_sidecar` — strategy in BACKTESTED but `add_indicator_request.json` missing → 422 with message naming the missing file.
3. `test_post_author_plugin_422_when_wrong_state` — strategy in PROPOSED → 422.
4. `test_post_evaluate_202_then_verified` — plugin in AUTHORED, contract uses scenarios our generators support, trivial passing plugin.py → 202; polling shows `agent_run.status="done"`; plugin in VERIFIED; one verification_run row with `status="passed"`.
5. `test_post_evaluate_422_when_not_authored` — plugin in SPECIFIED → 422.
6. `test_post_evaluate_failed_plugin_stays_authored` — plugin.py that returns wrong-length output → 202 still; polling shows agent_run done (the evaluation completed even though it failed); plugin still in AUTHORED; verification_run row with `status="failed"` + non-null `error_log_path`.
7. `test_get_verification_runs_returns_newest_first` — three runs (passed, failed, failed) for one plugin → list[0].status == "failed" (most recent), list[-1].status == "passed".
8. `test_m5_smoke_self_test` — import + run `scripts.m5_smoke.main()`, assert it returns 0; sandbox with `tmp_path` for `data/`.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/api/test_plugin_flow_api.py tests/scripts/test_m5_smoke.py -v
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run python scripts/m5_smoke.py
```
Expected smoke output ends with `[m5_smoke] PASSED`.
Commit: `feat(M5b): plugin_flow API endpoints + m5 smoke end-to-end`

---

## End-of-session housekeeping

1. Full pytest: `VIRTUAL_ENV= uv run pytest -q` → target ~217 passed (187 baseline + ~5 + ~8 + ~10 + ~7).
2. `alembic upgrade head` → at 0004.
3. `python -c "from fwbg_agents.main import app"` → app importable.
4. `graphify update .` in `~/Projekte/fwbg-agents/` to refresh.
5. Update design-doc implementation status table: M5b row with the four final commit hashes.
6. Update this plan's status header to `✓ done`.
7. Update `project_fwbg_agents.md` memory: M5b-block, final commits, test count.
8. Create `reference_fwbg_agents_m5b_plan.md` memory: commit table + locked-decisions reference.
9. Add MEMORY.md index lines for the new reference.

---

## M6 preview (NEXT session — do NOT implement now)

The M5b deliverable closes the plugin lifecycle from SPECIFIED → VERIFIED. M6 (per design doc roadmap) is the **Reiterate-with-plugin** loop: once a plugin VERIFIES, the parent strategy's next iteration should be able to reference it. Sketch only:

- `reiterate_with_plugin(session, strategy_id, plugin_slug)` — clones the strategy's latest spec, swaps in the new plugin slug at the requested phase, re-runs Translator validation against the catalog, increments `iteration_count`, kicks Runner.
- API: `POST /strategies/{id}/reiterate-with-plugin` `{plugin_slug}`.
- No new ORM. Reuses M4 reiterate machinery.
- Smoke variant: full circle `add_indicator → PluginAuthor → PluginEvaluator → reiterate → Runner → Analyst` would be the M6 acceptance test.
