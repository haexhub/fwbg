# fwbg-agents M5d — Planner/Implementer Split for Plugin-Authoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** ✓ done 2026-06-25 — seven commits `24b8be0`, `51c16c5`, `46b21c5`, `9a73136`, `1cefb65`, `979cb23`, `d85ad1c` in `~/Projekte/fwbg-agents/` on branch `feat/m5d-planner-implementer-split`. 371 tests green (323 → +48 net after retiring 8 M5b plugin_author tests). `scripts/m5d_smoke.py` self-test ends `[m5d_smoke] PASSED`. M5c smoke also still green (M5d Planner→Implementer drives the M5c reiterate flow via factory monkey-patches).

**Goal:** Split the M5b `PluginAuthor` into two distinct LLM agents — `PluginPlanner` (stronger model: `claude-opus-4-8`) emits a structured `PluginPlan`; `PluginImplementer` (weaker model: `claude-opus-4-7`) writes the actual Python module from that plan, with a deterministic refinement loop bounded by syntax+contract gates and a configurable max-rounds budget. Same external API envelope as M5b (`POST /strategies/{id}/author-plugin`), no change to M5c/M6b consumers.

**Architecture:**
- **PluginPlanner** runs exactly once per author-session. Reads `AnalystRecommendation` (AddIndicator-sidecar) + parent `strategy.json` + `PluginCatalog`-excerpt + the canonical `prompts/plugin_authoring.md` conventions doc. Emits a `PluginPlan` (slug, class_name, phase, params, feature_columns, algorithm_sketch, edge_cases, expected_test_names). One `AgentRun(kind="plugin_plan")` row.
- **PluginImplementer** runs up to `PLUGIN_IMPL_MAX_ROUNDS` times. Each round consumes the `PluginPlan` + the previous-round's code + the previous-round's gate error (if any). Emits `PluginAuthorResult` (slug, python_code, contract, spec_md) — identical shape to M5b so M5c-downstream stays untouched. One `AgentRun(kind="plugin_implement")` row with N `LlmCall` children, one per round.
- **Gate loop (deterministic, no LLM):** `validate_python_syntax(code)` → if fail, feed back to Implementer. Else `load_contract_from_code(code)` (M5a's `PluginContract` loader). If contract OK → persist Plugin → return. After N failed rounds → `PluginAuthorFailed("gates still failing after N rounds")`.
- **Persistence:** Plan-JSON written to `data/plugin-runs/<slug>/plan.json` (audit-trail, analog to `add_indicator_request.json`). Plugin row + plugin.py + tests.py + spec.md written on success exactly as in M5b. On budget-exhaust: both AgentRuns marked FAILED, no Plugin row created, last attempted code kept in the implement-AgentRun's `error_message` for debugging.
- **Settings:** `pydantic-settings`-based `AgentModels` class reads env vars `PLUGIN_PLANNER_MODEL` (default `claude-opus-4-8`), `PLUGIN_IMPLEMENTER_MODEL` (default `claude-opus-4-7`), `PLUGIN_IMPL_MAX_ROUNDS` (default `5`). Test-overridable via `monkeypatch.setenv()`.
- **Prompt-doc:** `fwbg-agents/prompts/plugin_authoring.md` is the canonical fwbg-Plugin-Konventionen doc, loaded into the Planner's system-prompt at runtime. Versioned with the code; updates require no agent-code change.

**Tech Stack:** Python 3.13, pydantic-ai (Planner+Implementer both pydantic-ai agents with `Model` override per env), `pydantic-settings` (NEW dep), SQLAlchemy 2.x async, FastAPI, pytest. No alembic migration needed (AgentRun.kind is already a free string).

**Locked Decisions** (from brainstorming session 2026-06-25):
- (A) **Refinement-Loop ja, deterministische Gates.** Loop endet bei (Syntax OK ∧ Contract OK) ODER bei `max_rounds`. Kein LLM-Self-Judgment.
- (B) **Failure-Routing: Implementer fixt.** Planner läuft genau einmal. Bei Gate-Fail bekommt der Implementer den Plan + last_code + last_error und versucht erneut. Begründung: Reasoning-Plan ist meist OK, Syntax/Naming/Imports schief.
- (C) **`max_rounds=5` default**, konfigurierbar via env.
- (D) **`PluginPlan`-Schema aus `BasePlugin`-Contract abgeleitet** — `PluginPhase`-SDK-Enum, `ParamSpec`-Liste mit Type/Default/Description/min/max/step/choices.
- (E) **Prompt-Doc lebt in `fwbg-agents/prompts/plugin_authoring.md`** — kein globaler Claude-Skill, keine Code-Duplikation.
- (F) **Kein Backwards-Compat-Fallback.** Alte `PluginAuthor`-Klasse wird ersetzt; M5b-Tests werden gesplittet. Single-path-Code.
- (G) **1 AgentRun pro logischer Phase** (plan + implement), N LlmCalls als Children unter dem Implement-Run. Retry-Loop ist Implementierungsdetail, nicht User-Visible-Operation.
- (H) **Concrete-only für Plugin-Authoring.** Keine Generalisierung auf Researcher/Analyst/Translator-Per-Agent-Model-Config in dieser Iteration. Wenn 2-3 weitere Use-Cases entstehen → eigenes Mini-Refactor.

**PluginPlan-Schema:**
```python
class ParamSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    type: Literal["int", "float", "bool", "string",
                  "list[int]", "list[float]", "list[string]", "choice"]
    default: int | float | bool | str | list | None
    description: str = Field(min_length=1)
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None
    choices: list[str] | None = None
    required: bool = True

class PluginPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slug: str = Field(min_length=2, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    class_name: str = Field(pattern=r"^[A-Z][A-Za-z0-9]+$")
    phase: PluginPhase
    version: str = "0.1.0"
    stateful: bool = False
    depends_on: list[str] = []
    params: list[ParamSpec]
    feature_columns: list[str] = Field(min_length=1)
    algorithm_sketch: str = Field(min_length=120)
    edge_cases: list[str] = Field(min_length=1)
    expected_test_names: list[str] = Field(min_length=3)
```

**Phase-Mapping (AnalystRecommendation.phase → PluginPhase SDK enum):**
```
"indicator"          → PluginPhase.INDICATORS
"feature_selection"  → PluginPhase.FEATURE_SELECTION
"preprocessing"      → PluginPhase.PREPROCESSING
"filter"             → PluginPhase.RISK_MANAGEMENT
```
(Filters live under `risk_management` in the SDK enum — confirmed via SDK `PluginPhase` definition. Mapping stays explicit, not derived.)

**Pre-checks (verify at session start):**
- `cd ~/Projekte/fwbg-agents && git log -1 --format="%H"` = `eccb07e...` (M6b final)
- `VIRTUAL_ENV= uv run pytest -q` = `323 passed`
- `VIRTUAL_ENV= uv run alembic current` = `0004 (head)` (no migration needed for M5d)

---

## Task 1 — `AgentModels` settings module (env-driven, pydantic-settings)

**Files:**
- New: `src/fwbg_agents/settings.py`
- Modify: `pyproject.toml` (+ `pydantic-settings` dep)
- New: `tests/orchestrator/test_settings.py`

**Behaviour:**
- `AgentModels(BaseSettings)` with three fields: `plugin_planner_model: str = "claude-opus-4-8"`, `plugin_implementer_model: str = "claude-opus-4-7"`, `plugin_impl_max_rounds: int = Field(default=5, ge=1, le=20)`.
- Env-prefix `model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")`. Env vars read uppercase: `PLUGIN_PLANNER_MODEL`, `PLUGIN_IMPLEMENTER_MODEL`, `PLUGIN_IMPL_MAX_ROUNDS`.
- Module-level singleton `agent_models = AgentModels()` for non-test callers. Tests import `AgentModels` and instantiate per-test.

**Test sketch:**
1. `test_defaults_when_no_env` — fresh `AgentModels()` returns the documented defaults.
2. `test_env_override_planner_model` — `monkeypatch.setenv("PLUGIN_PLANNER_MODEL", "claude-sonnet-4-6")` → instance picks it up.
3. `test_max_rounds_must_be_in_range` — `setenv("PLUGIN_IMPL_MAX_ROUNDS", "0")` → `ValidationError`. `"21"` → error. `"1"` and `"20"` → OK.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_settings.py -v
```

**Commit:** `feat(M5d): AgentModels pydantic-settings for per-agent model + max-rounds config`

---

## Task 2 — Canonical `prompts/plugin_authoring.md`

**Files:**
- New: `prompts/plugin_authoring.md` (top-level in `fwbg-agents`)
- New: `tests/agents/test_plugin_authoring_prompt.py` (smoke: file exists, sections present)

**Content sections (mandatory headings, used as regex anchors in the smoke-test):**
1. `# BasePlugin Contract` — class attrs (`name`, `phase`, optional `version`, `stateful`, `cacheable`, `depends_on`), `__init_subclass__` validation, lifecycle methods.
2. `# Phase-Specific Subclasses` — `BaseIndicator.compute()`, `BaseFeatureSelector.select()`, `BasePreprocessor.transform()`, `BaseRiskManager.evaluate()` signatures + return contracts. Pulled verbatim from SDK module docstrings.
3. `# Parameters: get_default_params + get_param_schema` — exact dict shape, allowed type strings, when to use `choices`.
4. `# Feature Columns: get_feature_columns` — naming rules (snake_case, plugin-name prefix recommended, no whitespace, must match columns actually produced by `compute()`/`transform()`).
5. `# Tests Convention (tests.py)` — minimum 3 tests, `test_<behaviour>` naming, no-lookahead pattern for indicators (assert `df[col].iloc[i]` only depends on `df.iloc[:i+1]`), use `pytest` not unittest.
6. `# File Layout` — `<slug>/__init__.py` (contains the plugin class), `<slug>/tests.py`, optional `<slug>/docs/README.md`.
7. `# Worked Examples` — points to `get_fwbg_plugin_examples(catalog, category=..., n=3)` (existing M5b helper) for runtime example injection. The doc lists slugs of 2 high-quality reference plugins per phase.

**Test sketch:**
1. `test_prompt_file_exists_at_canonical_path`.
2. `test_all_seven_headings_present` — read file, assert each `^# <heading>$` regex matches.
3. `test_no_phase_enum_drift` — parse the markdown, find the `PluginPhase` enum names mentioned, assert every name is a member of `fwbg_sdk.base.PluginPhase` (catches doc-drift after SDK enum changes).

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/agents/test_plugin_authoring_prompt.py -v
```

**Commit:** `docs(M5d): prompts/plugin_authoring.md canonical fwbg-Plugin-Konventionen for PluginPlanner`

---

## Task 3 — `PluginPlanner` agent

**Files:**
- New: `src/fwbg_agents/agents/plugin_planner.py`
- New: `tests/agents/test_plugin_planner.py`

**Behaviour:**
- pydantic-ai `Agent[None, PluginPlan]`. Model resolved at call time from `AgentModels().plugin_planner_model`.
- System prompt = static preamble + the contents of `prompts/plugin_authoring.md` (loaded via `importlib.resources` or `Path(__file__).parents[2] / "prompts" / "plugin_authoring.md"`).
- `run_plan(strategy, sidecar, catalog) -> PluginPlan` (async). Inputs:
  - `strategy: Strategy` — parent (BACKTESTED) — render excerpt of `strategy.json` (keys: name, pipeline, model, filters, validation, exit_strategies).
  - `sidecar: dict` — `AnalystRecommendation`-AddIndicator payload (phase, capability, rationale).
  - `catalog: PluginCatalog` — for `get_fwbg_plugin_examples(catalog, category=phase, n=3)` runtime examples.
- Validation after LLM call:
  - `plan.phase` must match `_PHASE_MAPPING[sidecar["phase"]]`. Mismatch → `PluginPlannerFailed("phase mismatch: sidecar says X, plan says Y")`.
  - `plan.slug` must NOT collide with existing catalog slugs. Collision → `PluginPlannerFailed("slug collision: X already exists")`.
  - All `ParamSpec` instances passed pydantic frozen-validation by virtue of model decode.
- Persist plan JSON to `data/plugin-runs/<plan.slug>/plan.json` immediately on success (audit). Directory created if missing.
- Raise `PluginPlannerFailed` (new class, extends `RuntimeError`) on all internal errors. Caller (`plugin_flow.author_plugin`) wraps in AgentRun status updates.

**Test sketch (uses `FunctionModel` for deterministic LLM responses):**
1. `test_planner_happy_path_indicator_phase` — sidecar phase="indicator" → plan.phase=INDICATORS, slug valid, params non-empty, all required schema fields present.
2. `test_planner_raises_on_phase_mismatch` — FunctionModel returns plan with phase=PREPROCESSING but sidecar phase="indicator" → `PluginPlannerFailed`.
3. `test_planner_raises_on_slug_collision` — catalog contains slug "x"; FunctionModel returns plan.slug="x" → `PluginPlannerFailed`.
4. `test_planner_writes_plan_json` — happy-path, then read `data/plugin-runs/<slug>/plan.json`, assert it round-trips into `PluginPlan`.
5. `test_planner_loads_authoring_prompt_doc` — assert that `prompts/plugin_authoring.md` content appears in `agent.system_prompt` (regex check on known heading).
6. `test_planner_model_from_env` — `monkeypatch.setenv("PLUGIN_PLANNER_MODEL", "claude-sonnet-4-6")` → `Planner._model_name == "claude-sonnet-4-6"`.
7. `test_planner_min_param_validations` — FunctionModel returns plan with `feature_columns=[]` → pydantic ValidationError surfaces as `PluginPlannerFailed`.
8. `test_planner_catalog_examples_injected` — assert that 2-3 sample plugins from `get_fwbg_plugin_examples` are present in the user-prompt body.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/agents/test_plugin_planner.py -v
```

**Commit:** `feat(M5d): PluginPlanner agent (opus-4-8 default) emits PluginPlan from sidecar + parent strategy`

---

## Task 4 — `PluginImplementer` agent with gate-loop

**Files:**
- New: `src/fwbg_agents/agents/plugin_implementer.py`
- New: `tests/agents/test_plugin_implementer.py`

**Behaviour:**
- pydantic-ai `Agent[None, PluginAuthorResult]`. Model resolved at call time from `AgentModels().plugin_implementer_model`.
- System prompt: short preamble ("You are writing a Python module that implements the given PluginPlan against the fwbg BasePlugin contract. Reuse the conventions from the plan. Tests must follow the listed expected_test_names.")
- `run_implement(plan, max_rounds) -> PluginAuthorResult` (async). Internal loop:
  ```python
  last_code: str | None = None
  last_err: str | None = None
  for round_idx in range(1, max_rounds + 1):
      user_prompt = render_prompt(plan, last_code, last_err, round_idx)
      result = await self._agent.run(user_prompt)
      code = result.output.python_code
      syntax = validate_python_syntax(code)
      if not syntax.ok:
          last_code, last_err = code, f"SyntaxError L{syntax.line}: {syntax.msg}"
          continue
      contract = try_load_contract_from_code(code)
      if not contract.ok:
          last_code, last_err = code, f"ContractError: {contract.msg}"
          continue
      return result.output
  raise PluginImplementerFailed(
      f"gates still failing after {max_rounds} rounds",
      last_code=last_code,
      last_err=last_err,
  )
  ```
- `PluginImplementerFailed` carries `last_code` and `last_err` attrs so the orchestrator can stash them in the AgentRun's `error_message`/`output_artifact_path` for post-mortem.
- `try_load_contract_from_code(code)` — exec the code in a sandbox dict, find the `BasePlugin` subclass, instantiate, assert `.phase`/`.name`/`get_default_params()` callable. Reuse M5a's `load_contract` if possible — likely refactor it to accept a code-string in addition to a Path.

**Test sketch (FunctionModel):**
1. `test_implementer_happy_path_first_round` — FunctionModel returns valid code first try → returns `PluginAuthorResult`, no retries logged.
2. `test_implementer_recovers_from_syntax_error` — FunctionModel returns broken code round 1, valid code round 2 → succeeds at round 2.
3. `test_implementer_recovers_from_contract_error` — code has wrong `phase` enum or missing `name` → round 1 fails contract, round 2 OK.
4. `test_implementer_exhausts_max_rounds` — FunctionModel always returns broken code, max_rounds=3 → `PluginImplementerFailed` with `last_code` and `last_err` set.
5. `test_implementer_last_err_in_round_prompt` — capture FunctionModel call sequence, assert round-2 prompt contains the round-1 error string.
6. `test_implementer_last_code_in_round_prompt` — round-2 prompt contains the round-1 code (truncated if very long).
7. `test_implementer_respects_max_rounds_env` — `setenv("PLUGIN_IMPL_MAX_ROUNDS", "2")` → loop caps at 2.
8. `test_implementer_model_from_env` — `setenv("PLUGIN_IMPLEMENTER_MODEL", ...)` → uses overridden model.
9. `test_implementer_no_lookahead_on_round_counter` — FunctionModel succeeds at round 1, assert exactly 1 `LlmCall`-equivalent invocation (counter doesn't pre-allocate budget).
10. `test_implementer_syntax_error_preserves_class_name_in_feedback` — error message includes filename or class hint, not just bare line number.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/agents/test_plugin_implementer.py -v
```

**Commit:** `feat(M5d): PluginImplementer agent (opus-4-7 default) with deterministic gate-loop max_rounds=5`

---

## Task 5 — Orchestrator wiring: `author_plugin()` flow

**Files:**
- Modify: `src/fwbg_agents/orchestrator/plugin_flow.py` — rewrite `author_plugin_from_strategy()` to use Planner→Implementer pipeline.
- Delete (after migration): `src/fwbg_agents/agents/plugin_author.py` (the old M5b single-agent class). Keep `get_fwbg_plugin_examples`, `validate_python_syntax`, `_read_plugin_source`, `FwbgPluginExample` — move to a new `src/fwbg_agents/agents/plugin_authoring_shared.py` module so Planner+Implementer share them without circular import.
- Modify: `src/fwbg_agents/api/plugins.py` — only if precondition error types changed (likely just import rename).
- Modify: `tests/agents/test_plugin_author.py` — split into `test_plugin_planner.py` + `test_plugin_implementer.py` (already covered in Tasks 3+4); this file shrinks to test the joined-flow integration: planner runs, implementer runs, plugin row created.
- New: `tests/orchestrator/test_plugin_flow_split.py` — full flow with mocked Planner + Implementer.

**Behaviour of `author_plugin_from_strategy(strategy_id, session) -> Plugin`:**
1. Precondition checks (existing M5b): strategy state ∈ {BACKTESTED}, sidecar exists, catalog loaded. Raise `AuthorPluginPreconditionError` as today.
2. Insert `AgentRun(kind="plugin_plan", strategy_id=parent.id, status=RUNNING, started_at=now)`. Commit.
3. Call `PluginPlanner().run_plan(parent, sidecar, catalog)`. On success: AgentRun status=SUCCESS, finished_at, output_artifact_path=plan.json. On `PluginPlannerFailed`: status=FAILED, error_message, re-raise.
4. Insert `AgentRun(kind="plugin_implement", strategy_id=parent.id, status=RUNNING, started_at=now, input_artifact_path=plan.json-path)`. Commit.
5. Call `PluginImplementer().run_implement(plan, max_rounds=agent_models.plugin_impl_max_rounds)`.
   - For each LLM round, persist one `LlmCall(agent_run_id=impl_run.id, request_payload=..., response_payload=..., model=..., tokens=..., ms=...)` row. This is the "N children under one AgentRun" pattern.
   - On success: AgentRun status=SUCCESS.
   - On `PluginImplementerFailed`: status=FAILED, error_message=last_err, output_artifact_path stores last_code (debug aid). Re-raise as `PluginAuthorFailed` for API-level handling.
6. On Implementer success: write plugin.py + tests.py + spec.md + contract.yaml to `data/plugin-runs/<slug>/`, insert `Plugin(state=PROPOSED, slug=plan.slug, agent_run_id=impl_run.id, ...)`, insert `Transition(plugin_id, None→PROPOSED, ...)`. Single commit.

**Test sketch:**
1. `test_full_flow_creates_two_agent_runs_and_plugin` — mock Planner+Implementer to succeed, assert: 2 AgentRuns (plan SUCCESS + implement SUCCESS), 1 Plugin row, plugin.py on disk, plan.json on disk.
2. `test_implementer_failure_marks_both_runs_correctly` — Planner OK, Implementer raises after 3 rounds. Assert: plan-run SUCCESS, implement-run FAILED with last_err in error_message, NO Plugin row created.
3. `test_planner_failure_short_circuits_implementer` — Planner raises. Assert: plan-run FAILED, NO implement-run created, NO Plugin row.
4. `test_n_llm_calls_under_implement_run` — Implementer needs 2 rounds. Assert: `session.execute(select(LlmCall).where(LlmCall.agent_run_id == impl_run.id)).all()` has exactly 2 rows.
5. `test_preconditions_block_before_any_agent_run` — strategy in PROPOSED state → `AuthorPluginPreconditionError`, no AgentRun inserted.
6. `test_api_envelope_unchanged` — `POST /strategies/{id}/author-plugin` returns the same 202 shape as M5b: `{agent_run_id, status: "scheduled"}`. (API-test, in `tests/api/test_plugins.py`.)

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_plugin_flow_split.py tests/api/test_plugins.py -v
```

**Commit:** `feat(M5d): author_plugin orchestrator runs Planner→Implementer with 2 AgentRuns + N LlmCalls`

---

## Task 6 — Cleanup: retire M5b `plugin_author.py`, finalize shared helpers

**Files:**
- Delete: `src/fwbg_agents/agents/plugin_author.py` (after Task 5 migration is green).
- Verify: `src/fwbg_agents/agents/plugin_authoring_shared.py` (created in Task 5) holds `get_fwbg_plugin_examples`, `validate_python_syntax`, `_read_plugin_source`, `FwbgPluginExample`, `_PHASE_MAPPING`.
- Verify: all callers updated. Grep for `from fwbg_agents.agents.plugin_author import` and update remaining references.
- Modify: `tests/agents/test_plugin_author.py` → delete this file after confirming Tasks 3+4+5 fully cover its assertions.

**Behaviour:**
- Pure delete + rename. No new logic.
- `PluginAuthorFailed` class moves to `plugin_flow.py` (where it's raised) or stays at the API-error layer — verify M5c imports it.

**Test sketch:** No new tests. Run full suite to ensure no breakage.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest -q
```
Expected: ~340 tests passing (323 baseline + ~3 settings + ~3 prompt-doc + ~8 planner + ~10 implementer + ~6 flow − ~10 deleted M5b plugin_author tests now covered elsewhere).

**Commit:** `refactor(M5d): retire M5b PluginAuthor monolith, move shared helpers to plugin_authoring_shared`

---

## Task 7 — End-to-end smoke `scripts/m5d_smoke.py`

**Files:**
- New: `scripts/m5d_smoke.py`

**Behaviour (mirrors `m5c_smoke.py` structure):**
1. Connect to dev DB. Reset state for a fresh slug `m5d_smoke_<timestamp>`.
2. Seed BACKTESTED parent strategy + `add_indicator_request.json` sidecar with phase="indicator", capability="mean-reversion-on-Z-score".
3. `POST http://localhost:8000/strategies/{id}/author-plugin` → expect 202 + AgentRun ID.
4. Poll AgentRun(s) until terminal state (FAILED or both SUCCESS). Timeout 5 min (real LLM calls).
5. Assertions on success:
   - Exactly 2 AgentRuns: kind="plugin_plan" SUCCESS, kind="plugin_implement" SUCCESS.
   - 1 ≤ LlmCall count under implement-run ≤ `PLUGIN_IMPL_MAX_ROUNDS`.
   - `data/plugin-runs/<slug>/plan.json` exists and round-trips into `PluginPlan`.
   - `data/plugin-runs/<slug>/plugin.py` + `tests.py` + `spec.md` + `contract.yaml` exist.
   - Plugin row state=PROPOSED.
6. Optional: run M5b `PluginEvaluator` on the result → expect VERIFIED state (proves downstream pipeline still works).
7. Print `[m5d_smoke] PASSED` or full failure dump.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && uvicorn fwbg_agents.api.main:app &
sleep 2
VIRTUAL_ENV= uv run python scripts/m5d_smoke.py
```

**Commit:** `feat(M5d): scripts/m5d_smoke.py end-to-end Planner→Implementer→evaluator smoke`

---

## Out-of-scope (parked / future M5e or later)

- **Per-agent model config for Researcher/Analyst/Translator.** Concrete-before-generic: revisit after 2-3 more use-cases.
- **Loop-convergence heuristics** (early abort on repeated identical failure profile). YAGNI; `max_rounds` is enough.
- **Skill at `~/.claude/skills/fwbg-plugin-authoring/`.** Repo-internal `prompts/plugin_authoring.md` is the single source of truth.
- **Planner-self-review** of Implementer output. Not needed — deterministic gates cover all failure modes that matter.
- **Streaming progress per-round.** Implementer-loop is fast (< 60s typical for 1-2 rounds with opus-4-7); no UX win from streaming.

## Acceptance Criteria (Definition of Done) — ✓ all met

- All 7 tasks committed (commit chain above).
- `pytest -q` in fwbg-agents: 371 passing (target was ≥340), 0 failures.
- `scripts/m5d_smoke.py` self-test (in pytest) ends `[m5d_smoke] PASSED`.
- M5c `scripts/m5c_smoke.py` still passes (downstream regression guard).
- Old `src/fwbg_agents/agents/plugin_author.py` deleted; no `from fwbg_agents.agents.plugin_author` remaining.
- `prompts/plugin_authoring.md` loaded by both Planner and Implementer at runtime.
- Plan-doc status updated above.
- Memory entry `project-fwbg-agents-m5d-sketch.md` was already superseded during planning by `reference-fwbg-agents-m5d-plan.md`.

## Implementation deviations from the original plan (worth noting)

- **Task 1 — Settings:** extended existing `config.py:Settings` (pydantic-settings) rather than creating a parallel `settings.py`. Surgical change, no parallel config layers. Tests live at `tests/test_config_plugin_authoring.py`.
- **Task 2 — Prompt-doc smoke test:** PluginPhase-drift test uses a pinned `EXPECTED_PHASE_NAMES` list rather than importing `fwbg_sdk.base.PluginPhase` (fwbg-agents intentionally does not depend on fwbg_sdk — kept HTTP/filesystem-decoupled). Update both the constant and the doc when the SDK enum changes.
- **Task 3 — Phase mapping:** `_PHASE_MAPPING` accepts both plural (per `AddIndicator.phase` Literal — "indicators"/"filters") and singular (per M5c Translator's `_PHASE_TO_FIELD` — "indicator"/"filter") sidecar phase values. Both forms exist in the codebase; tolerance is correct, not legacy-compat.
- **Task 3 — slug pattern:** relaxed to `^[a-z][a-z0-9_-]*$` (allow kebab-case) to match `PluginContract.name` convention; existing fwbg plugins use both snake_case and kebab-case.
- **Task 4 — contract gate:** AST-only static check (`contract_check`), no code execution. Plan-doc described a sandbox-exec approach; the static version is cheaper and avoids adding `fwbg_sdk` as an agents dependency.
- **Task 5 — three AgentRuns total per author session:** outer `plugin_author_flow` (created by the API for the user-facing poll target) + inner `plugin_planner` + inner `plugin_implementer`. Plan-doc described only the two inner runs; the outer was pre-existing.
- **Task 5 — `lookup_plugin_capability`:** reads only the `plugin_planner` AR (no legacy `plugin_author` fallback). The `tests/api/test_plugin_reiterate.py` fixture was migrated to seed a `plugin_planner` AR row.
- **Task 6 — smoke compat shim:** the m5c smoke's stub plugin code has both a top-level `compute()` function (for the M5b PluginEvaluator's exec path) and a `BaseIndicator` subclass (for the M5d contract gate). The fwbg_sdk imports are guarded with stub fallbacks so the module loads in the agents venv. Proper cleanup (evaluator update for class-style plugins) belongs in a future M5e.
- **Task 6 — `scripts/m5_smoke.py`:** soft-abandoned with a deprecation header pointing to `m5c_smoke.py` and `m5d_smoke.py` (per [[feedback-no-hard-delete]]).
- **Task 7 — smoke assertion filter:** `_assert_split_flow_artifacts` filters AgentRuns by `agent_name.in_(("plugin_planner", "plugin_implementer"))` to exclude the API's outer `plugin_author_flow` row from the 2-AR assertion.

## Final commit table

| # | Commit  | Description |
|---|---------|-------------|
| 1 | 24b8be0 | feat(M5d): Settings fields for per-agent model + max-rounds config |
| 2 | 51c16c5 | docs(M5d): prompts/plugin_authoring.md canonical fwbg-Plugin-Konventionen |
| 3 | 46b21c5 | feat(M5d): PluginPlanner agent emits PluginPlan from sidecar + parent strategy |
| 4 | 9a73136 | feat(M5d): PluginImplementer agent with deterministic gate-loop (max_rounds=5) |
| 5 | 1cefb65 | feat(M5d): author_plugin orchestrator runs Planner→Implementer with 2 AgentRuns |
| 6 | 979cb23 | refactor(M5d): retire M5b PluginAuthor, move shared helpers to plugin_authoring_shared |
| 7 | d85ad1c | feat(M5d): scripts/m5d_smoke.py end-to-end Planner→Implementer smoke |
