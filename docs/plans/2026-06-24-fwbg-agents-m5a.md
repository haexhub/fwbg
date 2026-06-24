# fwbg-agents M5a — Plugin Discovery + Contract + AddIndicator + Validator Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** ✓ done (2026-06-24, final commit `0418eb1`, 187 tests green)
**Repo:** `~/Projekte/fwbg-agents/`
**Baseline:** `45825f9` (M4 done, 159 tests green) → `0418eb1` (M5a done, 187 tests, +28)

**Commits:** Task 1 `291afa6` · Task 2 `0dd440b` (+ `c050fd5` in fwbg) · Task 3 `28c4180` · Task 4 `0418eb1`

**Goal:** Replace the hand-coded `KNOWN_*` frozensets in `strategy_validator.py` with a runtime catalog that discovers fwbg-core + fwbg-premium plugin manifests AND merges in agent-authored verified plugins from the DB. Add `AddIndicator` to the Analyst recommendation union so backtests can request a new plugin instead of inventing slugs. Define the `PluginContract` schema that M5b's Evaluator will check against.

**Architecture (M5a):**
- `orchestrator/plugin_catalog.py` — read-only discovery + DB merge. lru_cache per process, no invalidation. Deterministic.
- `orchestrator/plugin_contract.py` — pydantic schema for `contract.yaml`. Validates structure only; M5b's Evaluator runs the invariants.
- `agents/analyst.py` — extend Union with `AddIndicator{kind, phase, capability, reason}`. The Analyst sees the catalog in its prompt context (so it can't hallucinate existing capabilities).
- `orchestrator/recommendations.py` — handle `add_indicator` by writing `iteration_NNN/add_indicator_request.json` sidecar, no transition. Mirrors M4's `change_exit` sidecar pattern.
- `orchestrator/strategy_validator.py` — `validate_strategy_json(data, *, catalog)`. Translator/API caller injects catalog. Test default factory falls back to the M4 frozenset set for isolation.

**Tech Stack:** pydantic v2, pydantic-ai, SQLAlchemy 2 (async), pytest-asyncio, alembic. No new deps in M5a.

---

## Locked decisions (do NOT re-litigate)

| # | Decision |
|---|---|
| 1 | Plugin discovery scans fwbg-core + fwbg-premium roots. Missing root → log warn + return empty subset (no hard fail). |
| 2 | Catalog cache: `functools.lru_cache(maxsize=None)` on a module-level loader. Tests reset via `clear()`. |
| 3 | DB-side plugins included ONLY if `current_state IN (verified, adopted_in_fwbg)`. M5b authored-but-unverified plugins MUST NOT validate. |
| 4 | `PluginContract` schema is structural only. Invariant runtime checks live in M5b's `PluginEvaluator`. |
| 5 | `AddIndicator` writes sidecar `iteration_NNN/add_indicator_request.json` exactly like M4's `change_exit` sidecar. No state transition. |
| 6 | `strategy_validator.validate_strategy_json` keeps old signature backward-compatible: catalog is keyword-only with a default factory returning the legacy M4 frozenset set. Existing call sites work unchanged. |
| 7 | `PluginKind` enum extension (indicator → indicator/model/exit_strategy/.../data_loading) is DEFERRED to M5b's Migration 0004. M5a does not touch the ORM. |

---

## Task table

| # | Task | Files | Tests added (target) |
|---|---|---|---|
| 1 | PluginCatalog discovery + DB merge | `orchestrator/plugin_catalog.py` | ~12 |
| 2 | PluginContract schema | `orchestrator/plugin_contract.py` | ~6 |
| 3 | AddIndicator recommendation + sidecar | `agents/analyst.py`, `orchestrator/recommendations.py`, `agents/prompts/analyst.md` | ~6 |
| 4 | strategy_validator refactor to catalog-aware | `orchestrator/strategy_validator.py` + call sites | ~6 (refactor existing) |
| — | Smoke + housekeeping | — | 0 |

Target post-M5a: **~190 passed**. Each task = one commit (English, no Claude footer, green pytest + alembic clean).

---

## Task 1 — PluginCatalog (discovery + ORM merge)

**Files:**
- Create: `src/fwbg_agents/orchestrator/plugin_catalog.py`
- Create: `tests/orchestrator/test_plugin_catalog.py`
- Modify: `src/fwbg_agents/config.py` (add `fwbg_repo_root: Path` settings — defaults to `~/Projekte/fwbg`)

**Behaviour:**
- `PluginManifest(BaseModel)`: `name`, `category`, `provenance` (`"fwbg-core" | "fwbg-premium" | "agent-authored"`), `version` (str — semver from manifest, or "v1" for authored), `source_path` (Path).
- `PluginCatalog(BaseModel)`: `by_category: dict[str, dict[str, PluginManifest]]` — outer key = category slug (`indicators`, `models`, `exit_strategies`, `risk_management`, `entry_modifiers`, `preprocessing`, `feature_selection`, `data_loading`), inner key = plugin slug.
- Method `has(category, slug)`, `get(category, slug)`, `all_slugs_for(category)`.
- `discover_fwbg_plugins(fwbg_root: Path) -> dict[str, dict[str, PluginManifest]]`: reads `src/fwbg/plugins/*/manifest.json` (treats top-level manifest as bundle, expands `plugins: {indicators: [...], ...}` to per-slug `PluginManifest`s). Also walks `packages/fwbg-premium/src/fwbg_premium/plugins/*/manifest.json` (same bundle shape) — each leaf-dir `feature_selection/<slug>/manifest.json` IS its own plugin with category derived from the path segment.
- `merge_with_db(catalog, db_plugins: list[Plugin]) -> PluginCatalog`: only `verified` and `adopted_in_fwbg`, derives category from `Plugin.kind` (string), version from `spec_path` (`data/plugins/<slug>/v1/spec.md` → "v1"). DB plugins shadow fwbg-side plugins of the same slug (agent-authored wins after verification).
- `load_catalog(session) -> PluginCatalog`: top-level entry. Calls `_load_fwbg_cached(fwbg_root)` (lru_cached) then merges DB. Tests call `_load_fwbg_cached.cache_clear()` in fixture.
- Missing fwbg root or unreadable manifest.json → log.warning + that subset empty; never raise.

**Test sketch (behaviour-only):**
1. `discover_fwbg_plugins` over a tmp_path with a faked `src/fwbg/plugins/fwbg-core/manifest.json` containing `{"indicators": ["ema", "sma"]}` returns category→slug→manifest with `provenance="fwbg-core"`.
2. Same fixture with a premium leaf-dir layout (`packages/.../plugins/fwbg-premium/feature_selection/boruta/manifest.json`) returns the boruta entry under `feature_selection` with `provenance="fwbg-premium"`.
3. Missing fwbg root → returns empty `by_category`, no exception, warning logged.
4. `merge_with_db` skips a DB plugin in `SPECIFIED` or `AUTHORED`.
5. `merge_with_db` includes a DB plugin in `VERIFIED` and gives it `provenance="agent-authored"`.
6. DB plugin shadows fwbg-side same slug (agent-authored overrides).
7. `has("indicators", "ema")` True after discovery; `has("indicators", "nonexistent")` False.
8. `load_catalog` caches: second call with same fwbg_root reuses fs result (verify via mock count). After `cache_clear()`, hit-count resets.
9. Malformed manifest.json (invalid JSON) → category empty for that bundle, warning logged.
10. `all_slugs_for("indicators")` returns sorted list across fwbg-core + premium + DB.
11. `PluginCatalog` is pydantic-frozen — direct dict mutation forbidden.
12. `load_catalog` async signature: awaits `session.execute(select(Plugin)...)` — async test verifies one round trip per call.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_plugin_catalog.py -v
```
Commit: `feat(M5a): plugin discovery + DB merge catalog`

---

## Task 2 — PluginContract schema

**Files:**
- Create: `src/fwbg_agents/orchestrator/plugin_contract.py`
- Create: `tests/orchestrator/test_plugin_contract.py`
- Create: `docs/specs/plugin_contract.example.yaml` (a worked example for an indicator — used as fixture)

**Behaviour:**
- `PluginContractInput(BaseModel)`: `name: str` (parameter name), `dtype: Literal["float", "int", "bool", "series", "ohlcv"]`, `required: bool = True`, `description: str = ""`.
- `PluginContractOutput(BaseModel)`: `name`, `dtype` (`"series" | "scalar" | "boolean_series"`), `length_invariant: Literal["same_as_input", "trimmed", "any"] = "same_as_input"`.
- `PluginContractParam(BaseModel)`: `name`, `dtype: Literal["float", "int", "bool", "str"]`, `default: Any`, `min: float | None`, `max: float | None`, `description`.
- `PluginContractScenario(BaseModel)`: `name`, `data_path: str` (relative to plugin dir, e.g. `test_scenarios/trending_up.parquet`), `expected_outputs: dict[str, Any] | None = None` (optional pin — M5b's Evaluator uses approximate compare).
- `PluginContract(BaseModel)`: `name: str`, `kind: Literal["indicator", "model", "exit_strategy", "risk_management", "entry_modifier", "preprocessing", "feature_selection", "data_loading"]` (mirrors the M5b PluginKind enum extension — kept as Literal here so M5a doesn't depend on M5b's Migration 0004), `version: str = "v1"`, `inputs: list[PluginContractInput]`, `outputs: list[PluginContractOutput]`, `params: list[PluginContractParam]`, `invariants: list[str]` (free-text checks: "outputs[0] same length as inputs[0]", "outputs[0] never NaN past first 14 bars"), `test_scenarios: list[PluginContractScenario]`.
- `load_contract(path: Path) -> PluginContract`: reads YAML, validates via pydantic. Raises `PluginContractError` with a useful message on schema mismatch.
- `dump_contract(contract: PluginContract, path: Path) -> None`: writes YAML with stable key order.

**Test sketch:**
1. `load_contract` round-trips the example fixture without error and surfaces correct kind/inputs/params count.
2. `dump_contract` then `load_contract` gives back an equal contract (pydantic `.model_dump()` equality).
3. Missing required field (e.g. no `kind`) → `PluginContractError` with field path in message.
4. `kind="bogus"` → validation error.
5. Scenario list MAY be empty (some indicators have no test scenarios) → no error.
6. Invariants list MUST be non-empty for `kind="indicator"` → validation error if `kind=indicator AND len(invariants)==0` (model_validator).

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_plugin_contract.py -v
```
Commit: `feat(M5a): PluginContract schema for contract.yaml`

---

## Task 3 — AddIndicator recommendation + sidecar

**Files:**
- Modify: `src/fwbg_agents/agents/analyst.py` (add `AddIndicator` model, extend Union, update Analyst prompt-context to include catalog snapshot)
- Modify: `src/fwbg_agents/agents/prompts/analyst.md` (new section: "if no fwbg-side plugin covers what you need, emit add_indicator — DO NOT invent a slug for an existing capability")
- Modify: `src/fwbg_agents/orchestrator/recommendations.py` (`validate_and_apply` handles `add_indicator` → write sidecar, return `RecommendationOutcome(applied=False, sidecar_path=...)`)
- Modify: `tests/agents/test_analyst.py` (add `FunctionModel` stub returning AddIndicator)
- Modify: `tests/orchestrator/test_recommendations.py` (sidecar write + no transition)

**Behaviour:**
- `class AddIndicator(BaseModel)`: `kind: Literal["add_indicator"] = "add_indicator"`, `confidence: float [0,1]`, `reasoning: str`, `phase: Literal["feature_selection", "indicators", "preprocessing", "filters"]` (broad — Analyst nominates the pipeline phase), `capability: str` (free-text: "support/resistance zones from pivot points"), `category: Literal[...]` (one of the PluginContract.kind literals — the Analyst commits to which fwbg-category this belongs to).
- Analyst prompt receives a flattened catalog snapshot — list of `(category, slug)` pairs — so it can name an existing slug instead of fabricating a `capability`. The Analyst MUST emit `add_indicator` only after checking the catalog.
- `validate_and_apply` for `add_indicator`:
  - Increments `iteration_count` like M4's `change_exit` sidecar.
  - Writes `data/strategies/<slug>/iteration_NNN/add_indicator_request.json` with `{phase, capability, category, reason, requested_at}`.
  - NO transition, NO ABANDON. Strategy stays in BACKTESTED.
  - Returns outcome with `sidecar_path` set, `applied=False`.

**Test sketch:**
1. Analyst with FunctionModel stub returning `AddIndicator{...}` produces the right pydantic instance with discriminator routed correctly.
2. `validate_and_apply(AddIndicator)` writes the sidecar JSON at the expected path with the expected fields.
3. After `validate_and_apply(AddIndicator)`, strategy.current_state is unchanged (still BACKTESTED).
4. `iteration_count` increments by 1.
5. Analyst's prompt rendering interpolates a catalog snapshot (assert: the catalog count line appears in the rendered prompt).
6. Bad `category` value (not in the literal) → pydantic validation error at Analyst output stage (before recommendations.py sees it).

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/agents/test_analyst.py tests/orchestrator/test_recommendations.py -v
```
Commit: `feat(M5a): AddIndicator recommendation + sidecar request flow`

---

## Task 4 — strategy_validator refactor (catalog-aware)

**Files:**
- Modify: `src/fwbg_agents/orchestrator/strategy_validator.py`
- Modify: `src/fwbg_agents/agents/translator.py` (pass catalog into validator)
- Modify: `src/fwbg_agents/orchestrator/research_flow.py` (load catalog once per flow, pass through)
- Modify: any API callers that invoke validate_strategy_json
- Modify: `tests/orchestrator/test_strategy_validator.py` (cover catalog injection + fallback)

**Behaviour:**
- New signature: `validate_strategy_json(data: dict, *, catalog: PluginCatalog | None = None) -> None`.
- When `catalog is None`: use the legacy M4 frozenset set (`KNOWN_PIPELINES`, `KNOWN_MODELS`, `KNOWN_FILTERS`, `KNOWN_VALIDATIONS`, `KNOWN_RESOURCES`, `KNOWN_DATASOURCES`). Existing call sites continue to work; tests without DB still pass. **Keep the frozensets in the module** — they ARE the test-isolation fallback.
- When `catalog is not None`: lookup happens against the catalog. Mapping field→category:
  - `pipeline` → catalog phase "indicators" + "feature_selection" — but pipelines themselves aren't fwbg plugins, they're identifiers fwbg knows. M5a keeps `KNOWN_PIPELINES` hard-coded (still a frozenset literal) and only catalog-routes the plugin-shaped fields. Specifically: M5a routes ONLY `model` (catalog category `models`) and the per-item `exit_strategies[i].name` (category `exit_strategies`). `pipeline`/`filters`/`validation`/`resources`/`timeframe`/`datasource` remain frozenset-validated.
- Error messages MUST mention the catalog-derived suggestion list (top N most similar slugs via simple substring match) when validation fails AND a catalog was provided, so the Translator gets a helpful retry hint.
- Translator's `validate_strategy_json` calls now pass `catalog=session_catalog` (loaded once per flow).
- `research_flow.research_and_translate` loads catalog at entry and threads it through.

**Test sketch:**
1. Legacy call `validate_strategy_json(data)` (no catalog) still passes with the M4 frozenset set — existing tests already cover this; verify ZERO regressions.
2. With a catalog containing `models: {signal_orb_v1}` but NOT `bogus_model`, calling with `model="bogus_model"` raises with a message that includes "signal_orb_v1" (substring suggestion).
3. With a catalog adding a new `models: brand_new_model` that the legacy frozenset doesn't list, the new model passes validation when catalog is supplied (catalog supersedes frozenset).
4. `exit_strategies[0].name="fixed"` passes when catalog has `exit_strategies: {fixed}`.
5. `exit_strategies[0].name="trailing_stop"` fails when catalog lacks it, with a suggestion line.
6. Translator integration: `Translator.run_fresh` with FunctionModel-stub generating a valid strategy JSON validates against catalog from session — no regression vs M4 test_translator_fresh.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_strategy_validator.py tests/agents/test_translator_fresh.py tests/orchestrator/test_research_flow.py -v
```
Commit: `refactor(M5a): catalog-aware strategy_validator with frozenset fallback`

---

## End-of-session housekeeping

1. Full pytest run: `VIRTUAL_ENV= uv run pytest -q` → target ~190 passed.
2. `alembic upgrade head` → still at 0003 (M5a doesn't migrate).
3. `python -c "from fwbg_agents.main import app"` → app importable.
4. `graphify update .` in `~/Projekte/fwbg-agents/` → refresh graph.
5. Update design-doc implementation status table: add M5a row with final commit hashes.
6. Update this plan's status header to "✓ done".
7. Update `project_fwbg_agents.md` memory: M5a-block, final commit, test count.
8. Create `reference_fwbg_agents_m5a_plan.md` memory: commit table + locked decisions reference.
9. Add MEMORY.md index lines for the new references.

---

## M5b preview (NEXT session — do NOT implement now)

Sketched only so the design doc roadmap stays current:

- **Task 5: PluginAuthor agent** — pydantic-ai agent. Tool `get_fwbg_plugin_examples(category, n=3)`, tool `validate_python_syntax(code) -> ast.parse`. Writes `data/plugins/<slug>/v1/{plugin.py, contract.yaml, spec.md}`, INSERTs `Plugin(SPECIFIED)` then `transition_plugin(AUTHORED, payload={request_path, request_strategy_id})`. ~8 tests, FunctionModel-stub pattern.
- **Task 6: PluginEvaluator agent** — DETERMINISTIC in M5b. Generates synthetic test_scenarios as parquet (trending_up, trending_down, sideways, high_vola, sparse_data) via hand-curated np-seeded `scenario_generators.py`. Dynamic-imports `plugin.py`, runs against each scenario, checks contract invariants. New ORM table `verification_run(id, plugin_id, status, scenarios_run, scenarios_passed, error_log_path)` + Migration 0004 (also extends PluginKind to all 8 categories — additive string column, no DB enum). On full pass: `transition_plugin(VERIFIED, payload={verification_run_id})`. ~10 tests.
- **Task 7: API + plugin_flow.py glue** — `POST /strategies/{id}/author-plugin`, `POST /plugins/{id}/evaluate`, `GET /plugins/{id}/verification-runs`. M3/M4 pattern: 202 + BackgroundTasks + AgentRun-envelope.
- **Task 8: M5b smoke** — `scripts/m5_smoke.py` pre-seeds strategy with `add_indicator_request.json` (from M5a), drives author → evaluate → verified end-to-end. Verifies all files written, plugin reaches VERIFIED.
