# fwbg-agents M5c — reiterate-with-plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** ✓ done 2026-06-24 — eight commits `f55a4e3`, `0acf06f`, `7863ba4`, `6c24e18`, `d994766`, `69ab77d`, `66f5cac`, `22ce7e1` in `~/Projekte/fwbg-agents/`. 258 tests green (234 → +24). Manual `scripts/m5c_smoke.py` against dev DB ends `[m5c_smoke] PASSED`. M5c smoke surfaced a pre-existing M5a/M5b bug in `plugin_catalog.merge_with_db` (singular `Plugin.kind` vs plural bundle-manifest categories → DB-VERIFIED plugins invisible in catalog) — fixed in-range. M6=PaperTrading next.

**Goal:** Close the loop from a VERIFIED plugin back into strategy iteration: spawn a child Strategy whose `strategy.json` references the new plugin-slug at the slot indicated by the original `add_indicator_request.json` sidecar. Child starts PROPOSED with `parent_strategy_id` set; downstream Runner takes over as in M3/M4.

**Architecture:**
- `strategy.json` gets four NEW optional `list[str]` fields — `indicators`, `feature_selection`, `preprocessing`, `extra_filters` — additive to the existing single-string `pipeline`/`model`/`filters`/`validation`. AddIndicator.phase maps 1:1 to a list. M4 strategies remain valid (empty lists default).
- `Translator.run_reiterate_with_plugin(parent, plugin_slug, sidecar)` mirrors M4's deterministic `run_reiterate` shape. Deep-copies parent strategy.json, appends slug into the phase-matching list, inserts child Strategy(PROPOSED, parent_strategy_id, iteration_count=1), writes child `iteration_001/{strategy.json, hypothesis.json, spec.md}`. Hypothesis is inherited+template-annotated (Decision C1 — M5b API untouched). Catalog cache cleared before validation so freshly-VERIFIED plugins resolve.
- `orchestrator/plugin_flow.reiterate_with_plugin()` + `POST /strategies/{id}/reiterate-with-plugin` follow the M3/M4/M5b envelope: 422 preconditions, 202 + AgentRun + BackgroundTasks.
- One end-to-end smoke `scripts/m5c_smoke.py`: seed BACKTESTED parent + sidecar → author plugin → evaluate → reiterate-with-plugin → assert child exists with slug spliced correctly.

**Tech Stack:** Python 3.13, pydantic-ai (deterministic — no LLM in this milestone), SQLAlchemy 2.x async, FastAPI, pytest, alembic. No new dependencies.

**Locked Decisions** (from session intro):
- (A) **M5c**, M6=PaperTrading unverändert.
- (B) **Neue list-fields** (`indicators`/`feature_selection`/`preprocessing`/`extra_filters`) zusätzlich zu bestehenden single-strings — keine Union-Polymorphie, kein dict-Override.
- (C) **C1 inherited+annotated** — child hypothesis = parent-hypothesis-copy + template-rationale block; PluginAuthor API unangetastet → 234 M5b-Tests bleiben grün.
- (D) **Sidecar bleibt liegen** — append-only audit; reiterate liest, mutiert nicht.
- (E) **`_load_fwbg_cached.cache_clear()`** am Anfang von `reiterate_with_plugin`. Surgical, eine Zeile, deckt den Race "Plugin VERIFIED aber Catalog noch nicht geladen" ab.

**Phase-Mapping (AddIndicator.phase → strategy.json list-field):**
```
"indicator"          → indicators
"feature_selection"  → feature_selection
"preprocessing"      → preprocessing
"filter"             → extra_filters
```
(Translator raises `TranslatorFailed` for unknown phases. Phase values mirror the M5a `PluginKindLit` enum.)

**Pre-checks (already verified at session start):**
- HEAD = 36aca54 ✓
- `pytest -q` = 234 passed ✓
- `alembic current` = 0004 (head) ✓

---

## Task 1 — Schema extension for plugin-slot list-fields

**Files:**
- Modify: `src/fwbg_agents/orchestrator/strategy_validator.py` — add four optional list-fields with catalog-routing.
- Modify: `src/fwbg_agents/agents/translator.py` — extend `_TranslatorOutput` if it tries to populate the new fields (fresh-translate path stays empty-default; M5c only writes them in `run_reiterate_with_plugin`).
- Test: `tests/orchestrator/test_strategy_validator.py` (extend).

**Behaviour:**
- `validate_strategy_json` accepts four NEW optional top-level keys: `indicators: list[str]`, `feature_selection: list[str]`, `preprocessing: list[str]`, `extra_filters: list[str]`. Default: omitted == empty == valid.
- If present: must be `list[str]` (no nested objects, no None entries). Empty list is fine.
- Catalog-routing: when a `PluginCatalog` is passed AND has entries for the matching category, each slug in the list must be in the catalog. Catalog categories: `indicators` → "indicators", `feature_selection` → "feature_selection", `preprocessing` → "preprocessing", `extra_filters` → "filters". When no catalog or empty category → lax (no membership check). Mirrors the M5a `_check_field_with_catalog` pattern, no frozen-fallback (these fields are 100% plugin-authored, never fwbg-builtin).
- Existing M4 strategies (no list-fields) keep validating as before — additive change.

**Test sketch (behaviour-only):**
1. `test_strategy_with_no_list_fields_is_valid` — pure M4-shape strategy.json validates as before (regression guard).
2. `test_indicators_list_must_be_list_of_str` — `indicators: "adx"` (string) → `StrategyValidationError`. `indicators: [{"x": 1}]` (dict) → error.
3. `test_indicators_empty_list_is_valid` — `indicators: []` validates.
4. `test_indicators_slug_must_be_in_catalog_when_catalog_present` — catalog has `["adx-trend-strength"]` for "indicators"; payload has `indicators: ["adx-trend-strength"]` → OK. `indicators: ["made-up"]` → error with did-you-mean hint.
5. `test_extra_filters_routes_to_catalog_filters_category` — explicit assertion: `extra_filters: ["custom-filter-x"]` looks up category `"filters"` (not `"extra_filters"`), matches catalog entry.
6. `test_no_catalog_means_lax_membership_for_list_fields` — without catalog kwarg, arbitrary slugs in `indicators` pass (M4-compatible lax mode).

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_strategy_validator.py -v
```
Expected: all new tests pass, prior tests unchanged.

**Commit:** `feat(M5c): strategy_validator accepts plugin-slot list-fields (indicators/feature_selection/preprocessing/extra_filters)`

---

## Task 2 — Translator.run_reiterate_with_plugin

**Files:**
- Modify: `src/fwbg_agents/agents/translator.py` — add method `run_reiterate_with_plugin(parent, plugin_slug, sidecar) -> Strategy`.
- Test: `tests/agents/test_translator_reiterate_plugin.py` (new file — mirrors `test_translator_reiterate.py`).

**Behaviour:**
- Signature: `async def run_reiterate_with_plugin(self, parent: Strategy, plugin_slug: str, sidecar: dict) -> Strategy`.
- Preconditions (raise `TranslatorFailed` with informative msg):
  - `parent.current_state == BACKTESTED`. Otherwise → "reiterate_with_plugin requires parent in BACKTESTED, got {state}".
  - `sidecar["phase"]` ∈ `{"indicator", "feature_selection", "preprocessing", "filter"}`. Otherwise → "unknown phase".
  - Parent's latest iteration dir has `strategy.json`. Otherwise → "parent missing strategy.json at {path}".
  - **No** check on `Plugin.current_state == VERIFIED` here — that's the caller's (plugin_flow) job. Translator is a deterministic mechanical splice; validation happens via `validate_strategy_json(catalog=load_catalog())` which already raises if the slug isn't in the catalog (and the catalog only sees VERIFIED plugins per M5a).
- AgentRun envelope: insert `AgentRun(agent_name="translator", strategy_id=parent.id, status=RUNNING, input_artifact_path=str(parent_iteration_dir / "add_indicator_request.json"))`. On exception → status=FAILED, error, commit, re-raise (same shape as M4 `run_reiterate`).
- Splice logic:
  1. Deep-copy parent `strategy.json` payload.
  2. Map `sidecar["phase"]` → list field: `indicator→indicators`, `feature_selection→feature_selection`, `preprocessing→preprocessing`, `filter→extra_filters`.
  3. `child_payload.setdefault(list_field, []).append(plugin_slug)`.
  4. Generate child slug via existing `generate_slug(session, parent.strategy_family, parent.asset_class)`.
  5. Set `child_payload["name"] = child_slug`.
  6. Load PluginCatalog (M5a's `load_catalog()` — cache already cleared by caller). Validate with catalog. Failure → `TranslatorFailed`.
- DB writes (single commit at end of try-block):
  - Insert `Strategy(slug=child_slug, current_state=PROPOSED, iteration_count=1, parent_strategy_id=parent.id, asset_class=parent.asset_class, strategy_family=parent.strategy_family, created_at=now, updated_at=now)`.
  - Insert `Transition(strategy_id=child.id, from_state=None, to_state=PROPOSED, reason="translator: reiterate_with_plugin", payload={"parent_strategy_id": parent.id, "plugin_slug": plugin_slug, "sidecar": sidecar}, created_by="translator", created_at=now)`.
- File writes (under `strategy_dir(child_slug) / "iteration_001"`):
  - `strategy.json` — final payload.
  - `hypothesis.json` — parent hypothesis copy + appended iteration block (template C1):
    ```json
    {
      "...parent fields...": "...",
      "iterations": [
        {
          "iteration": 1,
          "action": "add_indicator",
          "plugin_slug": "<slug>",
          "phase": "<sidecar.phase>",
          "capability": "<sidecar.capability>",
          "rationale": "Iteration 1: added <slug> at <phase> per analyst recommendation: <capability>"
        }
      ]
    }
    ```
    If parent hypothesis already has `iterations`, append to it (don't overwrite). If parent hypothesis is missing entirely (edge case), child hypothesis becomes just the iterations block with a top-level `{"inherited_from": parent.slug, "iterations": [...]}` fallback.
  - `spec.md` — re-use existing `_write_spec_md(spec_path, strategy_slug=child_slug, hypothesis=hypothesis_data, strategy_json=child_payload)`.
  - `child.spec_path = str(spec_path)`, `child.updated_at = datetime.now(UTC)`.
- AgentRun finalisation: `status=DONE`, `output_artifact_path=str(child_strategy_path)`, `ended_at=now`. Commit.
- Return `child` Strategy ORM instance.

**Edge cases explicit:**
- Parent has no `add_indicator_request.json` in its latest iteration: not required by Translator itself (the sidecar is read by plugin_flow and passed in as dict). Translator never reads the sidecar file — it gets the parsed dict from the caller. This keeps Translator unit-testable without filesystem fixtures of the sidecar.
- Slug already exists in `indicators` list of parent (re-iterating with the SAME plugin twice): append anyway. M5c does NOT dedup — that's a Runner concern (duplicate-slug detection on the fwbg side). Worth a TODO comment but not a hard rule.
- Parent hypothesis.json may be a *string* in older payloads (M2/M3 era). Detect: if `parent_hypothesis_data` is a str, wrap as `{"inherited_text": str_value, "iterations": [...]}`. M4+ writes a dict already.

**Test sketch (behaviour-only — `tests/agents/test_translator_reiterate_plugin.py`):**
1. `test_run_reiterate_with_plugin_happy_path_indicator_phase` — parent in BACKTESTED with strategy.json + hypothesis.json; catalog has `["adx-trend-strength"]` under "indicators"; sidecar `{phase:"indicator", capability:"detect strong trends", ...}`. Assert child in PROPOSED, `child.parent_strategy_id == parent.id`, `child_strategy_json["indicators"] == ["adx-trend-strength"]`, child has iteration_001/{strategy.json,hypothesis.json,spec.md}, transition payload includes plugin_slug + parent_strategy_id.
2. `test_run_reiterate_with_plugin_feature_selection_phase` — same shape, sidecar phase=`"feature_selection"`; assert child_payload `feature_selection: ["..."]` populated (and other list-fields empty).
3. `test_run_reiterate_with_plugin_preprocessing_phase` — same shape, phase=`"preprocessing"`.
4. `test_run_reiterate_with_plugin_filter_phase` — phase=`"filter"`; assert lands in `extra_filters`, NOT in legacy `filters` single-string (which stays untouched).
5. `test_run_reiterate_with_plugin_rejects_unknown_phase` — sidecar phase=`"orchestration"`; assert `TranslatorFailed` raised, AgentRun status=FAILED, no child created.
6. `test_run_reiterate_with_plugin_rejects_parent_not_backtested` — parent in PROPOSED; assert `TranslatorFailed`, no child, AgentRun status=FAILED.
7. `test_run_reiterate_with_plugin_appends_to_existing_iterations` — parent hypothesis already has `iterations: [{iteration: 1, ...}]` (from a prior reiterate). New call adds iteration 2. Assert child hypothesis has BOTH iterations and `child_iterations[1].iteration == 2`.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/agents/test_translator_reiterate_plugin.py -v
```
Expected: 7 passed.

**Commit:** `feat(M5c): Translator.run_reiterate_with_plugin — deterministic slug splice into list-fields`

---

## Task 3 — plugin_flow.reiterate_with_plugin + API endpoint

**Files:**
- Modify: `src/fwbg_agents/orchestrator/plugin_flow.py` — add `reiterate_with_plugin(session, strategy_id, plugin_slug) -> int` + `ReiterateWithPluginPreconditionError`.
- Modify: `src/fwbg_agents/api/strategies.py` — add `POST /strategies/{strategy_id}/reiterate-with-plugin`.
- Test: `tests/api/test_plugin_reiterate.py` (new file).

**Behaviour:**
- `orchestrator/plugin_flow.reiterate_with_plugin(session, strategy_id, plugin_slug)`:
  1. SELECT parent Strategy by id. None → `ReiterateWithPluginPreconditionError("strategy {id} not found")` → 404 in API layer.
  2. Parent.current_state != BACKTESTED → `ReiterateWithPluginPreconditionError("strategy {slug} is in state {state}; reiterate-with-plugin requires BACKTESTED")`.
  3. SELECT Plugin by slug. None or current_state != VERIFIED → `ReiterateWithPluginPreconditionError("plugin {slug} not found or not VERIFIED")`.
  4. Locate latest `add_indicator_request.json` via reused `_find_latest_sidecar(parent.slug)`. None → `ReiterateWithPluginPreconditionError("no add_indicator_request.json found for {slug}")`.
  5. Parse sidecar JSON. Verify `sidecar["capability"] == plugin.capability_summary` (or equivalent — see plugin model field). If mismatch → `ReiterateWithPluginPreconditionError("plugin {slug} does not match the sidecar's requested capability")`. (Sanity guard: stops "splice random VERIFIED plugin into a strategy that asked for a different one".)
  6. **`_load_fwbg_cached.cache_clear()`** — Decision E. Imported from `fwbg_agents.orchestrator.plugin_catalog`. Ensures the just-VERIFIED plugin's manifest is loaded fresh.
  7. Instantiate `Translator(session)`, call `await translator.run_reiterate_with_plugin(parent, plugin_slug, sidecar)`. Returns child Strategy.
  8. Return `child.id`.
- API endpoint `POST /strategies/{strategy_id}/reiterate-with-plugin`:
  - Request body: `{"plugin_slug": str}` (pydantic model `ReiterateWithPluginRequest`).
  - 404 if strategy missing (catch precondition error, map message).
  - 422 if any other precondition fails.
  - 202: INSERT `AgentRun(agent_name="translator", strategy_id, status="running", started_at=now, input_artifact_path=<sidecar_path>)`. Schedule `_run_reiterate_with_plugin_background(agent_run_id, strategy_id, plugin_slug)`. Return envelope: `{"agent_run": {id, status, agent_name}, "strategy_id": parent_id}`.
- Background helper `_run_reiterate_with_plugin_background(agent_run_id, strategy_id, plugin_slug)`:
  - Open new AsyncSession. Re-load AgentRun by id.
  - try: call `reiterate_with_plugin(session, strategy_id, plugin_slug)` → child_id. Set `agent_run.status=DONE`, `output_artifact_path=str(strategy_dir(child.slug) / "iteration_001" / "strategy.json")`, payload-equivalent stored in transition row already.
  - except: status=FAILED, error=str(exc).
  - commit.
- Re-uses existing `_find_latest_sidecar` (M5b) → no duplication.

**Test sketch (`tests/api/test_plugin_reiterate.py`):**
1. `test_post_reiterate_with_plugin_returns_202_and_creates_child` — seed parent BACKTESTED + sidecar + plugin VERIFIED with matching capability. POST → 202 with agent_run envelope. Poll until done. Assert child Strategy exists, `parent_strategy_id==parent.id`, child slug in DB, child `strategy.json` on disk with slug in correct list-field.
2. `test_post_reiterate_with_plugin_404_strategy_missing` — POST with id=99999 → 404.
3. `test_post_reiterate_with_plugin_422_parent_not_backtested` — parent in PROPOSED → 422 with state-mismatch message.
4. `test_post_reiterate_with_plugin_422_plugin_not_verified` — plugin in AUTHORED → 422.
5. `test_post_reiterate_with_plugin_422_no_sidecar` — parent in BACKTESTED but no `add_indicator_request.json` → 422.
6. `test_post_reiterate_with_plugin_422_capability_mismatch` — sidecar capability="X", plugin.capability_summary="Y" → 422 with mismatch message.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/api/test_plugin_reiterate.py -v
```
Expected: 6 passed.

**Commit:** `feat(M5c): plugin_flow.reiterate_with_plugin + POST /strategies/{id}/reiterate-with-plugin`

---

## Task 4 — End-to-end smoke + integration

**Files:**
- Create: `scripts/m5c_smoke.py` (mirrors `scripts/m5_smoke.py` shape).
- Test: `tests/scripts/test_m5c_smoke.py` (self-test that runs the smoke via ASGI transport against a temporary DB + tmp_path data dir).

**Behaviour:**
- `scripts/m5c_smoke.py`:
  1. Spin app via `httpx.AsyncClient(transport=ASGITransport(app=app))`.
  2. Seed parent strategy in BACKTESTED with `add_indicator_request.json` sidecar (same seeding logic as M5b smoke, but with a fresh slug `smoke_m5c_parent` to avoid colliding with M5b dev-DB residue).
  3. `POST /strategies/{id}/author-plugin` → 202 → poll → done; capture `plugin_id`.
  4. `POST /plugins/{plugin_id}/evaluate` → 202 → poll → done; assert plugin VERIFIED.
  5. **`POST /strategies/{id}/reiterate-with-plugin` with body `{plugin_slug: <slug>}`** → 202 → poll → done.
  6. Assertions:
     - Child Strategy exists in DB.
     - `child.parent_strategy_id == parent.id`.
     - `child.current_state == "proposed"`.
     - Child `strategy.json` has plugin slug in the correct list-field (e.g. `indicators`).
     - Child `hypothesis.json` has the new iteration block with the template rationale.
     - Parent's `add_indicator_request.json` STILL exists (Decision D — append-only).
  7. Print `[m5c_smoke] PASSED` and `return 0`.
- Smoke uses `FunctionModel` stub for PluginAuthor LLM call (deterministic + offline — same pattern as M5b smoke).

**Test sketch (`tests/scripts/test_m5c_smoke.py`):**
1. `test_m5c_smoke_self_test` — import + run `scripts.m5c_smoke.main()` with `tmp_path` data dir + fresh in-memory SQLite; assert returns 0; assert child Strategy row exists; assert `child.parent_strategy_id` set.
2. `test_m5c_smoke_idempotent_against_existing_plugin` — pre-seed a Plugin with the slug the smoke would create; smoke should detect & error cleanly (NOT crash uncaught). This is the M5b-smoke regression we hit at session start.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/scripts/test_m5c_smoke.py -v
# After all tests green, manual run against dev DB:
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run python scripts/m5c_smoke.py
```
Expected: tests pass; manual script ends with `[m5c_smoke] PASSED`.

**Commit:** `feat(M5c): m5c smoke end-to-end (parent → plugin → reiterate → child)`

---

## End-of-session housekeeping

1. **Full pytest:** `cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest -q` → target ~257 passed (234 M5b baseline + 6 schema + 7 reiterate + 6 API + 2 smoke-self-test ≈ 21 new; final count may differ slightly with extra edge-case tests).
2. **alembic:** `VIRTUAL_ENV= uv run alembic current` → still at 0004 (no migration in M5c — schema unchanged, list-fields are JSON-payload-only).
3. **App importable:** `VIRTUAL_ENV= uv run python -c "from fwbg_agents.main import app"` → no error.
4. **graphify update:** `cd ~/Projekte/fwbg-agents && graphify update .` → refresh graph (AST-only, no API).
5. **Design-doc status:** add M5c row to implementation-status table in `fwbg/docs/plans/2026-06-23-fwbg-agents-design.md` with the four final commit hashes.
6. **This plan-doc:** update status header to `✓ done`.
7. **Memory:** update `project_fwbg_agents.md` with M5c-block (mirrors M5b-block structure: commits, test count, locked decisions, what's next).
8. **Memory:** create `reference_fwbg_agents_m5c_plan.md` — commit table + locked decisions (analog `reference_fwbg_agents_m5b_plan.md`).
9. **MEMORY.md index:** add line for the new reference memory.
10. **Manual smoke:** `VIRTUAL_ENV= uv run python scripts/m5c_smoke.py` against dev DB → `[m5c_smoke] PASSED`. (If dev DB has stale `smoke_m5c_*` rows from a re-run, the smoke errors cleanly per Task 4 Test 2 — manually `sqlite3 ... "DELETE FROM strategies WHERE slug LIKE 'smoke_m5c_%'"` to reset.)

**Discipline reminders:**
- One commit per task. English. NO Claude footers. Each commit: pytest green + alembic clean.
- Behaviour-only tests — no LLM mocking, no implementation-detail asserts. Schema validation tests pin the contract.
- Use `graphify query "..."` BEFORE reading source files (project convention).
- Use `superpowers:executing-plans` to drive the task list.

---

## M6 preview (next session — do NOT implement now)

**M6 = PaperTrading agent loop** (unchanged from M5b plan-doc preview):
- Polls `Strategy.current_state == PAPER_TRADING` rows.
- Pulls live-paper-trade metrics from fwbg paper-trading endpoint (TBD).
- Re-runs the Analyst on the paper-trade window using the same criteria YAML.
- Promote-to-live / Abandon / Tune sidecars same as M3 Analyst, but operating on paper-trade metrics rather than backtest.
- New `PROMOTE_LIVE_TRADING` state with a hard human-approval gate ([[feedback-risk-conscious-trading]]).
- ~3-4 tasks: poller infra, analyst-paper variant, promote-live gate, smoke.

After M5c is done, M5a + M5b + M5c form a closed plugin-authoring loop. M6 closes the deployment loop from paper to live.
