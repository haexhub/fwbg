# fwbg-agents M3 — Runner + Analyst Implementation Plan

**Status**: ✓ done 2026-06-24, final commit `df33384` (78 tests green).
Cross-reference: design doc Implementation-Status row for M3.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans / superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** After M3, a human can POST a strategy.json to fwbg-agents, the Runner deterministically drives fwbg's backtest via HTTP, persists results, transitions to `backtested`, then the Analyst (LLM) emits a typed recommendation that the orchestrator validates against hard rules and applies (promote / abandon / record-tuning).

**Architecture:**
- Runner = deterministic Python class (no LLM); LLM lives only in Analyst. Risk gate is the orchestrator's `validate_and_apply`, which re-uses `check_backtest_criteria` from M2.
- New tables `agent_run` + `llm_call` via Alembic 0003. agent_run tracks every Runner/Analyst invocation; llm_call records token/cost per LLM call.
- File layout: `data/strategies/<slug>/iteration_<NNN>/{strategy.json, fwbg_results.json, analyst_report.md}`. M3 always writes iteration 1 (no auto-increment yet — Translator/M4 owns iteration bumps).
- fwbg integration: write strategy.json into `~/fwbg/strategies/configs/<slug>__it<NNN>.json` (fwbg expects a name), call `POST /api/runs/start`, poll `GET /api/runs/<id>/progress`, fetch `GET /api/runs/<id>` on completion.

**Tech Stack:** FastAPI, SQLAlchemy 2 async + alembic, httpx (async), pydantic-ai (Claude via haex-claude-proxy), pytest + pytest-asyncio.

**Decisions captured for M3:**
1. `iteration_count`: Runner does NOT increment. Each new strategy.json = iteration 1. M4 Translator bumps it.
2. `llm_call` table lives in this migration (Analyst is the first LLM consumer).
3. Analyst system prompt lives in `src/fwbg_agents/agents/prompts/analyst.md` — easier to iterate without code changes.

---

## Task 1: fwbg HTTP client

**Files:**
- Create: `src/fwbg_agents/tools/fwbg_client.py`
- Create: `tests/tools/__init__.py`, `tests/tools/test_fwbg_client.py`
- Modify: `src/fwbg_agents/config.py` — add `fwbg_strategies_dir`, `runner_poll_interval_seconds`, `runner_poll_timeout_seconds`.

**Shape:**
```python
class FwbgClient:
    def __init__(self, base_url: str, http: httpx.AsyncClient | None = None): ...
    async def start_run(self, strategy_name: str, *, asset_classes: list[str] | None = None) -> dict: ...
    async def get_progress(self, run_id: str) -> dict: ...
    async def get_run(self, run_id: str) -> dict: ...
```

**Tests** (httpx_mock via `respx` or hand-rolled MockTransport):
1. `start_run` posts `{strategy_name, asset_classes?}` → returns the job dict
2. `get_progress` returns parsed progress dict (status/progress fields)
3. `get_run` returns parsed run results
4. Non-200 raises `FwbgClientError` containing status + body

**Commit:** `feat(M3): fwbg HTTP client wrapper + tests`

---

## Task 2: Migration 0003 — agent_run + llm_call + ORM

**Files:**
- Create: `src/fwbg_agents/persistence/migrations/versions/0003_agent_run.py`
- Modify: `src/fwbg_agents/persistence/models.py` — add `AgentRun`, `LlmCall`, enum `AgentRunStatus`.
- Create: `tests/persistence/__init__.py`, `tests/persistence/test_agent_run.py`

**agent_run columns:** `id, agent_name (str), strategy_id (int|null FK→strategy.id), plugin_id (int|null FK→plugin.id), status (enum: pending/running/done/failed), started_at, ended_at?, input_artifact_path?, output_artifact_path?, error?, created_at`

**llm_call columns:** `id, agent_run_id (FK→agent_run.id), model (str), input_tokens (int), output_tokens (int), cost_usd (numeric|null), latency_ms (int|null), created_at`

**Tests:**
1. Migration `alembic upgrade head` then `alembic downgrade -1` is reversible
2. Insert agent_run + child llm_call, FK enforced

**Commit:** `feat(M3): agent_run + llm_call tables + ORM`

---

## Task 3: POST /strategies seeding endpoint

**Files:**
- Modify: `src/fwbg_agents/api/strategies.py` — add `POST /strategies` (manual researcher-skip).
- Create: `tests/api/test_strategy_create.py`

**Body:** `{slug, asset_class, strategy_family, strategy_json, tags?}`
**Behavior:**
- Insert Strategy in PROPOSED, no transition row (this is creation, not state change).
- mkdir `data/strategies/<slug>/iteration_001/` and write `strategy.json`.
- Insert StrategyTag rows.
- Return `{id, slug, iteration_dir}`.
- Duplicate slug → 409.

**Tests:**
1. POST returns 201, strategy is queryable in `/strategies`, file on disk.
2. Tags persist.
3. Duplicate slug → 409.
4. POST does NOT create a transition row.

**Commit:** `feat(M3): POST /strategies for manual seeding`

---

## Task 4: Runner agent

**Files:**
- Create: `src/fwbg_agents/agents/runner.py`
- Create: `tests/agents/test_runner.py`

**Shape:**
```python
class RunnerResult(BaseModel):
    fwbg_run_id: str
    results_path: str            # path of saved fwbg_results.json
    metrics: dict                # extracted from results.json for the gate
    iteration_dir: str

class Runner:
    def __init__(self, fwbg_client: FwbgClient, session_factory, *, poll_interval_s: float, poll_timeout_s: float): ...
    async def run(self, strategy: Strategy) -> RunnerResult:
        # 1. Insert AgentRun(status=running, agent_name="runner", strategy_id=...)
        # 2. Read data/strategies/<slug>/iteration_001/strategy.json
        # 3. Copy to fwbg's strategies_dir as f"{slug}__it001.json" (sanitized)
        # 4. fwbg_client.start_run(name) → job_id
        # 5. Poll until status in {completed, failed} or timeout
        # 6. If failed/timeout → AgentRun.status=failed, raise RunnerFailed; NO transition
        # 7. Fetch full run, write fwbg_results.json into iteration dir
        # 8. Extract metrics dict (sharpe, profit_factor, mc_pvalue, etc.) from results
        # 9. transition_strategy(strategy, BACKTESTED, payload={"fwbg_run_id": ..., "results_path": ..., "backtest_metrics": metrics})
        # 10. AgentRun.status=done, output_artifact_path=results path
```

NOTE: The M2 guard `_guard_strategy_proposed_to_backtested` is currently a no-op. Runner attaches metrics to payload so they're queryable in transition history.

**Metrics extraction:** results.json contains `assets[symbol].unified_metrics` per symbol. For now, pick the **best symbol's metrics** (max sharpe) — single-symbol aggregation. This keeps M3 simple; refine in M5.

**Tests** (with a mock FwbgClient):
1. Happy path: poll returns "running" twice, then "completed". Strategy ends in BACKTESTED, agent_run.status=done, fwbg_results.json exists.
2. fwbg returns status="failed" → no transition, AgentRun.status=failed, RunnerFailed raised.
3. Polling exceeds timeout → no transition, status=failed.
4. Strategy missing iteration_001/strategy.json → ValueError.

**Commit:** `feat(M3): Runner agent + tests`

---

## Task 5: Analyst agent

**Files:**
- Create: `src/fwbg_agents/agents/analyst.py`
- Create: `src/fwbg_agents/agents/prompts/analyst.md` (system prompt with Jinja-style `{{ variable }}` placeholders)
- Create: `tests/agents/test_analyst.py`

**Recommendation union types** (pydantic-ai structured output):
```python
class Promote(BaseModel):
    kind: Literal["promote"] = "promote"
    confidence: float   # 0..1
    reasoning: str

class Abandon(BaseModel):
    kind: Literal["abandon"] = "abandon"
    confidence: float
    reasoning: str
    post_mortem_summary: str
    lessons: list[str]

class TuneParams(BaseModel):
    kind: Literal["tune_params"] = "tune_params"
    confidence: float
    reasoning: str
    param: str
    new_range: list[float | int]

class ChangeExit(BaseModel):
    kind: Literal["change_exit"] = "change_exit"
    confidence: float
    reasoning: str
    from_exit: str
    to_exit: str

AnalystRecommendation = Promote | Abandon | TuneParams | ChangeExit
```

**Shape:**
```python
class Analyst:
    def __init__(self, session_factory, model=None, prompt_path=None): ...
    async def analyze(self, strategy: Strategy) -> AnalystRecommendation:
        # 1. AgentRun(agent_name="analyst", status=running)
        # 2. Load latest iteration_dir, fwbg_results.json, criteria YAML
        # 3. Render prompt with strategy + metrics + criteria
        # 4. Call pydantic-ai Agent with output_type=AnalystRecommendation
        # 5. Persist LlmCall(agent_run_id=..., model=..., tokens, cost=null)
        # 6. AgentRun.status=done, output_artifact_path=analyst_report.md (rendered Markdown)
```

**Tests** (with pydantic-ai `TestModel` or `FunctionModel` returning fixed structured outputs):
1. TestModel returns Promote → analyst returns Promote, AgentRun done, LlmCall row written.
2. TestModel returns Abandon → analyst returns Abandon with post_mortem_summary.
3. TestModel returns TuneParams.
4. Missing fwbg_results.json → FileNotFoundError, AgentRun status=failed.

**Commit:** `feat(M3): Analyst agent (LLM) + recommendation schema + tests`

---

## Task 6: Recommendation validator

**Files:**
- Create: `src/fwbg_agents/orchestrator/recommendations.py`
- Create: `tests/orchestrator/test_recommendations.py`

**Shape:**
```python
async def validate_and_apply(
    session, strategy: Strategy, rec: AnalystRecommendation
) -> Transition | None:
    """Returns the transition row created, or None for tune/change_exit (record-only)."""
    match rec:
      case Promote():     # criteria-gated transition BACKTESTED → PAPER_TRADING
      case Abandon():     # write post_mortem.yaml, transition → ABANDONED
      case TuneParams() | ChangeExit():
                          # Recommendation recorded only — Translator (M4) re-iterates.
                          # M3 persists rec via an Insert into transition with same from/to=BACKTESTED?
                          # NO — transitions are state changes. Store as JSON in a sidecar
                          # data/strategies/<slug>/iteration_NNN/analyst_recommendation.json
```

**Tests:**
1. Promote w/ passing metrics → transition to PAPER_TRADING + payload contains recommendation.
2. Promote w/ failing metrics → InvalidTransition raised, no state change.
3. Abandon → post_mortem.yaml written, transition to ABANDONED, current_state=abandoned.
4. TuneParams → no transition, sidecar JSON written.
5. ChangeExit → same.

**Commit:** `feat(M3): recommendation validator + apply + tests`

---

## Task 7: Runs API endpoints

**Files:**
- Create: `src/fwbg_agents/api/runs.py`
- Modify: `src/fwbg_agents/main.py` — register router
- Create: `tests/api/test_runs.py`

**Endpoints:**
- `POST /strategies/{id}/run` — spawn Runner as background task (FastAPI `BackgroundTasks` or `asyncio.create_task`); 202 with `{agent_run_id}`.
- `POST /strategies/{id}/analyze` — spawn Analyst; 202 with `{agent_run_id}`. Requires latest iteration has `fwbg_results.json`.
- `GET /agents/runs/{id}` — Status + output paths + error.

**Tests** (with mocked Runner/Analyst that records calls but no actual fwbg HTTP):
1. POST /strategies/{id}/run → 202, agent_run row created, background task fired.
2. POST /strategies/{id}/analyze without results → 409.
3. GET /agents/runs/{id} returns current state.

**Commit:** `feat(M3): /strategies/{id}/run + /analyze + /agents/runs endpoints`

---

## Task 8: scripts/m3_smoke.py (real fwbg + real LLM)

**Files:**
- Create: `scripts/m3_smoke.py`

**Flow:**
1. POST /strategies with a real fwbg strategy.json (use one from `~/fwbg/strategies/configs/` as template — e.g. a simple orb_scalping variant).
2. POST /strategies/{id}/run → wait until agent_run done (poll).
3. Assert: strategy.current_state == backtested, fwbg_results.json exists.
4. POST /strategies/{id}/analyze → wait until done.
5. Print recommendation kind + reasoning. If Promote → orchestrator applies → assert state == paper_trading OR rejected with reason logged.

**Verification:** Run live against `localhost:8420` fwbg + haex-claude-proxy LLM. Don't fail the script if metrics don't allow promotion — that's a valid Abandon/TuneParams path.

**Commit:** `chore(M3): smoke script driving real fwbg + real LLM end-to-end`

---

## Final verification

- `VIRTUAL_ENV= uv run pytest -q` — expect ~60+ tests green.
- `VIRTUAL_ENV= uv run alembic upgrade head` — 5 tables now: calibration_run, strategy, plugin, transition, strategy_tag, agent_run, llm_call (7 actually).
- Run smoke script (manual).
- Update design doc + memory with M3 done + commit hash.
