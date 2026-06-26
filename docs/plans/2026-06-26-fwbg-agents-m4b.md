# fwbg-agents M4b — Researcher Search Resilience + Parallel Hypothesis Fan-out

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** ✓ done 2026-06-26 — five commits `9ad4cc2`, `cce8f28`, `4f46df5`, `9b6b6fc`, `053b144` in `~/Projekte/fwbg-agents/` on branch `feat/m4b-researcher-resilience` (off `develop`, see branch-strategy deviation below). 384 tests green (370 baseline → +14 net new). `scripts/m4_smoke.py` still exits 0 (clean skip, unchanged). `scripts/m4b_smoke.py` verified end-to-end against a migrated local DB through the real LLM-call boundary — see deviations below for why it doesn't print a literal `[m4b_smoke] PASSED` in this environment.

**Builds on:** M4 (`docs/plans/2026-06-24-fwbg-agents-m4.md`, done). Touches only `agents/researcher.py`, `tools/web_search.py`, `orchestrator/research_flow.py`, `api/research.py`, `config.py`. No schema migration.

**Motivation:** Comparing fwbg-agents' Researcher against bytedance/deer-flow's "lead agent + sub-agents" and pluggable-search-backend patterns surfaced two concrete, low-risk gaps (full discussion in chat 2026-06-26):

1. `tools/web_search.py` is Tavily-only. `search_web_tool` in `agents/researcher.py:118-133` silently returns `[]` when Tavily is unavailable or quota'd (`TavilyUnavailable` / any exception → empty list). The M4 design doc already named Brave as the documented secondary fallback (§10) but never built it.
2. `Researcher.run()` produces exactly one hypothesis per call. If `validate_hypothesis` rejects it (prior-art conflict without differentiation), the whole `AgentRun` fails — no second attempt within the same orchestration call. deer-flow's lead-agent pattern (spawn N sub-agents in parallel, synthesize) suggests generating several candidates per research request and keeping the first valid one instead of failing the run on one rejection.

**Explicit non-goal (locked by the user 2026-06-26):** No sandboxed code execution inside the Researcher / search tool. The Researcher's job stays "find + write up a hypothesis well enough for downstream agents to act on." Numeric/code validation of a strategy already happens downstream via the existing Runner (`agents/runner.py`, M3) against the real fwbg backtest tool — that pipeline is untouched by this plan. Also not revisiting MCP — already rejected in the original design doc's open-questions table ("MCP in fwbg einbauen? Nein — fwbg's HTTP-API ist schon die Schnittstelle").

**Goal:** (1) Researcher keeps producing literature-grounded sources when Tavily is down/quota'd, by falling back to a second search provider. (2) A single `/research/brief` call generates `RESEARCHER_FANOUT_N` candidate hypotheses concurrently and persists the first one that survives `validate_hypothesis`, instead of failing outright on the first rejection.

**Tech stack:** unchanged (Python 3.13, pydantic-ai, httpx, SQLAlchemy 2.x async, FastAPI, pytest). No new dependency for Brave (plain `httpx` REST call, same shape as `TavilyClient`).

---

## Locked decisions

- **(A) Two search backends only.** Tavily (primary) + Brave (secondary), per the M4 design doc's own §10 fallback note. The Anthropic built-in `web_search` tool stays parked — proxy compatibility still unverified, third backend is not worth the complexity right now.
- **(B) Fixed provider order, no smart routing.** `FallbackSearchClient` tries providers in the configured order and stops at the first that returns without raising. No scoring/merging of results across providers.
- **(C) `httpx.AsyncClient` is safe to share across concurrent fan-out tasks; `AsyncSession` is not.** Each fan-out candidate opens its own session via the existing `SessionLocal` sessionmaker (`persistence/database.py`) — mirrors how `api/research.py`'s background tasks already isolate sessions per top-level call. The search client (httpx-backed) is constructed once and shared across all concurrent candidates.
- **(D) Selection = first-valid-wins, in submission order.** No extra LLM call to rank multiple valid candidates — keep it deterministic and avoid the added token cost.
- **(E) Fan-out is config, not API-surface.** `RESEARCHER_FANOUT_N` lives in `Settings` (env-overridable), same pattern as M5d's `plugin_impl_max_rounds`. `/research/brief`'s request/response shape is unchanged.
- **(F) Rejected candidates get their own `AgentRun(status=FAILED)` row** (this already happens — each `Researcher.run()` call creates one). No new table for "rejected hypotheses"; existing audit trail is sufficient.
- **(G) Concrete-before-generic.** Fan-out is implemented only for the Researcher. Not generalized to Translator/Analyst/PluginPlanner (same reasoning as M5d's decision H).
- **(H) No semantic diversity prompting between candidates in v1.** Each candidate gets the identical `ResearcherInput`; diversity comes from the LLM's own sampling. Revisit only if duplicate hypotheses across candidates become a real problem.

---

## Task 1 — `SearchProvider` protocol + package restructure

**Files:**
- New: `src/fwbg_agents/tools/search/__init__.py` — re-exports `SearchResult`, `SearchProvider`, `SearchUnavailable`, `TavilyClient`, `BraveClient`, `FallbackSearchClient`.
- New: `src/fwbg_agents/tools/search/base.py` — `SearchResult` (moved verbatim from current `web_search.py`), `SearchProvider` Protocol (`name: str` attribute, `async def search(query, *, max_results=5, session=None, agent_run_id=None) -> list[SearchResult]`), `SearchUnavailable(RuntimeError)` (generalizes today's `TavilyUnavailable`).
- Move: `src/fwbg_agents/tools/web_search.py` → `src/fwbg_agents/tools/search/tavily.py`. `TavilyClient` unchanged in behavior, raises `SearchUnavailable` instead of `TavilyUnavailable`, `name = "tavily"`. Quota logging (`_log_quota`, `get_quota_usage`) moves with it but takes a `provider: str` parameter instead of the hardcoded `TAVILY_MODEL_NAME` constant — callers pass `f"{provider}-search"`.
- Delete: old `src/fwbg_agents/tools/web_search.py` once the move is verified (grep for remaining `from fwbg_agents.tools.web_search import` and update every hit — known call sites: `agents/researcher.py`, `orchestrator/research_flow.py`, `api/research.py`, `tests/tools/test_web_search.py`, `tests/agents/test_researcher.py`).

**Behavior:** Pure refactor, no behavior change for Tavily-only usage. `TavilyUnavailable` name is retired (not aliased — single-path code per M5d precedent); every catcher updates to `SearchUnavailable`.

**Test sketch:**
- Existing `tests/tools/test_web_search.py` content moves to `tests/tools/search/test_tavily.py`, import paths updated, all assertions unchanged.
- New `tests/tools/search/test_base.py`: `TavilyClient` (and later `BraveClient`) structurally satisfy the `SearchProvider` protocol (`isinstance(client, SearchProvider)` if using `@runtime_checkable`, or an explicit attribute/method check).

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/tools/search/ -v
```

**Commit:** `refactor(M4b): move tools/web_search.py into tools/search/ package, generalize TavilyUnavailable -> SearchUnavailable`

---

## Task 2 — `BraveClient` + `FallbackSearchClient`

**Files:**
- New: `src/fwbg_agents/tools/search/brave.py` — `BraveClient(api_key, base_url="https://api.search.brave.com/res/v1/web/search", http=None, timeout=30.0)`. `name = "brave"`. Raises `SearchUnavailable` if `api_key` is `None`. Sends `X-Subscription-Token` header; maps Brave's `web.results[].{url,title,description}` into `SearchResult(url, title, content_snippet=description, score=1.0)` (Brave has no relevance score field — fixed `1.0`, document this in a comment since it's a real API limitation, not an oversight). Same `_log_quota`/quota-window reuse as Tavily, with `provider="brave"`.
- New: `src/fwbg_agents/tools/search/fallback.py` — `FallbackSearchClient(providers: list[SearchProvider])`. `async def search(query, **kwargs) -> list[SearchResult]`: iterate `providers` in order, `try: return await provider.search(query, **kwargs)`, `except (SearchUnavailable, httpx.HTTPStatusError) as exc: log.warning(...); continue`. After exhausting all providers, `return []` (preserves today's graceful-degradation behavior — research continues without sources rather than failing the whole run). Logs which provider actually served each call (for quota analytics / debugging silent fallbacks).
- Modify: `src/fwbg_agents/config.py` — add `brave_api_key: str | None = None` next to `tavily_api_key` in the "Web research" section.

**Test sketch:**
- `test_brave_search_happy_path` — mocked httpx response, asserts mapping into `SearchResult`.
- `test_brave_raises_search_unavailable_without_key`.
- `test_fallback_uses_primary_when_healthy` — primary returns results, secondary never called (assert via mock call count).
- `test_fallback_falls_back_on_search_unavailable` — primary raises `SearchUnavailable`, secondary returns results.
- `test_fallback_falls_back_on_http_429` — primary raises `httpx.HTTPStatusError` (quota exceeded), secondary returns results.
- `test_fallback_returns_empty_when_all_providers_fail` — both raise, result is `[]`, no exception propagates.
- `test_fallback_logs_serving_provider_for_quota` — quota log row's `model` column is `"brave-search"` when Tavily failed over to Brave.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/tools/search/ -v
```

**Commit:** `feat(M4b): BraveClient + FallbackSearchClient — primary/secondary search resilience`

---

## Task 3 — Wire the fallback client into Researcher + API

**Files:**
- Modify: `src/fwbg_agents/agents/researcher.py` — constructor param `tavily: TavilyClient | None` → `search_client: SearchProvider | None`. `search_web_tool` calls `search_client.search(query, session=session, agent_run_id=agent_run_id)`, catches `SearchUnavailable` (not `TavilyUnavailable`). Guard for "not configured" becomes `if search_client is None`.
- Modify: `src/fwbg_agents/orchestrator/research_flow.py` — `research_and_translate(..., tavily: TavilyClient | None)` → `search_client: SearchProvider | None`.
- Modify: `src/fwbg_agents/api/research.py::_run_research_background` — build `search_client = FallbackSearchClient([TavilyClient(settings.tavily_api_key), BraveClient(settings.brave_api_key)])`; close both underlying clients on the way out (extend or replace the existing `tavily.aclose()` call).
- Modify: all existing tests referencing `tavily=` kwarg or `TavilyClient` directly in `Researcher(...)`/`research_and_translate(...)` calls — rename to `search_client=`.

**Test sketch:**
- `test_researcher_falls_back_to_brave_when_tavily_unavailable` — pass a `FallbackSearchClient([raising_tavily_stub, working_brave_stub])` as `search_client`, assert the emitted hypothesis's `sources` field contains the Brave-sourced URL.
- Existing researcher/research_flow/API tests pass unchanged after the rename (regression guard).

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/agents/test_researcher.py tests/orchestrator/test_research_flow.py tests/api/test_research.py -v
```

**Commit:** `feat(M4b): Researcher + research_flow + API use FallbackSearchClient instead of bare TavilyClient`

---

## Task 4 — Parallel hypothesis fan-out in `research_and_translate`

**Files:**
- Modify: `src/fwbg_agents/orchestrator/research_flow.py` — new private helper:
  ```python
  async def _generate_valid_hypothesis(
      input: ResearcherInput,
      *,
      model: Model | None,
      search_client: SearchProvider | None,
      fanout_n: int,
  ) -> ResearcherHypothesis:
      async def _one_candidate() -> ResearcherHypothesis:
          async with SessionLocal() as candidate_session:
              researcher = Researcher(candidate_session, model=model, search_client=search_client)
              return await researcher.run(input)

      results = await asyncio.gather(
          *(_one_candidate() for _ in range(fanout_n)),
          return_exceptions=True,
      )
      for result in results:
          if not isinstance(result, Exception):
              return result
      reasons = "; ".join(str(r) for r in results if isinstance(r, Exception))
      raise ResearcherFailed(f"all {fanout_n} candidates rejected: {reasons}")
  ```
  `research_and_translate` calls this instead of `Researcher(session, ...).run(input)` directly; everything after (slug generation, Strategy/Tag/Transition persistence, Translator) stays on the caller's existing `session`, unchanged.
- Modify: `src/fwbg_agents/config.py` — add `researcher_fanout_n: int = Field(default=2, ge=1, le=5, description="Parallel hypothesis candidates per /research/brief call; first to pass validate_hypothesis wins.")`.
- Modify: `src/fwbg_agents/api/research.py::_run_research_background` — pass `fanout_n=settings.researcher_fanout_n` through to `research_and_translate`.

**Test sketch:**
- `test_fanout_returns_first_valid_candidate` — `fanout_n=3`, candidates 1+2 raise `ResearcherFailed` via stubbed `Researcher.run` (monkeypatch or `FunctionModel` tuned to violate prior-art), candidate 3 succeeds → `research_and_translate` returns candidate 3's Strategy id.
- `test_fanout_creates_one_agent_run_per_candidate` — after the above, query `AgentRun.agent_name == "researcher"` rows: 2 `FAILED`, 1 `DONE`.
- `test_fanout_all_candidates_fail_raises_with_combined_reasons` — all N raise → `ResearcherFailed` message contains all N reasons.
- `test_fanout_n_equals_1_matches_today` — `fanout_n=1` behaves identically to pre-M4b `research_and_translate` (regression guard — single candidate, no parallelism overhead, same AgentRun count as before).
- `test_fanout_n_from_settings_env` — `monkeypatch.setenv("RESEARCHER_FANOUT_N", "3")` → `Settings().researcher_fanout_n == 3`.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/orchestrator/test_research_flow.py tests/test_config.py -v
```

**Commit:** `feat(M4b): research_and_translate fans out RESEARCHER_FANOUT_N parallel candidates, first-valid-wins`

---

## Task 5 — End-to-end smoke `scripts/m4b_smoke.py`

**Files:**
- New: `scripts/m4b_smoke.py` (mirrors `m4_smoke.py` structure — ASGI transport, graceful skip if neither `TAVILY_API_KEY` nor `BRAVE_API_KEY` set).

**Behavior:**
1. `POST /research/brief` with `RESEARCHER_FANOUT_N=2` env override.
2. Poll the returned `AgentRun` until terminal.
3. On success: assert ≥1 `AgentRun(agent_name="researcher")` row is `DONE`, assert the resulting Strategy has `hypothesis_path` set and the file round-trips into `ResearcherHypothesis`.
4. If both `TAVILY_API_KEY` and `BRAVE_API_KEY` are unset, assert the run still completes (Researcher proceeds with zero web sources, per existing graceful-degradation behavior) — proves the fallback chain doesn't hang or crash when fully unconfigured.
5. Print `[m4b_smoke] PASSED`.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && uvicorn fwbg_agents.api.main:app &
sleep 2
VIRTUAL_ENV= uv run python scripts/m4b_smoke.py
```

**Commit:** `feat(M4b): scripts/m4b_smoke.py end-to-end fallback-search + fan-out smoke`

---

## Out-of-scope

- **Sandboxed code execution / numeric backtesting inside the Researcher.** Explicitly rejected by the user 2026-06-26 — the Runner/Analyst pipeline (M3, against the real fwbg tool) already owns that, downstream of the hypothesis the Researcher hands off.
- **Semantic diversity prompting between fan-out candidates.** v1 relies on independent LLM sampling only (decision H).
- **LLM-judged ranking among multiple valid candidates.** First-valid-wins only (decision D).
- **A third search backend** (Anthropic built-in `web_search`). Still parked — proxy compatibility unverified (same open item as the original M4 design doc).
- **MCP.** Already rejected in the main design doc's open-questions table; deer-flow's MCP-tool-integration pattern doesn't change that answer for fwbg-agents' purposes.
- **Generalizing fan-out to Translator/Analyst/PluginPlanner.** Concrete-before-generic (decision G); revisit only after 2-3 more agents would benefit, per the M5d precedent.
- **deer-flow's skills system, IM channel integrations, multi-output-format generation (slides/web pages), multi-LLM-provider config.** None of these fit fwbg-agents' single-proxy, internal-API, dashboard-reviewed design — discussed and rejected in the 2026-06-26 chat before this plan was written.

## Acceptance criteria (Definition of Done) — ✓ met (one caveat, see deviations)

- All 5 tasks committed. ✓
- `uv run pytest -q` in fwbg-agents: ≥ 370 baseline + new tests (search package tests + fan-out tests), 0 failures. ✓ 384 passed, 1 skipped.
- `scripts/m4b_smoke.py` ends `[m4b_smoke] PASSED`. ⚠ Not literally reached in this sandbox — see deviations below. The script is correct and was verified live through the real LLM-call boundary.
- `scripts/m4_smoke.py` still passes (no regression in the non-fanout, non-fallback path when `RESEARCHER_FANOUT_N=1` and only Tavily configured). ✓ exits 0 (clean skip, `TAVILY_API_KEY` unset in this environment — same pre-existing limitation as before M4b).
- No remaining references to `TavilyUnavailable` or `tools/web_search.py` (grep clean). ✓
- `RESEARCHER_FANOUT_N=1` is byte-for-byte equivalent to pre-M4b single-candidate behavior (explicit regression test, Task 4). ✓ `test_fanout_n_equals_1_matches_today`.

## Pre-checks (verify at session start)

```bash
cd ~/Projekte/fwbg-agents && git log -1 --format="%H %s"
# expect: 879e6f7... Merge pull request #1 from haexhub/feat/m5d-planner-implementer-split
git status --short
# expect: only untracked Dockerfile / .dockerignore (unrelated, pre-existing — leave alone)
git branch --show-current
# expect: main
VIRTUAL_ENV= uv run pytest -q
# expect: 370 passed, 1 skipped
```

**Pre-check result at session start (2026-06-26):** `git log -1` and `git branch` did **not** match the expected values above — see deviations below (branch-strategy item). `pytest -q` matched exactly (370 passed, 1 skipped), confirming no functional drift, just infra/lint commits on top of the expected `879e6f7` (which is an ancestor of the actual `HEAD`). Flagged to the user before proceeding, per instruction; user chose to branch off `develop` instead of `main`.

## Implementation deviations from the original plan (worth noting)

- **Branch strategy:** the repo adopted Gitflow (`release-please.yml`: *"feature branches → PR → develop → PR → main → release"*) between when this plan was written and when it was implemented — `git branch --show-current` was `develop`, not `main`, and `git log -1` was several infra-only commits (CI bootstrap, dependabot bumps, ruff lint cleanup) ahead of the expected `879e6f7`. Confirmed `879e6f7` is an ancestor of `HEAD` and the pytest baseline matched exactly (370 passed, 1 skipped), so no functional drift — just process. Implemented on a new branch `feat/m4b-researcher-resilience` off `develop`, to be PR'd into `develop` rather than committing directly, per the user's explicit choice when asked.
- **Skill not installed:** the plan's `REQUIRED SUB-SKILL: superpowers:executing-plans` is not installed in this environment (no matching marketplace/plugin found). Implemented the plan task-by-task manually instead (read task → implement → run the task's Verify block → commit with the task's suggested message) — same workflow the skill would have driven.
- **Task 1 — exception naming:** `SearchUnavailable` (the plan's literal name) was renamed to `SearchUnavailableError` to satisfy this repo's `ruff` `N818` rule (exception classes must end in `Error`), enforced repo-wide since the recent lint cleanup (`39d5b76`) and consistent with every other exception in the codebase (`RunnerError`, `TranslatorError`, `ResearcherError`, etc. — no exception lacks the suffix).
- **Task 1/2 — quota-logging location:** `_log_quota` and `get_quota_usage` were generalized in place in `tools/search/tavily.py` (parameterized on `provider: str`, default `"tavily"` for backwards-compatible call sites) rather than moved to `base.py`. `brave.py` imports `_log_quota` from `tavily.py` directly — matches the plan's Task 2 wording ("Same `_log_quota`/quota-window reuse as Tavily") literally, and `get_quota_usage` was previously unused outside tests so the added `provider` kwarg is non-breaking.
- **Task 4 — exception naming:** the plan's pseudocode names the all-candidates-failed exception `ResearcherFailed` (likely copied from a stale view of the codebase — the actual exception is `ResearcherError`, not `*Failed`, after an earlier rename). Implemented as `ResearcherFanoutExhaustedError(RuntimeError)` in `research_flow.py`, for the same `N818` reason as above and to avoid clashing with `agents/researcher.py`'s existing `ResearcherError` (single-candidate validation failure) — the two are semantically distinct (one rejection vs. all-candidates-exhausted).
- **Task 4 — test infra:** `_generate_valid_hypothesis` opens fan-out candidate sessions via the module-level `SessionLocal` (decision C, as specified). This is the established repo pattern for background-task code under test (mirrors `tests/scripts/test_m5c_smoke.py` / `test_m5d_smoke.py`) — `tests/orchestrator/test_research_flow.py`'s `db` fixture now also monkeypatches `research_flow.SessionLocal` to the test's own tmp_path-bound sessionmaker, so fan-out candidates' `AgentRun` rows land in the same DB the test asserts against. Not called out explicitly in the plan's Task 4 file list, but required for the existing tests to keep working with the new fan-out helper.
- **Task 5 — smoke skip behavior:** the plan's one-line file description ("graceful skip if neither `TAVILY_API_KEY` nor `BRAVE_API_KEY` set") conflicts with its own detailed Behavior step 4 ("assert the run still completes ... proves the fallback chain doesn't hang or crash when fully unconfigured"). Implemented per the more specific Behavior section: the script never skips on missing search keys, it always runs the full flow and asserts graceful completion either way — more useful for local/CI runs without search-provider secrets configured.
- **Task 5 — live verification scope:** `scripts/m4b_smoke.py` was verified live against a freshly-migrated local `data/state.db` (user-approved `alembic upgrade head`) through `POST /research/brief` → background task → fan-out of 2 real `Researcher.run()` candidates → real (failed) LLM calls. Confirmed end-to-end: both candidates correctly raised on `Connection error` (haex-claude-proxy unreachable from this sandbox), `ResearcherFanoutExhaustedError` correctly combined both reasons, the orchestration `AgentRun` was correctly marked `failed` with that message, and the smoke script correctly detected the failure and exited 1. This validates every code path up to the external LLM boundary — the literal `[m4b_smoke] PASSED` output requires running it where the proxy is reachable, which the user will need to confirm separately.

## Final commit table

| # | Commit | Description |
|---|--------|-------------|
| 1 | `9ad4cc2` | refactor(M4b): move tools/web_search.py into tools/search/ package, generalize TavilyUnavailable -> SearchUnavailable |
| 2 | `cce8f28` | feat(M4b): BraveClient + FallbackSearchClient — primary/secondary search resilience |
| 3 | `4f46df5` | feat(M4b): Researcher + research_flow + API use FallbackSearchClient instead of bare TavilyClient |
| 4 | `9b6b6fc` | feat(M4b): research_and_translate fans out RESEARCHER_FANOUT_N parallel candidates, first-valid-wins |
| 5 | `053b144` | feat(M4b): scripts/m4b_smoke.py end-to-end fallback-search + fan-out smoke |
