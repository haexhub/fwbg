# fwbg-agents M4 — Researcher + Translator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. After each task, run the tests for that task; commit immediately on green. Do not batch commits across tasks.

**Status:** In progress (2026-06-24). Builds on M3 (commit `df33384`). Tasks 1-6 of 8 complete, latest commit `ed2b59a`. 147 tests green (78 baseline + 69 new).

**Done in this session:**
- Task 1 `1e9707e` — `orchestrator/prior_art.py` (tag-Jaccard similarity, no LLM, 11 tests)
- Task 2 `4d5041a` — `orchestrator/hypotheses.py` (ResearcherHypothesis schema, validate_hypothesis, generate_slug — 13 tests)
- Task 3 `3991d4f` — `tools/web_search.py` (Tavily client + quota via `llm_call(model='tavily-search')` — 9 tests)
- Task 4 `6dd3093` — `agents/researcher.py` + `agents/prompts/researcher.md` (LLM agent with lookup_prior_art + search_web tools, hard anti-redundancy gate — 5 tests)
- Task 5a `3d699da` — `orchestrator/strategy_validator.py` (lightweight structural check + plugin catalog — 21 tests)
- Task 5b `35324cf` — `agents/translator.py` fresh-mode + `agents/prompts/translator.md` (LLM, canonical slug enforced, spec.md written — 4 tests)
- Task 6 `ed2b59a` — Translator reiterate-mode (deterministic, parent_strategy_id lineage) + extended `Analyst.ChangeExit.new_exit_strategy` — 6 tests

**Open in next session:**
- Task 7 — `orchestrator/research_flow.py` (orchestration glue: Researcher → persist Strategy/Tags/hypothesis.json → Translator.run_fresh) + `api/research.py` (`POST /research/brief`, `POST /strategies/{id}/reiterate`, `GET /hypotheses`) + wire router in `api/__init__.py`
- Task 8 — `scripts/m4_smoke.py` + final verification (`pytest -q`, `alembic upgrade head` — no migration expected, `python -c "from fwbg_agents.api import app"`)
- Housekeeping — design doc M4 row + memory updates were done at the end of the prior session (this file's status reflects that)

**Goal:** Autonomous hypothesis-loop: Researcher (LLM + Tavily + `lookup_prior_art`) emits a typed `ResearcherHypothesis`; Translator (LLM) converts it into a valid fwbg `strategy.json` written to `data/strategies/<slug>/iteration_001/`. Re-iterate path: Translator consumes an Analyst sidecar (`TuneParams` / `ChangeExit` from M3) and produces a child strategy with `parent_strategy_id` set.

**Architecture:**
- LLM only in `Researcher` and `Translator`. Slug generation, prior-art lookup, hypothesis validation, strategy-json structural validation = deterministic Python.
- `lookup_prior_art` is a required tool the Researcher calls BEFORE LLM-emitting the hypothesis; `validate_hypothesis()` rejects any hypothesis where prior-art exists but `differentiates_from` is empty.
- Re-iterate creates a NEW `Strategy` row with `parent_strategy_id = old.id`, `iteration_count = 1`, state = `PROPOSED`. Old strategy stays frozen in `BACKTESTED` with its sidecar. No new state-machine edges — clean audit trail.
- Tavily quota tracking via `llm_call(model="tavily-search")` convention (no schema change).
- strategy.json validation in M4 is **lightweight** (required top-level keys + plugin-slug-catalog cross-check). Full pydantic validation against `fwbg.core.config.StrategyConfig` is deferred — would require adding `fwbg` as a heavy dependency to the agents repo. The Runner is still the ultimate validator (fwbg rejects bad configs at start).

**Tech Stack:** pydantic-ai, httpx (Tavily), sqlalchemy async, FastAPI BackgroundTasks. No new top-level dependencies beyond what M3 already had.

---

## Design Decisions (locked)

1. **Re-iterate = new child Strategy**, not state regression. `parent_strategy_id` carries lineage. iteration_count starts at 1 per child (NOT a global counter across the lineage — the design doc and ORM both treat each Strategy row as standalone).
2. **No `fwbg` Python import** in the agents repo for now. Lightweight structural validator + plugin-slug catalog. Documented limitation. Promotion to full validation is its own future task.
3. **Tavily logged in `llm_call`** with `model="tavily-search"`. No migration. Quota count = `SELECT count(*) FROM llm_call WHERE model = 'tavily-search' AND created_at > now() - INTERVAL 30 days`.
4. **Anthropic web_search fallback NOT built in M4.** Documented in code comment + TODO at the Tavily client. Decision deferred until proxy compatibility verified — see design doc §16 question 1.
5. **Plugin catalog hardcoded** in `translator.py`. M4 only validates against `{orb_simple_v1, signal_orb_v1, walk_forward_intraday_v1, orb_scalping_v1, standard_v1}` (plus `forexsb` for datasource and a few timeframes). M5 PluginAuthor opens this up.
6. **Spec.md is plain Markdown**, written by the Translator. Sections: Goal, Inputs, Outputs, Acceptance Criteria, Implementation Notes. No spec-kit subprocess in M4.

---

## Task 1: `lookup_prior_art` (deterministic, no LLM)

**Files:**
- Create: `src/fwbg_agents/orchestrator/prior_art.py`
- Test: `tests/orchestrator/test_prior_art.py`

**Behavior:**
- `async def lookup_prior_art(session, strategy_family: str, asset_class: str, tags: list[str]) -> list[PriorArtMatch]`
- `PriorArtMatch`: pydantic with `slug`, `current_state`, `strategy_family`, `asset_class`, `tags_overlap: list[str]`, `jaccard: float`, `post_mortem_path: str | None`, `post_mortem_summary: str | None` (first 240 chars from post_mortem.yaml if any).
- Query: join `strategy` with `strategy_tag` on `strategy_id`. Filter by `asset_class` equality. `strategy_family` is a soft signal (boosts jaccard if equal, doesn't filter — researcher should see neighbouring families).
- Compute Jaccard similarity between the given tags and each strategy's tags. Return only matches with `jaccard >= 0.2` OR `strategy_family` match, sorted desc by jaccard. Cap at 20.
- TODO comment for Layer-2 (sqlite-vec embedding similarity) — not built in M4.

**Step 1: Write the failing test**

```python
# tests/orchestrator/test_prior_art.py
import pytest
import pytest_asyncio
from datetime import UTC, datetime
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fwbg_agents.persistence.database import Base
from fwbg_agents.persistence.models import Strategy, StrategyTag, StrategyState
from fwbg_agents.orchestrator.prior_art import lookup_prior_art


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


async def _seed(session, slug, family, asset_class, tags, state=StrategyState.PROPOSED):
    now = datetime.now(UTC)
    s = Strategy(slug=slug, current_state=state.value, asset_class=asset_class,
                 strategy_family=family, created_at=now, updated_at=now)
    session.add(s)
    await session.flush()
    for t in tags:
        session.add(StrategyTag(strategy_id=s.id, tag=t))
    await session.commit()
    return s


@pytest.mark.asyncio
async def test_returns_only_strategies_for_same_asset_class(db):
    await _seed(db, "a", "ORB", "FOREX", ["intraday", "momentum"])
    await _seed(db, "b", "ORB", "INDEX", ["intraday", "momentum"])
    matches = await lookup_prior_art(db, "ORB", "FOREX", ["intraday", "momentum"])
    assert [m.slug for m in matches] == ["a"]


@pytest.mark.asyncio
async def test_jaccard_below_threshold_is_filtered(db):
    await _seed(db, "a", "RSI", "FOREX", ["mean_reversion", "rsi"])
    matches = await lookup_prior_art(db, "ORB", "FOREX", ["momentum", "intraday"])
    assert matches == []


@pytest.mark.asyncio
async def test_same_family_matches_even_without_tag_overlap(db):
    await _seed(db, "a", "ORB", "FOREX", ["breakout"])
    matches = await lookup_prior_art(db, "ORB", "FOREX", ["intraday"])
    assert [m.slug for m in matches] == ["a"]


@pytest.mark.asyncio
async def test_results_sorted_by_jaccard_desc(db):
    await _seed(db, "low", "ORB", "FOREX", ["intraday", "x"])
    await _seed(db, "high", "ORB", "FOREX", ["intraday", "momentum", "trend"])
    matches = await lookup_prior_art(db, "ORB", "FOREX", ["intraday", "momentum"])
    assert [m.slug for m in matches] == ["high", "low"]


@pytest.mark.asyncio
async def test_post_mortem_summary_loaded_when_path_exists(db, tmp_path):
    pm = tmp_path / "post_mortem.yaml"
    pm.write_text("strategy_family: ORB\nabandon_reason: no edge in any regime\n")
    s = await _seed(db, "abandoned_a", "ORB", "FOREX", ["intraday"])
    s.post_mortem_path = str(pm)
    s.current_state = StrategyState.ABANDONED.value
    await db.commit()
    matches = await lookup_prior_art(db, "ORB", "FOREX", ["intraday"])
    assert matches[0].post_mortem_summary is not None
    assert "no edge" in matches[0].post_mortem_summary
```

**Step 2:** Run `VIRTUAL_ENV= uv run pytest tests/orchestrator/test_prior_art.py -v`. Expected: FAIL (module missing).

**Step 3: Implement `prior_art.py`**

Implementation contract (no full code listing — keep it pydantic + sqlalchemy idiomatic, mirror M3's recommendations.py style):
- `PriorArtMatch(BaseModel)` with the 8 fields above.
- `lookup_prior_art(session, strategy_family, asset_class, tags)` selects `Strategy` joined with `StrategyTag` (left outer), filters by asset_class, groups tags into a dict `{strategy_id: set[tag]}`, computes `jaccard(input_tags, found_tags)`, keeps if `jaccard >= 0.2` OR `family == strategy_family`, sorts desc, caps at 20, loads post_mortem.yaml summary if `post_mortem_path` is set.
- TODO comment block at the bottom: `# Layer 2 (sqlite-vec embedding similarity) — deferred to post-M4 ...`

**Step 4:** Run test, expect PASS.

**Step 5: Commit**

```bash
git add src/fwbg_agents/orchestrator/prior_art.py tests/orchestrator/test_prior_art.py
git commit -m "feat(M4): prior-art lookup (tag-based, no LLM)"
```

---

## Task 2: Hypothesis validator + slug generator (deterministic)

**Files:**
- Create: `src/fwbg_agents/orchestrator/hypotheses.py`
- Test: `tests/orchestrator/test_hypotheses.py`

**Behavior:**
- `class HypothesisRejected(ValueError)`.
- `class ResearcherHypothesis(BaseModel)` — moved here (NOT in researcher.py, so the validator can import without circular). Fields: `title: str`, `asset_class: str`, `strategy_family: str`, `hypothesis: str`, `expected_edge_explanation: str`, `key_indicators: list[str]`, `tags: list[str]` (min_length 1), `sources: list[Source]`, `differentiates_from: list[str]` (default `[]`).
- `class Source(BaseModel)`: `url: str`, `title: str`, `why_relevant: str`.
- `validate_hypothesis(hypothesis: ResearcherHypothesis, prior_art: list[PriorArtMatch]) -> None` — raises `HypothesisRejected` if `prior_art` is non-empty AND `differentiates_from` is empty, OR if any slug in `differentiates_from` is not in `[m.slug for m in prior_art]`.
- `generate_slug(session, strategy_family: str, asset_class: str) -> str` — deterministic. Pattern: `<family_lc>__<asset_lc>__<NNN>` where NNN is the next free integer for that (family, asset_class) pair, found by querying max suffix in existing slugs. E.g. `orb__forex__001`. Strips non-alphanumeric from family for path safety.

**Step 1: Write failing tests**

```python
# tests/orchestrator/test_hypotheses.py
import pytest, pytest_asyncio
from datetime import UTC, datetime
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fwbg_agents.persistence.database import Base
from fwbg_agents.persistence.models import Strategy, StrategyState
from fwbg_agents.orchestrator.hypotheses import (
    HypothesisRejected, ResearcherHypothesis, Source,
    validate_hypothesis, generate_slug,
)
from fwbg_agents.orchestrator.prior_art import PriorArtMatch


def _hyp(**over):
    base = dict(
        title="t", asset_class="FOREX", strategy_family="ORB",
        hypothesis="h", expected_edge_explanation="e",
        key_indicators=["opening_range"], tags=["momentum"],
        sources=[Source(url="https://x", title="x", why_relevant="x")],
        differentiates_from=[],
    )
    base.update(over)
    return ResearcherHypothesis(**base)


def _match(slug="prev_orb_001"):
    return PriorArtMatch(
        slug=slug, current_state="abandoned", strategy_family="ORB",
        asset_class="FOREX", tags_overlap=["momentum"], jaccard=0.5,
        post_mortem_path=None, post_mortem_summary=None,
    )


def test_validate_passes_with_no_prior_art():
    validate_hypothesis(_hyp(), [])


def test_validate_rejects_when_prior_art_and_no_differentiates_from():
    with pytest.raises(HypothesisRejected):
        validate_hypothesis(_hyp(), [_match()])


def test_validate_passes_when_differentiates_from_covers_prior_art():
    validate_hypothesis(_hyp(differentiates_from=["prev_orb_001"]), [_match()])


def test_validate_rejects_when_differentiates_from_slug_unknown():
    with pytest.raises(HypothesisRejected):
        validate_hypothesis(_hyp(differentiates_from=["unrelated"]), [_match()])


def test_hypothesis_tags_min_length_enforced():
    with pytest.raises(Exception):  # pydantic ValidationError
        _hyp(tags=[])


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_generate_slug_starts_at_001(db):
    slug = await generate_slug(db, "ORB", "FOREX")
    assert slug == "orb__forex__001"


@pytest.mark.asyncio
async def test_generate_slug_increments_per_family_asset_pair(db):
    now = datetime.now(UTC)
    db.add(Strategy(slug="orb__forex__001", asset_class="FOREX",
                    strategy_family="ORB", current_state=StrategyState.PROPOSED.value,
                    created_at=now, updated_at=now))
    await db.commit()
    assert await generate_slug(db, "ORB", "FOREX") == "orb__forex__002"
    assert await generate_slug(db, "RSI", "FOREX") == "rsi__forex__001"


@pytest.mark.asyncio
async def test_generate_slug_strips_special_chars(db):
    assert await generate_slug(db, "RSI/EMA-Cross", "FOREX") == "rsiemacross__forex__001"
```

**Step 2:** Run, expect FAIL.

**Step 3: Implement `hypotheses.py`** — pydantic models + the two functions.

**Step 4:** Run, expect PASS.

**Step 5: Commit**

```bash
git add src/fwbg_agents/orchestrator/hypotheses.py tests/orchestrator/test_hypotheses.py
git commit -m "feat(M4): hypothesis schema + validator + deterministic slug-gen"
```

---

## Task 3: Tavily client (deterministic httpx wrapper)

**Files:**
- Create: `src/fwbg_agents/tools/web_search.py`
- Test: `tests/tools/test_web_search.py`

**Behavior:**
- `class SearchResult(BaseModel)`: `url: str`, `title: str`, `content_snippet: str`, `score: float`.
- `class TavilyClient`: takes `api_key: str | None`, `base_url: str = "https://api.tavily.com"`, optional `httpx.AsyncClient` for test injection.
- `async def search(query: str, *, max_results: int = 5, session: AsyncSession | None = None, agent_run_id: int | None = None) -> list[SearchResult]`:
  - POST `/search` body `{api_key, query, max_results, search_depth: "basic"}` (don't burn credits on "advanced" in M4).
  - Parse `results: [{url, title, content, score}]` into SearchResult.
  - If `session` and `agent_run_id` provided: insert a `LlmCall(agent_run_id=..., model="tavily-search", input_tokens=0, output_tokens=0, latency_ms=elapsed, cost=None)` row for quota tracking. Don't fail the search if the insert fails — log and continue (best-effort).
- Raise `TavilyUnavailable` if `api_key is None` — explicit, so the Researcher can short-circuit.
- TODO comment block: `# Anthropic web_search fallback (design §10) — proxy compatibility unverified; deferred to a later milestone. Brave Search secondary fallback also deferred.`
- `async def get_quota_usage(session, window_days: int = 30) -> int` — count of llm_call rows with `model = "tavily-search"` in the window.

**Step 1: Failing test** — uses `httpx_mock` (pytest-httpx; already a M3 dep? check; if not, the Tavily-client test uses a fake httpx.AsyncClient that intercepts via `transport=httpx.MockTransport(...)`. M3 used the latter — match it).

```python
# tests/tools/test_web_search.py — sketch
import pytest, pytest_asyncio, httpx
from datetime import UTC, datetime
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from fwbg_agents.persistence.database import Base
from fwbg_agents.persistence.models import AgentRun, AgentRunStatus, LlmCall
from fwbg_agents.tools.web_search import (
    TavilyClient, TavilyUnavailable, SearchResult, get_quota_usage,
)


def _mock_transport(payload):
    async def handler(req):
        return httpx.Response(200, json=payload)
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_search_parses_results():
    payload = {"results": [
        {"url": "https://x", "title": "X", "content": "snippet", "score": 0.9},
    ]}
    client = TavilyClient(api_key="k",
                          http=httpx.AsyncClient(transport=_mock_transport(payload)))
    results = await client.search("query")
    assert results == [SearchResult(url="https://x", title="X",
                                    content_snippet="snippet", score=0.9)]


@pytest.mark.asyncio
async def test_search_raises_when_api_key_missing():
    client = TavilyClient(api_key=None)
    with pytest.raises(TavilyUnavailable):
        await client.search("q")


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_search_logs_llm_call_for_quota(db):
    now = datetime.now(UTC)
    ar = AgentRun(agent_name="researcher", status=AgentRunStatus.RUNNING.value,
                  started_at=now, created_at=now)
    db.add(ar); await db.commit(); await db.refresh(ar)
    client = TavilyClient(api_key="k",
                          http=httpx.AsyncClient(transport=_mock_transport({"results": []})))
    await client.search("q", session=db, agent_run_id=ar.id)
    from sqlalchemy import select
    rows = (await db.execute(select(LlmCall).where(LlmCall.model == "tavily-search"))).scalars().all()
    assert len(rows) == 1
    assert rows[0].agent_run_id == ar.id


@pytest.mark.asyncio
async def test_get_quota_usage_counts_recent_rows(db):
    # quota count test (omitted for brevity — same pattern)
    ...
```

**Step 2-5:** Implement, run, commit.

```bash
git add src/fwbg_agents/tools/web_search.py tests/tools/test_web_search.py
git commit -m "feat(M4): Tavily client + quota tracking via llm_call"
```

---

## Task 4: Researcher agent (LLM, pydantic-ai)

**Files:**
- Create: `src/fwbg_agents/agents/researcher.py`
- Create: `src/fwbg_agents/agents/prompts/researcher.md`
- Test: `tests/agents/test_researcher.py`

**Behavior:**
- Mirrors M3 Analyst structurally (`agents/analyst.py:L122`).
- `ResearcherInput(BaseModel)`: `asset_class: str`, `strategy_family_hint: str | None = None`, `free_text_brief: str | None = None`.
- `class Researcher(model: Model, session: AsyncSession, tavily: TavilyClient | None, prompt_path: Path | None = None)`.
- pydantic-ai Agent with two registered tools (`@agent.tool`):
  - `search_web(ctx, query: str) -> list[SearchResult]` — proxies to `tavily.search(query, session=self.session, agent_run_id=self._current_run_id)`. Returns empty list and logs warning if `tavily is None` (i.e. TAVILY_API_KEY unset).
  - `lookup_prior_art(ctx, strategy_family: str, asset_class: str, tags: list[str]) -> list[PriorArtMatch]` — proxies to `lookup_prior_art(self.session, ...)`.
- Output type: `ResearcherHypothesis`.
- `async def run(input: ResearcherInput) -> ResearcherHypothesis`:
  - Insert `AgentRun(agent_name="researcher", status=RUNNING, ...)`; stash `id` in `self._current_run_id`.
  - Run pydantic-ai Agent with `system=<prompt>` and `user=<formatted brief>`.
  - Record `LlmCall(agent_run_id=ar.id, model=settings.anthropic_model, input_tokens=..., output_tokens=..., latency_ms=...)` for each underlying LLM call (capture via pydantic-ai `result.usage()`).
  - Call `validate_hypothesis(result, prior_art_seen)` — if it raises, mark AgentRun failed, re-raise as `ResearcherFailed`. Track `prior_art_seen` by intercepting the tool call (simplest: track in a list on `self`).
  - On success: mark AgentRun done. Return hypothesis. Caller (`research_flow.py`) is responsible for persisting it (Researcher itself doesn't know the slug yet).
- `researcher.md` system prompt explicit about: (a) MUST call `lookup_prior_art` before deciding on the hypothesis, (b) if prior-art returned, `differentiates_from` MUST list the slugs it deviates from with explicit reasoning in the hypothesis body, (c) tags should be specific (asset/timeframe/technique), (d) hypothesis text is for a human strategist — be precise about edge, not generic.

**Step 1: failing tests** with `pydantic_ai.models.function.FunctionModel` (M3 pattern at `tests/agents/test_analyst.py:L37`).

Test scenarios:
1. `test_happy_path_no_prior_art`: FunctionModel returns a hypothesis with empty `differentiates_from`; no prior art exists; Researcher returns hypothesis; AgentRun marked done; LlmCall recorded.
2. `test_hypothesis_rejected_when_prior_art_and_no_differentiates_from`: seed an existing strategy with overlapping tags; FunctionModel returns hypothesis with empty `differentiates_from`; expect `ResearcherFailed` raised, AgentRun marked failed.
3. `test_search_web_with_tavily_unset_returns_empty`: tavily=None; FunctionModel's first tool call hits `search_web` which returns `[]`; LLM emits a hypothesis based only on `lookup_prior_art`; AgentRun completes.
4. `test_lookup_prior_art_tool_invocable_from_agent`: assert that pydantic-ai's tool registry includes both tool names.

**Step 2-5:** Implement, run, commit.

```bash
git add src/fwbg_agents/agents/researcher.py \
        src/fwbg_agents/agents/prompts/researcher.md \
        tests/agents/test_researcher.py
git commit -m "feat(M4): Researcher agent (LLM + lookup_prior_art + Tavily)"
```

---

## Task 5: Translator agent — fresh mode (LLM, pydantic-ai)

**Files:**
- Create: `src/fwbg_agents/agents/translator.py`
- Create: `src/fwbg_agents/agents/prompts/translator.md`
- Create: `src/fwbg_agents/orchestrator/strategy_validator.py` (lightweight structural validator)
- Test: `tests/agents/test_translator_fresh.py`
- Test: `tests/orchestrator/test_strategy_validator.py`

**Behavior:**
- `class StrategyValidationError(ValueError)`.
- `KNOWN_PLUGINS` const in `strategy_validator.py`:
  ```python
  PIPELINES = frozenset({"orb_simple_v1"})
  MODELS = frozenset({"signal_orb_v1"})
  FILTERS = frozenset({"orb_scalping_v1"})
  VALIDATIONS = frozenset({"walk_forward_intraday_v1"})
  RESOURCES = frozenset({"standard_v1"})
  DATASOURCES = frozenset({"forexsb"})
  TIMEFRAMES = frozenset({"MINUTE_5", "MINUTE_15", "MINUTE_30", "HOUR_1"})
  ```
  These are intentionally small — M5 PluginAuthor expands the catalog. Document this as a known limitation.
- `validate_strategy_json(data: dict) -> None`:
  - required top-level keys: `name, datasource, pipeline, model, filters, validation, resources, timeframe, exit_strategies, tags, hypothesis`
  - each plugin-slug reference must be in its corresponding `KNOWN_PLUGINS` set
  - `exit_strategies` must be a non-empty list of objects each containing `name, params`
  - `tags` non-empty list of strings
- `class TranslatorInput`: `mode: Literal["fresh", "reiterate"]`, `strategy_id: int`.
- `class Translator(model, session)`:
  - pydantic-ai Agent. Output type = `dict` (strategy.json). System prompt is `translator.md`.
  - Tool: `get_known_plugins() -> dict[str, list[str]]` — returns `KNOWN_PLUGINS` so the LLM doesn't hallucinate slugs.
  - `async def run_fresh(strategy: Strategy) -> Path`:
    1. Insert `AgentRun(agent_name="translator", status=RUNNING, strategy_id=strategy.id)`.
    2. Read `hypothesis.json` from `strategy_dir(strategy.slug) / "iteration_001" / "hypothesis.json"`.
    3. Run LLM with hypothesis as user message.
    4. Validate output with `validate_strategy_json` — on failure, mark AgentRun failed, raise `TranslatorFailed` with the validation error.
    5. Force `name = strategy.slug` (LLM cannot rename).
    6. Write `iteration_001/strategy.json` (pretty-printed, sort_keys=True for stable diffs).
    7. Write `iteration_001/spec.md` (sections: Goal, Inputs, Outputs, Acceptance Criteria, Implementation Notes — content derived deterministically from hypothesis + the generated strategy.json).
    8. Set `strategy.spec_path = str(spec_md_path)` and commit (no transition — strategy stays PROPOSED).
    9. Mark AgentRun done. Return path to strategy.json.

**Step 1: failing tests** for `strategy_validator.py` first (pure functions, fast):
- valid example (lift the m3_smoke strategy.json) passes
- missing top-level key fails
- unknown pipeline slug fails
- exit_strategies missing fails
- empty tags fails

Then failing tests for `translator.run_fresh`:
- FunctionModel returns a valid dict → strategy.json + spec.md written, AgentRun done, name forced to slug
- FunctionModel returns invalid dict (missing key) → TranslatorFailed, AgentRun failed
- FunctionModel returns dict with unknown plugin slug → TranslatorFailed

**Step 2-5:** Implement (validator first, then translator), commit each separately:

```bash
git add src/fwbg_agents/orchestrator/strategy_validator.py tests/orchestrator/test_strategy_validator.py
git commit -m "feat(M4): lightweight strategy.json structural validator"

git add src/fwbg_agents/agents/translator.py \
        src/fwbg_agents/agents/prompts/translator.md \
        tests/agents/test_translator_fresh.py
git commit -m "feat(M4): Translator agent — fresh mode (hypothesis → strategy.json + spec.md)"
```

---

## Task 6: Translator — reiterate mode + child strategy creation

**Files:**
- Modify: `src/fwbg_agents/agents/translator.py` (add `run_reiterate`)
- Modify: `src/fwbg_agents/orchestrator/recommendations.py` — confirm the sidecar JSON format the Analyst writes (no code change expected; we read whatever it wrote in M3).
- Test: `tests/agents/test_translator_reiterate.py`

**Behavior:**
- `async def run_reiterate(parent: Strategy) -> Strategy`:
  1. Read sidecar JSON from `strategy_dir(parent.slug) / f"iteration_{parent.iteration_count:03d}" / "analyst_recommendation.json"` (M3 writes this for TuneParams/ChangeExit — verify the path by reading `recommendations.py:_rec_to_dict` during implementation; if path differs, this task fixes it).
  2. Read parent's last `strategy.json` from the same iteration dir.
  3. Compute `child_slug = generate_slug(session, parent.strategy_family, parent.asset_class)` — fresh number.
  4. Insert child Strategy row: `current_state=PROPOSED`, `iteration_count=1`, `parent_strategy_id=parent.id`, same family/asset_class.
  5. Copy parent's `hypothesis.json` into the child's iteration_001 (lineage preserved).
  6. **No LLM for TuneParams** — purely deterministic edit: read the recommendation `{kind: "tune_params", param: "...", new_range: [...], reason: "..."}`, mutate the parent's strategy.json (e.g. for a grid param replace its range), write to child's iteration_001/strategy.json.
  7. **No LLM for ChangeExit either** — read `{kind: "change_exit", new_exit_strategy: {...}}`, replace `exit_strategies` in the parent's strategy.json.
  8. Run `validate_strategy_json` on the new strategy.json. Reject if invalid.
  9. Write a new `spec.md` for the child (deterministic; just notes "iterated from parent_slug due to <reason>").
  10. Mark AgentRun done; insert `Transition(entity_id=child.id, from_state=None, to_state=PROPOSED, reason="translator: re-iterate from <parent_slug>", payload={parent_strategy_id, recommendation_kind})`.

**Decision recorded in code:** Re-iterate is **deterministic** in M4. Originally I considered using the LLM to re-translate, but: (a) we already have a working strategy.json; (b) the Analyst recommendation is structured; (c) determinism here means re-iterations are reproducible.

**Step 1: failing tests:**
- TuneParams sidecar → child strategy created with mutated param, parent untouched, Transition row exists, validator passes.
- ChangeExit sidecar → child has new exit_strategies entry.
- Missing sidecar → TranslatorFailed.
- Resulting child strategy.json fails validator → TranslatorFailed, child Strategy row not committed (or rolled back) — use `session.rollback()` on validator error.

**Step 2-5:** Implement, run, commit.

```bash
git add src/fwbg_agents/agents/translator.py tests/agents/test_translator_reiterate.py
git commit -m "feat(M4): Translator reiterate mode — child strategy from Analyst sidecar"
```

---

## Task 7: research_flow orchestration glue + research API

**Files:**
- Create: `src/fwbg_agents/orchestrator/research_flow.py`
- Create: `src/fwbg_agents/api/research.py`
- Modify: `src/fwbg_agents/api/__init__.py` — register the research router (mirror M3's runs router wiring).
- Test: `tests/api/test_research.py`

**Behavior:**
- `async def research_and_translate(session, input: ResearcherInput, *, model, tavily) -> int`:
  1. Run Researcher → ResearcherHypothesis (raises on rejection; we propagate).
  2. `slug = await generate_slug(session, hypothesis.strategy_family, hypothesis.asset_class)`.
  3. Insert Strategy row (PROPOSED, iteration_count=1; design says iteration_count starts at 1 once the first artifact is written — M3 Runner used iteration_001/ dir but left iteration_count=0; we fix this here for M4: set to 1 explicitly). Insert StrategyTag rows. Insert initial Transition(from=None, to=PROPOSED).
  4. Write `iteration_001/hypothesis.json` (pretty-printed) and `iteration_001/research_notes.md` (sources + reasoning, deterministic from hypothesis).
  5. Set `strategy.hypothesis_path = str(hypothesis_path)`.
  6. Run Translator(fresh) → strategy.json + spec.md.
  7. Return `strategy.id`.

- `async def reiterate(session, parent_id: int, *, model) -> int`:
  - Loads parent, calls Translator.run_reiterate. Returns child strategy id.

- API endpoints in `api/research.py`:
  - `POST /research/brief` — body: `ResearcherInput`. Returns 202 with `{agent_run_id, message}`. Schedules `research_and_translate` as `BackgroundTasks`. Each background task opens its own SessionLocal (M3 pattern).
  - `POST /strategies/{id}/reiterate` — schedules `reiterate`. Returns 202 `{agent_run_id, message}`.
  - `GET /hypotheses` — list of recent strategies that have `hypothesis_path` set, with their state and slug. Limit 50. Read-only.

**Step 1: failing tests** (mirror `tests/api/test_runs.py` from M3):
- `POST /research/brief` schedules → returns 202 → AgentRun row created in `pending`/`running`.
- `POST /strategies/{id}/reiterate` for a BACKTESTED strategy with sidecar → returns 202.
- `POST /strategies/{id}/reiterate` for a strategy without sidecar → 422 or 409.
- `GET /hypotheses` lists strategies with `hypothesis_path`.

Use `pydantic_ai.models.function.FunctionModel` for the LLM stubs in background tasks (M3 pattern: tests inject the model via a settings/factory override).

**Step 2-5:** Implement, run, commit.

```bash
git add src/fwbg_agents/orchestrator/research_flow.py \
        src/fwbg_agents/api/research.py \
        src/fwbg_agents/api/__init__.py \
        tests/api/test_research.py
git commit -m "feat(M4): /research/brief + /reiterate + /hypotheses API"
```

---

## Task 8: Smoke + final verification

**Files:**
- Create: `scripts/m4_smoke.py`

**Behavior:**
- Requires `TAVILY_API_KEY` set (script exits with a clear message otherwise).
- Builds the in-process FastAPI app, posts to `/research/brief` with a real brief (e.g. `"find a mean-reversion strategy for FOREX majors that we haven't tried yet"`).
- Polls `GET /agents/runs/{id}` until done. Asserts:
  - `hypothesis.json` + `research_notes.md` written into the new `data/strategies/<slug>/iteration_001/`.
  - `strategy.json` + `spec.md` written.
  - `validate_strategy_json` passes on the result.
  - At least one `LlmCall` row with `model="tavily-search"` exists.
- Does NOT run the Runner — that's already smoked in M3. Manual Runner kick can follow if desired.

**Final verification commands:**

```bash
VIRTUAL_ENV= uv run pytest -q  # expect ~100+ tests green (78 baseline + ~25 new)
VIRTUAL_ENV= uv run alembic upgrade head  # no new migration expected
VIRTUAL_ENV= uv run python -c "from fwbg_agents.api import app; print('app loads')"
```

**Optional smoke (only if `TAVILY_API_KEY` set):**

```bash
VIRTUAL_ENV= uv run python scripts/m4_smoke.py
```

**Step: Commit smoke**

```bash
git add scripts/m4_smoke.py
git commit -m "chore(M4): smoke script driving Researcher+Translator end-to-end"
```

---

## Out of M4 Scope (recorded so the executor doesn't drift)

- PluginAuthor (M5). Translator-fresh uses fixed plugin catalog.
- sqlite-vec embedding-similarity (deferred). Layer-1 tag matching only.
- PaperTrader / LiveTrader (M6/M7).
- Cron scheduling (M8 / later).
- Anthropic web_search fallback build (proxy compatibility verification first).
- Full `fwbg.core.config.StrategyConfig` pydantic validation (defer; promote when fwbg becomes a dep).
- Dashboard pages for hypotheses / research runs (own session).

---

## Post-implementation housekeeping

After all 7 commits land and tests green:
1. Update `~/Projekte/fwbg/docs/plans/2026-06-23-fwbg-agents-design.md` Implementation Status table: M4 done with final commit hash.
2. Update memory: `project_fwbg_agents.md` (M4 status), new `reference_fwbg_agents_m4_plan.md`, refresh `MEMORY.md` index.
