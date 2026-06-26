# fwbg-agents — Design Document

**Datum**: 2026-06-23
**Status**: M0+M1+M2+M3+M4+M4b+M5a+M5b+M5c+M5d+M6a+M6b implemented; M7 (Live Trading + Risk) next
**Author**: brainstorming session zwischen User und Claude

## Implementation Status

| Milestone | Status | Notes |
|---|---|---|
| M0 — Skeleton | ✓ done 2026-06-23 | Commit `5cc7649` in `~/Projekte/fwbg-agents/`. FastAPI boots, SQLite + alembic init, mock SSE, pydantic-ai LLM client. Proxy connection not end-to-end verified (port 8080 was occupied by another service during test). |
| M1 — Calibrator + Criteria | ✓ done 2026-06-23 | Calibrator scans `~/fwbg/test_results/` (not `~/Projekte/fwbg/test_results/`), groups by asset class via inlined symbol→class map (no fwbg runtime dep), writes section-6.1 YAMLs + raw baseline JSON. Endpoints: `GET /criteria`, `GET/PUT /criteria/{class}`, `POST /calibrate`, `GET /calibrate/runs`. Verified against real data: 79 runs scanned, 12 INDEX elites, calibration_run row persisted. Dashboard page at `/agents/criteria` (textarea editor, Recalibrate button). DSR/PBO/max_drawdown/profit_factor are absent from current fwbg results and intentionally omitted from generated YAML — schema is forward-compatible when fwbg starts emitting them. |
| M2 — Strategy-Lifecycle Skeleton | ✓ done 2026-06-24 | Commit `ade47ad`. ORM models (`Strategy`, `Plugin`, `Transition`, `StrategyTag`) + alembic 0002. Deterministic state machine in `orchestrator/lifecycle.py`: collapsed strategy lifecycle `proposed → backtested → paper_trading → live_trading` plus terminal `abandoned`, plugin lifecycle `specified → authored → verified → adopted_in_fwbg` plus `abandoned`. Guards: `backtested → paper_trading` evaluates criteria YAML via a small comparator parser (`>=`, `<=`, `>`, `<`, `==`, `!=`); `paper_trading → live_trading` requires `human_approval=True` in payload (UI gate ships M7, state machine enforces from day one); `→ abandoned` requires `post_mortem_path` (anti-redundancy for the Researcher's M4 prior-art lookup). Append-only: no cascade deletes, transition rows insert-only. Read-only API: `GET /strategies` (filters `?state=` `?asset_class=`), `GET /strategies/{id}` (detail + transitions + tags), `GET /strategies/{id}/transitions`; mirror for `/plugins`. 23 new tests (16 lifecycle + 7 API) all green. End-to-end smoke (`scripts/m2_smoke.py`) verified live. Dashboard pages deferred to a follow-up session. |
| M3 — Runner + Analyst | ✓ done 2026-06-24 | Commit `df33384`. Adds the first real iteration loop (manually triggered). New module `tools/fwbg_client.py` is a thin async httpx wrapper around fwbg's `/api/runs/start`, `/runs/{id}`, `/runs/{id}/progress`. New deterministic `agents/runner.py` (no LLM — Runner is on the critical path): copies the strategy.json into fwbg's strategies dir, posts the start, polls until terminal, fetches the full run, writes `fwbg_results.json` into the iteration dir, then `transition_strategy(s, BACKTESTED, payload={fwbg_run_id, results_path, backtest_metrics})`. New LLM-driven `agents/analyst.py`: pydantic-ai with structured output (`Promote | Abandon | TuneParams | ChangeExit`), system prompt in `agents/prompts/analyst.md` for easy iteration, every call recorded in `llm_call`. New `orchestrator/recommendations.py.validate_and_apply` runs hard rules between Analyst output and any state change (Promote re-checks criteria YAML; Abandon writes `post_mortem.yaml`; TuneParams/ChangeExit persist as sidecar JSON for M4 Translator). M3 endpoints: `POST /strategies` (manual seeding into PROPOSED with `iteration_001/strategy.json`), `POST /strategies/{id}/run`, `POST /strategies/{id}/analyze`, `GET /agents/runs/{id}`. Migration 0003 adds `agent_run` + `llm_call`. 78 tests green (was 42 after M2). Decisions captured: Runner does NOT bump `iteration_count` (M4 Translator owns iteration bumps); `llm_call` lives in M3 since Analyst is the first LLM consumer; prompt in `.md` file for prompt-iteration ergonomics. Smoke (`scripts/m3_smoke.py`) verified against live fwbg :8420 (strategy POSTed, Runner kicked, fwbg job accepted, polling working); Analyst LLM call exercised against the configured proxy (404 in current env — `haex-claude-proxy` not running, code path verified). |
| M4 — Researcher + Translator | ✓ done 2026-06-24 | Plan at `docs/plans/2026-06-24-fwbg-agents-m4.md`. Final commit `45825f9` in `~/Projekte/fwbg-agents/`. New modules: `orchestrator/prior_art.py` (tag-Jaccard similarity, `1e9707e`), `orchestrator/hypotheses.py` (`ResearcherHypothesis` + `validate_hypothesis` + `generate_slug`, `4d5041a`), `tools/web_search.py` (Tavily client with quota tracking via `llm_call(model='tavily-search')`, `3991d4f`), `agents/researcher.py` (LLM with `lookup_prior_art` + `search_web` tools and hard anti-redundancy gate, `6dd3093`), `orchestrator/strategy_validator.py` (lightweight structural validator + hardcoded plugin-slug catalog, `3d699da`), `agents/translator.py` (fresh-mode LLM: hypothesis → strategy.json + spec.md with canonical slug enforced, `35324cf`; reiterate-mode fully deterministic with `parent_strategy_id` lineage + extended `ChangeExit.new_exit_strategy`, `ed2b59a`), `orchestrator/research_flow.py` (Researcher → persist Strategy + StrategyTag + initial Transition + write hypothesis.json/research_notes.md → Translator.run_fresh, `73257f1`), `api/research.py` wired into main router (`POST /research/brief`, `POST /strategies/{id}/reiterate` with 422/409 preconditions, `GET /hypotheses`, `ed453a5`), `scripts/m4_smoke.py` (end-to-end via ASGI transport with graceful TAVILY_API_KEY skip, `45825f9`). 159 tests green (78 baseline + 81 new). No migration in M4 (Tavily reuses `llm_call`). Locked decisions: re-iterate via `parent_strategy_id` (not state-machine regression), Tavily quota via convention `model='tavily-search'` (no schema change), strategy.json validation is lightweight structural — full `fwbg.core.config.StrategyConfig` deferred until fwbg becomes runtime dep, Anthropic web_search fallback documented but not built pending proxy compat verification. |
| M4b — Researcher Search Resilience + Parallel Hypothesis Fan-out | ✓ done 2026-06-26 | Plan at `docs/plans/2026-06-26-fwbg-agents-m4b.md`. Five commits `9ad4cc2`, `cce8f28`, `4f46df5`, `9b6b6fc`, `053b144` in `~/Projekte/fwbg-agents/` on branch `feat/m4b-researcher-resilience` (off `develop` — repo adopted Gitflow between plan-writing and implementation). Moved `tools/web_search.py` into a `tools/search/` package behind a `SearchProvider` protocol (`SearchUnavailableError`, renamed from the plan's `SearchUnavailable` for this repo's ruff `N818` rule); added `BraveClient` + `FallbackSearchClient` (Tavily primary, Brave secondary, fixed order, no cross-provider scoring); `research_and_translate` now fans out `RESEARCHER_FANOUT_N` (default 2, env-configurable) parallel candidates via `asyncio.gather`, each in its own `SessionLocal` session, first to pass `validate_hypothesis` wins in submission order; all-candidates-exhausted raises `ResearcherFanoutExhaustedError` with every rejection reason. 384 tests green (370 baseline + 14 new). `scripts/m4b_smoke.py` verified live through the real LLM-call boundary (proxy unreachable in the implementing sandbox — see plan-doc deviations for the literal `[m4b_smoke] PASSED` caveat); `scripts/m4_smoke.py` unaffected. Out of scope, unchanged from the plan: no sandboxed code execution in the Researcher (stays downstream in Runner/Analyst against the real fwbg tool); no MCP; no generalizing fan-out to other agents yet (concrete-before-generic, per M5d's decision H). |
| M5a — Plugin Catalog + Contract + AddIndicator + Validator Refactor | ✓ done 2026-06-24 | Plan at `docs/plans/2026-06-24-fwbg-agents-m5a.md`. Final commit `0418eb1` in `~/Projekte/fwbg-agents/`. New modules: `orchestrator/plugin_catalog.py` (PluginManifest + PluginCatalog with two-root fwbg discovery — `src/fwbg/plugins/` and `packages/fwbg-premium/.../plugins/` — both using bundle manifests of shape `plugins: {category: [slug, ...]}`; `merge_with_db()` only includes verified/adopted plugins; `functools.lru_cache(maxsize=8)` per fwbg_root path, `291afa6`), `orchestrator/plugin_contract.py` + `docs/specs/plugin_contract.example.yaml` (pydantic schema for contract.yaml with typed inputs/outputs/params/invariants/test_scenarios, `kind` as Literal mirroring M5b's PluginKind extension, model_validator forces indicators to declare ≥1 invariant, `0dd440b` + `c050fd5`), `agents/analyst.py` AddIndicator recommendation kind + Analyst now renders flat catalog snapshot into its system prompt; `orchestrator/recommendations.py` writes `iteration_001/add_indicator_request.json` sidecar without transition (`28c4180`), `orchestrator/strategy_validator.py` refactored: `validate_strategy_json(data, *, catalog=None)` with frozenset fallback — when catalog has entries for `models` or `exit_strategies`, lookup is catalog-based + difflib "did you mean" hints in error messages (`0418eb1`). 187 tests green (159 baseline + 28 new). No migration in M5a (PluginKind enum extension + verification_run table deferred to M5b's Migration 0004). Settings adds `fwbg_repo_root: Path` defaulting to `~/Projekte/fwbg`. Locked decisions: PluginEvaluator deterministic in M5b; test_scenarios hand-curated np-seeded; PluginAuthor writes to `data/plugins/<slug>/v1/` never directly into fwbg; catalog cache process-lifetime (restart refreshes); plugin version fixed at `v1/`; PluginKind Literal in PluginContract avoids M5a needing a migration; validator catalog injection via optional kwarg keeps existing Translator/research_flow tests passing untouched. |
| M5b — PluginAuthor + Evaluator + API | ✓ done 2026-06-24 | Plan at `docs/plans/2026-06-24-fwbg-agents-m5b.md`. Four commits in `~/Projekte/fwbg-agents/`: Migration 0004 + PluginKind extension + `verification_run` ORM (`35d41c4`); `agents/plugin_author.py` (LLM-driven pydantic-ai agent, FunctionModel-stub tested, tools `get_fwbg_plugin_examples(category, n=3, hard_cap=5)` + deterministic `validate_python_syntax`, slug-collision guard against catalog AND DB, persists `data/plugins/<slug>/v1/{plugin.py, contract.yaml, spec.md}`, transitions SPECIFIED→AUTHORED) (`aaed198`); `agents/plugin_evaluator.py` + `orchestrator/scenario_generators.py` (deterministic — no LLM in M5b per locked decision; 5 hand-curated np-seeded OHLCV generators: trending_up/trending_down/sideways/high_vola/sparse_data; dynamic-imports `plugin.py` via `importlib.util.spec_from_file_location`, runs compute() against each scenario, checks `length_invariant=same_as_input`; structured JSON `error_log.json` on failure; full pass → transition AUTHORED→VERIFIED, fail → stay AUTHORED) (`4cdc1bd`); `orchestrator/plugin_flow.py` + `api/plugins.py` extension + `scripts/m5_smoke.py` (POST /strategies/{id}/author-plugin, POST /plugins/{id}/evaluate, GET /plugins/{id}/verification-runs — 202 + AgentRun envelope + BackgroundTasks; m5_smoke uses FunctionModel stub to drive the full HTTP path end-to-end and ended PASSED on the dev DB) (`36aca54`). 234 tests green (187 M5a baseline + 47 new: 7 migration/ORM + 8 author + 17 generator + 6 evaluator + 9 API). Deps added: `numpy>=2.1`, `pandas>=2.2`, `pyarrow>=18` (parquet engine). Per-plugin scenario layout `data/plugins/<slug>/v1/test_scenarios/<name>.parquet`. Failed verification policy: stay AUTHORED, manual retry, no counter, no auto-abandon (deferred to M8+). |
| M5c — Reiterate-with-Plugin (Bridge Plugin → Strategy iteration) | ✓ done 2026-06-24 | Plan at `docs/plans/2026-06-24-fwbg-agents-m5c.md`. Eight commits in `~/Projekte/fwbg-agents/`: `strategy_validator.py` extended with four optional plugin-slot list-fields (`indicators`, `feature_selection`, `preprocessing`, `extra_filters`) routed catalog-first with no frozen-fallback; M4 strategies remain valid via empty-list default (`f55a4e3`); `Translator.run_reiterate_with_plugin(parent, plugin_slug, sidecar) -> Strategy` — deterministic mirror of M4 `run_reiterate`, hard-coded phase→list-field map (indicator→indicators, feature_selection→feature_selection, preprocessing→preprocessing, filter→extra_filters), deep-copies parent strategy.json, appends slug, generates child slug via `generate_slug()`, validates with `validate_strategy_json(catalog=load_catalog())`, inserts child Strategy(PROPOSED, parent_strategy_id, iteration_count=1) + Transition row + writes iteration_001/{strategy.json, hypothesis.json, spec.md}, child hypothesis = parent-copy + Decision-C1 template-rationale block `f"Iteration {n}: added {slug} at {phase} per analyst recommendation: {capability}"` (`0acf06f`); `orchestrator/plugin_flow.reiterate_with_plugin(session, strategy_id, plugin_slug) -> int` with six preconditions (parent exists, parent==BACKTESTED, plugin exists, plugin==VERIFIED, sidecar present, capability-match-guard via `_lookup_plugin_capability` that locates the originating PluginAuthor AgentRun and reads its sidecar's `capability` field — Plugin has no capability column so we link via AgentRun.plugin_id + agent_name="plugin_author"); `_load_fwbg_cached.cache_clear()` invoked before Translator call (Decision E); `POST /strategies/{id}/reiterate-with-plugin` body `{plugin_slug: str}` returns 202+AgentRun-envelope (`agent_name="translator_reiterate_flow"`) + BackgroundTasks; 404 on missing strategy, 422 on all other preconditions (`7863ba4`); local-import hoist in api/plugins.py per CLAUDE.md style (`6c24e18`); `scripts/m5c_smoke.py` end-to-end: seed parent BACKTESTED + add_indicator_request sidecar → POST /author-plugin → POST /evaluate → POST /reiterate-with-plugin → poll → assert child Strategy + indicators=[slug] + hypothesis iteration block + parent sidecar still present (Decision D append-only audit); FunctionModel stub double-patches `author_plugin_from_strategy` in both `api.plugins` and `orchestrator.plugin_flow` since api.plugins binds at import time (`d994766`); drop unused `Path` import + document double-patch (`69ab77d`). **M5c smoke surfaced a pre-existing M5a/M5b bug**: `merge_with_db` used `p.kind` (singular per PluginContract.PluginKindLit) as the catalog bucket key while fwbg-side bundle manifests + the validator query plural categories, leaving every PluginAuthor-authored VERIFIED plugin invisible in `catalog.all_slugs_for("indicators"|"models"|...)`. Fixed with a hand-coded `_KIND_TO_CATEGORY` mapping in `plugin_catalog.py:merge_with_db` (singular→plural for the four where natural English plurals diverge: indicator/model/filter/exit_strategy; pass-through for multi-word categories feature_selection/preprocessing/risk_management/entry_modifier/data_loading); existing M5a tests using plural `kind` strings keep passing via the fallback (`66f5cac`). Smoke fixture now uses real fwbg model slug `xgboost` (was `signal_orb_v1` which only exists in M4 frozen-fallback, not real fwbg catalog); self-test seeds a VERIFIED xgboost Plugin row so the isolated test catalog matches (`22ce7e1`). Manual `python scripts/m5c_smoke.py` against dev DB ends `[m5c_smoke] PASSED` with child `orb__forex__001` in PROPOSED, `indicators=['smoke-m5c-rsi']`. 258 tests green (234 M5b baseline + 21 M5c task tests + 3 plugin_catalog regression tests for the kind→category mapping). No migration in M5c (schema unchanged — list-fields are JSON-payload-only, merge_with_db fix is in-process only). Locked decisions: (A) M5c bridges Plugin↔Strategy; M6=PaperTrading unchanged. (B) Additive list-fields, NOT `Union[str, list[str]]`, NOT `plugin_overrides: dict[phase, list[str]]`. (C1) Hypothesis inherited+template-annotated; PluginAuthor API untouched (M5b 234 tests stay green). (D) Sidecar preserved post-reiterate (append-only audit). (E) `_load_fwbg_cached.cache_clear()` in reiterate_with_plugin only (other M5c paths don't consume freshly-VERIFIED plugins). Capability-match guard = strict equality between sidecar.capability and the plugin's originating PluginAuthor sidecar's capability. M6-deferred minor items captured in `reference_fwbg_agents_m5c_plan.md`. |
| M6a — Live Paper-Trading Telemetry | ✓ done 2026-06-24 | Plan at `docs/plans/2026-06-24-fwbg-agents-m6.md`. Cross-repo: 6 commits in `~/Projekte/fwbg-agents/` (`b58a868` M5c-polish chore, `b765757` `Strategy.paper_account_id` + `paper_phase_target_days` columns + alembic 0005, `be74769` `tools/fwbg_paper_reader.py` with `PaperTradeSummary`/`PaperPositions` pydantic models + inline-formula aggregator over `trades.jsonl`+`status.json`+`positions.json`, `b5691e5` `GET /strategies/{id}/paper-summary` endpoint + `settings.fwbg_data_dir` field + `_require_paper_or_live_trading()` helper, `954e28c` `GET /strategies/{id}/paper-positions` endpoint reusing the helper, `fa0a92a` `scripts/m6a_smoke.py` end-to-end + `.gitignore` `data/smoke/`); 3 commits in `~/Projekte/fwbg/` (`0a8df61` `AssetConfig.strategy_slug` + CLI `--strategy-slug` flag + `TradingBot.strategy_slug` attribute, `3a4e1f1` per-strategy telemetry writer — `_execute_signal` appends `trades.jsonl`, `_write_status` overwrites `status.json`+`positions.json` under `data/account-trades/<strategy_slug>/`, equity_curve bounded to 200 with first+last preserved, write failures logged not raised, `Position` dataclass gains optional `stop_loss`/`take_profit`/`opened_at` + `BrokerAdapter.is_paper` property; `d1e25fe` IG adapter populates `Position.stop_loss`/`take_profit` from IG's `stopLevel`/`limitLevel` so dashboard sees actual SL/TP — both adapter copies wired). agents tests 258→282 (+24), fwbg tests 2528→2536 (+8). Manual `python scripts/m6a_smoke.py` ends `[m6a_smoke] PASSED` and is **re-run idempotent** (Stage-0 cleanup wipes prior smoke artefacts). Locked decisions: (A+C) cross-repo event-driven+periodic — fwbg writes, agents reads/aggregates, dashboard GETs (no auto-poller in agents); (B/D) DEFERRED to M6b — paper-criteria YAML + promote-live endpoint come with the Analyst; (E) `paper_phase_target_days` lives on Strategy NOW (M6a) with default 90, dashboard-configurable per-strategy, NO auto-abandon; (F) M5c-polish chore as first M6a commit; (G) M6 split into M6a/M6b; (H) logical isolation 1 Strategy = 1 fwbg-account-config (free-form string in `paper_account_id`, agents never reads fwbg accounts/ dir). |
| M6b — Paper-Analyst + Promote-Live | ✓ done 2026-06-24 | Plan at `docs/plans/2026-06-24-fwbg-agents-m6b.md`. Agents-only (no fwbg-side code, only one plan-doc edit `0b89947` in `~/Projekte/fwbg/`). 13 commits in `~/Projekte/fwbg-agents/`: `e558b17` alembic 0006 `Strategy.metadata_json` JSON column (generic vehicle for recommendation flags); `33301e4` hand-curated `data/criteria/paper/{equity,forex,crypto}.yaml` flat `required_all`+`hard_blockers` schema (looser than M2 backtest — equity sharpe≥0.8/dd≤0.25, forex sharpe≥1.0/dd≤0.15, crypto sharpe≥0.6/dd≤0.30, [[feedback-no-data-derived-thresholds]]); `ffa7dfb`+`ec2b07a` `orchestrator/criteria_paper.py` (concrete copy of `lifecycle._eval_comparator` per locked decision N — `_OPERATORS` list-of-tuples for M2 parity, `_`-prefix key skip mirrors M2, `dataclasses.asdict` for CriteriaEvalResult future-proofing, 17 tests incl. parametrized 6-operator coverage); `61b997c`+`54ac7dd` `agents/paper_analyst.py` + `agents/prompts/paper_analyst.md` (pydantic-ai discriminated-union `PromotePaperToLive | AbandonPaper | ContinueObservation` with deterministic validator: Promote requires `paper_criteria_eval.passed=True` else `PaperAnalystValidationError`, Abandon auto-fills `post_mortem_path` default via `model_copy`, ContinueObservation force-flips `stale=True` when `days_in_paper > paper_phase_target_days`; lazy prompt-load mirrors M3 Analyst, `json.dumps(default=str)` for LLM payload, 6 tests via FunctionModel stub matching M3 test pattern); `51fdf9c`+`0f20213` `orchestrator/paper_flow.py::paper_analyze(strategy_id, session, *, settings, analyst, existing_ar)` — loads on-disk telemetry, runs Analyst-with-validator, writes sidecar at `strategy_dir(slug)/paper_analyst_<ar.id>.json` (M3-style helper reuse), reassigns `metadata_json` flag NOT mutates (SQLA JSON change-tracking), `log.exception` before commit so original traceback always logged, inner try/except around commit so commit-failure doesn't replace original exception, 6 tests; `16d8003`+`ca00697` `POST /strategies/{id}/paper-analyze` mirroring M5c reiterate-with-plugin pattern (pre-creates AgentRun PENDING, kicks BackgroundTasks via `_run_paper_analyze_background` with fresh SessionLocal, envelope `{agent_run_id, status: "scheduled"}` honest about AR being PENDING at HTTP-return, `existing_ar` kwarg threaded into `paper_analyze` for backwards-compat), 4 tests; `f3daf61`+`d17807f` `POST /strategies/{id}/promote-live` triple-gated (body `human_approval=true` AND `metadata.paper_analyst_promote_recommended=True` AND state==PAPER_TRADING; M2's `_guard_strategy_paper_to_live` re-checks payload as defence-in-depth third gate), AgentRun(`agent_name="promote_live"`, status=DONE) staged BEFORE `transition_strategy` so both flush atomically inside lifecycle's commit, `operator_note` normalized via `.strip() or None` (empty/whitespace → None), stale `paper_analyst_promote_recommended` flag cleared + `promoted_live_at` ISO timestamp set on success, 8 tests; `eccb07e` `scripts/m6b_smoke.py` end-to-end via in-process httpx ASGITransport — monkeypatches `paper_flow.PaperAnalyst` with `_SmokeAnalyst` stub returning `PromotePaperToLive` (BackgroundTask shares process so patch reaches the orchestrator's fallback instantiation), idempotent stage-0 cleanup deletes Strategy+Transition+AgentRun rows AND `data/strategies/<slug>/`+`account-trades/<slug>/` artefacts, synthesised 50-trade fixture clears forex paper-criteria (sharpe≈5.2 > 1.0, win_rate=0.56 > 0.45, dd=0.007 < 0.15), final assertions cover sidecar+metadata flag cleared+promoted_live_at timestamp+Transition payload audit+AgentRun audit. agents tests 282→323 (+41). Manual `python scripts/m6b_smoke.py` ends `[m6b_smoke] PASSED` and is **re-run idempotent**. Locked decisions: (I) JSON metadata column (generic vehicle, not boolean columns — future flags need no migration); (J) 3 hand-curated paper-criteria YAMLs minimum (equity/forex/crypto, flat schema NOT nested under `paper_to_live` to avoid cross-coupling with calibrator-seeded M2 YAMLs); (K) PaperAnalyst NEVER transitions state — only writes sidecar + sets metadata flag; (L) Promote-live triple-gated, only the metadata flag is LLM-influenceable (and only positively — recommend not approve); (M) `paper_phase_target_days` is a soft warning forcing `stale=True` on Continue — humans decide, no auto-abandon; (N) Concrete-before-generic — `criteria_paper.py` does NOT share a base with M2's `check_backtest_criteria`, comparator parser duplicated (~15 LOC), extract only when a 3rd evaluator appears (M7 live-trading risk gates). |
| M5d — PluginAuthor Planner/Implementer Split | ✓ done 2026-06-25 | Plan at `docs/plans/2026-06-25-fwbg-agents-m5d.md`. Seven commits in `~/Projekte/fwbg-agents/` on branch `feat/m5d-planner-implementer-split` final `d85ad1c`: `24b8be0` `AgentModels` settings module (pydantic-settings, env vars `PLUGIN_PLANNER_MODEL`/`PLUGIN_IMPLEMENTER_MODEL`/`PLUGIN_IMPL_MAX_ROUNDS` defaults `claude-opus-4-8`/`claude-opus-4-7`/`5`); `51c16c5` `prompts/plugin_authoring.md` canonical fwbg-Plugin-Konventionen doc (loaded into Planner system-prompt at runtime, versioned with code); `46b21c5` `PluginPlan` + `ParamSpec` pydantic schemas (frozen, `extra="forbid"`, derived from BasePlugin contract — PluginPhase SDK enum, ParamSpec with type/default/min/max/step/choices); `9a73136` `agents/plugin_planner.py` (single-shot LLM, stronger model, reads AnalystRecommendation + parent strategy.json + PluginCatalog-excerpt, emits PluginPlan, persists `data/plugin-runs/<slug>/plan.json`); `1cefb65` `agents/plugin_implementer.py` (loop-bounded LLM, weaker model, consumes PluginPlan + prev-round code + prev-round gate error, emits identical `PluginAuthorResult` shape as M5b so M5c downstream untouched, deterministic gate-loop = `validate_python_syntax` → AST-based `load_contract_from_code` → if both OK persist + return, else feed error back, after N rounds → `PluginAuthorFailed` with last code in error_message); `979cb23` `orchestrator/plugin_flow.author_plugin_from_strategy` rewired to drive Planner→Implementer (3 AgentRuns per session: outer envelope + plan + implement, N LlmCalls as children under implement-run); `d85ad1c` retire 8 M5b plugin_author tests, `scripts/m5d_smoke.py` self-test ends `[m5d_smoke] PASSED`, M5c smoke still green via factory monkey-patches. 371 tests green (323 → +48 net after retiring 8 M5b plugin_author tests). No alembic migration (`AgentRun.kind` is free string). Phase-mapping AnalystRecommendation→PluginPhase: indicator→INDICATORS, feature_selection→FEATURE_SELECTION, preprocessing→PREPROCESSING, filter→RISK_MANAGEMENT (filters live under risk_management in SDK enum). Locked decisions: (A) Refinement-Loop with deterministic gates only — no LLM-self-judgment; (B) Planner runs exactly once, Implementer fixes (reasoning OK, syntax/naming/imports schief); (C) `max_rounds=5` default env-configurable; (D) `PluginPlan` schema derived from BasePlugin contract; (E) Prompt-Doc in repo, not global Claude skill; (F) No backwards-compat — old PluginAuthor replaced, M5b tests split, single-path code; (G) 1 AgentRun per logical phase, retry-loop = implementation detail not user-visible; (H) Concrete-only — no generalization to Researcher/Analyst/Translator per-agent-model-config until 2-3 more use cases appear. |
| M7 — Live Trading + Risk | pending | |
| M8 — Promotion + Polish | pending | |

### Late-binding design changes

- **LLM SDK**: switched from raw `anthropic` SDK to **`pydantic-ai`** during M0 — provider-neutral, typed, Vercel-AI-SDK-style. `AnthropicModel(base_url=...)` still routes through `haex-claude-proxy`. Decision recorded in section 15.
- **M5c milestone insertion** (2026-06-24): the original M5 was a single milestone "Plugin-System". Split into M5a (catalog + contract + AddIndicator + validator refactor), M5b (PluginAuthor + Evaluator + API + smoke), and **M5c (reiterate-with-plugin bridge)** because the plugin lifecycle ended at VERIFIED with no way to feed the freshly-VERIFIED slug back into a child strategy. M5c adds `Translator.run_reiterate_with_plugin` + `POST /strategies/{id}/reiterate-with-plugin` so the loop `add_indicator → author → evaluate → reiterate → child` closes. M6 (PaperTrading) and later milestones unchanged.
- **Plugin.kind plural-vs-singular convention** (2026-06-24, surfaced by M5c smoke, fixed at `66f5cac`): `PluginContract.PluginKindLit` is **singular** (`indicator`, `model`, `filter`, `exit_strategy`, ...) and PluginAuthor writes `Plugin.kind` verbatim from the contract. fwbg-side bundle manifests AND the strategy_validator query **plural** categories (`indicators`, `models`, `filters`, `exit_strategies`). The bridge lives in `plugin_catalog._KIND_TO_CATEGORY` map applied inside `merge_with_db`. M5a tests that pre-dated the M5c discovery still pass via pass-through (they used plural strings as kind already). M6+ work that introduces a new PluginKindLit must update `_KIND_TO_CATEGORY` if the singular form differs from the plural.
- **M6 split into M6a/M6b** (2026-06-24): the user's M6 expansion ("per-strategy paper/live account isolation + live position tracking with SL/TP visible on dashboard") sprengt the original ~4-day M6 scope. Split decision: **M6a** ships the live-telemetry data path (per-strategy account-id field, fwbg writes `trades.jsonl`+`status.json`+`positions.json` per strategy, agents reads/aggregates, dashboard polls GETs) — closed loop for "see what's running now". **M6b** adds the LLM-driven decision layer on top (paper-criteria YAML, Paper-Analyst with discriminated-union output, sidecar-recommendation pattern, separate `POST /promote-live` endpoint requiring `human_approval=true` AND prior analyst-recommendation flag). M5d (PluginPlanner+Implementer split with per-agent model config) is parked behind M6b — concrete request: stronger model (e.g. opus-4-8) for plugin design, weaker (e.g. opus-4-7) for implementation; configurable via env vars / settings yaml. See [[project-fwbg-agents-m5d-sketch]].
- **Cross-repo telemetry path** (M6a, 2026-06-24): the user's "live SL/TP on dashboard" requirement forced a cross-repo data path. fwbg's `TradingBot._write_status()` was already writing a dashboard status file; M6a adds per-strategy telemetry under `data/account-trades/<slug>/` with `trades.jsonl` (append-only, event-driven per `_execute_signal`) + `status.json` (overwrite-periodic, equity curve bounded to 200 with first+last preserved) + `positions.json` (overwrite-periodic, list of open positions with SL/TP/current_price). agents NEVER reads fwbg's accounts/ dir — only stores a free-form `paper_account_id` string on Strategy. Writes are best-effort (failures logged, never raised — telemetry can't abort trades). The dir naming `account-trades` (not `paper-trades`) was deliberate because the user wanted both paper AND live accounts tracked the same way.
- **Position SL/TP wiring in IG adapter** (M6a, 2026-06-24, `d1e25fe`): the existing IG adapter populated IG-named `stop_level`/`limit_level` fields on Position. M6a added M6a-named `stop_loss`/`take_profit` as separate fields (not aliases) on Position — both populated from the same IG `stopLevel`/`limitLevel` row keys. Both adapter copies wired (`src/fwbg/adapters/broker/ig/adapter.py` AND `packages/fwbg-broker-ig/src/fwbg_broker_ig/adapter.py`). Backward-compatible — existing readers of `stop_level`/`limit_level` unchanged.

## 1. Motivation & Goals

Erweitere das fwbg-Ökosystem um eine **autonome Agent-Schicht**, die eigenständig:

1. **Strategien recherchiert** im Internet (Tavily-API)
2. **Konkrete fwbg-Strategy-Configs generiert** aus der Recherche
3. **Backtests triggert** via fwbg-API und Ergebnisse auswertet
4. **Strategien iteriert**: Parameter, Indikatoren, Exits anpassen basierend auf Auswertung
5. **Vielversprechende Strategien auf Paper-Konto deployt** für ~3 Monate
6. **Bestandene Paper-Strategien auf Live-Konto** mit echtem Geld traden lässt
7. **Neue Plugins/Indikatoren generiert** wenn nötig
8. **Existierende Plugins kontinuierlich verifiziert** gegen Contracts

Ein **Webdashboard** erlaubt Konfiguration, Monitoring, manuelle Eingriffe.

### Non-Goals

- **Kein** generisches Agent-Framework. Wir bauen konkret für fwbg.
- **Kein** Multi-User, keine Auth, kein Multi-Tenant (single-user, single-machine zunächst).
- **Keine** Echtzeit-Datenverarbeitung im Agent-Layer — fwbg macht die Backtests, der Broker das Trading.
- **Kein** Worker-Queue (Redis/Celery) — asyncio reicht für unsere I/O-bound Workload.

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    fwbg-dashboard (Nuxt 4)                       │
│  /agents  /agents/runs  /agents/strategies  /agents/plugins ... │
└──────────────┬──────────────────────────────────┬───────────────┘
               │ REST + SSE                       │ REST (existing)
               ▼                                  ▼
┌──────────────────────────────────┐  ┌──────────────────────────┐
│      fwbg-agents (NEW Python)     │  │      fwbg (existing)     │
│  ┌────────────────────────────┐  │  │   POST /api/runs/start   │
│  │  FastAPI + asyncio          │  │  │   GET  /api/runs/{id}    │
│  │  • Orchestrator             │──┼─▶│   GET  /api/strategies   │
│  │  • Agent-Plugins            │  │  │   IG Broker Adapter      │
│  │  • Lifecycle State Machine  │  │  └──────────────────────────┘
│  │  • Risk Controls            │  │
│  └────────────────────────────┘  │
│  SQLite + Filesystem-Artefakte   │
└──────────────┬───────────────────┘
               │ Anthropic SDK (base_url override)
               ▼
┌──────────────────────────────────┐
│  haex-claude-proxy (Subscription) │
└──────────────────────────────────┘
```

**Drei Komponenten, drei klare Verantwortlichkeiten:**

- **fwbg-agents** (NEU) — FastAPI-Service. Lädt Agent-Plugins, pflegt Lifecycle-State-Machine, führt Promotion-Guards durch, enforced Risk-Controls. Spricht ausgehend mit fwbg-API (Backtests), Tavily (Recherche), haex-claude-proxy (LLM), IG-Broker (Paper/Live).

- **fwbg-dashboard** (existing) — wird erweitert um Agent-Pages. Nutzt weiterhin `@nuxt/ui` v4 (kein Wechsel zu shadcn-vue).

- **fwbg** (existing) — bleibt unverändert. Einzige offizielle Schnittstelle ist die HTTP-API. Nur für **Typen** wird `fwbg_sdk` importiert (Strategy-Schemas, Enums).

## 3. Komponentenwahl & Begründung

| Bereich | Wahl | Begründung |
|---|---|---|
| Backend-Sprache | Python 3.12+ | Konsistenz mit fwbg; LLM-Ökosystem ist Python-first |
| API-Framework | FastAPI | Async-native, SSE-fähig, pydantic-typed |
| State-Storage | SQLite + Filesystem | Queryable Metadata, inspizierbare Artefakte; kein Postgres-Overhead |
| Frontend | Nuxt 4 + `@nuxt/ui` v4 | Existing fwbg-dashboard erweitern statt neu bauen |
| LLM-Provider | **pydantic-ai** mit `AnthropicModel(base_url=haex-claude-proxy)` | Provider-neutral, typed Agent-Abstraktionen; heute Claude via Proxy für Subscription-Pricing, später Multi-Provider ohne Refactor |
| Agent-Orchestration | Custom Python | Spec-kit zu interaktiv, LangGraph zu schwer für unsere klar abgegrenzten Rollen |
| Spec-Workflow | Spec-Kit Artefakt-Format + Prompts | Konsistente Contracts; via direkte SDK-Calls (Variante 1) |
| Web Research | Tavily (primary) + Anthropic web_search (fallback) | LLM-optimierte Snippets; Fallback bei Quota-Reißen |
| Process-Modell | FastAPI + asyncio + Resumable State | Agents sind I/O-bound; State-Persistence für Multi-Tages-Runs |

## 4. Agent-Katalog

Jeder Agent ist eine Python-Klasse mit definiertem Interface (`Agent` Protocol). Drop-in als File in `fwbg_agents/agents/`. Initial-Katalog:

| Agent | Trigger | LLM? | Inputs | Outputs |
|---|---|---|---|---|
| **Researcher** | manual / scheduled | ✓ | Tavily web search | `research_notes.md`, `hypothesis.json` |
| **Translator** | Researcher done | ✓ | hypothesis.json, fwbg_sdk types, plugin catalog | `strategy.json`, `spec.md` |
| **PluginAuthor** | Translator: "indicator X missing" | ✓ | fwbg plugin docs + examples | Python file in `generated_plugins/` + `contract.yaml` + `spec.md` |
| **PluginEvaluator** | weekly cron + on-author | ✓ | plugin contract, code | `verification_run` record + synthetic `test_scenarios/*.parquet` |
| **Runner** | strategy.json ready | ✗ | strategy.json | fwbg run_id, polls progress, results.json |
| **Analyst** | Runner done | ✓ | results.json, fwbg metrics, diagnostic data | `analyst_report.md`, typed recommendation |
| **PaperTrader** | Analyst promotes | ✗ | strategy.json, IG-Demo connection | trades, `paper_metrics.jsonl` |
| **PaperMonitor** | daily cron | ✓ | paper_metrics, expected ranges | drift alerts, demote-recommendation |
| **LiveTrader** | **manual approval** | ✗ | strategy.json, IG-Live, risk-config | trades, `live_metrics.jsonl` |
| **LiveGuardian** | every second | ✗ | live_metrics, position state, risk-config | emergency-stop triggers |
| **PromoteAgent** | Plugin verified N times | ✓ | plugin file + tests | GitHub PR against fwbg-repo |
| **Calibrator** | first run + manual | ✗ | existing fwbg/test_results/ | auto-generated `criteria/<asset_class>.yaml` |

**Wichtig**: nicht alle "Agents" sind LLM-getrieben. Runner, LiveGuardian, PaperTrader, LiveTrader sind deterministisch. LLM-Inferenz nur dort, wo qualitatives Reasoning gebraucht wird. Kritische Pfade (Risk!) sind deterministisch.

**Agent Interface**:
```python
class Agent(Protocol):
    name: str
    requires: list[str]
    produces: list[str]

    async def run(self, ctx: AgentContext) -> AgentResult: ...
```

`AgentContext` injiziert: SQLite session, filesystem paths, anthropic client (proxy-pointed), fwbg HTTP client, plugin loader.

## 5. Lifecycle State Machines

### 5.1 Strategy Lifecycle

```
   proposed ─────► specified ─────► backtested_pending ─────► backtested_evaluated
   (researcher)    (translator)     (runner submitted)        (results in)
                                                                    │
                                                                    ▼
                          ┌──────────────────────────── analyst evaluates
                          │                                          │
                  ┌───────┴───────┐                                  │
                  ▼               ▼                                  │
            adjust+retry      abandon                                │
            (→ translator)    (with post_mortem)                     │
                                                     guard: DSR>0.8  │
                                                     AND PBO<0.5 ───►│
                                                     AND p<0.05      │
                                                     AND max_dd<15%  │
                                                                    │
                                                                    ▼
                                                          approved_for_paper
                                                                    │
                                                                    ▼
                                                            paper_trading
                                                            (3 Monate, cron-tick)
                                                                    │
                                                    ┌───────────────┴──────┐
                                                    ▼                      ▼
                                              revaluation_pending    passed paper criteria
                                              (drift detected)             │
                                                    │                      ▼
                                          analyst re-evaluates   awaiting_human_approval
                                          ┌──────┬──────┬─────┐  (Dashboard-Gate)
                                          ▼      ▼      ▼     ▼          │
                                       adjust  add    change abandon    ▼
                                       params  indi-  exit/  (only      live_trading
                                               cator  risk   confirmed)  │
                                                                         ▼ on breach
                                                                emergency_stopped
```

Wichtig: **`abandoned` ist immer eine bewusste Analyst-Entscheidung, nie automatisch nach Paper-Demote**.

### 5.2 Plugin Lifecycle

```
specified ──► authored ──► verified ──► adopted_in_fwbg
(spec.md)   (code+lint    (synthetic   (PR merged)
             passes)       tests pass)
                              │
                              ▼ weekly re-verify
                          suspect ──► deprecated
                          (drift)     (manual confirm)
```

### 5.3 Implementation

- States als Python `Enum`, persistent in SQLite `strategy.current_state`, `plugin.current_state`
- Übergänge via `transitions.yaml` deklariert, deterministische Guard-Funktionen in `fwbg_agents/orchestrator/guards.py`
- Jeder Übergang erzeugt Audit-Eintrag in `transition` Tabelle: from_state, to_state, triggered_by, evidence, timestamp
- **Live-Transitions** loggen Risk-Config-Snapshot — falls später was schiefgeht, ist exakt nachvollziehbar mit welchen Limits losging
- Orchestrator zentral; Agents triggern nur Transition-Requests

## 6. Success Criteria & Iteration Strategy

### 6.1 Scorecard pro Stage (deklarativ)

In `criteria/<asset_class>.yaml`:

```yaml
# criteria/FOREX.yaml (Beispiel)
backtest_to_paper:
  required_all:
    - dsr: ">= 0.8"
    - pbo: "<= 0.5"
    - mc_pvalue: "< 0.05"
  required_any:
    - { sharpe: ">= 1.2", min_trades: 100 }
    - { profit_factor: ">= 1.5", win_rate: ">= 0.55" }
  hard_blockers:
    - max_drawdown: "< 0.20"
    - tail_ratio: "> 1.0"

paper_to_live:
  realized_vs_backtest:
    sharpe_deviation_max: 0.40
    drawdown_breach_factor: 1.5
    win_rate_deviation_max: 0.10
  minimum_sample:
    trades: 30
    days_running: 60
  regime_check:
    require_no_distribution_shift: true
```

**Initial Calibration**: ein einmaliger `Calibrator` scannt existing `fwbg/test_results/`, berechnet Quantile, schreibt Schwellen-Vorschläge nach `criteria/*.yaml`. Im Dashboard nachjustierbar.

### 6.2 Iteration-Scoring

```
score(iteration) = w1·DSR + w2·(1-PBO) + w3·(1-max_dd) + w4·sharpe·log(trades)
                 - penalty(iteration_count)
```

**Kein Hard-Cap auf Iterationen**, aber:
- `iteration_count` fließt in DSR-Berechnung (Multiple-Testing-Korrektur)
- Plateau-Detection: 2 aufeinanderfolgende Iterationen ohne >3% Score-Verbesserung → automatischer Abandon
- Paper-Trading-Stage als ehrlicher OOS-Test

### 6.3 Analyst Diagnosis

Der Analyst kriegt **strukturierte Diagnostics** vor seiner Empfehlung:

```
Failed criteria: [sharpe], [max_drawdown]
Param-edge-of-grid: [tp_pips at 30 (max)]   ← Hinweis: Grid erweitern
Regime-breakdown: trending=ok, ranging=bad  ← Hinweis: Filter fehlt
Trade-distribution: 80% wins ≤ 3 pips       ← Hinweis: TP zu klein
Equity-curve segments: gut 0-50, schlecht 50-100  ← Hinweis: Regime-Shift
```

Output ist **typisierte Empfehlung** mit Konfidenz:
- `tune_params { param, new_range }` → zurück zum Runner
- `add_indicator { phase, capability }` → zum PluginAuthor
- `change_exit { from, to }` → zum Translator
- `abandon { reason }` → final mit Post-Mortem

**Orchestrator validiert die Empfehlung gegen Hard-Rules** (z.B. add_indicator nur erlaubt wenn iteration_count < 3). LLM kann nichts übersteuern.

### 6.4 Anti-Redundanz (Researcher Pre-Check)

Strategien werden **nie hart gelöscht**, nur als `abandoned` markiert. Gründe:
1. Traceability: nachvollziehen was getestet wurde
2. Anti-Redundanz: verhindern dass Researcher dieselbe Idee wieder vorschlägt

**Beim Abandon-Transition** wird zwingend ein `post_mortem.yaml` geschrieben:

```yaml
strategy_family: "ORB"
asset_class: "INDEX"
hypothesis_summary: "..."
abandon_reason: "no edge in any regime"
best_iteration: { dsr: 0.42, sharpe: 0.31 }
iterations_tried: 4
failure_pattern: "high IS sharpe, poor OOS — fitting noise"
lessons:
  - "ORB on M15 timeframe doesn't generalize on indices"
related_strategies: ["orb_dax_v2"]
```

**Researcher hat verpflichtenden Pre-Check** via `lookup_prior_art` Tool:
- Vor Hypothesen-Ausarbeitung query nach `{strategy_family, asset_class, key_indicators}`
- Bei Match: muss `differentiates_from: [...]` ausfüllen oder Recherche-Idee verwerfen

**Two-Layer Similarity**:
- Layer 1 (tag-based, deterministisch, sofort): `strategy_tag` Tabelle
- Layer 2 (embedding-based, später): sqlite-vec mit voyage-3 Embeddings

## 7. Data Model

### 7.1 SQLite Tabellen (`state.db`)

```sql
strategy(id, slug, current_state, iteration_count, parent_strategy_id,
         created_at, updated_at, asset_class, strategy_family,
         abandon_reason, post_mortem_path, failure_class, score_history JSON)

strategy_tag(strategy_id, tag, value)

plugin(id, slug, kind, current_state, contract_path, code_path,
       created_at, last_verified_at, generated_by_strategy_id)

transition(id, entity_type, entity_id, from_state, to_state,
           triggered_by, evidence JSON, timestamp)

agent_run(id, agent_name, strategy_id, plugin_id, status,
          started_at, ended_at, input_artifact_path, output_artifact_path,
          tokens_in, tokens_out, cost_usd, error)

llm_call(id, agent_run_id, model, tokens_in, tokens_out, latency_ms,
         cost_usd, timestamp)

trade_session(id, strategy_id, mode ENUM('paper','live'), broker, account_id,
              started_at, ended_at, status, risk_config_snapshot JSON)

trade(id, session_id, symbol, side, opened_at, closed_at,
      entry_price, exit_price, pnl, expected_pnl_at_open)

verification_run(id, plugin_id, status, test_scenarios_path,
                 results JSON, baseline_diff JSON, ran_at)

daily_metric(date, session_id, sharpe, drawdown, win_rate, trade_count,
             distribution_pvalue)

broker_connectivity_event(id, broker, account_id,
  state ENUM('connected','disconnected','reconnecting'),
  detected_at, recovered_at, error_message)

-- Später (optional):
strategy_embedding(strategy_id, hypothesis_embedding BLOB, model_version)
```

**Append-only Constraint**: `strategy`, `plugin`, `transition`, `trade` haben keine Cascade-Deletes. `DELETE` Endpoints existieren nicht.

### 7.2 Filesystem Layout (`data/`)

```
data/
├── strategies/<strategy-slug>/
│   ├── iteration_001/
│   │   ├── hypothesis.json           ← Researcher output
│   │   ├── spec.md                   ← Translator (spec-kit format)
│   │   ├── strategy.json             ← fwbg-kompatibel
│   │   ├── analyst_report.md
│   │   └── fwbg_results.json
│   └── iteration_002/...
│   └── post_mortem.yaml              ← bei abandon
│
├── plugins/<plugin-slug>/
│   ├── spec.md                       ← spec-kit
│   ├── plan.md
│   ├── tasks.md
│   ├── contract.yaml                 ← maschinen-verifizierbar
│   ├── code/__init__.py
│   ├── test_scenarios/
│   │   ├── positive_clear_breakout.parquet
│   │   └── edge_nan_gaps.parquet
│   └── verifications/<timestamp>/
│
├── criteria/
│   ├── FOREX.yaml
│   ├── INDEX.yaml
│   ├── CRYPTO.yaml
│   └── _calibration_baseline.json
│
└── transcripts/<agent_run_id>.jsonl  ← raw LLM I/O für Debugging
```

## 8. API + Dashboard Surface

### 8.1 FastAPI Endpoints

```
# Strategy Lifecycle
GET    /strategies
GET    /strategies/{slug}
POST   /strategies                          # manuell anlegen (Researcher-Skip, M3)
POST   /strategies/{id}/run                 # trigger Runner backtest (M3)
POST   /strategies/{id}/analyze             # trigger Analyst (M3)
POST   /strategies/{id}/reiterate           # deterministic re-iterate from sidecar (M4)
POST   /strategies/{id}/author-plugin       # PluginAuthor triggered from add_indicator sidecar (M5b)
POST   /strategies/{id}/reiterate-with-plugin  # splice VERIFIED plugin into child strategy (M5c)
POST   /research/brief                      # Researcher + Translator end-to-end (M4)
# NO DELETE — abandon-via-post_mortem only

# Agent Runs
GET    /agents                        # registry, configured?, enabled?
GET    /agents/runs                   # filterable
GET    /agents/runs/{id}              # full transcript
POST   /agents/runs/{id}/cancel       # asyncio task cancellation

# Plugin Lifecycle
GET    /plugins
GET    /plugins/{id}/verification-runs      # list verification runs (M5b)
POST   /plugins/{id}/evaluate               # trigger PluginEvaluator (M5b)
POST   /plugins/{slug}/promote              # trigger PromoteAgent (M8)

# Criteria & Config
GET/PUT /criteria/{asset_class}
GET/PUT /risk-config

# Real-time
GET    /events/stream                 # SSE: agent progress, transitions, alerts

# Calibration & Cost
POST   /calibrate
GET    /costs/summary
```

### 8.2 Dashboard Pages (in fwbg-dashboard)

| Route | Purpose |
|---|---|
| `/agents` | Agent-Plugin-Registry, enable/disable, prompts inspect |
| `/agents/strategies` | Kanban: proposed → backtest → paper → live mit Drag bei manual gates |
| `/agents/strategies/[slug]` | Iteration-History, Score-Verlauf, Artefakte |
| `/agents/plugins` | Generated Plugins, Verification-Status, Promote-Buttons |
| `/agents/plugins/[slug]` | spec.md/plan.md/tasks.md, Code-Diff, Test-Scenarios |
| `/agents/runs` | Live-Tail von SSE |
| `/agents/criteria` | YAML-Editor pro Asset-Class, Calibration-Trigger |
| `/agents/risk` | Live-Trading Guardrails |
| `/agents/costs` | LLM-Spending Charts |

**Header-Bar zeigt zwei LEDs** (IG-Demo / IG-Live Connectivity) prominent.

## 9. Spec-Kit Integration (Variante 1)

Spec-Kit's Skill-Markdown-Inhalte werden **direkt als System-Prompts via Anthropic SDK ausgeführt**:

```python
SPEC_PROMPT = read_speckit_skill("speckit-specify")
response = anthropic.messages.create(
    base_url=PROXY_URL,
    system=SPEC_PROMPT,
    messages=[{"role": "user", "content": brief}]
)
# Wir schreiben das Ergebnis selbst nach .specify/specs/<slug>/spec.md
```

**Vorteile**: schnell (~2-5s pro Step), keine claude-CLI-Subprocess-Dependency, testbar (SDK-mock).
**Nachteil**: Prompt-Updates aus spec-kit upstream müssen manuell synced werden (selten, ~5min Aufwand).

Spec-Kit wird nur für **Artefakt-Authoring** verwendet (Plugin-Specs, Strategy-Specs). Agent-Orchestrierung ist eigener Code.

**Reverse-Spec für existing Plugins**: ein einmaliger Initial-Run lässt den LLM für jedes existierende fwbg-Plugin (EMA, ORB, etc.) eine spec.md rückwirkend schreiben. Damit hat alles einen Vertrag.

## 10. Web Research

**Primary**: Tavily API (1000 free calls/month, LLM-optimized results).

**Fallback**: Anthropic web_search built-in tool (Status durch haex-claude-proxy unklar, muss in M4 verifiziert werden).

**Quota-Tracking**: `llm_call`-Tabelle erweitert. Ab 950 Calls/Monat (Buffer 50) automatischer Switch auf Fallback. Reset nach 30 Tagen. Dashboard zeigt Status.

Falls Anthropic-Fallback durch Proxy nicht funktioniert: Brave Search API (2000 free/month) als Sekundär-Fallback.

## 11. Risk Controls (Live Trading)

### 11.1 Layer 1: Pre-Trade Validation (deterministisch, blockierend)

Vor jedem Order an IG:

```python
def validate_order(order, account_state, risk_config) -> Result:
    if order.stop_loss is None: REJECT("no stop loss")
    if order.stop_distance > risk_config.max_stop_distance: REJECT
    if position_size_exceeds_strategy_limit(...): REJECT
    if position_size_exceeds_account_limit(...): REJECT
    if symbol_blacklisted(...): REJECT
    if cumulative_exposure_breach(...): REJECT
    if margin_insufficient(...): REJECT
    if in_blackout_window(...): REJECT  # NFP, FOMC
    if daily_loss_breached(...): REJECT
```

**Stop-Loss ist Pflicht, keine Ausnahme.** SL geht **atomar mit Entry-Order** raus. Optional Guaranteed Stop (kostet extra Spread).

### 11.2 Layer 2: LiveGuardian (asynchron, every second)

Eigener asyncio-Task. Bei Breach:
1. Emergency-stop triggern
2. close_all_positions (market-close, kein TP/SL warten)
3. State-Transition: live_trading → emergency_stopped
4. Alert (dashboard, email, push)

**Breach-Trigger** (`risk-config.yaml`):
```yaml
account_level:
  max_drawdown_daily: 0.03
  max_drawdown_total: 0.15
  min_free_margin_pct: 0.30
  max_concurrent_positions: 10
  max_correlated_exposure: 2.0

strategy_level:
  max_drawdown_daily: 0.02
  max_consecutive_losses: 5
  max_position_pct_equity: 0.05

blackout_windows:
  - { event: "NFP", before_min: 30, after_min: 60 }
  - { event: "FOMC", before_min: 60, after_min: 120 }
```

### 11.3 Layer 3: Restart-Resilience

Nach Crash/Restart:
1. LiveTrader liest aktive Sessions aus SQLite
2. Holt IG's *actual* Positionsliste
3. Vergleicht mit lokalem Trade-Log
4. Bei Diskrepanz → `awaiting_reconciliation`, keine neuen Orders, manuelle Klärung

### 11.4 Layer 4: Approval-Gate

Übergang `awaiting_human_approval → live_trading` erfordert **explizite Dashboard-Bestätigung**. Beim Approval wird `risk_config` Snapshot in `trade_session` festgeschrieben.

### 11.5 Initial-Position-Sizing

Erster Live-Trade einer neu promoteten Strategie: **10%** der konfigurierten Position-Size. Nach 30 erfolgreichen Trades oder 14 Tagen ohne Breach: schrittweise hoch auf 100%.

### 11.6 Dashboard Surface

- **STOP ALL LIVE Button** auf `/agents` Hauptseite
- Per-Session pause/resume
- Real-time Drawdowns als Progress-Bars gegen Limits
- Event-Log: jede abgelehnte Order mit Begründung sichtbar

## 12. Broker Connectivity Handling

**Disconnect ist ein scoped pause**, kein system-wide pause.

**Zwei unabhängige Verbindungen**: `ig_demo`, `ig_live`. Ausfall der einen pausiert nur die zugehörigen Trader.

**Affected by ig_demo disconnect**:
- PaperTrader ⏸ keine neuen demo-Orders
- PaperMonitor ✓ läuft weiter (liest stored metrics)

**Affected by ig_live disconnect**:
- LiveTrader ⏸ keine neuen live-Orders
- LiveGuardian ✓ läuft, kann nur warnen, nicht eingreifen (SL bei IG schützt)

**NOT affected by any broker disconnect** (kritisch!):
- Researcher, Translator, PluginAuthor, PluginEvaluator
- Runner (fwbg-Backtest braucht keinen Broker)
- Analyst, Calibrator, PromoteAgent

**Reconnect-Loop**: exponential backoff 5s → 10s → 30s → 60s max. Nach Reconnect: Reconciliation-Step.

**Server-side SL ist das einzige funktionierende Sicherheitsnetz bei Disconnect** — wenn wir offline sind, können wir per Definition nichts mehr aktiv tun.

## 13. Repository Layout

### 13.1 fwbg-agents (`~/Projekte/fwbg-agents/`)

```
fwbg-agents/
├── pyproject.toml              # Python 3.12+, uv-managed
├── README.md
├── CLAUDE.md                   # graphify-rules
├── docker-compose.yml          # ein Service: api (FastAPI)
│
├── src/fwbg_agents/
│   ├── main.py                 # FastAPI app entry
│   ├── config.py               # pydantic-settings
│   │
│   ├── orchestrator/
│   │   ├── lifecycle.py        # transition logic
│   │   ├── guards.py           # criteria evaluation
│   │   └── scheduler.py        # asyncio task management
│   │
│   ├── agents/
│   │   ├── base.py             # Agent Protocol
│   │   ├── researcher.py
│   │   ├── translator.py
│   │   ├── runner.py
│   │   ├── analyst.py
│   │   ├── plugin_author.py
│   │   ├── plugin_evaluator.py
│   │   ├── paper_trader.py
│   │   ├── paper_monitor.py
│   │   ├── live_trader.py
│   │   ├── live_guardian.py
│   │   ├── promote_agent.py
│   │   └── calibrator.py
│   │
│   ├── tools/
│   │   ├── anthropic_client.py # SDK wrapper mit proxy base_url
│   │   ├── tavily_search.py    # Web research mit Fallback
│   │   ├── fwbg_client.py      # HTTP client für fwbg-API
│   │   ├── ig_broker.py        # Wraps fwbg.adapters.broker.ig
│   │   └── github_pr.py        # für PromoteAgent
│   │
│   ├── speckit/
│   │   ├── prompts/            # *.md, kopiert aus upstream spec-kit
│   │   └── runner.py
│   │
│   ├── persistence/
│   │   ├── models.py           # SQLAlchemy ORM
│   │   ├── migrations/         # Alembic
│   │   └── artifacts.py        # FS-Layout helpers
│   │
│   ├── api/
│   │   ├── strategies.py
│   │   ├── plugins.py
│   │   ├── agents.py
│   │   ├── criteria.py
│   │   ├── risk.py
│   │   └── events.py           # SSE stream
│   │
│   └── risk/
│       ├── validators.py
│       └── guardian.py
│
├── data/                       # runtime state, gitignored
│   ├── state.db
│   ├── strategies/
│   ├── plugins/
│   ├── criteria/
│   └── transcripts/
│
└── tests/
```

### 13.2 fwbg-dashboard Erweiterungen

```
fwbg-dashboard/
├── pages/agents/               # NEU
│   ├── index.vue
│   ├── strategies/
│   │   ├── index.vue           # Kanban
│   │   └── [slug].vue          # Detail
│   ├── plugins/
│   ├── runs.vue                # SSE live-tail
│   ├── criteria.vue
│   ├── risk.vue
│   └── costs.vue
├── composables/
│   └── useAgentsApi.ts
└── stores/
    └── agents.ts               # Pinia: lifecycle, runs, events
```

## 14. Implementation Milestones

| # | Name | Aufwand | Wert nach Abschluss |
|---|---|---|---|
| M0 | Repo Setup | ~½ Tag | Skeleton, Proxy-Connection-Test, mock SSE |
| M1 | Calibrator + Criteria | ~1 Tag | Initial Erfolgs-Schwellen aus existing fwbg-Runs |
| M2 | Strategy-Lifecycle Skeleton | ~2 Tage | End-to-end Pipeline sichtbar (ohne LLMs) |
| M3 | Runner + Analyst | ~3 Tage | Iteration-Loop funktioniert manuell |
| M4 | Researcher + Translator | ~4 Tage | Vollautomatischer Research→Backtest Loop |
| M5 | Plugin-System | ~5 Tage | Agents können Indikatoren selbst bauen |
| M6 | Paper Trading | ~4 Tage | 3-Monats-Paper-Phase läuft |
| M7 | Live Trading + Risk | ~5 Tage | Production-ready Live-Pipeline |
| M8 | Promotion + Polish | ~2 Tage | Generated Plugins → fwbg-Core PRs |

**Gesamt-Schätzung**: ~26-30 Arbeitstage, 8-10 Wochen kalender.

**Funktionale Meilensteine**:
- Ab M3 (~6 Tage): manuelle Strategy-Iteration
- Ab M4 (~10 Tage): vollautomatischer Loop
- Ab M6 (~18 Tage): Paper-Trading bereit

## 15. Decision Log

| Frage | Entscheidung | Begründung |
|---|---|---|
| Separates Repo vs Monorepo? | Separates Repo `~/Projekte/fwbg-agents/` | Saubere Trennung Experimentation vs. Production-Code; eigene Iterationsgeschwindigkeit |
| Neues vs. existing Dashboard? | Existing `fwbg-dashboard` erweitern, @nuxt/ui behalten | Spart komplettes Frontend-Reimplement |
| Prozess-Modell? | FastAPI + asyncio + Resumable State | Agents sind I/O-bound; State-Persistence für Multi-Tages-Runs |
| Storage? | SQLite + Filesystem-Artefakte | Queryable Metadata + inspizierbare Artefakte; SQL-Komfort ohne Postgres-Overhead |
| LLM-Stack? | **pydantic-ai** (Anthropic provider) via `haex-claude-proxy` | Typed, provider-neutral, Vercel-AI-SDK-Philosophie (konsistent mit user's TS-Projekten); Proxy für Subscription-Pricing bleibt erhalten |
| Agent-Topologie? | Pipeline mit Rollen + Orchestrator | Klare Verantwortungen, einzeln debuggbar, vorhersagbar |
| Plugin-Generierung? | Separater Bereich + Human-Promote ins fwbg-Core | fwbg-Core bleibt sauber, alles auditierbar |
| Promotion-Logic? | Hard Rules + Human Approval für live | Sicher, transparent, nachvollziehbar |
| Spec-Kit Integration? | Variante 1: Artefakt-Format + Prompts via SDK | Schneller und robuster als Subprocess; nutzt spec-kit's Skills inhaltlich direkt |
| MCP in fwbg einbauen? | Nein | fwbg's HTTP-API ist schon die Schnittstelle; MCP wäre Indirection ohne Mehrwert |
| Skills? | Nicht relevant für autonome Agents | Skills sind für interactive Claude-Sessions |
| Web Research? | Tavily primary, Anthropic web_search Fallback | LLM-optimierte Results, kostenloser Tier; Fallback bei Quota |
| Erfolgs-Kriterien? | Iterativ aus existing fwbg runs ableiten | Datengetrieben, nicht theoretisch |
| Iteration-Limit? | Kein Hard-Cap, Plateau-Detection + DSR-Penalty | Maximal Freiheit, Multiple-Testing über DSR korrigiert |
| Analyst-Autonomie? | Strukturierte Empfehlung, Orchestrator validiert | LLM kann nicht übersteuern |
| Hard Delete? | Nein, nur soft-abandon | Traceability + Anti-Redundanz |
| Broker-Disconnect? | Scoped Pause; SL bei IG ist Sicherheitsnetz | Bei Disconnect können wir nichts mehr aktiv tun |

## 16. Open Questions / Verifizieren in Implementation

1. **Funktioniert Anthropic `web_search` durch `haex-claude-proxy`?** (Muss in M4 verifiziert werden.)
2. **Welche Asset-Classes initial?** Vermutlich FOREX, INDEX initial; CRYPTO später wenn fwbg's Crypto-Adapter fertig ist.
3. **GitHub-Token für PromoteAgent**: woher? PAT mit `repo` scope, vermutlich in env-var.
4. **Email/Push für Alerts**: später; initial nur Dashboard-Banner.
5. **Embedding-Modell für Layer-2-Similarity**: voyage-3 oder Claude-Embedding? Erst entscheiden wenn Layer 2 nötig wird.
6. **Test-Strategy für Pre-Trade-Validators**: golden-path Unit-Tests + Property-based Tests (hypothesis library)?

## 17. Future Work

- Multi-Provider LLM-Switch (LiteLLM-Wrapper) falls Anthropic-only zu beschränkend
- Multi-Account/Multi-Broker (über IG hinaus)
- Cloud-Deployment (Postgres statt SQLite, Auth)
- Hyperparameter-Optimization als eigener Agent
- Strategy-Portfolios statt einzelner Strategien
- Markt-Regime-aware Strategy-Aktivierung
