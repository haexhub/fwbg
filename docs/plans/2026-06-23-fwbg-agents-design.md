# fwbg-agents — Design Document

**Datum**: 2026-06-23
**Status**: M0+M1+M2+M3+M4 implemented, M5 next
**Author**: brainstorming session zwischen User und Claude

## Implementation Status

| Milestone | Status | Notes |
|---|---|---|
| M0 — Skeleton | ✓ done 2026-06-23 | Commit `5cc7649` in `~/Projekte/fwbg-agents/`. FastAPI boots, SQLite + alembic init, mock SSE, pydantic-ai LLM client. Proxy connection not end-to-end verified (port 8080 was occupied by another service during test). |
| M1 — Calibrator + Criteria | ✓ done 2026-06-23 | Calibrator scans `~/fwbg/test_results/` (not `~/Projekte/fwbg/test_results/`), groups by asset class via inlined symbol→class map (no fwbg runtime dep), writes section-6.1 YAMLs + raw baseline JSON. Endpoints: `GET /criteria`, `GET/PUT /criteria/{class}`, `POST /calibrate`, `GET /calibrate/runs`. Verified against real data: 79 runs scanned, 12 INDEX elites, calibration_run row persisted. Dashboard page at `/agents/criteria` (textarea editor, Recalibrate button). DSR/PBO/max_drawdown/profit_factor are absent from current fwbg results and intentionally omitted from generated YAML — schema is forward-compatible when fwbg starts emitting them. |
| M2 — Strategy-Lifecycle Skeleton | ✓ done 2026-06-24 | Commit `ade47ad`. ORM models (`Strategy`, `Plugin`, `Transition`, `StrategyTag`) + alembic 0002. Deterministic state machine in `orchestrator/lifecycle.py`: collapsed strategy lifecycle `proposed → backtested → paper_trading → live_trading` plus terminal `abandoned`, plugin lifecycle `specified → authored → verified → adopted_in_fwbg` plus `abandoned`. Guards: `backtested → paper_trading` evaluates criteria YAML via a small comparator parser (`>=`, `<=`, `>`, `<`, `==`, `!=`); `paper_trading → live_trading` requires `human_approval=True` in payload (UI gate ships M7, state machine enforces from day one); `→ abandoned` requires `post_mortem_path` (anti-redundancy for the Researcher's M4 prior-art lookup). Append-only: no cascade deletes, transition rows insert-only. Read-only API: `GET /strategies` (filters `?state=` `?asset_class=`), `GET /strategies/{id}` (detail + transitions + tags), `GET /strategies/{id}/transitions`; mirror for `/plugins`. 23 new tests (16 lifecycle + 7 API) all green. End-to-end smoke (`scripts/m2_smoke.py`) verified live. Dashboard pages deferred to a follow-up session. |
| M3 — Runner + Analyst | ✓ done 2026-06-24 | Commit `df33384`. Adds the first real iteration loop (manually triggered). New module `tools/fwbg_client.py` is a thin async httpx wrapper around fwbg's `/api/runs/start`, `/runs/{id}`, `/runs/{id}/progress`. New deterministic `agents/runner.py` (no LLM — Runner is on the critical path): copies the strategy.json into fwbg's strategies dir, posts the start, polls until terminal, fetches the full run, writes `fwbg_results.json` into the iteration dir, then `transition_strategy(s, BACKTESTED, payload={fwbg_run_id, results_path, backtest_metrics})`. New LLM-driven `agents/analyst.py`: pydantic-ai with structured output (`Promote | Abandon | TuneParams | ChangeExit`), system prompt in `agents/prompts/analyst.md` for easy iteration, every call recorded in `llm_call`. New `orchestrator/recommendations.py.validate_and_apply` runs hard rules between Analyst output and any state change (Promote re-checks criteria YAML; Abandon writes `post_mortem.yaml`; TuneParams/ChangeExit persist as sidecar JSON for M4 Translator). M3 endpoints: `POST /strategies` (manual seeding into PROPOSED with `iteration_001/strategy.json`), `POST /strategies/{id}/run`, `POST /strategies/{id}/analyze`, `GET /agents/runs/{id}`. Migration 0003 adds `agent_run` + `llm_call`. 78 tests green (was 42 after M2). Decisions captured: Runner does NOT bump `iteration_count` (M4 Translator owns iteration bumps); `llm_call` lives in M3 since Analyst is the first LLM consumer; prompt in `.md` file for prompt-iteration ergonomics. Smoke (`scripts/m3_smoke.py`) verified against live fwbg :8420 (strategy POSTed, Runner kicked, fwbg job accepted, polling working); Analyst LLM call exercised against the configured proxy (404 in current env — `haex-claude-proxy` not running, code path verified). |
| M4 — Researcher + Translator | ✓ done 2026-06-24 | Plan at `docs/plans/2026-06-24-fwbg-agents-m4.md`. Final commit `45825f9` in `~/Projekte/fwbg-agents/`. New modules: `orchestrator/prior_art.py` (tag-Jaccard similarity, `1e9707e`), `orchestrator/hypotheses.py` (`ResearcherHypothesis` + `validate_hypothesis` + `generate_slug`, `4d5041a`), `tools/web_search.py` (Tavily client with quota tracking via `llm_call(model='tavily-search')`, `3991d4f`), `agents/researcher.py` (LLM with `lookup_prior_art` + `search_web` tools and hard anti-redundancy gate, `6dd3093`), `orchestrator/strategy_validator.py` (lightweight structural validator + hardcoded plugin-slug catalog, `3d699da`), `agents/translator.py` (fresh-mode LLM: hypothesis → strategy.json + spec.md with canonical slug enforced, `35324cf`; reiterate-mode fully deterministic with `parent_strategy_id` lineage + extended `ChangeExit.new_exit_strategy`, `ed2b59a`), `orchestrator/research_flow.py` (Researcher → persist Strategy + StrategyTag + initial Transition + write hypothesis.json/research_notes.md → Translator.run_fresh, `73257f1`), `api/research.py` wired into main router (`POST /research/brief`, `POST /strategies/{id}/reiterate` with 422/409 preconditions, `GET /hypotheses`, `ed453a5`), `scripts/m4_smoke.py` (end-to-end via ASGI transport with graceful TAVILY_API_KEY skip, `45825f9`). 159 tests green (78 baseline + 81 new). No migration in M4 (Tavily reuses `llm_call`). Locked decisions: re-iterate via `parent_strategy_id` (not state-machine regression), Tavily quota via convention `model='tavily-search'` (no schema change), strategy.json validation is lightweight structural — full `fwbg.core.config.StrategyConfig` deferred until fwbg becomes runtime dep, Anthropic web_search fallback documented but not built pending proxy compat verification. |
| M5 — Plugin-System | pending | |
| M6 — Paper Trading | pending | |
| M7 — Live Trading + Risk | pending | |
| M8 — Promotion + Polish | pending | |

### Late-binding design changes

- **LLM SDK**: switched from raw `anthropic` SDK to **`pydantic-ai`** during M0 — provider-neutral, typed, Vercel-AI-SDK-style. `AnthropicModel(base_url=...)` still routes through `haex-claude-proxy`. Decision recorded in section 15.

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
POST   /strategies                    # manuell anlegen (Researcher-Skip)
POST   /strategies/{slug}/transition  # state-transition (z.B. paper→live)
# NO DELETE

# Agent Runs
GET    /agents                        # registry, configured?, enabled?
GET    /agents/runs                   # filterable
GET    /agents/runs/{id}              # full transcript
POST   /agents/runs/{id}/cancel       # asyncio task cancellation

# Plugin Lifecycle
GET    /plugins
GET    /plugins/{slug}/verifications
POST   /plugins/{slug}/verify         # ad-hoc re-verification
POST   /plugins/{slug}/promote        # trigger PromoteAgent

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
