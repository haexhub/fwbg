# fwbg-agents M6a — Live Paper-Trading Telemetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status:** ✓ done 2026-06-24 — 8 tasks (+ 1 fix-up) across two repos. fwbg-agents commits: `b58a868`, `b765757`, `be74769`, `b5691e5`, `954e28c`, `fa0a92a` (282 tests, +24). fwbg commits: `0a8df61`, `3a4e1f1`, `d1e25fe` (2536 tests, +8). Manual `scripts/m6a_smoke.py` against dev DB ends `[m6a_smoke] PASSED` twice in a row (idempotent). M6b (Paper-Analyst + Promote-Live) and M5d (PluginAuthor planner/implementer split) sketched at bottom for next sessions.

**Goal:** Make every paper/live trading account observable in real time per strategy. fwbg writes per-trade-event + periodic position-snapshot to disk in per-strategy directories. fwbg-agents reads them and exposes two GET endpoints the dashboard can poll. Each Strategy maps to exactly one fwbg account-config (1:1).

**Architecture:**
- **Per-Strategy Account Isolation:** new column `Strategy.paper_account_id: str | null` references an existing `~/fwbg/accounts/<id>.yaml` config. fwbg-side TradingBot is launched with both `account_config_path` AND `strategy_slug` — bot now knows which strategy it's executing for. Multiple bots run in parallel (one per Strategy↔Account pair) — fwbg's `run_bot_for_account()` already supports this; we extend the entry-point to pipe `strategy_slug` through.
- **fwbg writes three files per strategy, all under `data/paper-trades/<strategy_slug>/`:**
  - `trades.jsonl` — append-only, one JSON line per executed trade. Event-driven (TradingBot._execute_signal appends after each successful broker call).
  - `status.json` — overwritten per `_write_status` call (already runs periodically). Contains current_equity, starting_equity, last_update_at, equity_curve_sample (bounded last 200 points).
  - `positions.json` — overwritten periodically (every status-write). Contains list of currently-open Positions: symbol, side, qty, entry_price, current_price (if streaming), stop_loss, take_profit, unrealised_pnl.
- **agents-side:** new `tools/fwbg_paper_reader.py` reads all three. Computes PaperTradeSummary (sharpe, dd, win-rate, etc.) from trades.jsonl + status.json. Returns PaperPositions raw from positions.json. No fwbg-utils import — formulas inline (mirrors `utils/metrics.py`).
- **Two new agents endpoints, both pure GET, no LLM, no BackgroundTasks:**
  - `GET /strategies/{id}/paper-summary` → PaperTradeSummary
  - `GET /strategies/{id}/paper-positions` → PaperPositions (currently-open positions with SL/TP)
  - Both return 409 if strategy is not in PAPER_TRADING or LIVE_TRADING.
  - Both return 404 if no on-disk data yet (bot hasn't started or hasn't written anything).

**Tech Stack:** Python 3.13 (both repos), SQLAlchemy 2.x async, FastAPI, pytest, alembic. NO LLM in M6a — pure plumbing + telemetry. No new dependencies.

**Locked Decisions:**
- (A+C) **PATH 2 Cross-Repo, event-driven + periodic.** fwbg writes; agents reads; agents aggregates; dashboard polls. No webhook, no Auto-Poller in agents.
- (B) **DEFERRED to M6b** — paper-criteria YAML only needed by Paper-Analyst.
- (D) **DEFERRED to M6b** — promote-live endpoint comes with the Analyst.
- (E) **paper_phase_target_days column added NOW (M6a) but only as a configurable hint** — the Analyst will USE it in M6b. Dashboard can already let user set the number now.
- (F) **M5c-polish chore commit FIRST.**
- (G) **M6 split: M6a = Live Telemetry (this session), M6b = Paper-Analyst + Promote-Live (next session).**
- (H) **Logical isolation: 1 Strategy = 1 fwbg-account-config.** No broker-sub-account magic; we just pair strategies to accounts in config + start one bot per pair.

**Pre-checks (verified at session start):**
- agents HEAD = `22ce7e1` ✓
- agents `VIRTUAL_ENV= uv run pytest -q` = 258 passed ✓
- agents `VIRTUAL_ENV= uv run alembic current` = `0004 (head)` ✓
- fwbg accounts/ dir exists at `~/fwbg/accounts/` — primary repo path is `~/fwbg/` (graphify-out lives at `~/Projekte/fwbg/`, but source is in `~/fwbg/`)

**File-layout overview (agents-side adds):**
```
src/fwbg_agents/
  tools/fwbg_paper_reader.py          (new — reads trades.jsonl + status.json + positions.json)
  api/strategies.py                   (modify — add 2 endpoints: GET /paper-summary, GET /paper-positions)
  db/models.py                        (modify — add Strategy.paper_account_id + Strategy.paper_phase_target_days)
alembic/versions/
  0005_paper_telemetry.py             (new — migration)
scripts/
  m6a_smoke.py                        (new — end-to-end smoke against dev DB + synthetic on-disk fixtures)
```

**File-layout overview (fwbg-side adds — repo `~/fwbg/`):**
```
src/fwbg/
  bot.py                              (modify — _execute_signal appends trades.jsonl; _write_status writes per-strategy status.json + positions.json)
  __main__.py or core/config.py       (modify — AssetConfig gets strategy_slug field; run_bot_for_account pipes it through)
  adapters/broker/__init__.py         (modify — Position dataclass gets optional stop_loss, take_profit, current_price fields)
data/paper-trades/<slug>/             (new runtime artefact — append-only trades.jsonl + overwritten status.json + positions.json)
tests/
  test_paper_trade_writer.py          (new — fwbg-side writer tests)
```

---

## Task 0 — Chore: M5c-deferred polish items (1 commit, agents-only)

**Files (all in `~/Projekte/fwbg-agents/`):**
- Modify: `src/fwbg_agents/agents/translator.py` — hoist `_PHASE_TO_FIELD` to module scope.
- Modify: `src/fwbg_agents/orchestrator/plugin_catalog.py` — add public `reset_fwbg_cache()` helper.
- Modify: reiterate-flow AgentRun creator (likely `orchestrator/plugin_flow.py`) — set `ar.plugin_id` on reiterate-flow runs.
- Modify: `src/fwbg_agents/agents/translator.py` — extract `validate_reiterate_preconditions()` (or publicize `_lookup_plugin_capability`).
- Modify: `scripts/m5b_smoke.py`, `scripts/m5c_smoke.py` — lift Stage-0 idempotency check above `_seed_parent_strategy()`.

**Behaviour:** surface-level refactors, no behaviour change. Tests stay green.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest -q
```
Expected: 258 passed.

**Commit:** `chore(M5c-polish): hoist _PHASE_TO_FIELD, add reset_fwbg_cache helper, set plugin_id on reiterate, extract validate_reiterate_preconditions, lift smoke idempotency check`

---

## Task 1 — Strategy migration: paper_account_id + paper_phase_target_days

**Files:**
- Modify: `src/fwbg_agents/db/models.py` — add two columns to Strategy:
  - `paper_account_id: Mapped[str | None] = mapped_column(String, nullable=True)`
  - `paper_phase_target_days: Mapped[int] = mapped_column(Integer, nullable=False, server_default="90")`
- Create: `alembic/versions/0005_paper_telemetry.py` (autogen + review).
- Test: `tests/db/test_strategy_paper_columns.py` (new).

**Behaviour:**
- `paper_account_id` is a free-form string. Dashboard sets it to e.g. `"ig-demo-account-001"` matching a `~/fwbg/accounts/ig-demo-account-001.yaml`. agents NEVER reads the fwbg accounts/ dir — it only stores the string. Cross-repo wiring stays loose.
- `paper_phase_target_days` default 90, configurable per Strategy. M6a doesn't USE it yet — that's M6b's Analyst. Adding here so dashboard can already show + edit it.
- Migration upgrade: ADD COLUMN both. Downgrade: DROP COLUMN both.

**Test sketch:**
1. `test_new_strategy_has_null_paper_account_id` — newly-created Strategy → `.paper_account_id is None`.
2. `test_paper_account_id_can_be_set` — set to `"foo-account"`, persist, reload → value matches.
3. `test_new_strategy_has_default_paper_phase_target_days_90` — `.paper_phase_target_days == 90`.
4. `test_paper_phase_target_days_can_be_overridden` — set 120 on create, persist, reload → 120.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run alembic upgrade head
VIRTUAL_ENV= uv run pytest tests/db/test_strategy_paper_columns.py -v
VIRTUAL_ENV= uv run alembic current
```
Expected: `0005 (head)`, 4 tests pass.

**Commit:** `feat(M6a): Strategy.paper_account_id + paper_phase_target_days columns (per-strategy account isolation + paper-phase timing)`

---

## Task 2 — fwbg-side: AssetConfig.strategy_slug + bot startup wiring

**Files (all in `~/fwbg/`):**
- Modify: `src/fwbg/bot.py` (AssetConfig dataclass at L49 per graphify) — add `strategy_slug: str | None = None`.
- Modify: `src/fwbg/__main__.py` (`run_bot_for_account()` at L70 per graphify, `main()` at L122) — accept `--strategy-slug` CLI arg, pipe into AssetConfig.
- Modify: `src/fwbg/bot.py.run_bot_from_config()` (L762) — accept strategy_slug from config-file or kwarg.
- Test: `tests/test_strategy_slug_wiring.py` (new).

**Behaviour:**
- AssetConfig gains an optional `strategy_slug: str | None` field. None = legacy mode (no paper-trade telemetry written — backward-compat for existing accounts).
- CLI: `python -m fwbg --account accounts/ig-demo.yaml --strategy-slug orb__forex__001` sets the slug. Without `--strategy-slug` → legacy mode.
- AssetConfig is passed through to TradingBot via its constructor (already accepts AssetConfig per graphify L102).
- TradingBot exposes `self.strategy_slug` for downstream writer (Task 3) to read.

**Test sketch:**
1. `test_asset_config_strategy_slug_defaults_none` — `AssetConfig()` → `.strategy_slug is None`.
2. `test_asset_config_strategy_slug_can_be_set` — `AssetConfig(strategy_slug="foo")` → preserved.
3. `test_trading_bot_exposes_strategy_slug_from_config` — bot constructed with config containing slug → `bot.strategy_slug == "foo"`.
4. `test_run_bot_for_account_accepts_strategy_slug_kwarg` — call signature accepts kwarg; bot receives it.
5. `test_run_bot_for_account_legacy_mode_when_slug_none` — no slug → bot.strategy_slug is None → no telemetry side-effects (verified in Task 3).

**Verify:**
```bash
cd ~/fwbg && uv run pytest tests/test_strategy_slug_wiring.py -v
```
Expected: 5 tests pass. Full fwbg suite still green.

**Commit (fwbg repo):** `feat(M6a): AssetConfig.strategy_slug + bot startup wiring (1-strategy-per-bot for paper-trade telemetry isolation)`

---

## Task 3 — fwbg-side: per-strategy paper-trade writer

**Files (all in `~/fwbg/`):**
- Modify: `src/fwbg/bot.py`:
  - `_execute_signal()` (L485) — after successful adapter call, IF `self.strategy_slug` is set AND bot runs in paper mode (`adapter.is_paper` or equivalent — investigate, add property if missing), append one JSON line to `data/paper-trades/{strategy_slug}/trades.jsonl`.
  - `_write_status()` (L728) — extended to ALSO write per-strategy `status.json` and `positions.json` when `self.strategy_slug` is set.
- Modify: `src/fwbg/adapters/broker/__init__.py` — `Position` dataclass (L73) gains three OPTIONAL fields: `stop_loss: float | None = None`, `take_profit: float | None = None`, `current_price: float | None = None`.
- Modify: `src/fwbg/adapters/broker/__init__.py` — `BrokerAdapter` base gains an abstract property `is_paper: bool` (default `True` in adapter implementations against demo accounts; child adapters can override).
- Test: `tests/test_paper_trade_writer.py` (new).

**trades.jsonl line schema:**
```json
{"trade_id": "uuid", "strategy_slug": "orb__forex__001", "symbol": "EURUSD",
 "side": "buy", "entry_time": "2026-06-24T10:30:00Z", "exit_time": "2026-06-24T11:15:00Z",
 "entry_price": 1.0823, "exit_price": 1.0867, "pnl_pct": 0.0041,
 "quantity": 1000, "fees": 0.4}
```

**status.json schema (overwritten per write):**
```json
{"strategy_slug": "orb__forex__001", "updated_at": "ISO8601",
 "current_equity": 10421.50, "starting_equity": 10000.00,
 "equity_curve_sample": [{"t": "ISO8601", "equity": 10000.00}, ...up to 200]}
```

**positions.json schema (overwritten per write):**
```json
{"strategy_slug": "orb__forex__001", "updated_at": "ISO8601",
 "positions": [
   {"symbol": "EURUSD", "side": "buy", "quantity": 1000, "entry_price": 1.0823,
    "current_price": 1.0851, "stop_loss": 1.0790, "take_profit": 1.0900,
    "unrealised_pnl_pct": 0.0026, "opened_at": "ISO8601"}
 ]}
```

**Behaviour:**
- File creates parent dir on first write (`Path.mkdir(parents=True, exist_ok=True)`).
- Telemetry writes are BEST-EFFORT: failures logged via `logger.warning(...)`, NOT raised. Trade execution must NOT abort because of telemetry.
- `equity_curve_sample`: last N=200 observations. Downsample older points if curve longer (every 2nd, then every 4th, etc.).
- `positions.json` is sourced from `self.adapter.get_positions()` plus the bot's internal SL/TP tracking dict (TradingBot will need a small dict `self._sl_tp_by_symbol: dict[str, tuple[float | None, float | None]]` populated on each entry signal — investigate during impl; if SL/TP comes from signal-generation, store at that point).
- Bot in LIVE mode (`adapter.is_paper == False`) currently STILL writes to the same dir but is gated separately — TBD whether live-trading writes go to `data/live-trades/<slug>/` (parallel tree). **M6a decision: live-mode also writes to `data/paper-trades/<slug>/` because the user explicitly said "jedes paper/live trade konto" — the dir name "paper-trades" is misleading then.** Rename dir to `data/account-trades/<slug>/` to cover both. (Note: M5c uses `data/strategies/...` for strategy-side artefacts. Keep the per-account-trade telemetry separate under `data/account-trades/...`.)

**Test sketch:**
1. `test_execute_signal_with_strategy_slug_appends_trades_jsonl` — bot with slug set, simulate signal → `data/account-trades/foo/trades.jsonl` has one line.
2. `test_execute_signal_without_strategy_slug_writes_nothing` — bot with slug=None → no file created.
3. `test_write_status_writes_status_and_positions_json` — call _write_status with slug set → both files exist with expected shape.
4. `test_status_json_equity_curve_bounded_to_200` — pump 500 equity points → file's `equity_curve_sample` has ≤200 entries.
5. `test_positions_json_includes_sl_tp_from_adapter` — adapter mock returns Position(stop_loss=1.08, take_profit=1.09) → positions.json line has those fields.
6. `test_write_failures_are_logged_not_raised` — patch open() to raise → bot continues normally, warning logged.
7. `test_position_dataclass_accepts_optional_sl_tp_current_price` — `Position(symbol="EURUSD", side="buy", quantity=1, entry_price=1.08, stop_loss=1.07, take_profit=1.10, current_price=1.085)` → all fields preserved.
8. `test_broker_adapter_is_paper_default_true_for_demo_adapter` — IGBrokerAdapter pointed at demo creds → `.is_paper is True`.

**Verify:**
```bash
cd ~/fwbg && uv run pytest tests/test_paper_trade_writer.py -v
```
Expected: 8 tests pass. Full fwbg suite still green.

**Commit (fwbg repo):** `feat(M6a): per-strategy paper/live-trade telemetry — trades.jsonl + status.json + positions.json under data/account-trades/<slug>/`

---

## Task 4 — agents-side: fwbg_paper_reader (Summary + Positions)

**Files (all in `~/Projekte/fwbg-agents/`):**
- Create: `src/fwbg_agents/tools/fwbg_paper_reader.py` — two functions:
  - `read_paper_summary(strategy_slug, fwbg_data_dir) -> PaperTradeSummary | None`
  - `read_paper_positions(strategy_slug, fwbg_data_dir) -> PaperPositions | None`
- Test: `tests/tools/test_fwbg_paper_reader.py` (new).

**PaperTradeSummary (pydantic.BaseModel):**
```python
class PaperTradeSummary(BaseModel):
    strategy_slug: str
    sharpe_paper: float
    max_dd_paper: float        # 0.0..1.0
    trades_total: int
    trades_today: int
    days_in_paper: int
    win_rate: float            # 0.0..1.0
    last_trade_at: datetime | None
    current_equity: float
    starting_equity: float
    equity_curve_sample: list[dict[str, Any]]   # [{"t": iso, "equity": float}, ...]
```

**PaperPositions (pydantic.BaseModel):**
```python
class PaperPosition(BaseModel):
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    entry_price: float
    current_price: float | None
    stop_loss: float | None
    take_profit: float | None
    unrealised_pnl_pct: float | None
    opened_at: datetime

class PaperPositions(BaseModel):
    strategy_slug: str
    updated_at: datetime
    positions: list[PaperPosition]
```

**Behaviour:**
- `read_paper_summary("foo", Path(".../fwbg/data"))` reads `account-trades/foo/trades.jsonl` and `account-trades/foo/status.json`. Returns None if neither exists.
- If only status.json exists (no trades yet): returns Summary with `trades_total=0, sharpe_paper=0.0, win_rate=0.0`, max_dd computed from equity_curve_sample.
- Sharpe inline: `mean(per-trade pnl_pct) / std(per-trade pnl_pct) * sqrt(252)`. Std=0 → sharpe=0.
- Max DD: walks equity_curve_sample running max, computes `max((peak - eq) / peak)`. Empty curve → 0.0.
- Win rate: `winning_count / total`. Empty → 0.0.
- `days_in_paper`: `(now - first_trade.entry_time).days`. No trades → 0.
- `trades_today`: count where `entry_time.date() == now.date()` UTC.
- `read_paper_positions(slug, dir)`: reads `account-trades/foo/positions.json`. Returns None if missing.

**Test sketch:**
1. `test_summary_returns_none_when_no_files_exist` — empty dir → None.
2. `test_summary_with_status_only_zero_trades` — status.json only → trades_total=0, dd from curve.
3. `test_summary_computes_sharpe_from_trades` — fixture with 30 trades, known pnl distribution → sharpe within tolerance.
4. `test_summary_computes_max_dd_from_equity_curve` — curve [100,120,90,105] → dd ≈ 0.25.
5. `test_summary_win_rate_from_trades` — 6 winners 4 losers → 0.6.
6. `test_summary_days_in_paper_from_first_trade` — first trade 45 days ago → ≥45.
7. `test_summary_trades_today_filters_by_utc_date` — mixed timestamps → only today's count.
8. `test_positions_returns_none_when_file_missing`.
9. `test_positions_parses_sl_tp_current_price`.
10. `test_positions_empty_list_when_file_has_empty_positions`.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/tools/test_fwbg_paper_reader.py -v
```
Expected: 10 tests pass.

**Commit:** `feat(M6a): fwbg_paper_reader aggregates trades.jsonl + status.json + positions.json into PaperTradeSummary + PaperPositions`

---

## Task 5 — agents-side: GET /strategies/{id}/paper-summary endpoint

**Files:**
- Modify: `src/fwbg_agents/api/strategies.py` (or wherever per-strategy router lives — investigate first).
- Test: `tests/api/test_strategies_paper_summary.py` (new).

**Endpoint behaviour:**
- 200 → returns PaperTradeSummary JSON when on-disk data exists.
- 404 → `{"detail": "no paper-trade data on disk for strategy {slug}"}` when reader returns None.
- 409 → `{"detail": "strategy not in PAPER_TRADING or LIVE_TRADING state, got {state}"}` for strategies in other states.
- Pure read, no LLM, no BackgroundTasks. Dashboard polls this endpoint.
- `fwbg_data_dir` resolved from `settings.FWBG_DATA_DIR` env var (default `~/fwbg/data`).

**Test sketch:**
1. `test_returns_200_with_summary_when_on_disk_data_exists` — seed strategy in PAPER_TRADING + fixture files → 200.
2. `test_returns_404_when_no_on_disk_data` — strategy exists but no files → 404.
3. `test_returns_409_when_strategy_in_proposed_state` — PROPOSED strategy → 409.
4. `test_accepts_live_trading_state` — strategy in LIVE_TRADING + files exist → 200.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/api/test_strategies_paper_summary.py -v
```
Expected: 4 tests pass.

**Commit:** `feat(M6a): GET /strategies/{id}/paper-summary endpoint (dashboard-polled, no LLM)`

---

## Task 6 — agents-side: GET /strategies/{id}/paper-positions endpoint

**Files:**
- Modify: `src/fwbg_agents/api/strategies.py`.
- Test: `tests/api/test_strategies_paper_positions.py` (new).

**Endpoint behaviour:**
- 200 → returns PaperPositions JSON when positions.json exists.
- 404 → `{"detail": "no positions snapshot on disk for strategy {slug}"}` when file missing.
- 409 → same state-guard as paper-summary endpoint.
- Returns empty positions list (200) if file exists but no open positions.

**Test sketch:**
1. `test_returns_200_with_positions_when_file_exists`.
2. `test_returns_200_with_empty_positions_when_no_open_positions`.
3. `test_returns_404_when_file_missing`.
4. `test_returns_409_when_strategy_in_proposed_state`.
5. `test_positions_payload_includes_sl_tp_current_price` — fixture file has SL=1.07 TP=1.09 current=1.085 → all visible in response.

**Verify:**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run pytest tests/api/test_strategies_paper_positions.py -v
```
Expected: 5 tests pass.

**Commit:** `feat(M6a): GET /strategies/{id}/paper-positions endpoint (live open positions with SL/TP for dashboard)`

---

## Task 7 — End-to-end M6a smoke (m6a_smoke.py)

**Files:**
- Create: `scripts/m6a_smoke.py`.

**Smoke flow:**
1. Stage 0: idempotency check + cleanup of prior smoke artifacts.
2. Stage 1: seed Strategy in PAPER_TRADING with `slug="orb__forex__m6a__001"`, `asset_class="forex"`, `paper_account_id="ig-demo-001"`, `paper_phase_target_days=90`.
3. Stage 2: synthesise on-disk fixtures under a tmp data-dir (override FWBG_DATA_DIR env via settings):
   - `account-trades/orb__forex__m6a__001/trades.jsonl` — 30 trades over 45 days, sharpe ~0.9, dd ~0.10.
   - `account-trades/orb__forex__m6a__001/status.json` — equity 10000→11200 curve.
   - `account-trades/orb__forex__m6a__001/positions.json` — 2 open positions with SL/TP.
4. Stage 3: GET `/strategies/{id}/paper-summary` → assert 200, `sharpe_paper ~ 0.9`, `trades_total == 30`.
5. Stage 4: GET `/strategies/{id}/paper-positions` → assert 200, 2 positions, first has SL/TP set.
6. Stage 5: state-guard check — change strategy state to PROPOSED, retry endpoints → both 409.
7. Stage 6: 404-check — restore PAPER_TRADING but delete fixture files → both 404.
8. Print `[m6a_smoke] PASSED`.

**Verify (manual, against dev DB):**
```bash
cd ~/Projekte/fwbg-agents && VIRTUAL_ENV= uv run python scripts/m6a_smoke.py
```
Expected: `[m6a_smoke] PASSED` at the end.

**Commit:** `feat(M6a): scripts/m6a_smoke.py — end-to-end live-telemetry smoke (summary + positions + state-guards)`

---

## End-of-Session Housekeeping (after Task 7 green)

1. **Update design-doc** `docs/plans/2026-06-23-fwbg-agents-design.md`:
   - Implementation-Status-Table → M6a row with final commits + ✓ done.
   - "Late-binding design changes" note: M6 split into M6a/M6b; account-trades dir naming; per-strategy account-id field.
2. **Mark this plan-doc done:** Status → `✓ done YYYY-MM-DD — N commits in fwbg-agents (and M in fwbg)`.
3. **Update memory:** `project_fwbg_agents.md` gains an M6a-done block.
4. **New reference memory:** `reference_fwbg_agents_m6a_plan.md` with commit-table.
5. **Save new memory:** [[reference-fwbg-agents-m5d-plan]] **placeholder** linking to a future M5d plan-doc (see Next Milestones below).
6. **MEMORY.md index update.**
7. **graphify update** in both repos.
8. **Final manual smoke** against dev DB → `[m6a_smoke] PASSED`.
9. **fwbg-side commit style check** before Task 2/3 commits: `git log --oneline -10` in fwbg.

---

## Expected test growth (M6a)

- Task 0: 0 new (refactor; agents stays at 258).
- Task 1: +4 (agents 262).
- Task 2 (fwbg-repo): +5 (fwbg gains 5).
- Task 3 (fwbg-repo): +8 (fwbg gains 8 — cumulative +13 fwbg).
- Task 4: +10 (agents 272).
- Task 5: +4 (agents 276).
- Task 6: +5 (agents 281).
- Task 7: smoke is manual, no pytest count.

**Final agents test count target: 281 passed.** fwbg gains +13.

---

## Commit-discipline reminder

- One commit per task. English. NO Claude footers.
- Per commit: green pytest + clean alembic.
- agents-side commits in `~/Projekte/fwbg-agents/`, fwbg-side commits in `~/fwbg/`.
- Tasks 2+3 are the only ones touching fwbg.

---

# Next Milestones (parked for future sessions)

## M6b — Paper-Analyst + Promote-Live (~5-6h, agents-only, ~+20 tests)

**Goal:** Add the LLM-driven decision layer on top of M6a's telemetry. The Analyst decides Promote / Abandon / ContinueObservation; Promote sets a metadata flag; a separate manual endpoint guards the paper→live transition behind explicit human_approval.

**Scope:**
- `data/criteria/paper/<class>.yaml` — hand-curated per-asset-class thresholds (looser than backtest).
- `src/fwbg_agents/orchestrator/criteria_paper.py` — `load_paper_criteria(class)` + `evaluate_paper_criteria(summary, criteria)`.
- `src/fwbg_agents/agents/paper_analyst.py` (+ `prompts/paper_analyst.md`) — pydantic-ai agent emitting discriminated-union output Promote | Abandon | ContinueObservation. Deterministic validator: Promote requires criteria-pass; Abandon requires post_mortem_path; ContinueObservation auto-flags `stale=True` if `days_in_paper > paper_phase_target_days`.
- `src/fwbg_agents/orchestrator/paper_flow.py` — `paper_analyze(strategy_id)` writes sidecar + sets `Strategy.metadata.paper_analyst_promote_recommended=True` on Promote. NO state transition.
- `POST /strategies/{id}/paper-analyze` — manual trigger, BackgroundTasks, AgentRun lifecycle.
- `POST /strategies/{id}/promote-live` — requires `payload.human_approval=true` AND `Strategy.metadata.paper_analyst_promote_recommended=true`. Calls M2's `transition_strategy(paper_trading → live_trading, human_approval=true)`. Writes audit AgentRun.
- End-to-end `scripts/m6b_smoke.py`.

**Locked decisions (from M6a session):**
- (B) Separate `data/criteria/paper/<class>.yaml`, hand-curated, audited.
- (D) Sidecar-recommendation + separate promote-live endpoint. Analyst NEVER transitions.
- (E) `paper_phase_target_days` already in DB (M6a). Soft warning, no auto-abandon.

## M5d — PluginAuthor Plan-Agent + Implement-Agent split (~4-6h, agents-only, ~+15 tests)

**Goal:** Split the M5b PluginAuthor into two distinct LLM agents: a **PluginPlanner** (designs the indicator/plugin: what calculation, signature, edge cases) and a **PluginImplementer** (writes the actual Python module). Different models for each — typically stronger model for planning, weaker for implementation.

**Motivation (user, 2026-06-24):** "für die implementierung von neuen indikatoren/plugins möchte ich für die plannungsphase auf das stärkste model setzen und für die konkrete implementierung tendenziell ein etwas schwächeres. Konkretes beispiel wenn wir mit claude arbeiten: dann möchte ich konfigurieren können dass der plan-agent derzeit opus 4.8 und der implementator opus 4.7 ist."

**Scope sketch (DETAILED PLAN in separate doc later):**
- `src/fwbg_agents/agents/plugin_planner.py` (+ `prompts/plugin_planner.md`) — emits a structured `PluginPlan` object (function signature, params, edge cases, test expectations) using a stronger configurable model (env `PLUGIN_PLANNER_MODEL`, default `claude-opus-4-8`).
- Rename `src/fwbg_agents/agents/plugin_author.py` → `plugin_implementer.py`; it now CONSUMES a PluginPlan + emits Python code via a weaker model (env `PLUGIN_IMPLEMENTER_MODEL`, default `claude-opus-4-7`).
- `src/fwbg_agents/orchestrator/plugin_flow.py.author_plugin()` chains: Planner → PluginPlan sidecar → Implementer → plugin module → Evaluator (M5b unchanged).
- Two AgentRuns per author-session: `kind="plugin_plan"` then `kind="plugin_implement"`. Both linked to the same Plugin row via `plugin_id`.
- Settings file pattern: `~/.fwbg-agents/agent_models.yaml` (or env vars) — central place to assign per-agent models. Forward-compatible with future per-agent model overrides for Researcher/Analyst/Translator.
- Backwards-compatibility: if `PLUGIN_PLANNER_MODEL` env var is unset, fall back to current single-agent flow (call only the Implementer with implicit planning). This keeps M5b smokes green during migration.

**Open questions for M5d planning session:**
- Single PluginPlan schema or per-kind (indicator/feature_selection/preprocessing/filter) variations?
- Does Implementer get to PUSH BACK on a plan (e.g., "your spec is incoherent")? If yes, what's the loop?
- Settings file format: YAML, JSON, env-only? Memory says pydantic-ai — pydantic-settings can read all three.

**Concrete-before-generic guard:** start with PluginPlanner+Implementer specifically. DON'T abstract `BaseAgent.with_model(...)` until 2-3 agents need the same pattern.
