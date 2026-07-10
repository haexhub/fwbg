# Plan: Enforce a mandatory stop-loss on every entry order (deterministic broker gate)

> **Executor instructions**: Follow this plan step by step. Run every verify
> command and confirm the expected result before moving on. If a STOP condition
> occurs, stop and report — do not improvise.
>
> **Drift check (run first)**: the line numbers below come from a read on
> 2026-07-10 of branch `feat/plugin-register-endpoint`. The repo has active
> feature work, so RE-CONFIRM every cited symbol/line before editing:
> `git -C . log --oneline -5`
> `git grep -n "def submit_order" src/fwbg/adapters packages/`
> `git grep -n "else 50" src/fwbg packages/`
> On any mismatch with "Current state" below, re-locate the symbol; do not edit blind.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (touches live-order placement — verify with the broker tests)
- **Category**: security / safety
- **Origin**: 2026-07-10 read-only audit from the fwbg-agents session (plan 001
  fallout). Verdict was **PARTIAL** — see "Why this matters".

## Why this matters

The governing safety rule (fwbg-agents `CLAUDE.md`, maintainer-confirmed): *"Stop-loss
is mandatory for every order, paper or live. Pre-trade validators reject orders
without SL. SL is sent atomically with entry."* Backtest is exempt (separate engine).

Today that is only **PARTIAL**:
- A truly naked order is *practically* avoided because `bot.py` always computes an
  ATR stop AND the IG adapter substitutes a hardcoded 50-point stop when none is
  passed. SL is also sent **atomically** (good — no naked window).
- BUT there is **no deterministic reject-without-SL gate**. The interface declares
  `stop_distance` optional, the range validator is skipped when it is `None`, and a
  missing/zero stop is silently replaced by an arbitrary 50-point default rather
  than rejected. Safety rests on convention + a silent default, not a hard gate —
  which is exactly what the rule forbids.

Goal: make *"no entry order without a validated stop-loss, sent atomically"* a
single deterministic gate at the broker boundary, uniform across every adapter,
impossible to bypass per-adapter. Exits (`close_position`) stay exempt.

## Current state (verified 2026-07-10 — RE-CONFIRM before editing)

- **Entry caller (only one that opens exposure)** — `src/fwbg/bot.py`: `_check_signal`
  (~:462) → `_execute_signal` (~:510) computes `sl_dist = max(10, int((atr *
  cfg.sl_mult) / cfg.point_value))` (~:563) and calls
  `self.adapter.submit_order(..., stop_distance=sl_dist, limit_distance=tp_dist)`
  (~:573-579). Confirm no other production caller of `submit_order` opens a position.
- **Broker base class** — `src/fwbg/adapters/broker/__init__.py`: abstract
  `submit_order(..., stop_distance: float = None, ...)` (~:171-179; docstring ~:24
  mentions `sl` but the contract does not require it). `close_position` (~:293) sends
  a counter-order via `submit_order` with no stop (~:308) — legitimate (it is an exit).
- **IG adapter** — `src/fwbg/adapters/broker/ig/adapter.py`: `submit_order` (~:478);
  pre-trade guards (~:504-521) check size range and stop/limit range, but the stop
  check is **conditional on presence**: `if stop_distance is not None and not (MIN
  <= stop_distance <= MAX)` (~:510) — skipped entirely when `None`. Silent fallback:
  `sl_dist = int(stop_distance) if stop_distance else 50` (~:526). Atomic submit:
  `self._ig.create_open_position(..., guaranteed_stop=False, stop_distance=sl_dist,
  limit_distance=tp_dist)` (~:534-544).
- **Adapter wiring** — `src/fwbg/adapters/__init__.py:27-29` re-exports
  `IGBrokerAdapter` from `.broker`; `src/fwbg/adapters/broker/__init__.py:362`
  `from .ig import IGBrokerAdapter`; `src/fwbg/__main__.py` `create_adapter()`
  (~:56-68) builds it. paper vs live = only `env` (DEMO/LIVE); identical order code.
- **Duplicate adapter (latent risk)** — `packages/fwbg-broker-ig/src/fwbg_broker_ig/adapter.py`:
  `submit_order` (~:405), same `else 50` fallback (~:433), but **lacks** the
  size/range guards. If the plugin/registry ever loads this package adapter instead
  of the in-tree one, the path is strictly weaker. Confirm whether it is loadable.
- **Result type** — CONFIRM the exact `OrderResult` / `OrderStatus` types and that a
  `REJECTED` status (or equivalent) exists: `git grep -n "class OrderResult\|class OrderStatus\|REJECTED"`.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Establish test runner | check `[tool.pytest.ini_options]` in `pyproject.toml` | note `testpaths` |
| Targeted tests | `uv run pytest tests/<broker tests> -q` (or `pytest` — confirm) | pass |
| New gate tests | `uv run pytest tests/test_broker_sl_gate.py -q` | pass |
| Full suite | `uv run pytest -q` | pass |
| Lint (if configured) | check `pyproject.toml` for ruff | exit 0 |

## Scope

**In scope**:
- `src/fwbg/adapters/broker/__init__.py` (add the gate to the base class)
- `src/fwbg/adapters/broker/ig/adapter.py` (rename impl, drop `else 50`)
- `packages/fwbg-broker-ig/src/fwbg_broker_ig/adapter.py` (converge on the base gate)
- new test file, e.g. `tests/test_broker_sl_gate.py`

**Out of scope**:
- `bot.py` signal/SL-computation logic (it already always computes a stop).
- The backtest/simulation engine (`src/fwbg/simulation/*`) — no order-level SL there, acceptable.
- `guaranteed_stop` policy (gap protection) — a separate, optional follow-up.

## Git workflow

- Branch off the repo's integration base (CONFIRM whether that is `develop` or `main`).
- Conventional commit, e.g. `feat(broker): reject entry orders without a stop-loss`.
- Do NOT push or open a PR unless the operator says so.

## Steps

### Step 1 — Confirm the contracts
Confirm `OrderResult`/`OrderStatus` shape (does a `REJECTED` member exist? what does a
rejected `OrderResult` look like?), and which IG adapter the runtime actually loads
(in-tree vs `packages/`). Write down the answers. If a rejected `OrderResult` cannot be
expressed cleanly → STOP.

### Step 2 — Add the deterministic gate to the base class
In `src/fwbg/adapters/broker/__init__.py`:
- Rename the abstract `submit_order(...)` to an abstract `_submit_order_impl(...)`.
- Add a concrete, **non-overridable** `submit_order(...)` template method that, before
  delegating, rejects an entry lacking a stop:
  ```python
  if stop_distance is None or stop_distance <= 0:
      return OrderResult(success=False, status=OrderStatus.REJECTED,
                         message="Rejected: stop-loss is mandatory for entry orders")
  return self._submit_order_impl(...)
  ```
- Keep exits exempt: `close_position` (~:308) must call `_submit_order_impl(...)`
  directly (a counter-order legitimately has no stop).

### Step 3 — IG adapter
In `src/fwbg/adapters/broker/ig/adapter.py`:
- Rename `submit_order` (~:478) → `_submit_order_impl`.
- **Delete the `else 50` fallback** (~:526) — a missing SL can no longer arrive here
  (the base gate guarantees presence). Keep the range validation; it no longer needs
  the `is not None` conditional (~:510).
- Leave the atomic `create_open_position(..., stop_distance=..., limit_distance=...)`
  as-is; add a one-line comment that SL is forwarded in the same request.

### Step 4 — Converge the duplicate adapter
`packages/fwbg-broker-ig/src/fwbg_broker_ig/adapter.py`: rename its `submit_order`
(~:405) → `_submit_order_impl`, drop its `else 50` (~:433). Prefer making it inherit
the base-class gate (and, ideally, the shared size/stop-range validation) so it cannot
ship a laxer path than the in-tree adapter.

### Step 5 — Tests
Add `tests/test_broker_sl_gate.py` (mirror the stub-adapter pattern in
`tests/test_paper_trade_writer.py`):
- `submit_order(stop_distance=None)` → returns `REJECTED` and the underlying
  `create_open_position` is **never called** (mock/spy it, assert not called).
- `submit_order(stop_distance=0)` → also `REJECTED` (no silent 50).
- A valid `stop_distance` → forwarded in the single `create_open_position` call
  (atomic; assert the kwarg is present in that one call).
- `close_position(...)` → still works with no stop.

### Step 6 — Full suite + lint
`uv run pytest -q` → pass. Lint if configured.

## Done criteria

- [ ] `submit_order` on the base class rejects entries without a positive SL; the check
      cannot be bypassed by an adapter override.
- [ ] No `else 50` (or equivalent silent SL default) remains in any adapter.
- [ ] `close_position` (exits) still submits with no stop.
- [ ] New gate tests exist and pass; full suite green.
- [ ] Both IG adapters go through the same gate.

## STOP conditions

- Any production caller legitimately submits an **entry** order without a stop
  (would now be rejected) — report it; the gate assumption is wrong.
- `OrderResult`/`OrderStatus` cannot express a clean rejection — report the shape.
- The `packages/` duplicate adapter cannot inherit the base gate without a larger
  refactor — report; do not ship the gate on only one adapter silently.
- The runtime can load an adapter that bypasses the base class entirely — report.

## Maintenance notes / follow-up

- `guaranteed_stop=False` (ig/adapter.py ~:541) means a normal stop that can slip on a
  price gap — protected against outage, not against gaps. Consider a
  `FWBG_IG_GUARANTEED_STOP` env for live as a separate change.
- Long-term: converge the two IG adapters (in-tree + `packages/`) so there is a single
  order path to reason about.
