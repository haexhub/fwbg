# Jev Signal Provider

## Status: Stage A failed — not pursued further (2026-09-25, revised)

Feasibility check (Evaluation Stage A/A0 below) ran against real EUR/USD H1
data (`scripts/jev_batch_query.py`, ~1040 live calls across 5 configurations,
0 API errors). Result: both providers perform **at or below random-chance
level** on every configuration tested — including after fixing a real
confound found partway through (see below).

| Config | Setup | Ground truth split | `jev_official` acc. | `jev_semif` acc. |
|---|---|---|---|---|
| 1 | Fixed 54p/36p (spread-mult defaults), raw price deltas in prompt | 88/8/5 | 30–40% | 7.5–10% |
| 2 | Same barriers, explicit target/stop prices + pip count | 88/8/5 | 27.5–32.5% | 10–27.5% |
| 3 | Symmetric 1.5×ATR(14), 8h horizon (tuned for class balance) | 40/35/25 | 27.5% | 27.5–30% |
| 4 | Curated narrative state (5-8 semantic signals, session context) + qualitative directional question (no price arithmetic), criteria order alternated per bar | 40/35/25 | 27.5–30% | 32.5% |

**Important correction to configs 1-3's `jev_official` numbers:** a
dedicated diagnostic (`orderbias` mode) found `jev_official`'s `choice`
answer is dominated by **JSON key position**, not content — swapping
`criteria` key order (down_first before up_first) flipped the dominant
answer from `down_first` (90%) to `up_first` (55%), in both cases the key
in position 2. Configs 1-3 always used the same key order, so their
`jev_official` accuracy numbers reflect this positional artifact, not
genuine (mis)calibration. Config 4 alternates key order per bar to cancel
this out — its number is the first trustworthy `jev_official` result, and
it's still at chance (27.5-30%, vs. ~33% random / 40% majority-baseline).

`jev_semif` doesn't show the same clean positional flip (65%→75% down_first
across the swap, i.e. real but weaker order-sensitivity) but shows an
equally concerning **format-sensitivity**: it answered `neither` on 2% of
bars under the arithmetic price-threshold framing (config 3) vs. 92% of
bars under the narrative/qualitative framing (config 4) — for the *same*
underlying bars. Its choice tracks prompt style more than market state.

Taken together: not a setup artifact anymore. Five configurations addressing
barrier width, class balance, price-framing clarity, indicator density/
format, question type (arithmetic vs. qualitative), and JSON positional bias
all land at/below chance, and both providers show answer patterns driven
more by surface prompt structure than by input content. Cost was never the
constraint (whole check <$0.10 total); calibration quality was.

Kept for traceability — see `## Open unknowns` (now resolved) and the raw
per-config caches under `data/jev_cache/eurusd_h1_results*.csv` for the
full per-bar breakdown if this is revisited later (e.g. a fundamentally
different question mechanism such as decomposed independent `noul`
questions, a different asset/timeframe, or moving to a reasoning-capable
model instead of a "System 1" typed-decision one).

## Problem

Jev-style "System One" decision models (typed probabilistic answers instead
of generated text) are fast and cheap enough to query for every historical
bar in a backtest. Question: does asking one — "will price touch +X before
-Y in the next N bars" — produce a signal with real, calibrated edge for
EUR/USD, and does it fit into fwbg's existing pipeline without new framework
code?

Two candidate providers exist:

- `jev_official` — TypeSafe AI's commercial Jev API (RLCD-trained, paid,
  `$0.042/M` input tokens). No direct TypeSafe account; reached via an
  existing OpenRouter key/account instead — `POST
  https://openrouter.ai/api/v1/systemone`, `model: "typesafe/jev-1.13"`
  (OpenRouter resolves this to the dated model id, e.g.
  `typesafe/jev-1.13-20260917`). An Abacus.ai (RouteLLM) account was tried
  first and does **not** carry Jev/TypeSafe — its `/v1/models` list is
  conventional chat models only (GPT/Claude/Gemini/Grok/DeepSeek/...), and
  the Abacus key is rejected (401) directly against `api.typesafe.ai`.
- `jev_semif` — itemis's own hosted instance at
  `https://llms.itemis.cloud/typesafe/v1/systemone`, built on the independent
  open-source SemIf/OpenJev project (small open models — Qwen3 0.6B /
  MiniCPM5 2B / Qwen3.5 4B — read via direct choice-logit readout). SemIf is
  **not affiliated with TypeSafe**; it approximates the same "typed decision"
  interface with different (open, self-hosted) models.

Both expose the same request shape:

```json
{
  "state": "<free text>",
  "questions": {
    "<name>": {
      "type": "noul | choice | score",
      "instructions": "<question>",
      "criteria": { "<label>": "<description>", "...": "..." }
    }
  }
}
```

### Response schema (confirmed via live calls, 2026-09-25)

Both providers return the same shape; `criteria` is a required field for
every question `type` (`choice`, `score`, `noul` alike — omitting it is a
422, not just needed for `choice`):

```json
{
  "model": "<provider/model id>",
  "answers": {
    "<question name>": {
      "type": "choice",
      "choice": "<top label>",
      "probabilities": { "<label>": 0.0, "...": "..." },
      "confidence": 0.0
    }
  },
  "usage": { "input_tokens": 0, "output_tokens": 0, "cost": 0.0 }
}
```

Two live-call findings that change how `confidence` and `output_tokens`
should be used downstream:

- **`confidence` is not equivalent across providers.** For `jev_semif` it
  equals the top-category probability exactly (e.g. `0.525` chosen,
  `confidence: 0.525`). For `jev_official` it does **not**
  (`down_first: 0.74` chosen, but `confidence: 0.6`) — the two providers
  compute this field differently. **Use `probabilities` for the A0/A
  evaluation, not `confidence`.**
- **`output_tokens` differs by provider as expected**: `jev_semif` returns
  `0` (pure choice-logit readout, no generation); `jev_official` returns a
  nonzero count (`46` in the test call) — it is not a pure logit readout on
  TypeSafe's own hosted model, which affects latency/cost modeling per
  provider (`jev_official`'s per-call cost in the test, $0.0000184 for 437
  input tokens, lines up with the documented `$0.042/M` input-token price).

## Approach: reuse the existing `signal` model plugin, unmodified

fwbg already has exactly the right shape of plugin for an externally computed
probability: `models/signal` reads a precomputed `_composed_signal_long` /
`_composed_signal_short` column (values in `[0, 1]`) and exposes it as a
class-probability array — no training, no fit. That is precisely Jev's
interface (no fit, a calibrated probability per point in time). No new model
plugin is needed.

```
Batch script (offline, one-time per config)
    → calls jev_official + jev_semif with identical state+questions per bar
    → caches both providers' outputs to CSV, keyed by timestamp
    ↓
New DataSource entry (reads the cached CSV — same pattern as macro_data/cot_positioning)
    ↓
New DataLoader plugin `jev_signal` (fwbg-core or custom)
    → maps raw probability → _composed_signal_long / _composed_signal_short
    → mandatory 1-bar shift via shift_features() (lookahead prevention)
    ↓
Existing `models/signal` plugin — unchanged
    ↓
Existing exit_strategies/fixed (tp/sl/timeout_bars) + validation phase — unchanged
```

Jev is called **once per historical bar per provider**, cached, and reused
across every fold / `ct` threshold in the grid search. `tp`/`sl` values
*are not* free to sweep this way (see below) — only `ct` is.

## Label definition: triple-barrier, matches fwbg's native exit strategy

Originally scoped as a plain endpoint comparison ("price at t+30min vs t"),
revised to match fwbg's existing `exit_strategies/fixed`, which already
implements a de Prado-style triple barrier (upper = TP, lower = SL, vertical
= `timeout_bars`). No framework change needed — just configuration:

- `timeout_bars` is the prediction horizon, **configurable**, not hardcoded
  to 30 minutes (that was only the horizon used in the original example).
- `tp` / `sl` in fwbg are spread multiples, not raw pips — the batch script
  converts to actual price/pip distance per bar before building the prompt.

Because "P(TP touched before SL)" depends on the concrete TP/SL distance, a
single cached Jev call per bar only answers *one* tp/sl combination. Scope
for the first run: **one fixed tp/sl pair**, not the full grid — sweeping
tp/sl later means additional batch runs (one per combination). `ct` remains
freely sweepable against the same cached column.

### Prompt / question shape

**Updated during implementation** (this section originally described a single
`choice` question; the implementation task found two independent `noul`
questions to be the better match — see below for why): one bar produces two
independent yes/no questions, mirroring `FixedExitStrategy.compute_targets()`'s
own two independent `targets_long`/`targets_short` arrays (each direction is
its own hypothetical trade from the same bar, not a single mutually-exclusive
outcome — see `docs/plans/2026-09-24-jev-signal-provider-plan.md` Task 3 for
the source-verified detail):

```json
{
  "state": "<OHLC window + indicator values as text>",
  "questions": {
    "is_long_win": {
      "type": "noul",
      "instructions": "Opening a long position now with take-profit X pips and stop-loss Y pips: is take-profit reached before stop-loss or before N bars pass?",
      "criteria": { "true": "Take-profit reached first", "false": "Stop-loss reached first, or neither within the horizon" }
    },
    "is_short_win": {
      "type": "noul",
      "instructions": "Opening a short position now with take-profit X pips and stop-loss Y pips: is take-profit reached before stop-loss or before N bars pass?",
      "criteria": { "true": "Take-profit reached first", "false": "Stop-loss reached first, or neither within the horizon" }
    }
  }
}
```

`_composed_signal_long = P(is_long_win)`, `_composed_signal_short = P(is_short_win)`.

X/Y here are **pips**, not fwbg's internal spread-multiplier convention —
the batch script converts `tp`/`sl` (spread multiples, same unit
`compute_labels()` uses) to actual pips via `get_asset(symbol).spread` /
`.point` before building this prompt, so the barrier Jev is asked about
matches the barrier the realized labels are computed against. (This
conversion was originally missing — found and fixed during the final
whole-branch review, see plan.md's commit history.)

## Scope (phase 1 / feasibility)

- Asset: EUR/USD only.
- Prompt content: OHLC + existing fwbg-core indicators only (trend, momentum,
  volatility, price_action, ...). No news/sentiment yet — fwbg has no text
  data source today (only macro_data/cot_positioning in premium). News is a
  deliberate phase-2 addition (new DataSource), gated on phase 1 showing
  signal.
- TP/SL: one fixed pair, not the full grid.

## Evaluation: three stages, cheapest first

- **A0 — provider agreement.** Per bar, correlation between
  `jev_official` and `jev_semif` output distributions + top-category
  agreement rate. Answers "do the two providers even agree with each
  other" independent of whether either is right.
- **A — calibration/accuracy.** Computed directly from the cached batch
  output + realized triple-barrier outcomes: Brier score, reliability
  diagram, accuracy vs. baseline (majority class / coin flip). No optimizer
  run needed — cheap, fast, the actual go/no-go gate.
- **B — full fwbg pipeline.** Only if A looks promising: `models/signal`
  through `fwbg --assets EURUSD` with the `ct` grid, exercising the existing
  walk-forward CV + DSR + PBO/CSCV + Monte-Carlo permutation tests. This is
  a trading-strategy-robustness check (relevant because `ct`/`tp`/`sl` grid
  search happens regardless of where the signal comes from — the overfitting
  protection is not wasted effort here even though Jev itself has no
  hyperparameters to overfit).

## What stays unchanged

- `models/signal` plugin — used as-is.
- `exit_strategies/fixed`, walk-forward CV, DSR/PBO/Monte-Carlo validation —
  used as-is.
- All existing indicator plugins — reused for prompt construction, no changes.

## Open unknowns / to verify before implementation

- ~~Exact response schema of both APIs~~ — **confirmed 2026-09-25**, see
  "Response schema" above.
- ~~Which model size itemis's `jev_semif` instance actually runs~~ —
  **confirmed 2026-09-25**: `semif-qwen3.5-4b` (the Qwen3.5 4B option).
- ~~Batch script needs network access to `llms.itemis.cloud`~~ —
  **confirmed 2026-09-25**: reachable from the current dev environment while
  on the itemis VPN.
- Latency and rate limits under batch load — single test calls only so far
  (one bar each, sequential). Not yet measured: per-provider throughput,
  concurrency limits, retry/backoff behavior needed for a full historical
  backtest run (thousands of bars).
- `confidence` field semantics differ per provider (see "Response schema"
  above) — batch script/evaluation must key off `probabilities`, not
  `confidence`, to stay comparable across providers.
