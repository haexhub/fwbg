# Jev Signal Provider

## Status: Stage A failed — not pursued further (2026-09-25)

Feasibility check (Evaluation Stage A/A0 below) ran against real EUR/USD H1
data (`scripts/jev_batch_query.py`, 720 live calls across 3 independent
configurations, 0 API errors). Result: both providers perform **at or below
random-chance level**, not just below the majority-class baseline.

| Config | Barriers | Ground truth split | `jev_official` acc. | `jev_semif` acc. |
|---|---|---|---|---|
| 1 | Fixed 54p/36p (spread-mult defaults), raw price deltas in prompt | 88/8/5 (neither-heavy) | 30–40% | 7.5–10% |
| 2 | Same barriers, explicit target/stop prices + pip count in prompt | 88/8/5 | 27.5–32.5% | 10–27.5% |
| 3 | Symmetric 1.5×ATR(14), 8h horizon (tuned for class balance) | 40/35/25 (balanced) | 27.5% | 27.5–30% |

Config 3 rules out the two most likely confounds (unrealistic barrier width,
under-specified prompt) — `jev_official` still answers `down_first` in
90–98% of cases regardless of the (now symmetric) input, i.e. it behaves
close to a constant classifier rather than responding to state. Not pursued
further: three independently-varied configurations landing at/below chance
is treated as a robust negative, not a setup artifact. Cost was never the
constraint (whole check <$0.05); calibration quality was.

Kept for traceability — see `## Open unknowns` (now resolved) and the raw
per-config caches under `data/jev_cache/eurusd_h1_results_v*.csv` for the
full per-bar breakdown if this is revisited later (e.g. with a different
question formulation, a different indicator subset, or a different asset).

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

One `choice` question per bar covers both directions in a single call:

```json
{
  "state": "<OHLC window + indicator values as text>",
  "questions": {
    "barrier_outcome": {
      "type": "choice",
      "instructions": "Which barrier is reached first in the next N bars?",
      "criteria": {
        "up_first": "Price rises by at least X pips before falling Y pips",
        "down_first": "Price falls by at least Y pips before rising X pips",
        "neither": "Neither threshold is reached within the horizon"
      }
    }
  }
}
```

`_composed_signal_long = P(up_first)`, `_composed_signal_short = P(down_first)`.

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
