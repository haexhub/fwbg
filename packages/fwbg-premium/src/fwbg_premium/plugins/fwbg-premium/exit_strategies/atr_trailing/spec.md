# Plugin Spec — atr_trailing

**Kind**: exit_strategy  •  **Version**: 0.1.0

## Capability

Simulates ATR-scaled TP/SL trade exits with a breakeven stop and ATR-scaled trailing stop, producing per-bar long/short win labels (optionally with trade durations).

## Summary

Extends the ATR-based exit strategy with two additional risk-management mechanics inside a single Numba-compiled per-bar simulation: a breakeven stop that pulls SL to entry once price has moved `breakeven_trigger × tp_distance` in favour, and a trailing stop that follows the best price seen since (breakeven) activation at a distance of `trail_atr_mult × ATR`. TP and SL distances are computed as `ATR × tp_mult` / `ATR × sl_mult` with floors of `spread × min_tp_pips` / `spread × min_sl_pips`; entry is on the next bar's open adjusted by spread + half-spread slippage; an optional `timeout_bars` closes the trade at the close of bar `entry_idx + timeout_bars - 1` with the sign of PnL deciding win/loss. Returns two float64 arrays `targets_long`/`targets_short` (1.0 = win, 0.0 = loss/timeout-loss/no-exit), or four arrays including `durations_long`/`durations_short` when `return_durations=True`. ATR is taken from the `_atr` or `vol_atr` column when present, otherwise computed on the fly with `ta.volatility.average_true_range` using `atr_period`. Parameters may be supplied directly or overridden via `GridParams` (`tp_value`, `sl_value`, `timeout_bars`, and `extra` for `atr_period` / `min_tp_pips` / `min_sl_pips` / `breakeven_trigger` / `trail_atr_mult`).

## Inputs

- df with OHLC columns O, H, L, C (float-coercible)
- optional df column `_atr` or `vol_atr` (pre-computed ATR); otherwise computed from H/L/C via `ta.volatility.average_true_range`
- SimulationContext providing `spread`, `max_trade_bars`, and `exit_params` (used by resolve_distances)
- optional GridParams with `tp_value`, `sl_value`, `timeout_bars`, and `extra` dict overriding tp_mult/sl_mult/timeout_bars/atr_period/min_tp_pips/min_sl_pips/breakeven_trigger/trail_atr_mult

## Parameters

- `tp_mult` (float, default=3): ATR multiplier for the take-profit distance.
- `sl_mult` (float, default=1.5): ATR multiplier for the stop-loss distance.
- `atr_period` (int, default=14): ATR window used only when neither `_atr` nor `vol_atr` is present on df.
- `min_tp_pips` (int, default=8): Minimum TP distance expressed as a multiple of ctx.spread.
- `min_sl_pips` (int, default=12): Minimum SL distance expressed as a multiple of ctx.spread.
- `timeout_bars` (int, default=None): Close the trade at close after this many bars if neither TP nor SL/trail was hit; disabled when None or 0.
- `breakeven_trigger` (float, default=0.5): Fraction of TP distance the price must travel in favour before SL is pulled to entry (0.0 disables breakeven and starts trailing immediately).
- `trail_atr_mult` (float, default=0.5): ATR multiplier for the trailing-stop distance behind the best price; 0.0 disables trailing (breakeven-only).
- `return_durations` (bool, default=False): When true, additionally return per-bar long/short trade durations.

## Outputs

- targets_long: np.ndarray[float64] of length len(df); 1.0 for winning simulated long trades, 0.0 otherwise
- targets_short: np.ndarray[float64] of length len(df); 1.0 for winning simulated short trades, 0.0 otherwise
- durations_long: np.ndarray[int64] of length len(df) (only when return_durations=True); bars from signal to exit, or max_bars if no exit
- durations_short: np.ndarray[int64] of length len(df) (only when return_durations=True); bars from signal to exit, or max_bars if no exit

## Acceptance Criteria

- AC-001: Returns a tuple (targets_long, targets_short) of float64 arrays of length len(df) when return_durations=False, and (targets_long, targets_short, durations_long, durations_short) when return_durations=True.
- AC-002: Target values are 1.0 for a winning simulated trade and 0.0 otherwise (loss, timeout-loss, or no exit within max_bars).
- AC-003: TP distance is max(ATR × tp_mult, spread × min_tp_pips); SL distance is max(ATR × sl_mult, spread × min_sl_pips).
- AC-004: Entry price for a long is opens[i+1] + spread + slippage; for a short it is opens[i+1] - spread - slippage, with slippage = ctx.spread × 0.5.
- AC-005: When breakeven_trigger > 0 and the best-seen price reaches entry ± breakeven_trigger × tp_distance, the SL is moved to entry (never worse than the current SL) and trailing becomes active; when breakeven_trigger <= 0 trailing is active from entry.
- AC-006: When trail_atr_mult > 0 and trailing is active, the SL is ratcheted to best_price − trail_distance (long) or best_price + trail_distance (short), where trail_distance = ATR × trail_atr_mult and SL only moves in the trade's favour.
- AC-007: An SL hit counts as a win when the SL has been ratcheted past the entry price (long: sl > entry, short: sl < entry), otherwise as a loss.
- AC-008: TP hit always counts as a win; on the same bar the SL check runs before the TP check so a simultaneous SL+TP touch resolves as SL.
- AC-009: When timeout_bars > 0 and bar index j reaches entry_idx + timeout_bars − 1 (capped at n−1), the trade exits at closes[j] and is a win iff PnL > 0.
- AC-010: Uses the `_atr` column if present, else `vol_atr`, else computes ATR via `ta.volatility.average_true_range` with `atr_period`; NaN ATR values are replaced with 0.0.
- AC-011: GridParams overrides direct kwargs: params.tp_value → tp_mult, params.sl_value → sl_mult, params.timeout_bars → timeout_bars (if not None), and params.extra keys atr_period/min_tp_pips/min_sl_pips/breakeven_trigger/trail_atr_mult override their kwargs when present.
- AC-012: resolve_distances returns per-bar tp/sl distance arrays using the same ATR-multiplier + minimum-pip formula as compute_targets.
- AC-013: get_cache_key returns a string of the form `atr_trail_tp{tp:.2f}_sl{sl:.2f}_be{be:.2f}_tr{trail:.2f}_to{timeout|none}`.

## Edge Cases

- Last bar (i == n−1) or i+1 >= n: no simulation is performed and both targets stay 0.0.
- NaN ATR values are coerced to 0.0, so TP and SL distances fall back to their respective spread × min_*_pips floors.
- breakeven_trigger = 0.0 and trail_atr_mult > 0.0: trailing is active from entry with no breakeven pull.
- breakeven_trigger > 0.0 and trail_atr_mult = 0.0: only the breakeven-to-entry move happens; SL is not further ratcheted.
- breakeven_trigger = 0.0 and trail_atr_mult = 0.0: behaves like fixed ATR TP/SL with no trailing/breakeven adjustments.
- timeout_bars is None or <= 0: no timeout exit; trade only closes on TP, SL/trail/BE, or end of max_bars window.
- Trade reaches end_idx (entry_idx + max_bars, capped at n) without any exit: returns 0.0 as target and duration = max_bars in the durations variant.
- ctx.max_trade_bars falsy: falls back to max_bars = len(df).
- Neither `_atr` nor `vol_atr` column present: ATR is computed via `ta.volatility.average_true_range` with the configured atr_period.

## Assumptions

- _none_
