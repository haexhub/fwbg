# Plugin Spec — orb_based

**Kind**: exit_strategy  •  **Version**: 0.1.0

## Capability

Simulates ORB breakout trade exits with SL from orb_sl_dist (ATR fallback) and TP from ATR or ORB range, with session-aware, trailing, breakeven, and entry-modifier dispatch.

## Summary

Exit strategy for ORB (Opening Range Breakout) setups. Stop-loss distance is read from an `orb_sl_dist` (or `*_sl_dist`) column — the full OR range — with `sl_mult=1.0` placing SL at the opposite ORB boundary and `sl_mult>1.0` adding a buffer beyond; falls back to `ATR * sl_mult` if no column is available. Take-profit distance defaults to `ATR * tp_mult` (`tp_mode="atr"`) or `range * tp_mult` when `tp_mode="range"`. Trades are simulated bar-by-bar for both long and short via numba kernels, with three dispatch modes: trailing/breakeven (`_simulate_trade_trailing_numba`), session-aware (`_simulate_trade_session_numba`), or plain (`_simulate_trade_numba`). When `ctx.entry_modifier` is set, the strategy pre-computes TP/SL/trail distance arrays and delegates entirely to the entry modifier's `compute_targets`. Enforces `min_tp_pips`/`min_sl_pips` floors (in spread multiples). Also exposes `resolve_distances` returning (tp_dists, sl_dists) arrays and a deterministic `get_cache_key` derived from tp/sl multipliers and timeout.

## Inputs

- df['O']
- df['H']
- df['L']
- df['C']
- df['_atr'] or df['vol_atr'] (optional; else computed via ta.volatility.average_true_range)
- df['orb_sl_dist'] or df['*_sl_dist'] (optional SL distance column; falls back to ATR*sl_mult)
- df['orb_range'] or df['*_orb_range'] or df['*_range'] excluding '*_vs_*' (optional; used when tp_mode='range')
- ctx.spread
- ctx.max_trade_bars
- ctx.exit_params (sl_dist_column, tp_mode, range_column, atr_period, min_tp_pips, min_sl_pips, breakeven_trigger, breakeven_offset, trail_pips)
- ctx.session_start_hour / ctx.session_end_hour / ctx.exit_session_start_hour / ctx.exit_session_end_hour (optional)
- ctx.entry_modifier + ctx.entry_modifier_params (optional dispatch)
- ctx.exit_modifier + ctx.exit_modifier_params (legacy trailing/breakeven override)
- params: GridParams (tp_value, sl_value, timeout_bars, extra dict) — optional override of tp_mult/sl_mult/timeout/atr_period/min_tp_pips/min_sl_pips

## Parameters

- `tp_mode` (choice, default='atr'): TP source: 'atr' = ATR * tp_mult, 'range' = range column * tp_mult (for ORB midpoint-entry setups).
- `tp_mult` (float, default=2): TP multiplier applied to ATR (tp_mode='atr') or the range column (tp_mode='range').
- `sl_mult` (float, default=1): Buffer multiplier on orb_sl_dist (1.0 = SL at opposite ORB boundary, >1.0 = beyond). Also used as ATR fallback multiplier when no SL column is available. Not applied when sl_dist_column is set explicitly via exit_params.
- `atr_period` (int, default=14): ATR window used for TP (tp_mode='atr') and SL fallback when no ATR column is present in df.
- `min_tp_pips` (int, default=8): Minimum TP distance expressed in spread multiples (floor applied via np.maximum / max).
- `min_sl_pips` (int, default=5): Minimum SL distance expressed in spread multiples (floor applied via np.maximum / max).
- `timeout_bars` (int, default=None): Close trade after N bars if neither TP nor SL is hit; None/0 = no timeout applied by the numba kernel.
- `sl_level` (choice, default='none'): SL anchored at a structural ORB level ('none', 'or_midpoint', 'or_high', 'or_low'). Declared in schema; not consumed by compute_targets in this file.
- `entry_delay` (int, default=1): Bars between signal and entry (0 = signal-bar close for breakout stop-orders, 1 = next-bar open, no lookahead). Declared in schema; not consumed inside compute_targets.
- `max_trades_per_signal` (int, default=1): Max trades per contiguous confidence>=ct signal event; 0 = unlimited. Declared in schema; not consumed inside compute_targets.
- `breakeven_trigger` (float, default=0): Fraction of TP at which SL moves to breakeven (0.0 = disabled). Enables trailing/breakeven kernel when >0.
- `breakeven_offset` (float, default=0): Fraction of TP distance added above entry when breakeven triggers (0.0 = true breakeven). Only active when breakeven_trigger > 0.
- `trail_pips` (int, default=0): Trailing stop distance in spread multiples. When >0 forces the trailing kernel; otherwise trailing falls back to range (tp_mode='range') or ATR*trail_atr_mult from exit_modifier_params.

## Outputs

- compute_targets → (targets_long: np.ndarray[float64], targets_short: np.ndarray[float64]) where 1.0 marks a winning trade at bar i
- compute_targets(return_durations=True) → (targets_long, targets_short, durations_long: np.ndarray[int64], durations_short: np.ndarray[int64])
- resolve_distances → (tp_dists: np.ndarray, sl_dists: np.ndarray) per-bar distances after floors
- get_cache_key → deterministic string 'orb_tp{tp:.2f}_sl{sl:.2f}_to{timeout|none}'

## Acceptance Criteria

- AC-001: compute_targets returns two float64 arrays of length len(df); positions where the long/short trade hit TP before SL/timeout equal 1.0, all others 0.0.
- AC-002: When return_durations=True, compute_targets additionally returns int64 duration arrays where a filled bar equals (exit_bar - i), and a never-exited trade equals max_bars.
- AC-003: The final bar (i = n-1) never opens a trade (loop iterates 0..n-2), so targets_long[-1] == targets_short[-1] == 0.0.
- AC-004: When df contains an 'orb_sl_dist' or '*_sl_dist' column and no explicit sl_dist_column is set, SL distance = raw_column * sl_mult with NaN rows replaced by ATR*sl_mult, then floored to min_sl_pips*spread.
- AC-005: When exit_params['sl_dist_column'] names a present column, its value is used as an ABSOLUTE SL distance (no sl_mult applied); NaN rows fall back to ATR*sl_mult.
- AC-006: When no *_sl_dist column exists in df, SL distance = ATR * sl_mult, floored to min_sl_pips*spread.
- AC-007: When tp_mode='atr' (default), TP distance = max(atr[i]*tp_mult, min_tp_pips*spread).
- AC-008: When tp_mode='range', TP distance = max(range_column[i]*tp_mult, min_tp_pips*spread) using the resolved range column (exit_params['range_column'] > 'orb_range' > '*_orb_range' > '*_range' excluding '*_vs_*').
- AC-009: ATR array is read from df['_atr'] if present, else df['vol_atr'], else computed via ta.volatility.average_true_range; NaNs replaced with 0.0.
- AC-010: When ctx defines integer (exit_)session_start_hour and (exit_)session_end_hour, session mask is built via compute_session_mask and the session-aware numba kernel is used (unless trailing is also enabled, which takes precedence).
- AC-011: When breakeven_trigger>0, trail_pips>0, or an exit_modifier supplies trail_atr_mult>0, the trailing-numba kernel is used with trail distance = trail_pips*spread, else range[i] (tp_mode='range'), else atr[i]*trail_atr_mult.
- AC-012: When ctx.entry_modifier is a non-empty string, compute_targets delegates to that modifier's compute_targets, passing precomputed tp/sl/trail/trail_tp distance arrays, spread, slippage=0.5*spread, max_bars, timeout, breakeven_trigger, and entry_modifier_params.
- AC-013: params (GridParams) overrides tp_mult (tp_value), sl_mult (sl_value), and — when set — timeout_bars; params.extra may override atr_period/min_tp_pips/min_sl_pips.
- AC-014: slippage passed to numba kernels equals 0.5 * ctx.spread.
- AC-015: max_bars defaults to ctx.max_trade_bars, falling back to len(df) when the ctx value is falsy.
- AC-016: resolve_distances returns per-bar (tp_dists, sl_dists) arrays applying the same floors, tp_mode branching, and SL column resolution as compute_targets, without simulating trades.
- AC-017: get_cache_key(params) returns the exact string 'orb_tp{tp_mult:.2f}_sl{sl_mult:.2f}_to{timeout_bars or "none"}'.
- AC-018: Plugin is registered under the name 'orb_based' via @register_exit_strategy.

## Edge Cases

- df with fewer than 2 rows: loop range(n-1) yields no iterations, so both targets arrays are all zeros (and duration arrays all zeros if requested).
- No orb_sl_dist / *_sl_dist column present: SL falls back to pure ATR*sl_mult (still floored).
- sl_dist_column set to a name that is NOT in df.columns: treated as auto-detected — the code searches for orb_sl_dist / *_sl_dist instead, applying sl_mult as a buffer multiplier.
- ATR column contains NaNs: replaced with 0.0 by np.nan_to_num, so TP/SL floors (min_*_pips*spread) become the effective distances on those bars.
- SL column value is NaN on a bar: replaced by ATR*sl_mult fallback in both explicit and auto-detected paths.
- tp_mode='range' but no matching range column is present: _get_range returns zeros, so TP distance collapses to the min_tp_pips*spread floor.
- Auto-detected range column excludes any column containing '_vs_' (e.g. '_vs_atr', '_vs_range').
- ctx.exit_params is None or missing keys: defaults are used (tp_mode='atr', atr_period=14, min_tp_pips=8, min_sl_pips=5, sl_dist_column=None).
- ctx.max_trade_bars is None/0/falsy: falls back to len(df).
- timeout_bars is None: passed to numba kernels as 0 (i.e. no timeout).
- Only one of session_start_hour / session_end_hour is an int (or either is None): session mask is not built, session kernel is not used.
- Trailing config takes precedence over session config: when both are configured, the trailing kernel is dispatched.
- ctx.entry_modifier is set to a truthy non-string (bool, int, etc.): the dispatch branch is skipped because it requires isinstance(str).
- Legacy exit_modifier is set: breakeven_trigger defaults to 0.5 and trail_atr_mult to 0.5 unless overridden in exit_modifier_params, overriding direct exit_params values for be_trigger/trail.
- No ATR column and 'ta' package unavailable: ImportError propagates from _get_atr.

## Assumptions

- df provides OHLC columns named 'O','H','L','C' (uppercase).
- ctx.spread is a positive float; slippage is derived as 0.5 * spread.
- compute_session_mask, _simulate_trade_numba, _simulate_trade_trailing_numba, _simulate_trade_session_numba, and get_entry_modifier are importable from fwbg.simulation / fwbg.core.
- GridParams exposes tp_value, sl_value, timeout_bars, and extra (dict) attributes when params is provided.
- sl_level, entry_delay, and max_trades_per_signal are declared in the param schema for downstream tooling but are NOT consumed inside compute_targets in this file (interpreted higher up in the pipeline).

## Needs Clarification

- [NEEDS CLARIFICATION: sl_level, entry_delay, and max_trades_per_signal appear in the schema/defaults but are not referenced in compute_targets/resolve_distances — confirm whether they are consumed elsewhere in the pipeline (e.g. by the signal-to-trade layer) or are dead schema entries.]
- [NEEDS CLARIFICATION: The trail_atr_mult / trail_tp_atr_mult / breakeven_trigger inside the entry-modifier branch read from ctx.exit_modifier_params even though no exit_modifier is required in that branch — confirm whether this precomputation should also honor exit_params['breakeven_trigger'] / trail_pips as the non-modifier branch does.]
