# Plugin Spec — atr_based

**Kind**: exit_strategy  •  **Version**: 0.1.0

## Capability

Simulates long/short trades with ATR-scaled TP/SL distances (with pip floors) and optional volatility-adaptive per-trade timeout to produce per-bar win/loss targets and durations.

## Summary

Exit strategy that sizes take-profit and stop-loss distances as ATR multiples (with pip-based floors) and simulates long/short trades to produce per-bar win targets, optionally adapting the per-trade timeout to the current volatility relative to its long-run mean. Delegates to a configured entry-modifier or exit-modifier when one is set on the SimulationContext.

## Inputs

- df: pandas DataFrame with columns 'O', 'H', 'L', 'C' and optionally a precomputed ATR column named '_atr' or 'vol_atr'
- ctx: SimulationContext supplying spread, max_trade_bars, optional exit_modifier / exit_modifier_params, optional entry_modifier / entry_modifier_params, and (for resolve_distances) exit_params
- params: optional GridParams whose tp_value, sl_value, timeout_bars, and extra dict override the corresponding keyword arguments

## Parameters

- `tp_mult` (float, default=2): ATR multiplier for take-profit distance (tp_distance = ATR * tp_mult).
- `sl_mult` (float, default=1.5): ATR multiplier for stop-loss distance (sl_distance = ATR * sl_mult).
- `atr_period` (int, default=14): ATR lookback period in bars; used only as fallback when no precomputed ATR column is present.
- `min_tp_pips` (int, default=10): Minimum TP distance in spread-multiples (pips) to prevent too-tight targets in low-volatility regimes.
- `min_sl_pips` (int, default=15): Minimum SL distance in spread-multiples (pips) to prevent too-tight stops in low-volatility regimes.
- `timeout_bars` (int, default=None): Fixed per-trade timeout: close the simulated trade after N bars if neither TP nor SL is hit. Ignored when adaptive_timeout=True; treated as 0 when None.
- `adaptive_timeout` (bool, default=False): When True, timeout is computed per trade from the ratio of current ATR to its moving average instead of using timeout_bars.
- `base_timeout` (int, default=48): Base timeout in bars at average volatility; scaled by the (clamped) vol ratio in adaptive mode.
- `min_timeout` (int, default=12): Floor for the adaptive per-trade timeout (in bars), applied after scaling base_timeout.
- `max_timeout` (int, default=96): Ceiling for the adaptive per-trade timeout (in bars), applied after scaling base_timeout.
- `atr_ma_period` (int, default=200): Window length of the rolling mean over the ATR series used to compute the vol ratio in adaptive-timeout mode.
- `return_durations` (bool, default=False): When True, compute_targets returns a 4-tuple including per-bar durations for long and short trades; otherwise only the two target arrays are returned.

## Outputs

- targets_long: np.ndarray[float64] of length len(df), 1.0 where the simulated long trade opened at bar i hit its TP, else 0.0
- targets_short: np.ndarray[float64] of length len(df), 1.0 where the simulated short trade opened at bar i was a win, else 0.0
- durations_long: np.ndarray[int64] of length len(df), bars-to-exit for the long trade at bar i (max_bars when no exit), returned only when return_durations=True
- durations_short: np.ndarray[int64] of length len(df), bars-to-exit for the short trade at bar i (max_bars when no exit), returned only when return_durations=True

## Acceptance Criteria

- AC-001: Returns a tuple (targets_long, targets_short) of float64 arrays by default; returns a 4-tuple (targets_long, targets_short, durations_long, durations_short) when return_durations=True.
- AC-002: Per-bar take-profit distance is computed as max(atr[i] * tp_mult, ctx.spread * min_tp_pips); per-bar stop-loss distance as max(atr[i] * sl_mult, ctx.spread * min_sl_pips).
- AC-003: ATR series is sourced from the df '_atr' column if present, else the 'vol_atr' column, else computed as ta.volatility.average_true_range over H/L/C with window=atr_period; NaNs are replaced with 0.0.
- AC-004: Slippage is set to ctx.spread * 0.5; the simulation upper bound max_bars uses ctx.max_trade_bars, falling back to len(df) when it is falsy.
- AC-005: When adaptive_timeout=True, the per-trade timeout is int(base_timeout * (atr_ma[i]/atr[i])) with the vol ratio clamped to [0.25, 4.0] and the resulting timeout clamped to [min_timeout, max_timeout]; when atr[i] or atr_ma[i] is not > 0 the timeout falls back to base_timeout.
- AC-006: When adaptive_timeout=False, timeout_bars (or 0 if None/falsy) is passed as a fixed timeout to the trade simulation.
- AC-007: The moving-average ATR used for the adaptive vol ratio is a pandas rolling mean over the ATR series with window=atr_ma_period and min_periods=1.
- AC-008: If a GridParams object is supplied, tp_mult is overridden by params.tp_value, sl_mult by params.sl_value, timeout_bars by params.timeout_bars (when not None), and params.extra may override atr_period, min_tp_pips, min_sl_pips, adaptive_timeout, base_timeout, min_timeout, max_timeout, and atr_ma_period.
- AC-009: When ctx.entry_modifier is set to a non-empty string, compute_targets pre-computes tp_dist_arr, sl_dist_arr, trail_dist_arr, and trail_tp_dist_arr and delegates to the resolved entry modifier's compute_targets, passing ctx.entry_modifier_params and exit_modifier_params trailing/breakeven fields; this dispatch takes precedence over exit-modifier dispatch.
- AC-010: When ctx.exit_modifier is set (and no entry modifier is active), compute_targets delegates to the resolved exit modifier's compute_targets with the OHLC arrays, ATR array, multipliers, spread, slippage, min distances, max_bars, timeout_val, return_durations, and ctx.exit_modifier_params.
- AC-011: Trade duration for a leg is (exit_bar - i) when the simulator returns a non-negative exit index, otherwise max_bars; a target is set to 1.0 only when the simulator returns result == 1.0 (win).
- AC-012: resolve_distances(df, tp, sl, ctx) returns per-bar (tp_dists, sl_dists) computed as max(atr * tp, ctx.spread * min_tp_pips) and max(atr * sl, ctx.spread * min_sl_pips), using ctx.exit_params for atr_period/min_tp_pips/min_sl_pips with defaults 14/10/15.
- AC-013: get_cache_key returns a string of the form 'atr_tp{tp_mult:.2f}_sl{sl_mult:.2f}_to{timeout}' where {timeout} is str(timeout_bars) if truthy else 'none'.
- AC-014: If a Numba-cached kernel raises ModuleNotFoundError during load, the plugin deletes stale *.nbi/*.nbc files under its __pycache__ and retries the call once.

## Edge Cases

- Leading NaNs in the ATR series (before atr_period bars have accumulated) are replaced with 0.0, which collapses tp/sl distances to the min_tp_pips/min_sl_pips floors via ctx.spread.
- Neither '_atr' nor 'vol_atr' present in df: ATR is recomputed on the fly with ta.volatility.average_true_range using atr_period.
- adaptive_timeout=True but atr[i] == 0 or atr_ma[i] == 0 (e.g. warm-up region): the adaptive branch falls back to using base_timeout for that bar.
- adaptive_timeout=True with extreme volatility: the vol ratio atr_ma/atr is clamped to [0.25, 4.0] before scaling base_timeout, and the resulting timeout is further clamped to [min_timeout, max_timeout].
- timeout_bars is None (default) and adaptive_timeout is False: timeout_val is passed as 0 to the numba kernel (interpreted downstream as no fixed timeout).
- ctx.max_trade_bars is None or 0: max_bars falls back to len(df), allowing simulation to run to the end of the series.
- Both ctx.entry_modifier and ctx.exit_modifier are set: the entry-modifier dispatch takes precedence because its kernel handles scale-in and trailing stop internally.
- Stale Numba pickle cache after a package rename raises ModuleNotFoundError: _clear_numba_cache() deletes the plugin's *.nbi/*.nbc files and the call is retried once, forcing recompilation.
- Last bar of the series: the inner loop iterates only to n-1, so targets_long[-1], targets_short[-1] and their durations remain at their zero-initialized values.
- GridParams.timeout_bars is None: the caller's timeout_bars argument is preserved rather than overridden.

## Assumptions

- _none_
