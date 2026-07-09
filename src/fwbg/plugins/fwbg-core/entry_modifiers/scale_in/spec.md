# Plugin Spec — scale_in

**Kind**: entry_modifier  •  **Version**: 0.1.0

## Capability

Adds additional positions at configurable retracement levels (fraction of Entry→SL distance) with a per-scale-in quantity multiplier, adjusting TP via the underlying scale-in trade simulation.

## Summary

Entry modifier that simulates scale-in trades: at each bar it runs long and short trade simulations that add positions when price retraces toward SL at configured level fractions, producing binary target arrays (and optional durations) via a Numba-JIT scale-in simulator.

## Inputs

- opens: np.ndarray
- closes: np.ndarray
- highs: np.ndarray
- lows: np.ndarray
- tp_dist_arr: np.ndarray
- sl_dist_arr: np.ndarray
- trail_dist_arr: np.ndarray
- spread: float
- slippage: float
- max_bars: int
- timeout_val: int
- return_durations: bool (default False)
- breakeven_trigger: float (default 0.0, pass-through to exit logic)
- trail_tp_dist_arr: np.ndarray (default zeros_like(tp_dist_arr), pass-through)

## Parameters

- `levels` (list[float], default=[0.2, 0.4, 0.6]): Retracement-Levels als Bruchteil der Entry→SL-Distanz. 0.2 = Nachkauf bei 20% Retracement Richtung SL.
- `qty_multiplier` (float, default=1): Positionsgröße pro Nachkauf relativ zur Initial-Position. 1.0 = gleiche Größe, 0.5 = halbe Größe.

## Outputs

- targets_long: np.ndarray[float64] of shape (n,) — 1.0 where a long scale-in trade would win, else 0.0
- targets_short: np.ndarray[float64] of shape (n,) — 1.0 where a short scale-in trade would win, else 0.0
- durations_long: np.ndarray[int64] of shape (n,) — bars-to-exit for long (max_bars if no exit); only when return_durations=True
- durations_short: np.ndarray[int64] of shape (n,) — bars-to-exit for short (max_bars if no exit); only when return_durations=True

## Acceptance Criteria

- AC-001: compute_targets returns a tuple (targets_long, targets_short) of float64 np.ndarrays with the same length as closes when return_durations=False.
- AC-002: compute_targets returns a 4-tuple (targets_long, targets_short, durations_long, durations_short) when return_durations=True; duration arrays are int64 and equal max_bars when no exit is found (exit_bar < 0).
- AC-003: targets_long[i] / targets_short[i] is set to 1.0 iff the underlying _simulate_trade_scale_in_numba call for that bar returns result == 1.0; otherwise 0.0.
- AC-004: The last bar (index n-1) is never simulated — its target and duration entries remain at their zero-initialized defaults.
- AC-005: When the caller passes levels=None, the modifier substitutes the default list [0.2, 0.4, 0.6] before packing.
- AC-006: When the caller passes trail_tp_dist_arr=None, the modifier substitutes np.zeros_like(tp_dist_arr).
- AC-007: Levels are packed into a fixed-size np.float64 array of length _MAX_LEVELS=10, filled with -1.0 sentinels; only the first min(len(levels), 10) entries are populated and n_levels reflects that count.
- AC-008: qty_multiplier and breakeven_trigger are cast to float before being passed to the Numba kernel.
- AC-009: On ModuleNotFoundError from a stale Numba cache, the plugin clears *.nbi/*.nbc files in its __pycache__ and retries the JIT call once (via _call_numba).
- AC-010: get_default_params() returns {'levels': [0.2, 0.4, 0.6], 'qty_multiplier': 1.0}.
- AC-011: get_param_schema() declares qty_multiplier with min=0.1, max=3.0, step=0.1 and levels as list[float] with the same default.

## Edge Cases

- levels=None → falls back to the default [0.2, 0.4, 0.6].
- levels list longer than 10 entries → silently truncated to the first 10 (excess levels ignored).
- Empty levels list → n_levels=0 and the packed array is all -1.0 sentinels; behavior then depends on the downstream _simulate_trade_scale_in_numba (no scale-ins performed).
- trail_tp_dist_arr=None → replaced by np.zeros_like(tp_dist_arr) so trailing TP is effectively disabled.
- Very short input arrays (n <= 1) → the loop `for i in range(n-1)` does not execute and all-zero targets/durations are returned.
- Simulation returns exit_bar < 0 (no exit within horizon) → duration is set to max_bars for that bar in the with-durations variant.
- Stale Numba on-disk cache raising ModuleNotFoundError → cache files are unlinked (OSError on unlink is swallowed) and the JIT call is retried exactly once.

## Assumptions

- fwbg.simulation._simulate_trade_scale_in_numba is the authoritative scale-in trade simulator; its 7-tuple return contract (result, exit_bar, ...) is stable and result==1.0 denotes a winning trade.
- tp_dist_arr, sl_dist_arr, trail_dist_arr are aligned per-bar arrays of the same length as opens/closes/highs/lows.
- Callers pass trailing parameters (breakeven_trigger, trail_tp_dist_arr) through from the configured exit strategy; the modifier itself does not decide trailing behavior.

## Needs Clarification

- [NEEDS CLARIFICATION: Whether qty_multiplier is actually consumed by _simulate_trade_scale_in_numba (it is forwarded, but its effect on position sizing is defined in that external simulator, not in this file).]
