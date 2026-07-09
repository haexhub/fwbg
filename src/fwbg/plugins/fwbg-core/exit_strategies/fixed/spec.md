# Plugin Spec — fixed

**Kind**: exit_strategy  •  **Version**: 0.1.0

## Capability

Computes trade exit targets using fixed take-profit and stop-loss distances derived from spread multipliers.

## Summary

Exit strategy that sets TP and SL as constant distances equal to the context spread multiplied by user-supplied integer pip multipliers, with optional bar-based timeout and support for entry-modifier-driven trailing exits.

## Inputs

- df: pandas DataFrame with OHLC columns O, H, L, C
- ctx: SimulationContext providing spread, max_trade_bars, and optional entry_modifier / entry_modifier_params / exit_modifier_params
- params: optional GridParams with tp_value, sl_value, timeout_bars (falls back to kwargs tp/sl/timeout_bars)

## Parameters

- `tp` (int, default=30): Take-profit distance as a spread multiplier (e.g. 30 = 30 pips at spread 0.0001).
- `sl` (int, default=20): Stop-loss distance as a spread multiplier (e.g. 20 = 20 pips at spread 0.0001).
- `timeout_bars` (int, default=None): Close the trade after N bars if neither TP nor SL is hit; None disables the timeout.

## Outputs

- targets_long: np.ndarray of per-bar long-trade outcomes
- targets_short: np.ndarray of per-bar short-trade outcomes
- durations_long, durations_short: np.ndarray durations returned only when return_durations=True
- resolve_distances(df, tp, sl, ctx): tuple of two length-n arrays holding constant tp and sl distances

## Acceptance Criteria

- AC-001: compute_targets returns (targets_long, targets_short) as np.ndarrays when return_durations is False.
- AC-002: compute_targets returns (targets_long, targets_short, durations_long, durations_short) when return_durations is True.
- AC-003: tp_distances and sl_distances are constant arrays of length len(df) equal to ctx.spread * tp and ctx.spread * sl respectively.
- AC-004: Slippage passed to the numba kernel equals ctx.spread * 0.5.
- AC-005: max_bars passed to the numba kernel equals ctx.max_trade_bars when truthy, otherwise len(df).
- AC-006: timeout_val passed to the numba kernel equals timeout_bars when truthy, otherwise 0.
- AC-007: When params is provided, tp/sl/timeout_bars are taken from params.tp_value, params.sl_value, params.timeout_bars; otherwise from kwargs with defaults tp=30, sl=20, timeout_bars=None.
- AC-008: resolve_distances(df, tp, sl, ctx) returns two np.full arrays of length len(df) with values ctx.spread*tp and ctx.spread*sl.
- AC-009: get_cache_key returns the string 'fixed_tp{int(tp)}_sl{int(sl)}_to{timeout_or_none}' where timeout_or_none is 'none' when timeout_bars is falsy.
- AC-010: get_default_params returns {'tp': 30, 'sl': 20, 'timeout_bars': None}.
- AC-011: When ctx.entry_modifier is a non-empty string, the registered entry modifier's compute_targets is invoked with tp_mult=0.0, sl_mult=0.0, the computed ATR array, spread, slippage, tp_distances[0], sl_distances[0], max_bars, timeout_val, return_durations, and trailing params (breakeven_trigger, trail_atr_mult, trail_tp_atr_mult) pulled from ctx.exit_modifier_params.
- AC-012: Under an entry modifier, ATR is sourced from df['_atr'] if present, else df['vol_atr'] if present, else computed via ta.volatility.average_true_range with window=14; NaNs in the ATR array are replaced with 0.0.

## Edge Cases

- ctx.max_trade_bars is None or 0 -> max_bars falls back to len(df).
- timeout_bars is None or 0 -> timeout_val is 0 (no timeout enforced by the kernel).
- params is None -> tp/sl/timeout_bars come from kwargs with defaults 30/20/None.
- ctx.entry_modifier is None or non-string -> the entry-modifier branch is skipped and the plain numba kernel is used.
- ctx.entry_modifier_params or ctx.exit_modifier_params is None -> treated as empty dict {}.
- ATR computation when neither '_atr' nor 'vol_atr' columns exist -> falls back to ta.volatility.average_true_range with window=14.
- NaN values in the ATR array -> replaced with 0.0 via np.nan_to_num before being passed to the entry modifier.
- get_cache_key with timeout_bars=None or 0 -> timeout segment is 'tonone'.

## Assumptions

- ctx.spread is a positive float used as the base pip size for scaling tp and sl.
- df contains float-coercible columns 'O', 'H', 'L', 'C' aligned by index.
- compute_targets_numba returns a 4-tuple (targets_long, targets_short, durations_long, durations_short).
- Registered entry modifiers expose a compute_targets signature matching the positional and keyword arguments used here.

## Needs Clarification

- [NEEDS CLARIFICATION: Param schema declares tp/sl min=1, but get_cache_key coerces via int() and no runtime validation is performed here — is the schema enforced upstream?]
- [NEEDS CLARIFICATION: timeout_bars schema advertises max=500, but the code accepts any truthy int; is the upper bound enforced by the caller/grid?]
