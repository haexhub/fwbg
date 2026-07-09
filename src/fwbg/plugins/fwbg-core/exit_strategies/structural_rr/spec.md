# Plugin Spec — structural_rr

**Kind**: exit_strategy  •  **Version**: 0.1.0

## Capability

Derives per-bar TP/SL distances for exit simulation by reading a structural SL distance from indicator columns and setting TP as a fixed R-multiple of that SL.

## Summary

Exit strategy that reads a pre-computed structural stop-loss distance from indicator columns (long/short) and derives take-profit as a fixed R-multiple of that SL, with pip-based floors for both TP and SL and optional per-trade timeout, delegating trade simulation to `compute_targets_numba`.

## Inputs

- OHLC columns `O`, `H`, `L`, `C` from the dataframe.
- Structural SL distance columns `sl_dist_column_long` (default `tpm_sl_dist_long`) and `sl_dist_column_short` (default `tpm_sl_dist_short`), typically produced by the `pullback_momentum` indicator.
- `SimulationContext` fields: `spread`, `max_trade_bars`, `exit_params` (used by `resolve_distances`).
- Optional grid `params` object with `tp_value` (→ r_multiple) and `timeout_bars` attributes.

## Parameters

- `r_multiple` (float, default=2): Risk-reward ratio. TP = structural SL distance × r_multiple. Maps to 'tp' in the grid config for optimization.
- `sl_dist_column_long` (string, default='tpm_sl_dist_long'): Column containing the structural SL distance for long trades.
- `sl_dist_column_short` (string, default='tpm_sl_dist_short'): Column containing the structural SL distance for short trades.
- `min_tp_pips` (int, default=10): Minimum TP distance in pips (spread multiples). Floor for very tight structures.
- `min_sl_pips` (int, default=5): Minimum SL distance in pips (spread multiples). Floor for very tight structures.
- `timeout_bars` (int, default=None): Close trade after N bars if TP/SL not hit. None = no timeout.

## Outputs

- Per-bar long-side target outcome array from `compute_targets_numba` (`tgt_l`).
- Per-bar short-side target outcome array from `compute_targets_numba` (`tgt_s`).
- When `return_durations=True`: additionally per-bar long/short duration arrays (`dur_l`, `dur_s`).
- Via `resolve_distances`: per-bar TP distance array and per-bar SL distance array (floored by `min_tp_distance` / `min_sl_distance`).

## Acceptance Criteria

- AC-001: Reads per-bar SL distance from `sl_dist_column_long` at long-entry bars and `sl_dist_column_short` at short-entry bars, merging them into a single per-bar SL array (long value preferred when both are set).
- AC-002: Take-profit distance is computed as `sl_dist * r_multiple` on a per-bar basis.
- AC-003: Enforces a minimum SL distance of `ctx.spread * min_sl_pips` and a minimum TP distance of `ctx.spread * min_tp_pips` as floors.
- AC-004: Delegates the actual TP/SL simulation to `compute_targets_numba` using OHLC arrays, `ctx.spread`, half-spread slippage, and `ctx.max_trade_bars` (with optional `timeout_bars`).
- AC-005: Parameter resolution order: `params.tp_value` (grid) → `kwargs['r_multiple']` overrides for `r_multiple`; `kwargs` overrides for column names and pip floors; default `r_multiple = 2.0`.
- AC-006: `resolve_distances` uses `tp` as the R multiple (falling back to 2.0 when falsy) and ignores the `sl` argument, reading column/floor settings from `ctx.exit_params`.
- AC-007: Cache key is formed as `structural_rr_r{r_multiple}_to{timeout_bars or 'none'}`.
- AC-008: When `return_durations=True`, `compute_targets` returns the full 4-tuple `(tgt_l, tgt_s, dur_l, dur_s)` from the numba kernel; otherwise it returns `(tgt_l, tgt_s)`.

## Edge Cases

- NaN values in the merged SL distance array are replaced with `min_sl_distance` (spread × min_sl_pips).
- If `sl_dist_column_long` or `sl_dist_column_short` is absent from the dataframe, that side's SL array is treated as all-NaN and effectively falls back to the min-SL floor.
- If both long and short SL columns are non-NaN on the same bar (not expected in normal operation), the long SL value is used.
- SL distances below `min_sl_distance` are clamped up to that floor; TP distances below `min_tp_distance` are clamped up to that floor.
- `timeout_bars=None` (default) is passed to the numba kernel as `0`, meaning no timeout is applied beyond `ctx.max_trade_bars`.
- `ctx.max_trade_bars` falsy → falls back to `len(df)` as the maximum trade duration.
- `r_multiple` falsy in `resolve_distances` (e.g. `tp=0`) → defaults to 2.0.
- `params.tp_value` being `None` leaves `r_multiple` at its default (2.0) rather than overriding it.

## Assumptions

- _none_
