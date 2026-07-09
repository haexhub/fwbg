# Plugin Spec — cci

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes the Commodity Channel Index (CCI) — the deviation of the Typical Price from its SMA — for one or more configurable lookback periods.

## Summary

CCI indicator that produces one `cci_{period}` feature column per configured period, computed via `ta.trend.cci` on H/L/C and shifted by one bar to prevent lookahead. Overbought at >100, oversold at <-100.

## Inputs

- df with columns H, L, C indexed by timestamp

## Parameters

- `periods` (list[int], default=[14, 20]): Periods for CCI calculation. CCI measures deviation of Typical Price from its SMA. Values > 100 indicate overbought, < -100 oversold.

## Outputs

- cci_{period} for each period in params['periods'] (defaults: cci_14, cci_20)

## Acceptance Criteria

- AC-001: compute() returns the input df concatenated with one `cci_{period}` column per period in `params['periods']`.
- AC-002: When `periods` is not supplied (or None), the defaults [14, 20] are used, producing `cci_14` and `cci_20`.
- AC-003: Each `cci_{period}` column is computed via `ta.trend.cci(H, L, C, window=period)`.
- AC-004: All feature columns are shifted by one bar via `shift_features` so that row i's feature depends only on data up to bar i-1 (no lookahead).
- AC-005: `get_feature_columns(params)` returns `[f'cci_{p}' for p in params['periods']]`, falling back to defaults when absent.
- AC-006: `get_signal_columns()` and `get_overlay_columns()` return empty lists.
- AC-007: `get_column_group_labels()` maps `'cci'` to the human label `'CCI (Commodity Channel Index)'`.
- AC-008: Plugin is registered under the name `'cci'` via `@register_indicator('cci')`.

## Edge Cases

- `periods=None` is treated as the default [14, 20].
- An empty `periods` list yields no CCI feature columns (df is returned essentially unchanged aside from concat).
- Initial rows within the CCI warmup window (and the extra bar from `shift_features`) will contain NaN values.
- Period larger than the number of rows results in an all-NaN column for that period.
- Duplicate values in `periods` produce duplicate column names as-is (no dedup).

## Assumptions

- Input df provides the high/low/close price columns under the names 'H', 'L', 'C'.
- `shift_features` applies a 1-bar shift and returns a DataFrame aligned to `df.index`.
