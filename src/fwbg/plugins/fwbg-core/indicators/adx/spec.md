# Plugin Spec — adx

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes the Average Directional Index (ADX) trend-strength value (0-100) over one or more configurable periods from H/L/C price data.

## Summary

ADX indicator that measures trend strength on a 0-100 scale (direction-agnostic) for each configured period, emitting one shifted feature column per period (adx_{period}) computed via ta.trend.adx over the high/low/close columns.

## Inputs

- df['H'] (high price series)
- df['L'] (low price series)
- df['C'] (close price series)

## Parameters

- `periods` (list[int], default=[7, 14, 21]): Periods for ADX calculation. ADX measures trend strength on a 0-100 scale regardless of direction. Shorter periods react faster, longer periods smooth out noise.

## Outputs

- adx_{period}: ADX trend-strength value (0-100) for each configured period, shifted by one bar to prevent lookahead

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame concatenated with one adx_{period} column per entry in the periods parameter
- AC-002: When periods is None or omitted, the default [7, 14, 21] is used, producing columns adx_7, adx_14, adx_21
- AC-003: Feature values are computed via ta.trend.adx using df['H'], df['L'], df['C'] and the given window
- AC-004: All emitted feature columns are shifted by one bar via shift_features so no row depends on future data
- AC-005: get_feature_columns(params) returns the list of adx_{period} names matching the effective periods (defaults merged with overrides)
- AC-006: get_signal_columns() and get_overlay_columns() return empty lists
- AC-007: get_column_group_labels() returns {'adx': 'ADX (Average Directional Index)'}

## Edge Cases

- periods argument is None → falls back to the default [7, 14, 21]
- A period larger than the number of rows in df → ta.trend.adx will produce NaN values for those rows
- Empty periods list → compute() returns the input DataFrame unchanged (no feature columns added) and get_feature_columns returns an empty list
- Missing required H/L/C columns in df → KeyError raised by df['H'|'L'|'C'] lookup
- Very short periods (e.g. below 2) may be rejected by the schema min=2 constraint at validation time

## Assumptions

- Input DataFrame contains uppercase OHLC columns 'H', 'L', 'C'
- The ta library's ta.trend.adx implementation is the source of truth for the ADX numerical values
- shift_features applies a one-bar shift to every feature column to enforce no-lookahead
