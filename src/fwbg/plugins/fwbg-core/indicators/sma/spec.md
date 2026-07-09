# Plugin Spec — sma

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes Simple Moving Average lines over configurable OHLC source prices, producing distance-to-close features, cross-line crossing features, and overlay lines.

## Summary

SMA indicator built on BaseMovingAverageIndicator. For each configured line (period + OHLC source), computes an SMA overlay line, its normalized distance from close, and optionally boolean crossing features for every pair of lines (including cross-source pairs).

## Inputs

- OHLC price columns (open, high, low, close) — the 'source' param selects which column feeds each SMA line

## Parameters

- `lines` (list[string], default=[{'period': 20, 'source': 'C'}, {'period': 50, 'source': 'C'}, {'period': 200, 'source': 'C'}]): List of SMA line configs. Each entry has 'period' (int, >= 2) and 'source' (one of 'O', 'H', 'L', 'C'). Source defaults to 'C' if omitted.
- `crossings` (bool, default=True): Compute crossing features for all SMA line pairs (including cross-source).

## Outputs

- sma_dist_* — SMA distance features normalized as percent of close (per configured line)
- sma_crossing_* — boolean crossing features for all SMA line pairs, including cross-source pairs (only when crossings=True)
- _sma_* — SMA overlay lines, underscore-prefixed so they are not treated as ML features

## Acceptance Criteria

- AC-001: Produces one SMA overlay line (prefixed '_sma') per entry in the 'lines' parameter using ta.trend.sma_indicator with the given period and source column.
- AC-002: Emits one 'sma_dist_*' feature per configured line, normalized as percentage of close.
- AC-003: When crossings=True, emits 'sma_crossing_*' boolean features for every pair of configured SMA lines, including pairs whose sources differ.
- AC-004: When crossings=False, no crossing features are emitted.
- AC-005: Overlay columns are prefixed with underscore so they are excluded from ML features.
- AC-006: Default configuration produces SMA lines with periods 20, 50, and 200, all sourced from close.
- AC-007: Column group labels expose 'SMA Distance', 'SMA Crossings', and 'SMA Lines' groups for UI grouping.

## Edge Cases

- Period longer than the input series — leading rows for that line's SMA are NaN (ta.trend.sma_indicator behavior).
- Only one line configured — crossings=True yields no crossing features because no pairs exist.
- Line entry omits 'source' — defaults to 'C' (close) per the param schema.
- Multiple lines with identical (period, source) — crossings between them are degenerate; behavior follows the base class's pair enumeration.
- Non-close source (O/H/L) — distance is still normalized against close, not against the source column.

## Assumptions

- BaseMovingAverageIndicator handles overlay-column emission, distance-feature computation, crossing-pair enumeration, and lookahead-shift as it does for the sibling EMA indicator.
- 'ta' library's sma_indicator returns a series aligned to the input index with NaN warmup rows.
- Distance normalization ('% vom Close') is performed by the base class, not this subclass.

## Needs Clarification

- [NEEDS CLARIFICATION: Exact output column naming pattern (e.g., 'sma_dist_20_c' vs 'sma_dist_c_20') is delegated to BaseMovingAverageIndicator and not visible in this file — confirm against the base class before pinning column names in the contract.]
- [NEEDS CLARIFICATION: Whether the no-lookahead shift is applied inside BaseMovingAverageIndicator (assumed) or must be added here — verify in the base class to satisfy Constitution §IV.]
