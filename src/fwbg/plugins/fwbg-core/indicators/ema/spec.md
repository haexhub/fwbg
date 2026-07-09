# Plugin Spec — ema

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes Exponential Moving Averages on configurable OHLC source columns and derives Close-to-EMA distance features plus pairwise EMA crossing features.

## Summary

EMA indicator that produces multiple EMA lines (each with its own period and OHLC source), a normalized distance feature between Close and each EMA, optional binary crossing features for every EMA pair (including cross-source pairs), and raw EMA overlay lines prefixed with `_` for charting. Extends BaseMovingAverageIndicator; the EMA itself is delegated to `ta.trend.ema_indicator`.

## Inputs

- ohlcv dataframe with columns Open, High, Low, Close (sources selected per line via 'O'/'H'/'L'/'C')

## Parameters

- `lines` (list[string], default=[{'period': 8, 'source': 'C'}, {'period': 21, 'source': 'C'}, {'period': 50, 'source': 'C'}, {'period': 100, 'source': 'C'}, {'period': 200, 'source': 'C'}]): List of EMA line definitions. Each entry is an object with 'period' (int, >= 2) and 'source' (one of 'O','H','L','C'; defaults to 'C' when omitted). Defaults to periods [8, 21, 50, 100, 200] on source 'C'.
- `crossings` (bool, default=True): If true, emits binary crossing features for all EMA line pairs (including cross-source pairs), indicating whether the shorter-period EMA is above the longer-period EMA.

## Outputs

- ema_dist_{period}  — Close-to-EMA distance (normalized) when source is 'C'
- ema_dist_{period}_{source_lower}  — Close-to-EMA distance for non-'C' sources
- ema_{period_a}[_{src_a}]_above_{period_b}[_{src_b}]  — binary crossing feature per EMA pair (when crossings=True)
- _ema_{period}[_{source_lower}]  — raw EMA overlay line (prefixed with '_', not a ML feature)

## Acceptance Criteria

- AC-001: For each entry in `lines`, an EMA is computed via `ta.trend.ema_indicator` on the selected OHLC source with the given period.
- AC-002: A distance feature `ema_dist_{period}` (source 'C') or `ema_dist_{period}_{source_lower}` (non-'C' sources) is produced for every configured line, representing Close normalized against the EMA.
- AC-003: When `crossings=True`, a binary `..._above_...` feature is emitted for every pair of configured EMA lines, encoding whether the shorter-period EMA is above the longer-period EMA (cross-source pairs included).
- AC-004: A raw EMA overlay column is emitted for every line, named `_ema_{period}` for source 'C' or `_ema_{period}_{source_lower}` otherwise; because of the leading underscore it is an overlay, not a ML feature.
- AC-005: Column naming omits the source suffix exclusively when source is 'C'; other sources always append `_{source_lower}`.
- AC-006: `get_default_params()` returns `lines` = the module-level DEFAULT_LINES (periods 8/21/50/100/200 on 'C') and `crossings` = True.
- AC-007: `get_param_schema()` declares `lines` as list[object] with per-item schema (period int in [2, 1000]; source enum O/H/L/C defaulting to 'C') and `crossings` as bool.
- AC-008: `get_column_group_labels()` maps the prefixes `ema_dist`, `ema_crossing`, and `_ema` to human labels 'EMA Distance', 'EMA Crossings', and 'EMA Lines' respectively.
- AC-009: Registered under the name 'ema' via `@register_indicator('ema')`; class attribute `name == 'ema'`, `version == '1.0.0'`, `_human_label == 'EMA'`.

## Edge Cases

- A line entry omits the 'source' key — defaults to 'C' per the param schema, and the resulting columns use the no-suffix naming form.
- Two identical `lines` entries (same period + source) — behaviour of duplicate line definitions is not explicitly guarded in this file.
- `crossings=False` — no `..._above_...` features are emitted; only distance and overlay columns remain.
- A single-line configuration — no EMA pairs exist, so no crossing features are produced even when `crossings=True`.
- Cross-source pairs (e.g. EMA(5, H) vs EMA(200, C)) — crossings are still computed across differing sources.
- Warm-up window: EMA values at the start of the series are NaN/undefined until enough bars accumulate for the given period (delegated to `ta.trend.ema_indicator`).

## Assumptions

- The dataframe passed to `compute()` contains the standard OHLC columns (`Open`, `High`, `Low`, `Close`) that BaseMovingAverageIndicator maps from the 'O'/'H'/'L'/'C' source codes.
- BaseMovingAverageIndicator handles the naming conventions described in the module docstring (dist/crossing/overlay prefixes) uniformly across all MA subclasses; this file only supplies the EMA computation and defaults.

## Needs Clarification

- [NEEDS CLARIFICATION: Exact formula used for `ema_dist_*` (e.g. `(Close - EMA) / Close` vs `(Close - EMA) / EMA`) is defined in BaseMovingAverageIndicator, not in this file.]
- [NEEDS CLARIFICATION: Whether feature shifting for no-lookahead is applied inside BaseMovingAverageIndicator (`shift_features`) — cannot be confirmed from this source alone.]
- [NEEDS CLARIFICATION: Whether crossings enumerate ordered pairs or only shorter-vs-longer period pairs, and how ties (equal periods with different sources) are ordered — determined by BaseMovingAverageIndicator.]
