# Plugin Spec — cot_positioning

**Kind**: data_loading  •  **Version**: 1.0.0

## Capability

Computes CFTC COT positioning features (52-week z-scores, extreme-long/short flags, crowded-trade flag, and multi-horizon net-position momentum) from macro_cot_* base columns.

## Summary

Consumes pre-loaded macro_cot_* net-position columns and derives, per instrument prefix: a rolling z-score over a configurable week window, extreme long/short indicators (|z|>2), a crowded-trade flag (|z|>1.5), and pct_change momentum over configurable week-based lookbacks. All produced features are shifted by one bar to prevent lookahead, then written back onto ctx.df.

## Inputs

- ctx.df columns starting with 'macro_cot_' (net positioning series per instrument)

## Parameters

- `indicators` (string, default=None): Mapping of COT data file stems to column prefixes (e.g. COT_EURUSD_DAY -> cot_eurusd). Default in code is a dict of the seven majors {COT_EURUSD_DAY: cot_eurusd, COT_USDJPY_DAY: cot_usdjpy, COT_GBPUSD_DAY: cot_gbpusd, COT_USDCAD_DAY: cot_usdcad, COT_AUDUSD_DAY: cot_audusd, COT_USDCHF_DAY: cot_usdchf, COT_NZDUSD_DAY: cot_nzdusd}. Values are used by get_feature_columns as feature-column prefixes.
- `lookbacks_weeks` (list[int], default=[1, 4, 12, 26]): Week-based lookback periods used to compute pct_change momentum on the net position series (converted to H1 bars as weeks*5*24).
- `zscore_window_weeks` (int, default=52): Rolling window in weeks for z-score normalization of net positions (converted to H1 bars as weeks*5*24; min_periods = window//4; std is clipped to a 1e-6 floor).

## Outputs

- {prefix}_zscore
- {prefix}_extreme_long
- {prefix}_extreme_short
- {prefix}_crowded
- {prefix}_chg_{lb}w for each lb in lookbacks_weeks

## Acceptance Criteria

- AC-001: For every column in ctx.df starting with 'macro_cot_', produces a {prefix}_zscore column where prefix is the source column with the 'macro_' stripped.
- AC-002: Produces {prefix}_extreme_long = 1.0 where zscore > 2.0 else 0.0, and {prefix}_extreme_short = 1.0 where zscore < -2.0 else 0.0.
- AC-003: Produces {prefix}_crowded = 1.0 where |zscore| > 1.5 else 0.0.
- AC-004: For each lb in lookbacks_weeks, produces {prefix}_chg_{lb}w = net.pct_change(lb*5*24) * 100.
- AC-005: Rolling z-score uses window = zscore_window_weeks * 5 * 24 bars with min_periods = window // 4, and rolling std is clipped to a lower bound of 1e-6 to avoid divide-by-zero.
- AC-006: All produced cot_* feature columns (those starting with 'cot_' and not starting with 'macro_cot_') are shifted by 1 bar before being returned to prevent lookahead.
- AC-007: Returns ctx with ctx.df carrying the added feature columns; existing df columns are preserved.
- AC-008: get_feature_columns returns the deterministic list of expected feature names derived from indicators.values() and lookbacks_weeks, regardless of what is actually present in ctx.df.

## Edge Cases

- ctx.df contains no 'macro_cot_' columns: no features are added and ctx is returned unchanged.
- Net position series is (near-)constant over the rolling window: rolling std would be ~0, but is clipped to 1e-6 so zscore is finite (extremes/crowded flags evaluate against a very large magnitude).
- Fewer than window//4 non-NaN observations at the start of the series: rolling mean/std return NaN, so zscore is NaN and the derived boolean-cast columns become 0.0 (NaN>2.0 is False -> 0.0) while the shift(1) still propagates NaN at the first bar.
- lookbacks_weeks is empty: no {prefix}_chg_{lb}w columns are produced, but zscore/extreme/crowded columns are still produced.
- The 'indicators' param passed to get_feature_columns does not match the macro_cot_* columns actually present in ctx.df: get_feature_columns lists names based on indicators.values(), which may diverge from the columns execute() actually creates.

## Assumptions

- Underlying bar frequency is H1, so weeks are converted to bars via weeks * 5 * 24 (5 trading days x 24 H1 bars).
- macro_cot_* base columns are provided upstream by a DataSource; this loader does not read COT files itself despite the 'indicators' mapping in params.
- ctx exposes a mutable .df attribute of type pandas.DataFrame supporting rolling, pct_change, and shift.

## Needs Clarification

- [NEEDS CLARIFICATION: The 'indicators' param is declared as type 'string' in get_param_schema but its default is a dict, and execute() ignores it entirely (columns are discovered via the 'macro_cot_' prefix) while get_feature_columns uses indicators.values() as the source of truth. Intended type and role of this param need clarification.]
- [NEEDS CLARIFICATION: Whether execute() should honor the 'indicators' mapping (e.g. to filter which macro_cot_* columns to process) instead of processing every 'macro_cot_' column found in ctx.df.]
