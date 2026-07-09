# Plugin Spec — macro_surprise

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Derives OHLC "information-flow" features: overnight-gap stats, overnight-vs-intraday return decomposition, range/return surprise vs rolling vol, vol-break ratios, and streak counters.

## Summary

Indicator that derives 21 features characterizing "information arrival" in the market: gap analysis vs. previous close (including whether the gap is filled or extended), decomposition of the total return into overnight and intraday parts, detection of surprise moves whose actual range or return z-score exceeds a threshold multiple of the rolling-vol expectation, volatility-break ratios and z-scores, and streak counters for gap direction and surprise persistence. All feature columns are shifted by one bar via `shift_features` to prevent lookahead bias.

## Inputs

- df: pandas DataFrame with OHLC columns 'O', 'H', 'L', 'C'

## Parameters

- `vol_lookback` (int, default=20): Rolling-window length (in bars) used to estimate historical volatility, average true range, expected return std, and volatility-spike statistics.
- `surprise_threshold` (float, default=2): Number of standard deviations (or multiples of the expected move) beyond which a range move or return z-score is flagged as a surprise (macro_is_surprise / macro_return_surprise).
- `gap_ma_period` (int, default=10): Rolling-window length (in bars) used to compute macro_gap_avg and macro_gap_std over gap_pct.

## Outputs

- macro_gap
- macro_gap_pct
- macro_gap_normalized
- macro_gap_up
- macro_gap_down
- macro_gap_filled
- macro_gap_extended
- macro_gap_avg
- macro_gap_std
- macro_total_return
- macro_overnight_return
- macro_intraday_return
- macro_overnight_ratio
- macro_range_surprise
- macro_is_surprise
- macro_return_zscore
- macro_return_surprise
- macro_vol_ratio
- macro_vol_zscore
- macro_gap_streak
- macro_surprise_streak

## Acceptance Criteria

- AC-001: compute(df) returns the original DataFrame concatenated with all 21 feature columns listed in get_feature_columns().
- AC-002: All 21 feature columns are shifted by one bar via shift_features so no feature at bar i uses information from bar i or later.
- AC-003: macro_gap equals O - C.shift(1); macro_gap_pct equals that gap divided by the previous close.
- AC-004: macro_gap_normalized equals the gap divided by the vol_lookback-period rolling mean of (H - L).
- AC-005: macro_gap_up and macro_gap_down are 1.0/0.0 flags derived from the sign of the gap.
- AC-006: macro_gap_filled is 1.0 when an up-gap's close is at or below the previous close, or when a down-gap's close is at or above the previous close. When gap == 0 (flat open, neither up nor down), the np.where else-branch applies and macro_gap_filled = (c >= c_prev), so a zero-gap bar can be reported as filled even though no directional gap exists.
- AC-007: macro_gap_extended is 1.0 when the close moves further in the direction of the gap relative to the open. When gap == 0 (flat open), the np.where else-branch applies and macro_gap_extended = (c < o), so a zero-gap bar can be reported as extended even though no directional gap exists.
- AC-008: macro_gap_avg and macro_gap_std are the gap_ma_period rolling mean and std of macro_gap_pct.
- AC-009: macro_total_return, macro_overnight_return, and macro_intraday_return decompose the return using (C - C_prev)/C_prev, (O - C_prev)/C_prev, and (C - O)/O respectively.
- AC-010: macro_overnight_ratio equals |overnight_return| / |total_return|, with zero total-return bars mapped to NaN via replace(0, np.nan).
- AC-011: macro_range_surprise equals (H - L) / (rolling_std(pct_change, vol_lookback) * C_prev), with zero expected moves mapped to NaN.
- AC-012: macro_is_surprise is 1.0 iff the actual (H - L) exceeds surprise_threshold * expected_move.
- AC-013: macro_return_zscore equals pct_change divided by its vol_lookback rolling std; macro_return_surprise is 1.0 iff |zscore| > surprise_threshold.
- AC-014: macro_vol_ratio equals a 5-bar mean of |returns| divided by its vol_lookback rolling mean (expected vol), with zero-denominator bars mapped to NaN.
- AC-015: macro_vol_zscore equals (realized_vol - expected_vol) / rolling_std(realized_vol, vol_lookback).
- AC-016: macro_gap_streak counts consecutive bars with the same sign of the gap; macro_surprise_streak counts consecutive bars where macro_is_surprise is 1; both are 0 where the underlying value is 0 and NaN where it is NaN.
- AC-017: get_default_params() returns {'vol_lookback': 20, 'surprise_threshold': 2.0, 'gap_ma_period': 10}.
- AC-018: get_feature_columns() returns exactly the 21 column names produced by compute(), in the documented order.

## Edge Cases

- Zero previous close would divide by zero in gap_pct / return computations — inherits pandas division semantics (produces inf/NaN) and is not explicitly guarded.
- Zero total return: macro_overnight_ratio uses replace(0, NaN) on |total_return| to avoid division-by-zero, yielding NaN rather than inf.
- Zero average range in the rolling window would make macro_gap_normalized division-by-zero (inf/NaN); not explicitly guarded via safe_divide.
- Zero expected_move (all-flat returns window) is guarded via replace(0, NaN) for macro_range_surprise; expected_vol likewise for macro_vol_ratio.
- First vol_lookback / gap_ma_period bars: rolling means/stds yield NaN — features are NaN at the head of the series (and further shifted by 1 due to shift_features).
- Streak helper sets streak to 0 whenever the underlying series value equals 0 (e.g., gap sign == 0 for exactly-flat opens) and NaN whenever the series value is NaN.
- gap_filled / gap_extended are computed for every bar including bars where gap == 0; because np.where takes the 'else' branch (c >= c_prev), a flat-gap bar can still register as 'filled'.
- Single-row / very short DataFrames: c.shift(1) is NaN so all gap- and return-derived features are NaN for the first bar.

## Assumptions

- Input DataFrame contains uppercase OHLC columns named exactly 'O', 'H', 'L', 'C'.
- The DataFrame index is monotonically ordered in time (compute treats row i-1 as the immediately preceding bar).
- shift_features(features, df.index) applies a single-bar forward shift to every feature to enforce no-lookahead.
- 'Overnight' is interpreted as the O - C_prev transition between adjacent bars; the indicator does not distinguish calendar sessions.

## Needs Clarification

- [NEEDS CLARIFICATION: Whether divisions that are not wrapped in replace(0, NaN) (e.g., gap / c_prev, gap / avg_range) should additionally be routed through safe_divide to satisfy the constitution's 'use safe_divide for all divisions' rule.]
- [NEEDS CLARIFICATION: Whether the gap == 0 case should be treated as a distinct third category rather than falling into the down-gap branches of macro_gap_filled / macro_gap_extended.]
