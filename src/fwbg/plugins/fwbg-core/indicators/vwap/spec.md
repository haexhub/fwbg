# Plugin Spec — vwap

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Computes session-reset volume-weighted average price (VWAP) and derived mean-reversion features (normalized deviation, rolling z-scores, ±Nσ volume-weighted bands, band position, above-VWAP flag).

## Summary

VWAP indicator with configurable session-reset (session_start_hour), rolling z-score windows of the VWAP deviation, and volume-weighted standard-deviation bands. Uses typical price (H+L+C)/3 weighted by volume when available, otherwise falls back to equal weighting. Emits VWAP, normalized deviation, one z-score per configured window, upper/lower band per configured multiplier, a normalized band position within ±1σ, and a binary above-VWAP flag. All feature columns are shifted by one bar to prevent lookahead bias.

## Inputs

- H
- L
- C
- V (optional; falls back to equal-weight if missing, all-NaN, or all-non-positive)
- DatetimeIndex (used to derive hour for session boundaries)

## Parameters

- `session_start_hour` (int, default=9): Hour of day (0-23), in `session_tz`, at which VWAP is reset.
- `session_tz` (string, default="UTC"): IANA time zone of `session_start_hour` (e.g. America/New_York, Europe/Berlin); follows daylight saving time. A naive index is read as UTC.
- `zscore_windows` (list[int], default=[20, 50]): Rolling window sizes for the z-score of the VWAP deviation; one vwap_zscore_<window> column is produced per entry. Rolling min_periods is max(window // 2, 2).
- `band_multipliers` (list[float], default=[1, 2]): Standard-deviation multipliers for the VWAP bands (volume-weighted std). One vwap_upper_<label>/vwap_lower_<label> pair is emitted per entry; integer values use the int label, non-integer values replace '.' with '_'.

## Outputs

- vwap
- vwap_deviation
- vwap_zscore_20
- vwap_zscore_50
- vwap_upper_1
- vwap_lower_1
- vwap_upper_2
- vwap_lower_2
- vwap_band_pos
- vwap_above

## Acceptance Criteria

- AC-001: compute() returns the input df concatenated with the VWAP feature columns; original columns are preserved.
- AC-002: vwap equals the session-cumulative sum of (typical_price * weight) divided by the session-cumulative sum of weight, where typical_price = (H + L + C) / 3.
- AC-003: A bar's session is the wall-clock date (in session_tz) of the last session_start_hour o'clock at or before it, so a new session starts at session_start_hour even when no bar exists in that hour, and every bar of daily or coarser data is its own session (VWAP = the bar's typical price).
- AC-004: The weights at each bar use volume accumulated during the session up to that bar. Use equal weights until the first positive volume in the session. Zero, negative and missing volumes add no weight thereafter.
- AC-005: vwap_deviation = safe_divide(C - vwap, |vwap|).
- AC-006: For each window w in zscore_windows, vwap_zscore_<w> = safe_divide(deviation - rolling_mean(deviation, w), rolling_std(deviation, w)) using min_periods = max(w // 2, 2).
- AC-007: vwap_std = sqrt(clip(cum(w*tp^2)/cum(w) - vwap^2, lower=0)) per session, i.e. the volume-weighted std of the session's typical prices up to the bar (evaluated centred on the session's first typical price for numerical stability); for each multiplier m in band_multipliers, vwap_upper_<label> = vwap + m*vwap_std and vwap_lower_<label> = vwap - m*vwap_std, where <label> is int(m) if m is integer-valued else str(m).replace('.', '_').
- AC-008: vwap_band_pos = safe_divide(C - (vwap - vwap_std), 2 * vwap_std).
- AC-009: vwap_above = 1.0 when C > vwap else 0.0 (float dtype).
- AC-010: All emitted feature columns are shifted by one bar via shift_features before being concatenated, so row i features depend only on data up to and including row i-1 in the produced frame.
- AC-011: get_feature_columns(params) returns the columns compute() produces for the same params (ten by default); get_signal_columns() returns an empty list (vwap_above is shifted); get_default_params() returns session_start_hour=9, session_tz='UTC', zscore_windows=[20,50], band_multipliers=[1.0,2.0].

## Edge Cases

- V column missing entirely -> equal-weight fallback (weights = 1.0).
- V column present but all NaN or all <= 0 -> equal-weight fallback.
- With mixed volumes, switching to volume weighting affects the bar that first has positive volume and subsequent bars. Earlier equal-weight values stay unchanged.
- The first bar always opens a session.
- vwap_variance is clipped to a lower bound of 0 before sqrt to guard against tiny negative values from floating-point error.
- Volume-weighted moments are used only when cumulative weight is positive. Normalized features use safe_divide.
- Non-integer band multipliers (e.g. 1.5) produce column labels like vwap_upper_1_5 / vwap_lower_1_5 rather than vwap_upper_1.5.
- Rolling z-score columns can be NaN at the very start of the series until min_periods = max(window // 2, 2) observations are available, and after the one-bar lookahead shift the first valid value moves one row later.

## Assumptions

- df.index is a DatetimeIndex (naive = UTC), required to derive session boundaries.
- Columns 'H', 'L', 'C' are always present; 'V' is optional.
- shift_features shifts every produced feature column by exactly one bar to enforce the no-lookahead invariant.
- safe_divide returns NaN for zero or near-zero denominators.
