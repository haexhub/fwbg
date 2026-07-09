# Plugin Spec — efficiency_ratio

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes Kaufman's Efficiency Ratio (net price change divided by summed absolute bar-to-bar changes) and its change over half the period for each configured lookback.

## Summary

Kaufman's Efficiency Ratio indicator that measures trend quality on close prices. For each period N, produces `er_{N}` in [0,1] (near 1 = clean trend, near 0 = choppy/sideways) and `er_{N}_chg` which is the ER minus its value N//2 bars ago. Uses `safe_divide` for the ratio and `shift_features` to shift all outputs by one bar to prevent lookahead.

## Inputs

- df['C'] (close price series)

## Parameters

- `periods` (list[int], default=[10, 20, 50]): Lookback periods for Efficiency Ratio. ER near 1 = clean trend, near 0 = choppy sideways. Change features (er_N_chg) measure ER momentum over half the period.

## Outputs

- er_{period} — Efficiency Ratio in [0,1] for each configured period
- er_{period}_chg — change in ER over period//2 bars

## Acceptance Criteria

- AC-001: For each period in `periods`, produces exactly two feature columns: `er_{period}` and `er_{period}_chg`.
- AC-002: `er_{period}` is computed as |C - C.shift(period)| / rolling(period).sum(|C.diff()|) using `safe_divide` for zero-volatility protection.
- AC-003: `er_{period}_chg` equals `er_{period} - er_{period}.shift(period // 2)`.
- AC-004: All feature columns are passed through `shift_features(...)` before being concatenated to the input DataFrame, enforcing the no-lookahead invariant.
- AC-005: `get_feature_columns` returns the same column names produced by `compute()` for the resolved `periods` parameter.
- AC-006: `get_signal_columns` and `get_overlay_columns` return empty lists.
- AC-007: Default `periods` is `[10, 20, 50]` when not supplied (both when `periods=None` in `compute` and via `get_default_params`).
- AC-008: Column group label maps the `er` prefix to `"Efficiency Ratio (Kaufman)"`.

## Edge Cases

- First `period` bars have insufficient history for the rolling window and yield NaN for `er_{period}`.
- First `period + period//2` bars yield NaN for `er_{period}_chg` due to the additional shift.
- Zero total volatility over the window (e.g., a flat constant-price series) is handled by `safe_divide` rather than raising ZeroDivisionError.
- Passing `periods=None` explicitly falls back to the default `[10, 20, 50]`.
- A single-element `periods` list produces exactly two feature columns.

## Assumptions

- Input DataFrame contains a `C` (close) column.
- The DataFrame index is suitable for `shift_features` (typically a monotonic time index).
