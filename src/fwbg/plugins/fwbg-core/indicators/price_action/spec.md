# Plugin Spec — price_action

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes candle-structure, trend-structure (HH/LL/HL/LH), gap, streak, range-expansion, inside/outside-bar and optional volume features from OHLC(V) bars.

## Summary

Price-action indicator that derives a family of features directly from OHLC bars: candle body/shadow ratios and direction, Higher-Highs / Lower-Lows / Higher-Lows / Lower-Highs rolling counters and a combined trend-structure score, gap size / direction / fill flag, consecutive bullish/bearish streaks, range expansion vs a 20-bar mean, inside- and outside-bar flags, and — when a volume column (V or Volume) is present and compute_volume is True — OBV change over 5 bars, Money Flow Index, relative volume vs a 20-bar mean, and a volume-weighted body-direction "price trend". All features are shifted by one bar before being returned to prevent lookahead bias.

## Inputs

- df: pandas DataFrame with OHLC columns O, H, L, C (and optionally a volume column named V or Volume)

## Parameters

- `hh_ll_period` (int, default=5): Rolling window for Higher Highs / Lower Lows / Higher Lows / Lower Highs counters and trend structure score. Shorter values capture micro-structure, longer values capture swing structure.
- `compute_volume` (bool, default=True): Whether to compute volume-based features (OBV change, MFI, relative volume, volume-price trend). Requires a volume column (V or Volume) in the DataFrame; features are skipped if the column is absent.

## Outputs

- pa_range_pos
- pa_body_ratio
- pa_body_dir
- pa_upper_shadow
- pa_lower_shadow
- pa_hh
- pa_ll
- pa_hl
- pa_lh
- pa_trend_structure
- pa_gap
- pa_gap_abs
- pa_gap_dir
- pa_gap_filled
- pa_bullish_streak
- pa_bearish_streak
- pa_range_expansion
- pa_inside_bar
- pa_outside_bar
- vol_obv_change
- vol_mfi
- vol_relative
- vol_price_trend

## Acceptance Criteria

- AC-001: pa_range_pos equals (C - L) / (H - L) via safe_divide, with 0 when H == L.
- AC-002: pa_body_ratio equals |C - O| / (H - L) via safe_divide.
- AC-003: pa_body_dir equals sign(C - O): +1 bullish, -1 bearish, 0 when C == O.
- AC-004: pa_upper_shadow equals (H - max(C, O)) / (H - L) and pa_lower_shadow equals (min(C, O) - L) / (H - L), both via safe_divide.
- AC-005: pa_hh, pa_ll, pa_hl, pa_lh are rolling sums over hh_ll_period of the boolean events H>H[-1], L<L[-1], L>L[-1], H<H[-1] respectively.
- AC-006: pa_trend_structure equals (pa_hh + pa_hl) - (pa_ll + pa_lh).
- AC-007: pa_gap equals (O - C[-1]) / C[-1] via safe_divide; pa_gap_abs equals |pa_gap|.
- AC-008: pa_gap_dir equals +1 when pa_gap > 0.001, -1 when pa_gap < -0.001, else 0.
- AC-009: pa_gap_filled equals 1 when an up-gap bar's L <= previous close or a down-gap bar's H >= previous close, else 0 (and 0 when there is no significant gap).
- AC-010: pa_bullish_streak counts consecutive bars with C > O (0 on non-bullish bars); pa_bearish_streak is computed with the same grouping logic on bearish bars and is masked by the bullish flag (existing behaviour of the code).
- AC-011: pa_range_expansion equals current bar range divided by its 20-bar rolling mean.
- AC-012: pa_inside_bar equals 1 iff H < H[-1] AND L > L[-1]; pa_outside_bar equals 1 iff H > H[-1] AND L < L[-1].
- AC-013: When compute_volume is True and a V or Volume column exists, vol_obv_change is (OBV - OBV[-5]) / OBV[-5], vol_mfi is ta.volume.money_flow_index(H, L, C, V), vol_relative is V / V.rolling(20).mean(), and vol_price_trend equals pa_body_dir * vol_relative.
- AC-014: When compute_volume is False or no volume column is present, the vol_* columns are not added by compute().
- AC-015: All feature columns are shifted by one bar (shift_features) before being concatenated with df, so row i features only depend on data up to row i-1.
- AC-016: get_feature_columns() always lists all 23 feature names including the four vol_* columns (vol_obv_change, vol_mfi, vol_relative, vol_price_trend), regardless of compute_volume or whether a volume column is present; callers must not assume vol_* columns exist in the output DataFrame when compute_volume is False or no volume column is available (see AC-014 and volume edge case). get_signal_columns() returns pa_body_dir, pa_gap_dir, pa_gap_filled, pa_inside_bar, pa_outside_bar.
- AC-017: get_default_params() returns {'hh_ll_period': 5, 'compute_volume': True}.

## Edge Cases

- Doji bar with H == L: safe_divide keeps pa_range_pos, pa_body_ratio, pa_upper_shadow, pa_lower_shadow finite (0) instead of NaN/inf.
- First bar has no previous close, so pa_gap, pa_gap_abs, pa_gap_dir, pa_gap_filled, and HH/LL/HL/LH shift-based features are NaN/0 for the initial rows before being shifted again.
- First hh_ll_period bars produce NaN for pa_hh/pa_ll/pa_hl/pa_lh (rolling sum warmup) and consequently for pa_trend_structure.
- First 20 bars produce NaN for pa_range_expansion and (when volume is used) vol_relative due to rolling(20).mean() warmup.
- Previous close of zero would make gap and vol_obv_change divisions unsafe; safe_divide guards these to avoid inf/NaN.
- compute_volume=True but no V/Volume column present: volume features are silently skipped and their columns are absent from the returned DataFrame (though still listed in get_feature_columns()).
- Flat bar with C == O: pa_body_dir is 0, so it counts as neither bullish nor bearish for streaks (both streaks reset via the groupby of (flag != flag.shift()).cumsum()).
- Gap exactly at the 0.001 threshold: treated as no significant gap (strict > / <).
- Because every feature is shifted by 1 after computation, the very last row's features are set to the pre-shift values of the previous-to-last bar's computation — i.e. no future information leaks into any row.

## Assumptions

- Input DataFrame uses uppercase single-letter OHLC column names O, H, L, C.
- Volume column, if provided, is named either V or Volume.
- shift_features shifts every feature by exactly one bar and returns a DataFrame aligned to df.index.
- safe_divide returns 0 (or a defined finite value) when the denominator is zero, matching the fwbg_sdk convention.

## Needs Clarification

- [NEEDS CLARIFICATION: pa_bearish_streak is computed from the bearish groupby but then multiplied by `bullish` rather than `bearish` in the source — this appears to zero out bearish streaks on bearish bars. Confirm whether this is the intended behaviour or a bug to be fixed in a later revision (spec currently documents the code as-written).]
- [NEEDS CLARIFICATION: get_feature_columns() unconditionally lists the four vol_* columns, but compute() only produces them when compute_volume is True and a volume column exists. Confirm whether the declared feature set should reflect the runtime-dependent columns.]
