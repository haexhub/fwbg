# Plugin Spec — ichimoku

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes Ichimoku Cloud components (Tenkan, Kijun, Senkou A/B) and derived cloud position, TK-cross, kumo-twist, and composite bullish/bearish/neutral signals.

## Summary

Full Ichimoku Kinko Hyo feature bundle derived from ta.trend.IchimokuIndicator: base lines (Tenkan/Kijun/Senkou A/Senkou B), cloud position/thickness, above/below/in-cloud flags, TK cross magnitude and bullish/bearish cross events, price-to-Kijun distance, Kijun-flat flag, cloud color (bullish/bearish), Kumo twist detection, Chikou-above flag, distance to cloud, and composite strong-bullish/strong-bearish/neutral signals. All feature columns are shifted by one bar to prevent lookahead bias.

## Inputs

- df[H]
- df[L]
- df[C]

## Parameters

- `tenkan_period` (int, default=9): Tenkan-sen (Conversion Line) window; midpoint of high/low over this many bars.
- `kijun_period` (int, default=26): Kijun-sen (Base Line) window; also used as the Chikou-span lag for ichi_chikou_above.
- `senkou_b_period` (int, default=52): Senkou Span B window; midpoint of high/low over this many bars, projected forward for the cloud.

## Outputs

- ichi_tenkan
- ichi_kijun
- ichi_senkou_a
- ichi_senkou_b
- ichi_cloud_thick
- ichi_cloud_pos
- ichi_above_cloud
- ichi_below_cloud
- ichi_in_cloud
- ichi_tk_cross
- ichi_tk_bullish_cross
- ichi_tk_bearish_cross
- ichi_price_kijun
- ichi_kijun_flat
- ichi_bullish_cloud
- ichi_kumo_twist
- ichi_chikou_above
- ichi_strong_bullish
- ichi_strong_bearish
- ichi_neutral
- ichi_dist_to_cloud

## Acceptance Criteria

- AC-001: compute() returns the original df concatenated with all 21 columns listed in get_feature_columns().
- AC-002: ichi_tenkan, ichi_kijun, ichi_senkou_a, ichi_senkou_b come from ta.trend.IchimokuIndicator using window1=tenkan_period, window2=kijun_period, window3=senkou_b_period on df['H']/df['L'].
- AC-003: ichi_cloud_thick = safe_divide(cloud_top - cloud_bottom, df['C']) where cloud_top/cloud_bottom are the elementwise max/min of Senkou A and B.
- AC-004: ichi_cloud_pos = safe_divide(C - cloud_bottom, cloud_top - cloud_bottom).
- AC-005: ichi_above_cloud == 1 iff C > cloud_top; ichi_below_cloud == 1 iff C < cloud_bottom; ichi_in_cloud == 1 iff cloud_bottom <= C <= cloud_top.
- AC-006: ichi_tk_cross = safe_divide(tenkan - kijun, df['C']); ichi_tk_bullish_cross fires the bar tenkan crosses above kijun; ichi_tk_bearish_cross fires the bar it crosses below.
- AC-007: ichi_price_kijun = safe_divide(C - kijun, C).
- AC-008: ichi_kijun_flat == 1 when |kijun.diff()| < 0.0001 * C.
- AC-009: ichi_bullish_cloud == 1 iff Senkou A > Senkou B; ichi_kumo_twist == 1 on bars where that relationship flips vs. the previous bar.
- AC-010: ichi_chikou_above == 1 iff C > C.shift(kijun_period).
- AC-011: ichi_strong_bullish == 1 iff above_cloud & tk_cross>0 & bullish_cloud; ichi_strong_bearish == 1 iff below_cloud & tk_cross<0 & ~bullish_cloud.
- AC-012: ichi_dist_to_cloud = (C - cloud_bottom)/C above the cloud, -(cloud_top - C)/C below the cloud, else 0.
- AC-013: All feature columns are shifted by one bar via shift_features() before being concatenated to df (no-lookahead).

## Edge Cases

- Warm-up bars where any of Tenkan/Kijun/Senkou A/Senkou B are NaN — dependent features (cloud_pos, tk_cross, price_kijun, dist_to_cloud) inherit NaNs from the ta indicator; boolean/int flags derived via `.astype(int)` coerce NaN comparisons to 0.
- Zero-thickness cloud (cloud_top == cloud_bottom) — ichi_cloud_pos denominator is 0 and is handled by safe_divide.
- Zero or NaN close price — all safe_divide normalisations (cloud_thick, tk_cross, price_kijun, dist_to_cloud) fall back to the safe_divide default rather than dividing by zero.
- First bar for shift-based features (tk_bullish_prev, cloud_bullish_prev) — .shift(1).fillna(False) treats the missing prior state as bearish, so a bar starting bullish counts as a tk_bullish_cross / kumo_twist.
- Fewer than kijun_period bars — ichi_chikou_above compares against C.shift(kijun_period) which is NaN, yielding 0 after astype(int).

## Assumptions

- df has uppercase OHLC columns 'H', 'L', 'C' (no 'O' is consumed).
- df has a monotonic bar index compatible with shift_features().
- ta (ta-lib python wrapper) is available and ta.trend.IchimokuIndicator behaves per its documented windows.

## Needs Clarification

- [NEEDS CLARIFICATION: Senkou A and Senkou B are used as returned by ta.trend.IchimokuIndicator (i.e. not displaced forward on the time axis) — confirm whether the intended cloud semantics require the +kijun_period displacement that the classic Ichimoku definition uses, since ichi_above_cloud/ichi_cloud_pos here compare the current close against the non-displaced cloud.]
