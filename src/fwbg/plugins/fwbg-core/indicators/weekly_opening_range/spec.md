# Plugin Spec — weekly_opening_range

**Kind**: indicator  •  **Version**: 1.1.0

## Capability

Computes weekly opening range features (range, position, breakout flags, ATR-normalized distances, SL distance) plus rolling stats over past weeks' opening ranges.

## Summary

Weekly Opening Range (WOR) intraday indicator: derives per-bar features relative to the opening range defined by the first `range_bars` bar periods of each trading week (the week starts at its first bar: Sunday evening for FX, Monday for exchanges; the week key is the ISO week of `timestamp + 12h`), and — when enabled — rolling statistics (avg range, range-vs-avg, breakout rate) over the previous `stat_window` weeks. Supports single or multiple `range_bars` values (emits a `wor_rb{N}_` column set per value). Skips daily/weekly timeframes (returns df unchanged when the median bar spacing is >= 20 hours).

## Inputs

- OHLC DataFrame with pandas DatetimeIndex
- Column H (high)
- Column L (low)
- Column C (close)

## Parameters

- `range_bars` (list[int], default=2): Number of bars defining the Weekly Opening Range. On M15: 2 = first 30 min of the trading week (scalping default), 4 = first hour (more stable, fewer trades). Accepts a list (e.g. [2, 4]) to emit multiple wor_rb{N}_ column sets.
- `atr_period` (int, default=14): ATR period used to normalize WOR size, distance-to-boundary features, and the reload zone.
- `stat_window` (int, default=12): Rolling window (in weeks) for statistic features. 12 ≈ 3 months, 26 ≈ 6 months. Only past weeks contribute (shifted by 1 to avoid lookahead).
- `enable_stats` (bool, default=True): Enable historical statistics features (wor_stat_avg_range, wor_stat_range_vs_avg, wor_stat_breakout_rate).

## Outputs

- wor_rb{N}_range — WOR high-low range normalized by close
- wor_rb{N}_position — close position within the WOR (0=low, 1=high)
- wor_rb{N}_breakout_up — 1 if close > WOR high, else 0 (NaN before WOR established)
- wor_rb{N}_breakout_down — 1 if close < WOR low, else 0 (NaN before WOR established)
- wor_rb{N}_range_vs_atr — WOR range normalized by ATR
- wor_rb{N}_dist_to_high — signed ATR-normalized distance (or_high - close)/atr
- wor_rb{N}_dist_to_low — signed ATR-normalized distance (close - or_low)/atr
- wor_rb{N}_sl_dist — full WOR range, used as SL distance for orb_based exit strategy
- wor_stat_avg_range — rolling mean of first-bar week ranges over past `stat_window` weeks (when enable_stats)
- wor_stat_range_vs_avg — current week's first-bar range vs rolling average (when enable_stats)
- wor_stat_breakout_rate — fraction of past `stat_window` weeks whose price broke out of the first-bar range (when enable_stats)

## Acceptance Criteria

- AC-001: All emitted feature columns are shifted by one bar via `shift_features` before being concatenated to the returned DataFrame (no lookahead).
- AC-002: For each `rb` in `range_bars`, emits exactly the 8 columns: wor_rb{rb}_{range, position, breakout_up, breakout_down, range_vs_atr, dist_to_high, dist_to_low, sl_dist}.
- AC-003: WOR high/low per week are computed as the max H / min L over the time window `[first bar of the trading week, + range_bars * bar_step)` (a missing bar shrinks the range); features are NaN until that window has ended. Sunday-evening FX bars belong to the week that starts on Monday, not to the previous ISO week.
- AC-004: When `enable_stats=True`, adds wor_stat_avg_range, wor_stat_range_vs_avg, wor_stat_breakout_rate whose rolling window uses `stat_window` past weeks and is shifted by 1 week (excludes current week). The weekly range used by the statistics is the same `range_bars` window as the features (the first value when `range_bars` is a list); the breakout flag looks at closes after that window.
- AC-005: get_signal_columns() returns an empty list: the `_breakout_up` / `_breakout_down` columns are shifted like every feature.
- AC-006: `get_feature_columns()` returns the cached list after `compute()` runs; when called with explicit `params` (or before compute), they reconstruct the column names from `range_bars` and `enable_stats`.
- AC-007: Raises `ValueError` when the input DataFrame does not have a `DatetimeIndex`.
- AC-008: Returns the input DataFrame unchanged when the median index spacing is >= 20 hours (daily / weekly bars).
- AC-009: Uses `safe_divide` for all normalizations to avoid division-by-zero.
- AC-010: Registered under the name `weekly_opening_range` via `@register_indicator`.

## Edge Cases

- DataFrame without a DatetimeIndex → raises ValueError.
- Daily or weekly bars (median index diff >= 20h) → returns the input DataFrame unchanged, adding no features.
- Bars inside the range window of the week → all wor_rb{N}_* features are NaN for that bar.
- First week(s) in the data → wor_stat_* features are NaN until the rolling window has at least `max(1, stat_window // 2)` prior weeks.
- `range_bars` passed as an int vs a list → both are accepted; a list produces one column set per element.
- Zero-width opening range (or_high == or_low) → safe_divide yields NaN/0 instead of raising; `wor_position` becomes NaN.
- DataFrame of length <= 1 → the daily/weekly guard is skipped; features are computed but the opening range validity mask filters out all rows.
- `enable_stats=False` → the three wor_stat_* columns are not emitted.

## Assumptions

- Input DataFrame has columns H, L, C (OHLC convention used elsewhere in the SDK).
- The DatetimeIndex is timezone-consistent; the 12-hour rollover assumes the weekend gap contains Saturday (data on UTC or exchange time).
- Intraday timeframe (M1–H4) as documented; behavior on exotic spacings (e.g. 12h bars) is untested but the 20h guard treats them as intraday.

## Needs Clarification

- [NEEDS CLARIFICATION: Docstring mentions a 'Reload-Zone' feature (price returning to the WOR boundary after breakout), but no dedicated reload-zone column is emitted by the current implementation — is this intentional or a missing feature?]
