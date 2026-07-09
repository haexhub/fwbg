# Plugin Spec — balance_zones

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Computes ATR-normalised PBD balance-zone features (in-balance, in-zone, width, edge distances, breakouts, fake-breakout rejections, consecutive in-zone bars) from rolling candle-body extremes.

## Summary

Builds a rolling balance zone from candle body extremes (max/min of Open/Close over `lookback` bars) and emits ten ATR-normalised features describing the zone width, whether the close is inside the zone, distances to zone edges, strict breakouts above/below the previous zone, fake-breakout rejections against the two-bar-old zone, and a normalised count of consecutive in-zone bars. All features are shifted one bar to prevent lookahead.

## Inputs

- df['O']
- df['H']
- df['L']
- df['C']

## Parameters

- `lookback` (int, default=10): Rolling window (in bars) used to build the balance zone: zone_top = rolling_max(max(O,C), lookback), zone_bottom = rolling_min(min(O,C), lookback).
- `balance_atr_threshold` (float, default=2): Maximum zone_width / ATR ratio for the zone to be considered 'in balance' (sets bz_in_balance = 1).
- `atr_period` (int, default=14): ATR lookback period used to normalise width and distance features.
- `balance_bars_max` (int, default=20): Cap used to normalise bz_balance_bars: min(consecutive_in_zone, balance_bars_max) / balance_bars_max.

## Outputs

- bz_in_balance
- bz_in_zone
- bz_zone_width
- bz_zone_top_dist
- bz_zone_bottom_dist
- bz_breakout_bull
- bz_breakout_bear
- bz_fake_bear
- bz_fake_bull
- bz_balance_bars

## Acceptance Criteria

- AC-001: Registers under the name 'balance_zones' via @register_indicator and exposes the ten features listed in get_feature_columns().
- AC-002: Zone top/bottom at bar i are computed as rolling max/min of body_top/body_bottom over the last `lookback` bars with min_periods=1, and per-bar features use the zone from bar i-1 (prev_zt / prev_zb) so no bar depends on its own zone.
- AC-003: bz_zone_width[i] = (prev_zt - prev_zb) / atr_i; bz_in_balance[i] = 1 iff that width <= balance_atr_threshold, else 0.
- AC-004: bz_in_zone[i] = 1 iff prev_zb <= close[i] <= prev_zt, else 0.
- AC-005: bz_zone_top_dist[i] = max(0, (prev_zt - close[i]) / atr_i) and bz_zone_bottom_dist[i] = max(0, (close[i] - prev_zb) / atr_i), so each is 0 when price is on the 'wrong' side of that edge.
- AC-006: bz_breakout_bull[i] = 1 iff close[i] > prev_zt; bz_breakout_bear[i] = 1 iff close[i] < prev_zb (strict inequalities, mutually exclusive with in-zone).
- AC-007: Fake-breakout flags use the two-bar-old zone (i-2) and require i >= 2: bz_fake_bear[i] = 1 iff close[i-1] > old_zt AND close[i] <= old_zt; bz_fake_bull[i] = 1 iff close[i-1] < old_zb AND close[i] >= old_zb.
- AC-008: bz_balance_bars[i] = min(consecutive_in_zone, balance_bars_max) / balance_bars_max, where consecutive_in_zone increments while bz_in_zone==1 and resets to 0 otherwise; output is bounded in [0, 1].
- AC-009: ATR is a simple rolling mean of True Range with min_periods=1, with tr[0] = high[0]-low[0]; when atr_i <= EPSILON the normaliser falls back to 1.0 to avoid division blow-ups.
- AC-010: All emitted features are passed through shift_features(...) before being concatenated to df, guaranteeing no lookahead bias.
- AC-011: get_signal_columns() returns ['bz_breakout_bull', 'bz_breakout_bear', 'bz_fake_bear', 'bz_fake_bull'].
- AC-012: get_default_params() returns {'lookback': 10, 'balance_atr_threshold': 2.0, 'atr_period': 14, 'balance_bars_max': 20}, matching the compute() defaults.

## Edge Cases

- First bar (i=0): all features remain at their initialised values (zeros, or NaN for width/distances) because the compute loop starts at i=1.
- Second bar (i=1): fake_bear / fake_bull are always 0 because the i >= 2 guard prevents access to a two-bar-old zone.
- Flat ATR (atr_i <= EPSILON, e.g. a constant-price window): the code substitutes 1.0 for atr_i so zone_width and distance features are computed in raw price units without dividing by zero.
- Close exactly on a zone edge (close == prev_zt or close == prev_zb): treated as in-zone (bz_in_zone = 1) and neither breakout flag fires; the corresponding distance feature is 0.
- Zero-width zone (prev_zt == prev_zb): bz_zone_width = 0, bz_in_balance = 1 (assuming threshold > 0), and the close is in-zone only if it equals that single price.
- First `lookback`-1 bars: rolling max/min use min_periods=1 so zone edges are defined from bar 1 onward (built from fewer bars than the full window).
- Long stretches in-zone: bz_balance_bars saturates at 1.0 once consecutive_in_zone reaches balance_bars_max and stays there until the close leaves the zone.
- A single bar leaving the zone resets consecutive_in_zone to 0, so bz_balance_bars drops back to 0 immediately on that bar.

## Assumptions

- Input DataFrame contains numeric columns 'O', 'H', 'L', 'C' aligned on df.index (no explicit validation is performed in compute()).
- shift_features(features, df.index) applies a one-bar forward shift to each feature column to enforce the no-lookahead invariant required for indicators.
- EPSILON imported from fwbg_sdk is a small positive constant used only as an ATR safety floor.
