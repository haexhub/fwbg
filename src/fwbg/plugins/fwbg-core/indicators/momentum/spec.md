# Plugin Spec — momentum

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes a bundle of momentum oscillator features (RSI, Stochastic K/D, Williams %R, Ultimate Oscillator, Rate of Change) at multiple lookback periods from OHLC price data.

## Summary

Momentum indicator plugin that appends 13 momentum-oscillator feature columns to an OHLC DataFrame: RSI at 3 configurable periods, Stochastic %K and %D at 2 configurable periods, Williams %R at 2 configurable periods, a single Ultimate Oscillator, and Rate of Change at 3 configurable periods. All features are shifted by one bar via shift_features to prevent lookahead bias, and the original DataFrame is returned with the new columns concatenated.

## Inputs

- OHLC DataFrame with columns O, H, L, C (Stochastic/Williams/Ultimate Oscillator use H, L, C; RSI and ROC use only C)

## Parameters

- `rsi_periods` (list[int], default=[7, 14, 21]): Periods for RSI (Relative Strength Index) calculation. RSI oscillates 0-100 measuring the speed and magnitude of price changes. 14 is the classic Wilder period; shorter periods (7) are more sensitive to recent moves, longer periods (21) smoother.
- `stoch_periods` (list[int], default=[14, 21]): Lookback periods for Stochastic Oscillator (%K and %D). Measures where price closed relative to its high-low range over N bars. Shorter periods react faster to price swings, longer periods filter out noise.
- `williams_periods` (list[int], default=[14, 21]): Lookback periods for Williams %R. Similar to Stochastic but inverted (0 to -100 scale). Identifies overbought (near 0) and oversold (near -100) conditions within the given lookback window.
- `roc_periods` (list[int], default=[5, 10, 20]): Periods for Rate of Change (ROC) calculation. Measures the percentage change in price over N bars. Short periods (5) capture immediate momentum, longer periods (20) capture swing momentum.

## Outputs

- mom_rsi_7
- mom_rsi_14
- mom_rsi_21
- mom_stoch_k_14
- mom_stoch_d_14
- mom_stoch_k_21
- mom_stoch_d_21
- mom_williams_14
- mom_williams_21
- mom_uo
- mom_roc_5
- mom_roc_10
- mom_roc_20

## Acceptance Criteria

- AC-001: compute() returns the input DataFrame with 13 additional feature columns appended: mom_rsi_{7,14,21}, mom_stoch_k_{14,21}, mom_stoch_d_{14,21}, mom_williams_{14,21}, mom_uo, mom_roc_{5,10,20}.
- AC-002: RSI columns are computed via ta.momentum.rsi on df['C'] with window=period for each period in rsi_periods.
- AC-003: Stochastic %K and %D columns are computed via ta.momentum.StochasticOscillator on (H, L, C) with window=period for each period in stoch_periods.
- AC-004: Williams %R columns are computed via ta.momentum.williams_r on (H, L, C) with lbp=period for each period in williams_periods.
- AC-005: mom_uo is computed via ta.momentum.ultimate_oscillator on (H, L, C) using library defaults (single column, not parameterized).
- AC-006: ROC columns are computed via ta.momentum.roc on df['C'] with window=period for each period in roc_periods.
- AC-007: All computed feature columns are passed through shift_features(..., df.index) before being concatenated onto the returned DataFrame, so feature values at bar i reflect data available strictly before bar i (no lookahead).
- AC-008: get_feature_columns() returns the fixed list of 13 default column names regardless of the params actually passed to compute().
- AC-009: get_default_params() returns {'rsi_periods': [7,14,21], 'stoch_periods': [14,21], 'williams_periods': [14,21], 'roc_periods': [5,10,20]}.
- AC-010: When any of rsi_periods, stoch_periods, williams_periods, or roc_periods is passed as None, compute() substitutes the corresponding default list.
- AC-011: The plugin registers under the name 'momentum' via @register_indicator and exposes name='momentum', version='2.0.0'.

## Edge Cases

- Any of the period-list parameters passed explicitly as None is replaced with its default list inside compute().
- Non-default period lists change which feature columns are actually produced by compute(), but get_feature_columns() still returns the fixed 13-name default list — mismatch between produced columns and declared feature columns when non-default periods are used.
- Early rows (index < max lookback period) of each oscillator column will contain NaN values produced by the underlying ta library.
- The one-bar shift via shift_features means the first row of every feature column is NaN even after the ta library warmup, and the last raw computed value is dropped from the shifted output.
- The Ultimate Oscillator is emitted as a single mom_uo column and is not parameterized by any of the exposed period lists.
- compute() requires df to contain columns 'O', 'H', 'L', 'C' — missing H/L/C (used by Stochastic, Williams %R, Ultimate Oscillator) or missing C (used by RSI, ROC) will raise from the ta library.

## Assumptions

- Input DataFrame uses uppercase OHLC column names ('O', 'H', 'L', 'C') as consumed by compute().
- shift_features(features_dict, index) returns a DataFrame of the same feature columns shifted by one bar and aligned to the provided index (used here to enforce the no-lookahead invariant required for indicators).
- The ta library's ultimate_oscillator default short/medium/long windows (7/14/28) are the intended behaviour, since they are not exposed as parameters.

## Needs Clarification

- [NEEDS CLARIFICATION: get_feature_columns() returns a hard-coded 13-name list matching only the default period configuration; whether this should dynamically reflect the actual params (or whether callers are expected to always use defaults for column-name purposes) is not stated in the source.]
- [NEEDS CLARIFICATION: Ultimate Oscillator's short/medium/long windows are not exposed as params — whether they were intentionally fixed at ta-library defaults or simply not surfaced yet is not stated.]
