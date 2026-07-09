# Plugin Spec — macro_data

**Kind**: data_loading  •  **Version**: 1.0.0

## Capability

Computes derived macro feature columns (hourly/daily pct-change lookbacks, yield-curve/ratio spreads, and interest-rate differentials) from macro_* base columns already present in ctx.df.

## Summary

Pure-computation data_loading plugin that transforms already-loaded macro_* base columns into a wide feature set: hourly and daily pct_change lookbacks for every macro base column, configurable derived spreads/ratios (yield curves, risk ratios, credit spread proxy, international yield spreads), daily lookbacks on those derived spreads, and interest-rate differentials. Mutates ctx.df in place and returns ctx.

## Inputs

- ctx.df with macro_* base columns already loaded (e.g. macro_vix, macro_tnx, macro_us2y, macro_fed_rate, macro_ecb_rate, ...)

## Parameters

- `indicators` (object, default={VIX_DAY: vix, VVIX_DAY: vvix, SKEW_DAY: skew, ...}): Mapping of macro data file stems to column prefixes (e.g. VIX_DAY -> vix). Default dict covers volatility (VIX/VVIX/SKEW/VXN), US yields (TNX/TYX/FVX/IRX/US2Y/US5Y/US30Y), FX (DXY), commodities (gold/oil/silver), US equity indices (SPX/NASDAQ/DOW/RUSSELL), international indices (NIKKEI/HANGSENG/FTSE/DAX), sectors (XLF/XLE/XLK/XLU/XLP), bonds (TLT/HYG/LQD), and international 10Y yields (DE/JP/GB/AU). Used by get_feature_columns to enumerate expected output columns. Note: get_param_schema() declares this as type 'string' — that is a source bug; the actual runtime type is a dict/object.
- `lookbacks_hours` (list[int], default=[1, 2, 4, 8, 12, 24]): Hourly lookback periods (in bars) for pct_change features on each macro_* base column.
- `lookbacks_days` (list[int], default=[2, 5, 10, 20, 60]): Daily lookback periods for pct_change features on macro base and derived columns; converted to bars as 24*lb.
- `derived_features` (list[object], default=[{name, op, a, b}, ...]): List of derived feature specs, each with name, op ('subtract' or 'ratio'), and two source columns a and b. Default set: yield curves (10y-3m/10y-5y/10y-2y/30y-5y), ratios (vix/vvix, spx/tlt, hyg/lqd credit spread proxy, russell/spx smallcap, xlk/xlu tech-defensive), and US-vs-international yield spreads (DE/JP/GB/AU). Ratio ops use a / (b + 1e-10). Note: get_param_schema() declares this as type 'string' — that is a source bug; the actual runtime type is a list of dicts.
- `interest_rates` (list[object], default=[{name, file, lookbacks_days}, ...]): Interest rate data source metadata (file name and lookback periods). Default: FED_RATE.csv and ECB_RATE.csv with lookbacks_days [30, 90, 180]. Declared but not consumed by execute() — used by the orchestrator/loader upstream. Note: get_param_schema() declares this as type 'string' — that is a source bug; the actual runtime type is a list of dicts.
- `interest_rate_diffs` (list[object], default=[{name, a, b}]): Interest-rate differential specs: produces df[a] - df[b] as a new column when both inputs are present. Default: macro_rate_diff_usd_eur = macro_fed_rate - macro_ecb_rate. Note: get_param_schema() declares this as type 'string' — that is a source bug; the actual runtime type is a list of dicts.

## Outputs

- macro_{prefix}_chg_{lb}h for each indicator prefix and each hourly lookback
- macro_{prefix}_chg_{lb}d for each indicator prefix and each daily lookback
- Derived feature columns per derived_features spec (e.g. macro_yield_curve_10y_3m, macro_vix_vvix_ratio, macro_credit_spread_proxy, macro_yield_spread_us_de, ...)
- {derived_name}_chg_{lb}d for each derived feature and each daily lookback
- Interest-rate differential columns per interest_rate_diffs spec (e.g. macro_rate_diff_usd_eur)

## Acceptance Criteria

- AC-001: Computes hourly pct_change features (multiplied by 100) for each macro_* base column at each lookback in lookbacks_hours, named {col}_chg_{lb}h
- AC-002: Computes daily pct_change features (multiplied by 100, using 24*lb bar offset) for each macro_* base column at each lookback in lookbacks_days, named {col}_chg_{lb}d
- AC-003: Identifies macro base columns as any df column starting with 'macro_' and not containing '_chg_'
- AC-004: Computes derived features per spec: 'subtract' produces a - b, 'ratio' produces a / (b + 1e-10); only if both a and b columns exist in df
- AC-005: Computes daily-lookback pct_change features for each derived feature column, named {derived_name}_chg_{lb}d
- AC-006: Computes interest rate differentials per spec as df[a] - df[b], only when both columns are present
- AC-007: Mutates ctx.df in place with the new columns and returns ctx
- AC-008: get_feature_columns returns the deterministic list of expected column names derived from indicators, lookbacks_hours, lookbacks_days, and derived_features

## Edge Cases

- Derived-feature spec is silently skipped when either input column a or b is missing from df
- Interest-rate-diff spec is silently skipped when either input column a or b is missing from df
- Ratio derived features add 1e-10 to the denominator to avoid division by zero
- Base-column detection excludes any existing column containing '_chg_', so pre-existing change columns are not re-processed as bases
- pct_change over insufficient history yields NaN for the leading rows (no explicit handling)
- Derived-feature daily-lookback change column is only added when the chg_{lb}d key is not already present in df

## Assumptions

- ctx.df is hourly-bar data (daily lookbacks assume 24 bars per day)
- macro_* base columns (including macro_fed_rate / macro_ecb_rate used by rate-diff specs) are loaded upstream by the orchestrator before this plugin runs
- Base-column detection ('macro_' prefix, no '_chg_') reliably distinguishes source columns from any previously computed change columns

## Needs Clarification

- [NEEDS CLARIFICATION: execute() does not apply the mandatory no-lookahead shift required by the constitution for feature-producing plugins — confirm whether shifting is expected to be applied downstream or is a gap in this plugin]
- [NEEDS CLARIFICATION: The 'indicators' and 'interest_rates' params are declared in get_default_params/get_param_schema but not consumed inside execute() — confirm whether that is intentional (orchestrator-facing metadata) or a latent bug]
