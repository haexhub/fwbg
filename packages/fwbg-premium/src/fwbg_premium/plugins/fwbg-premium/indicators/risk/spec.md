# Plugin Spec — risk

**Kind**: indicator  •  **Version**: 2.0.0

## Capability

Computes risk/tail-risk features including drawdown state, VaR/CVaR, vol-of-vol, a composite crash-probability proxy, and optional SPX/VIX correlation features.

## Summary

Risk indicator producing drawdown depth/duration/recovery, rolling VaR and CVaR at configurable percentiles/windows, ATR-based vol-of-vol with z-score and trend, a composite crash-probability score (kurtosis + vol-of-vol + correlation decoupling + CVaR), and — when macro columns are present — SPX correlation/beta/lead-lag and VIX correlation/lead signals. All emitted features are shifted by one bar to prevent lookahead bias.

## Inputs

- df['C'] (close price)
- df['H'] (high price)
- df['L'] (low price)
- df['macro_spx'] (optional, enables SPX correlation/beta/lead-lag features)
- df['macro_vix'] or df['sent_vix'] (optional, enables VIX correlation/lead-lag features)
- df['dist_kurt_50'] and optionally df['dist_kurt_50_z'] (optional, contributes to crash probability)

## Parameters

- `dd_windows` (list[int], default=[50, 100, 200]): Rolling windows used for drawdown percentage and drawdown ratio; time-since-peak and recovery features require 200 to be present.
- `cvar_windows` (list[int], default=[50, 100]): Rolling windows for VaR and CVaR (Expected Shortfall) computation on returns.
- `cvar_percentiles` (list[int], default=[5, 1]): Tail percentiles (in percent) at which VaR/CVaR are evaluated; tail-ratio and 5%-change features require both 1 and 5 to be present.
- `vov_windows` (list[int], default=[20, 50, 100]): Rolling windows for vol-of-vol (std of ATR percent change); z-score requires 100, trend requires 50.
- `compute_correlations` (bool, default=True): When True and macro columns are available, computes SPX and VIX correlation, beta, and lead-lag features.

## Outputs

- risk_dd_pct_{w} for each w in dd_windows
- risk_dd_ratio_{w} for each w in dd_windows
- risk_bars_since_peak, risk_bars_since_peak_log, risk_recovery_ratio (only when 200 in dd_windows)
- risk_var_{p}_{w} and risk_cvar_{p}_{w} for each p in cvar_percentiles and w in cvar_windows
- risk_cvar_tail_ratio, risk_cvar_5_change (only when 100 in cvar_windows and both 1 and 5 in cvar_percentiles)
- risk_vol_of_vol_{w} for each w in vov_windows
- risk_vol_of_vol_zscore (only when 100 in vov_windows)
- risk_vol_of_vol_trend (only when 50 in vov_windows)
- risk_crash_probability, risk_crash_prob_change, risk_crash_regime
- corr_spx_{20,50,100}, corr_spx_stability, corr_spx_decoupling, lead_lag_spx, beta_spx_{50,100} (only when macro_spx present and compute_correlations=True)
- corr_vix_{20,50}, lead_lag_vix, vix_lead_signal (only when macro_vix or sent_vix present and compute_correlations=True)

## Acceptance Criteria

- AC-001: For each w in dd_windows, produces risk_dd_pct_{w} = (close - rolling_max_w) / rolling_max_w and risk_dd_ratio_{w} = dd_pct / rolling_min(dd_pct, w) via safe_divide.
- AC-002: When 200 is in dd_windows, produces risk_bars_since_peak (bars since last new 200-bar high), its log1p, and risk_recovery_ratio clipped to [0, 1]. Because the rolling max uses min_periods=1, the very first bar always qualifies as a peak (rolling_max_200[0] == close[0]), so _bars_since_event sets only that first bar to NaN. In practice, risk_bars_since_peak is NaN only for the first row and begins counting from 1 at the second row (before the 1-bar shift; after the shift, the first two rows are NaN). It does NOT remain NaN until a full 200-bar window has elapsed.
- AC-003: For each (percentile p, window w) in cvar_percentiles × cvar_windows, produces risk_var_{p}_{w} as a rolling quantile of returns and risk_cvar_{p}_{w} as the mean of returns in the lower p-percent tail (NaN when window has fewer than 10 observations).
- AC-004: When 100 is in cvar_windows and both 1 and 5 are in cvar_percentiles, produces risk_cvar_tail_ratio = cvar_1_100 / cvar_5_100 and risk_cvar_5_change = cvar_5_100 - cvar_5_100.shift(20).
- AC-005: For each w in vov_windows, produces risk_vol_of_vol_{w} as the rolling std over w of ATR(14).pct_change(); when 100 is present, adds risk_vol_of_vol_zscore against a 200-bar mean/std; when 50 is present, adds risk_vol_of_vol_trend = vov_50 - vov_50.shift(10).
- AC-006: Always emits risk_crash_probability (average of available components: kurtosis z-score clipped to [0,3]/3, vol-of-vol z-score clipped to [0,3]/3, correlation decoupling normalized/clipped to [0,1], and negated CVaR z-score clipped to [0,3]/3), risk_crash_prob_change = crash_prob - crash_prob.shift(10), and risk_crash_regime = int(crash_prob > 0.6); returns a zero series when no components are available.
- AC-007: When compute_correlations=True and df['macro_spx'] is present, emits corr_spx_{20,50,100}, corr_spx_stability (corr_spx_50 - shift(10)), corr_spx_decoupling (abs of stability), lead_lag_spx (20-bar momentum diff), and beta_spx_{50,100} = cov(asset_ret, spx_ret) / var(spx_ret) via safe_divide.
- AC-008: When compute_correlations=True and df['macro_vix'] or df['sent_vix'] is present (macro_vix preferred), emits corr_vix_{20,50}, lead_lag_vix = vix.pct_change(10) + close.pct_change(10), and vix_lead_signal = vix.pct_change(10).shift(5) * 100.
- AC-009: All columns whose names start with 'risk_', 'corr_', 'lead_lag_', or 'vix_lead_' are shifted by one bar before the DataFrame is returned so features at bar i reflect information available at bar i-1.

## Edge Cases

- Rolling CVaR windows with fewer than 10 valid observations return NaN (calc_cvar early-exits on len(x) < 10).
- risk_bars_since_peak is NaN only for the first row (before the output shift), because the rolling max uses min_periods=1 and close[0] >= rolling_max_200[0] is always True — the first bar is always a peak. After the 1-bar output shift, the first two rows are NaN. This is not a 200-bar warm-up; counting begins immediately from the second bar.
- If df['C'] never sets a new 200-bar high after the first bar (i.e. the series is monotonically declining), bars_since_peak still starts counting from the first bar's event and increases indefinitely.
- Recovery ratio is explicitly clipped to [0, 1] to guard against division artifacts.
- safe_divide is used for every ratio (dd_ratio, cvar_tail_ratio, vov_zscore, beta, decoupling normalization, crash-component z-scores) to avoid divide-by-zero.
- Crash probability gracefully skips components whose source columns are missing; if no components are available, it returns a constant 0.0 series (and crash_regime stays 0).
- Correlation block is fully skipped when compute_correlations=False; SPX sub-block requires df['macro_spx']; VIX sub-block accepts macro_vix or falls back to sent_vix, and is skipped when neither is present.
- The kurtosis component of crash probability prefers a pre-computed dist_kurt_50_z column when available and otherwise computes a 100-bar z-score of dist_kurt_50; if dist_kurt_50 is absent, the kurtosis component is skipped entirely.
- The one-bar output shift means the first row of every emitted feature column is NaN even when the underlying rolling window would have produced a value.

## Assumptions

- Input DataFrame has OHLC columns named 'C', 'H', 'L' (uppercase single-letter convention used throughout fwbg).
- Returns are computed as simple pct_change on close; no dividend/adjustment handling is performed here.
- The plugin does not itself register or require the presence of the distribution or macro plugins — it simply consumes their columns opportunistically.

## Needs Clarification

- [NEEDS CLARIFICATION: cvar_percentiles semantics: values are treated as percent (e.g. 5 → 5th percentile) via percentile/100; whether values outside (0, 100) should be validated is unspecified.]
- [NEEDS CLARIFICATION: Whether emitting VaR/CVaR/vol-of-vol features whose auxiliary columns (tail_ratio, zscore, trend) require specific window/percentile values to be present is intentional gating or should raise on misconfiguration is not documented.]
