# Plugin Spec — regime_cluster

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Combines orthogonal regime and volatility inputs into a rolling z-scored composite score and assigns quantile-based regime cluster labels (unfavorable/neutral/favorable).

## Summary

Aggregates pre-computed regime structure signals (Hurst persistence, entropy, variance ratio, ATR percent rank, Hurst divergence, and optionally a macro risk composite) via rolling z-score normalization into an equally-weighted composite score. The score is then bucketed into n_regimes clusters using rolling quantile thresholds, producing a discrete regime label alongside the raw score, its 24-bar change, and the number of contributing inputs. All feature columns are shifted by one bar to avoid lookahead bias.

## Inputs

- regime_hurst_200
- regime_entropy_100
- regime_vr_200_5
- vol_atr_pct_14_rank
- regime_hurst_divergence
- regime_risk_composite (optional)

## Parameters

- `zscore_window` (int, default=200): Rolling window (bars) used to z-score each input series before averaging.
- `quantile_window` (int, default=500): Rolling window (bars) used to compute the quantile thresholds that map the composite score to cluster labels.
- `n_regimes` (int, default=3): Number of regime clusters (quantile bins) the composite score is discretized into; produces labels 0..n_regimes-1.

## Outputs

- regime_cluster_score
- regime_cluster_label
- regime_cluster_score_chg
- regime_cluster_n_inputs

## Acceptance Criteria

- AC-001: Emits exactly the four columns regime_cluster_score, regime_cluster_label, regime_cluster_score_chg, regime_cluster_n_inputs.
- AC-002: The composite score is the equally-weighted nanmean of rolling z-scored inputs, where each input is multiplied by its declared sign (entropy is flipped) and variance-ratio inputs (columns starting with 'regime_vr_') are centered by subtracting 1.0 before signing.
- AC-003: Only inputs whose source columns are present in the input DataFrame contribute to the score; missing inputs are silently skipped, and regime_cluster_n_inputs records the count actually used.
- AC-004: If no core or optional input columns are present, all of SCORE, LABEL, SCORE_CHG are filled with NaN and N_INPUTS is 0 for the entire frame.
- AC-005: Cluster labels are assigned by rolling quantile thresholds at k/n_regimes for k in 1..n_regimes-1 over a quantile_window window (min_periods = quantile_window // 2); rows with a valid score below the lowest threshold receive label 0.
- AC-006: regime_cluster_score_chg equals the composite score minus its value 24 bars earlier.
- AC-007: All output feature columns are shifted by one bar via shift_features to eliminate lookahead bias.
- AC-008: Rolling z-scoring uses min_periods = zscore_window // 2 and a 1e-10 denominator epsilon to avoid divide-by-zero.

## Edge Cases

- None of the CORE_INPUTS or OPTIONAL_INPUTS columns are present in df — all score/label/change outputs are NaN and n_inputs is 0.
- Only a subset of inputs are present — score is computed from what exists and n_inputs reflects the reduced count.
- Warm-up period shorter than zscore_window // 2 — rolling z-score returns NaN, propagating NaN into the score for those bars.
- Warm-up period shorter than quantile_window // 2 — cluster label remains NaN until enough history is available.
- Composite score is NaN for a bar — cluster label for that bar is left as NaN (not forced to 0).
- Constant input series producing zero rolling std — the 1e-10 epsilon prevents divide-by-zero and yields a near-zero z-score.
- Variance-ratio input columns are auto-detected by the 'regime_vr_' prefix and re-centered at 0 before signing.
- First 24 bars have NaN score_chg because the 24-bar lag is undefined.

## Assumptions

- _none_
