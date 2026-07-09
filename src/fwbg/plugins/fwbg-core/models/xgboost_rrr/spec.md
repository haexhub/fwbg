# Plugin Spec — xgboost_rrr

**Kind**: model  •  **Version**: 1.0.0

## Capability

Trains one XGBoost classifier over multiple reward-risk-ratio variants with RRR as an input feature, then per sample picks the RRR with the highest predicted win probability.

## Summary

Single XGBoost classifier trained across multiple RRR (reward-risk ratio) variants by adding RRR as an input feature. Each variant contributes its own OHLC-derived binary Win/Loss labels (TP = rrr * base_sl_atr * ATR, SL = base_sl_atr * ATR, entry at next-bar open, horizon 50 bars). At inference, all variants are scored and the RRR with the highest predicted win probability is selected per sample, with the chosen RRR exposed for downstream per-trade TP/SL sizing.

## Inputs

- Training DataFrame with OHLC columns ('O', 'H', 'L') and an ATR column ('_atr' preferred, 'vol_atr' as fallback) used to compute per-variant Win/Loss targets.
- Feature DataFrame at inference — arbitrary feature columns; the model appends an 'rrr' column per variant during scoring.
- Trade direction ('long' or 'short') supplied by the training pipeline to _compute_variant_targets.
- Optional per-sample ATR array passed to get_per_trade_params to translate the selected RRR into absolute TP/SL distances.

## Parameters

- `n_estimators` (int, default=100): Number of boosting rounds passed to XGBClassifier.
- `max_depth` (int, default=6): Maximum tree depth for the underlying XGBClassifier.
- `learning_rate` (float, default=0.1): Boosting learning rate (eta) for XGBClassifier.
- `rrr_variants` (list[float], default=[1.5, 2, 2.5, 3, 4]): RRR (TP/SL) variants to train and score jointly; each becomes a value of the 'rrr' input feature.
- `base_sl_atr` (float, default=2): Base stop-loss distance expressed as a multiple of ATR; TP distance = rrr * base_sl_atr * ATR, SL distance = base_sl_atr * ATR.

## Outputs

- Predicted class probabilities of shape (n, 2) — for each sample, the probability row from the RRR variant with the highest win-class probability.
- selected_rrr — per-sample float array of the RRR variant chosen at inference (exposed via the `selected_rrr` property).
- Per-trade TP/SL distances of shape (n, 2) from get_per_trade_params: column 0 = selected_rrr * base_sl_atr * ATR (TP), column 1 = base_sl_atr * ATR (SL).

## Acceptance Criteria

- AC-001: Trains a single XGBClassifier where each training row is duplicated across all rrr_variants with the variant appended as an 'rrr' feature column.
- AC-002: Computes per-variant binary Win/Loss targets by simulating trades with TP = rrr * base_sl_atr * ATR and SL = base_sl_atr * ATR, entering at the next bar's open (entry_delay=1) and scanning up to max_bars=50 bars ahead.
- AC-003: For long direction, target is 1 if high reaches entry+TP before low reaches entry-SL; for short direction, target is 1 if low reaches entry-TP before high reaches entry+SL.
- AC-004: Reads the ATR series from the '_atr' column, falling back to 'vol_atr' if '_atr' is absent; raises ValueError if neither exists.
- AC-005: Skips samples where ATR is NaN or non-positive (target stays 0) and skips the final bar (no next-bar entry available).
- AC-006: At inference, scores every RRR variant by appending it as the 'rrr' feature, then per-sample selects the variant with the highest win-class probability and returns its full probability row.
- AC-007: Exposes the per-sample selected RRR via the `selected_rrr` property after each predict_probability call.
- AC-008: get_per_trade_params returns an (n, 2) array of [TP_distance, SL_distance] = [selected_rrr * base_sl_atr * atr, base_sl_atr * atr] when both selected_rrr and atr are available, else None.
- AC-009: Pops 'base_sl_atr' out of hyperparameters (default 2.0) before passing the remaining params to XGBClassifier, and consumes 'rrr_variants' via the base class variant hook.

## Edge Cases

- ATR column '_atr' missing but 'vol_atr' present — silently falls back to 'vol_atr'.
- Both '_atr' and 'vol_atr' missing — raises ValueError instructing the user to add the 'volatility' indicator to the pipeline.
- ATR value is NaN or <= 0 at a given bar — that sample's target is left as 0 (no trade simulated).
- The final bar of the DataFrame has no next-bar entry, so its target is always 0.
- Neither TP nor SL is hit within max_bars=50 bars — target is 0 (treated as a loss).
- predict_probability called before predict — `selected_rrr` starts as None.
- get_per_trade_params called with atr=None or before any predict_probability call — returns None instead of a TP/SL array.
- rrr_variants not provided in hyperparameters — falls back to the class default [2.0, 3.0] via BaseStackedXGBoostModel; the get_default_params dict advertises [1.5, 2.0, 2.5, 3.0, 4.0] instead.

## Assumptions

- _none_
