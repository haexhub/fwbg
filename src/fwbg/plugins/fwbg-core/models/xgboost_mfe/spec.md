# Plugin Spec — xgboost_mfe

**Kind**: model  •  **Version**: 1.0.0

## Capability

Predicts Maximum Favorable Excursion (MFE) in ATR multiples with XGBoost regression across stacked stop-loss variants, selecting per-sample the SL with the best predicted MFE/SL ratio.

## Summary

An XGBRegressor-based model that treats each candidate SL (in ATR multiples) as a stacked training variant with its own MFE targets computed from OHLC. At inference, every SL variant is scored, and per sample the SL that maximizes predicted_MFE/SL is chosen; the predicted MFE is exposed via the second column of the (n,2) probability array (repurposed so the confidence-threshold mechanism acts as an MFE threshold), and per-trade TP/SL are derived as predicted_MFE*ATR and selected_SL*ATR.

## Inputs

- features: pd.DataFrame of engineered features (must accept an added sl_atr column at inference)
- train_df: OHLC DataFrame used by compute_mfe_targets to produce per-variant MFE targets
- direction: 'long' or 'short' trade direction selecting which MFE target series to use
- atr: optional np.ndarray of per-sample ATR values used to translate ATR-multiple predictions into absolute TP/SL

## Parameters

- `n_estimators` (int, default=100): Number of boosting rounds passed to XGBRegressor.
- `max_depth` (int, default=6): Maximum tree depth for XGBRegressor.
- `learning_rate` (float, default=0.1): Boosting learning rate for XGBRegressor.
- `sl_variants` (list[float], default=[1.5, 2, 2.5, 3]): Candidate stop-loss values in ATR multiples stacked as the sl_atr feature during training and scanned at inference to pick the best MFE/SL ratio.

## Outputs

- probs: (n, 2) np.ndarray whose column 1 holds the predicted MFE (in ATR multiples) for the selected SL variant (column 0 is zeros; not a true probability)
- selected_sl_atr: (n,) np.ndarray of the SL (in ATR multiples) chosen per sample as maximizing predicted MFE/SL
- predicted_mfe: (n,) np.ndarray of predicted MFE (ATR multiples, floored at 0) for the selected SL
- per_trade_params: optional (n, 2) np.ndarray of [TP, SL] in price units = [predicted_mfe*atr, selected_sl_atr*atr]
- trained_classes: np.array([0, 1]) pseudo-classes for pipeline compatibility

## Acceptance Criteria

- AC-001: Registers under the name 'xgboost_mfe' via @register_model and inherits from BaseStackedXGBoostModel.
- AC-002: Builds an XGBRegressor via _build_xgb with objective 'reg:squarederror' added by _extra_default_params.
- AC-003: Stacks training data across the SL variants declared in the sl_variants param, exposing the current variant as the 'sl_atr' feature column.
- AC-004: For each variant, _compute_variant_targets calls fwbg.optimization.targets.compute_mfe_targets(train_df, sl_atr=variant, max_bars=50) and returns mfe_long or mfe_short based on the direction argument; falls back to a copy of fallback_targets when train_df is None.
- AC-005: _post_process_stacked_targets clips stacked training targets at 0.0 (no negative MFE).
- AC-006: At inference, _predict_probability_impl copies the input features, sets df_copy['sl_atr']=sl for each variant, predicts MFE (clipped at 0), and keeps per-sample the SL with the highest predicted_mfe/sl ratio.
- AC-007: Stores the per-sample selected SL in _selected_sl_atr (accessible via the selected_sl_atr property) and predicted MFE in _predicted_mfe after every predict call.
- AC-008: Returns an (n, 2) float64 probability array whose column 0 is zero and column 1 is the best predicted MFE per sample.
- AC-009: get_per_trade_params returns an (n, 2) array with column 0 = predicted_mfe*atr (TP) and column 1 = selected_sl_atr*atr (SL) when predicted MFE, selected SL, and atr are all available; returns None otherwise.
- AC-010: _trained_classes_impl returns np.array([0, 1]) as pseudo-classes for pipeline compatibility.
- AC-011: get_default_params returns {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'sl_variants': [1.5, 2.0, 2.5, 3.0]}.

## Edge Cases

- train_df is None during target computation: _compute_variant_targets returns a copy of the provided fallback_targets instead of recomputing.
- Negative raw MFE targets from OHLC: floored to 0.0 by _post_process_stacked_targets before training.
- XGBRegressor predicts a negative MFE at inference: np.maximum(..., 0.0) clips it to 0, so ratio is 0 and that variant will not out-rank a positive-MFE variant.
- Ties or all-zero predicted MFE across variants: best_ratio starts at -1.0 so the first variant with ratio >= 0 wins; if all variants tie at ratio 0, the first (smallest sl in _variants order) is retained.
- get_per_trade_params called before any predict() (predicted_mfe / selected_sl_atr still None) or with atr=None: returns None.
- sl_variants list is user-overridable; the plugin falls back to _default_variants=[1.5, 2.0, 2.5, 3.0] when none is supplied (handled by BaseStackedXGBoostModel via _variant_param_name).
- Direction other than 'long' is treated as 'short' by _compute_variant_targets (only 'long' selects mfe_long).

## Assumptions

- BaseStackedXGBoostModel provides the stacking machinery, self._variants (populated from the sl_variants param), self._model (the fitted XGBRegressor), and calls the hook methods overridden here.
- fwbg.optimization.targets.compute_mfe_targets returns a (mfe_long, mfe_short) pair of arrays aligned with train_df rows and expressed in ATR multiples.
- The features DataFrame passed to _predict_probability_impl accepts an added 'sl_atr' column without breaking the underlying XGBoost feature schema (i.e., 'sl_atr' is part of the training feature set).

## Needs Clarification

- [NEEDS CLARIFICATION: Whether 'sl_atr' is guaranteed to be part of the training feature schema, or whether _predict_probability_impl relies on XGBoost tolerating an extra column.]
- [NEEDS CLARIFICATION: Whether the max_bars=50 horizon used inside compute_mfe_targets should be a configurable param rather than a hard-coded value.]
