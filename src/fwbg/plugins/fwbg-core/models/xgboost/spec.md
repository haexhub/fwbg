# Plugin Spec — xgboost

**Kind**: model  •  **Version**: 1.0.0

## Capability

Trains an XGBoost gradient boosting classifier on tabular features and produces class probability predictions, with automatic CUDA-to-CPU fallback on GPU failure.

## Summary

A BaseModel implementation wrapping xgboost.XGBClassifier. Merges caller hyperparameters with defaults from fwbg.utils.xgb_config (including n_jobs and GPU/tree_method settings), fits the classifier with optional sample weights from the TrainingContext, and if fitting raises a CUDA/GPU/device-related error, disables GPU and refits on CPU using tree_method='hist'. Exposes predict_proba, trained classes, sklearn-estimator access, and per-feature importances. Also provides a reduced-hyperparameter variant (halved n_estimators) for inner-CV speed.

## Inputs

- features: pandas DataFrame of numeric feature columns
- targets: numpy array of class labels
- training_context: TrainingContext (may provide sample_weights)
- hyperparameters: keyword arguments forwarded to XGBClassifier

## Parameters

- `n_estimators` (int, default=100): Number of boosting rounds
- `max_depth` (int, default=6): Maximum tree depth
- `learning_rate` (float, default=0.1): Boosting learning rate (eta)
- `subsample` (float, default=0.8): Subsample ratio of training instances
- `colsample_bytree` (float, default=0.8): Subsample ratio of features per tree

## Outputs

- class probability matrix from predict_proba (shape n_samples x n_classes)
- trained class labels via _trained_classes_impl
- underlying sklearn-compatible XGBClassifier via _as_sklearn_estimator_impl
- per-feature importance mapping from get_feature_importance

## Acceptance Criteria

- AC-001: train() populates self._model with a fitted XGBClassifier and sets self._fitted=True and self._feature_names to the training feature column list
- AC-002: Hyperparameters passed via **hyperparameters override defaults, but random_state defaults to 42, verbosity defaults to 0, n_jobs is set from get_xgboost_n_jobs(), and get_xgboost_params() values are applied on top
- AC-003: When training_context.sample_weights is not None, it is forwarded to XGBClassifier.fit as sample_weight
- AC-004: If fit() raises an exception whose message (lowercased) contains 'cuda', 'gpu', or 'device', the model disables GPU, rebuilds XGBClassifier with tree_method='hist' and device='cpu' (dropping any prior 'device'/'tree_method'), and refits
- AC-005: If fit() raises a non-GPU exception, the error is logged and re-raised
- AC-006: _predict_probability_impl delegates to self._model.predict_proba and returns its output unchanged
- AC-007: _trained_classes_impl returns self._model.classes_ and _as_sklearn_estimator_impl returns self._model
- AC-008: get_feature_importance returns a dict mapping feature name to importance when the model is fitted and feature names are known, otherwise None
- AC-009: get_default_params returns n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8
- AC-010: get_reduced_hyperparameters returns a copy with n_estimators = max(10, original_n_estimators // 2), defaulting to 100 if absent

## Edge Cases

- get_feature_importance returns None before training (self._model is None)
- get_feature_importance returns None when self._feature_names has not been set
- get_reduced_hyperparameters floors n_estimators at 10 (e.g. input 10 or missing key both yield 10 or 50 respectively; input 5 // 2 = 2 would be clamped to 10)
- Non-GPU exceptions during fit are re-raised rather than triggering CPU fallback
- sample_weights being None omits the sample_weight kwarg from fit entirely
- GPU-fallback path constructs the CPU model from the already-merged params (including get_xgboost_params overrides) minus 'device' and 'tree_method'

## Assumptions

- xgboost is installed and importable at train() time
- fwbg.utils.xgb_config provides get_xgboost_params(), get_xgboost_n_jobs(), and disable_gpu()
- BaseModel supplies self.progress, self.logger, and self._fitted attributes
- TrainingContext exposes a sample_weights attribute (possibly None)
