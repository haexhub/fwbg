"""Shared base classes for model plugins.

Hosts scaffolding shared by xgboost_rrr (classifier with RRR as feature)
and xgboost_mfe (regressor predicting MFE in ATR multiples). Both stack a
copy of the feature matrix per variant (`rrr` / `sl_atr`), compute
per-variant targets, fit a single XGBoost model on the stacked data, and
at inference time score every variant and pick the best per sample.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fwbg_sdk.models import BaseModel, TrainingContext


class BaseStackedXGBoostModel(BaseModel, ABC):
    """Train one XGBoost model on a per-variant-stacked dataset.

    Subclass contract:
        - set `name`, `version`
        - set `_variant_param_name` (e.g. "rrr_variants", "sl_variants")
        - set `_variant_feature_name` (e.g. "rrr", "sl_atr")
        - set `_default_variants`
        - implement `_build_xgb(params)` returning a fresh XGBoost estimator
        - implement `_compute_variant_targets(train_df, variant, direction,
          fallback_targets)` returning np.ndarray for that variant
        - implement `_predict_probability_impl(features)` — variant selection
          differs (max prob vs max MFE/SL ratio)
        - implement `_trained_classes_impl` (model.classes_ vs pseudo)
    """

    # ---- Subclass overrides ----------------------------------------------

    _variant_param_name: str = ""
    _variant_feature_name: str = ""
    _default_variants: List[float] = []

    @abstractmethod
    def _build_xgb(self, params: Dict[str, Any]):
        """Return a fresh XGBoost estimator (classifier / regressor)."""

    @abstractmethod
    def _compute_variant_targets(
        self,
        train_df: Optional[pd.DataFrame],
        variant: float,
        direction: str,
        fallback_targets: np.ndarray,
    ) -> np.ndarray:
        """Compute the targets vector for one variant."""

    def _post_process_stacked_targets(self, y_stacked: np.ndarray) -> np.ndarray:
        """Optional hook — e.g. clamp negatives to 0 (MFE)."""
        return y_stacked

    def _extra_default_params(self) -> Dict[str, Any]:
        """Optional hook — extra XGBoost defaults (e.g. objective for regressor)."""
        return {}

    # ---- Shared lifecycle -------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._variants: List[float] = []

    def train(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        training_context: TrainingContext,
        **hyperparameters: Any,
    ) -> None:
        from fwbg.utils.xgb_config import get_xgboost_params, get_xgboost_n_jobs

        self.progress.begin_training()

        self._variants = list(
            hyperparameters.pop(self._variant_param_name, self._default_variants)
        )
        self._on_variant_params_loaded(hyperparameters)

        fold_info = training_context.fold_information or {}
        train_df = fold_info.get("train_df")
        direction = training_context.direction or "long"

        X_stacked, y_stacked, sample_weight = self._build_stacked_dataset(
            features, targets, training_context, train_df, direction,
        )

        self.progress.begin_stage(
            "prepare_parameters", "Preparing XGBoost parameters"
        )
        params = hyperparameters.copy()
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", 0)
        for k, v in self._extra_default_params().items():
            params.setdefault(k, v)
        params["n_jobs"] = get_xgboost_n_jobs()
        params.update(get_xgboost_params())
        self.progress.complete_stage("prepare_parameters")

        self.progress.begin_stage("fitting", "Fitting XGBoost model on stacked data")
        self._model = self._build_xgb(params)
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        try:
            self._model.fit(X_stacked, y_stacked, **fit_kwargs)
            self.progress.complete_stage("fitting")
        except Exception as error:
            error_message = str(error).lower()
            if "cuda" in error_message or "gpu" in error_message:
                self._handle_gpu_fallback(X_stacked, y_stacked, params, fit_kwargs)
            else:
                raise

        self._fitted = True
        self._feature_names = list(X_stacked.columns)
        total_duration = self.progress.complete_training()
        self.logger.info(
            f"Trained: {len(y_stacked)} stacked samples, "
            f"{len(self._feature_names)} features (incl {self._variant_feature_name}), "
            f"{total_duration:.2f}s"
        )

    def _on_variant_params_loaded(self, hyperparameters: Dict[str, Any]) -> None:
        """Hook to extract additional hyperparameters before stacking (e.g. base_sl_atr)."""

    def _build_stacked_dataset(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        training_context: TrainingContext,
        train_df: Optional[pd.DataFrame],
        direction: str,
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
        self.progress.begin_stage(
            "stacking", "Stacking dataset with per-variant targets"
        )
        stacked_features: List[pd.DataFrame] = []
        stacked_targets: List[np.ndarray] = []
        for variant in self._variants:
            df_copy = features.copy()
            df_copy[self._variant_feature_name] = variant
            variant_targets = self._compute_variant_targets(
                train_df, variant, direction, targets
            )
            stacked_features.append(df_copy)
            stacked_targets.append(variant_targets)

        X_stacked = pd.concat(stacked_features, ignore_index=True)
        y_stacked = self._post_process_stacked_targets(np.concatenate(stacked_targets))

        sample_weight = None
        if training_context.sample_weights is not None:
            sample_weight = np.tile(
                training_context.sample_weights, len(self._variants)
            )

        self.progress.complete_stage("stacking")
        self.logger.info(
            f"Stacked {len(features)} samples x {len(self._variants)} variants "
            f"= {len(X_stacked)} rows"
        )
        return X_stacked, y_stacked, sample_weight

    def _handle_gpu_fallback(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        params: Dict[str, Any],
        fit_kwargs: Dict[str, Any],
    ) -> None:
        from fwbg.utils.xgb_config import disable_gpu

        self.progress.begin_stage("gpu_fallback", "GPU failed, falling back to CPU")
        self.logger.warning("CUDA error — falling back to CPU")
        disable_gpu()
        cpu_params = {
            k: v
            for k, v in params.items()
            if k not in ("device", "tree_method")
        }
        cpu_params["tree_method"] = "hist"
        cpu_params["device"] = "cpu"
        self._model = self._build_xgb(cpu_params)
        self._model.fit(X, y, **fit_kwargs)
        self.progress.complete_stage("gpu_fallback")

    # ---- Shared overrides for BaseModel ----------------------------------

    def _as_sklearn_estimator_impl(self) -> Any:
        return self._model

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None or self._feature_names is None:
            return None
        importance = self._model.feature_importances_
        return dict(zip(self._feature_names, importance.tolist()))

    @classmethod
    def get_required_indicators(cls) -> List[str]:
        return ["volatility"]

    @classmethod
    def get_reduced_hyperparameters(
        cls, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        reduced = hyperparameters.copy()
        reduced["n_estimators"] = max(10, reduced.get("n_estimators", 100) // 2)
        return reduced
