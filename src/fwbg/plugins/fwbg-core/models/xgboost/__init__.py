"""XGBoost model plugin for FWBG."""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from fwbg_sdk.models import BaseModel, TrainingContext
from fwbg_sdk.registry import register_model


@register_model("xgboost")
class XGBoostModel(BaseModel):
    """XGBoost gradient boosting classifier with GPU fallback."""

    name = "xgboost"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._feature_names: Optional[List[str]] = None

    def train(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        training_context: TrainingContext,
        **hyperparameters: Any,
    ) -> None:
        from xgboost import XGBClassifier
        from fwbg.utils.xgb_config import get_xgboost_params, get_xgboost_n_jobs

        self.progress.begin_training()

        # Stage: prepare_parameters
        self.progress.begin_stage("prepare_parameters", "Preparing XGBoost parameters")
        params = hyperparameters.copy()
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", 0)
        params["n_jobs"] = get_xgboost_n_jobs()
        params.update(get_xgboost_params())
        self.progress.complete_stage("prepare_parameters")

        # Stage: fitting
        self.progress.begin_stage("fitting", "Fitting XGBoost model")
        self._model = XGBClassifier(**params)
        fit_kwargs = {}
        if training_context.sample_weights is not None:
            fit_kwargs["sample_weight"] = training_context.sample_weights

        try:
            self._model.fit(features, targets, **fit_kwargs)
            self.progress.complete_stage("fitting")
        except Exception as error:
            error_message = str(error).lower()
            if "cuda" in error_message or "gpu" in error_message or "device" in error_message:
                self._handle_gpu_fallback(features, targets, params, fit_kwargs)
            else:
                self.logger.error(f"Training failed: {error}")
                raise

        self._fitted = True
        self._feature_names = list(features.columns)
        total_duration = self.progress.complete_training()
        self.logger.info(
            f"Trained: {len(targets)} samples, {len(features.columns)} features, "
            f"{total_duration:.2f}s"
        )

    def _handle_gpu_fallback(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        original_params: dict,
        fit_kwargs: dict,
    ) -> None:
        from fwbg.utils.xgb_config import disable_gpu
        from xgboost import XGBClassifier

        self.progress.begin_stage("gpu_fallback", "GPU failed, falling back to CPU")
        self.logger.warning("CUDA error — falling back to CPU")
        disable_gpu()
        cpu_params = {
            k: v for k, v in original_params.items()
            if k not in ("device", "tree_method")
        }
        cpu_params["tree_method"] = "hist"
        cpu_params["device"] = "cpu"
        self._model = XGBClassifier(**cpu_params)
        self._model.fit(features, targets, **fit_kwargs)
        self.progress.complete_stage("gpu_fallback")

    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(features)

    @property
    def _trained_classes_impl(self) -> np.ndarray:
        return self._model.classes_

    def _as_sklearn_estimator_impl(self) -> Any:
        return self._model

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None:
            return None
        importance = self._model.feature_importances_
        if self._feature_names is None:
            return None
        return dict(zip(self._feature_names, importance.tolist()))

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    @classmethod
    def get_param_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "description": "Number of boosting rounds",
                "min": 10,
                "max": 2000,
                "step": 10,
            },
            "max_depth": {
                "type": "int",
                "default": 6,
                "description": "Maximum tree depth",
                "min": 1,
                "max": 15,
                "step": 1,
            },
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "description": "Boosting learning rate (eta)",
                "min": 0.001,
                "max": 1.0,
                "step": 0.01,
            },
            "subsample": {
                "type": "float",
                "default": 0.8,
                "description": "Subsample ratio of training instances",
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
            },
            "colsample_bytree": {
                "type": "float",
                "default": 0.8,
                "description": "Subsample ratio of features per tree",
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
            },
        }

    @classmethod
    def get_reduced_hyperparameters(
        cls, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """XGBoost-specific: halve n_estimators for inner CV speed."""
        reduced = hyperparameters.copy()
        reduced["n_estimators"] = max(10, reduced.get("n_estimators", 100) // 2)
        return reduced
