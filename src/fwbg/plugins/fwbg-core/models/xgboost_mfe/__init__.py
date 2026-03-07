"""XGBoost MFE Regression model plugin.

Predicts Maximum Favorable Excursion (how far a breakout runs) using
XGBRegressor instead of binary Win/Loss classification. MFE is normalized
by ATR for regime robustness.

SL is provided as an input feature (multiple SL variants are stacked),
letting the model learn which SL works best per setup. At inference,
the SL variant with the best predicted MFE/SL ratio is selected.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from fwbg_sdk.models import BaseModel, TrainingContext
from fwbg_sdk.registry import register_model


@register_model("xgboost_mfe")
class XGBoostMFEModel(BaseModel):
    """XGBoost regressor predicting MFE in ATR multiples."""

    name = "xgboost_mfe"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._sl_variants: List[float] = []
        self._selected_sl_atr: Optional[np.ndarray] = None
        self._predicted_mfe: Optional[np.ndarray] = None
        self._classes = np.array([0, 1])  # pseudo-classes for pipeline compat

    @property
    def selected_sl_atr(self) -> Optional[np.ndarray]:
        """Per-sample selected SL (ATR mult) from last predict call."""
        return self._selected_sl_atr

    def train(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        training_context: TrainingContext,
        **hyperparameters: Any,
    ) -> None:
        from xgboost import XGBRegressor
        from fwbg.utils.xgb_config import get_xgboost_params, get_xgboost_n_jobs

        self.progress.begin_training()

        # Extract MFE-specific params
        self._sl_variants = hyperparameters.pop("sl_variants", [1.5, 2.0, 2.5])

        self.progress.begin_stage("stacking", "Stacking dataset for SL variants")

        # Stack: duplicate dataset for each SL variant, add sl_atr column
        stacked_features = []
        stacked_targets = []
        for sl in self._sl_variants:
            df_copy = features.copy()
            df_copy["sl_atr"] = sl
            stacked_features.append(df_copy)
            stacked_targets.append(targets.copy())

        X_stacked = pd.concat(stacked_features, ignore_index=True)
        y_stacked = np.concatenate(stacked_targets)

        # Clamp negative MFE targets to 0
        y_stacked = np.maximum(y_stacked, 0.0)

        sample_weight = None
        if training_context.sample_weights is not None:
            sample_weight = np.tile(
                training_context.sample_weights, len(self._sl_variants)
            )

        self.progress.complete_stage("stacking")
        self.logger.info(
            f"Stacked {len(features)} samples x {len(self._sl_variants)} SL variants "
            f"= {len(X_stacked)} rows"
        )

        # Prepare XGBoost params
        self.progress.begin_stage(
            "prepare_parameters", "Preparing XGBoost parameters"
        )
        params = hyperparameters.copy()
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", 0)
        params.setdefault("objective", "reg:squarederror")
        params["n_jobs"] = get_xgboost_n_jobs()
        params.update(get_xgboost_params())
        self.progress.complete_stage("prepare_parameters")

        # Train regressor
        self.progress.begin_stage(
            "fitting", "Fitting XGBRegressor on stacked data"
        )
        self._model = XGBRegressor(**params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        try:
            self._model.fit(X_stacked, y_stacked, **fit_kwargs)
            self.progress.complete_stage("fitting")
        except Exception as error:
            error_message = str(error).lower()
            if "cuda" in error_message or "gpu" in error_message:
                self._handle_gpu_fallback(
                    X_stacked, y_stacked, params, fit_kwargs
                )
            else:
                raise

        self._fitted = True
        self._feature_names = list(X_stacked.columns)
        total_duration = self.progress.complete_training()
        self.logger.info(
            f"Trained MFE regressor: {len(y_stacked)} stacked samples, "
            f"{len(self._feature_names)} features (incl sl_atr), "
            f"{total_duration:.2f}s"
        )

    def _handle_gpu_fallback(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        params: dict,
        fit_kwargs: dict,
    ) -> None:
        from fwbg.utils.xgb_config import disable_gpu
        from xgboost import XGBRegressor

        self.progress.begin_stage(
            "gpu_fallback", "GPU failed, falling back to CPU"
        )
        self.logger.warning("CUDA error — falling back to CPU")
        disable_gpu()
        cpu_params = {
            k: v
            for k, v in params.items()
            if k not in ("device", "tree_method")
        }
        cpu_params["tree_method"] = "hist"
        cpu_params["device"] = "cpu"
        self._model = XGBRegressor(**cpu_params)
        self._model.fit(X, y, **fit_kwargs)
        self.progress.complete_stage("gpu_fallback")

    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        """Score all SL variants, select best MFE/SL ratio per sample.

        Returns (n, 2) array where column 1 = predicted MFE in ATR.
        This is NOT a probability -- it is repurposed so the CT mechanism
        acts as an MFE threshold.
        """
        n = len(features)
        best_mfe = np.zeros(n, dtype=np.float64)
        best_ratio = np.full(n, -1.0)
        best_sl = np.full(n, self._sl_variants[0])

        for sl in self._sl_variants:
            df_copy = features.copy()
            df_copy["sl_atr"] = sl
            pred_mfe = np.maximum(self._model.predict(df_copy), 0.0)
            ratio = pred_mfe / sl

            better = ratio > best_ratio
            best_mfe[better] = pred_mfe[better]
            best_ratio[better] = ratio[better]
            best_sl[better] = sl

        self._selected_sl_atr = best_sl
        self._predicted_mfe = best_mfe

        # Return as (n, 2) pseudo-probability: col 0 = 0, col 1 = predicted MFE
        probs = np.zeros((n, 2), dtype=np.float64)
        probs[:, 1] = best_mfe
        return probs

    @property
    def _trained_classes_impl(self) -> np.ndarray:
        return self._classes

    def _as_sklearn_estimator_impl(self) -> Any:
        return self._model

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._model is None or self._feature_names is None:
            return None
        importance = self._model.feature_importances_
        return dict(zip(self._feature_names, importance.tolist()))

    def get_per_trade_params(
        self,
        features: pd.DataFrame,
        atr: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return per-trade TP/SL based on predicted MFE and selected SL."""
        if self._predicted_mfe is None or self._selected_sl_atr is None or atr is None:
            return None

        n = len(features)
        ptp = np.zeros((n, 2), dtype=np.float64)
        ptp[:, 0] = self._predicted_mfe * atr   # TP = predicted_mfe * atr
        ptp[:, 1] = self._selected_sl_atr * atr  # SL = selected_sl_atr * atr
        return ptp

    @classmethod
    def get_reduced_hyperparameters(
        cls, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        reduced = hyperparameters.copy()
        reduced["n_estimators"] = max(
            10, reduced.get("n_estimators", 100) // 2
        )
        return reduced

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "sl_variants": [1.5, 2.0, 2.5, 3.0],
        }
