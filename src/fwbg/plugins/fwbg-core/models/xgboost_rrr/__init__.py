"""XGBoost RRR-as-Feature model plugin.

Trains a single XGBClassifier on multiple RRR variants simultaneously.
RRR (reward-risk ratio = tp/sl) is added as an input feature, letting
the model learn which RRR works best for each market setup.

Each RRR variant gets its own binary Win/Loss targets computed from OHLC
data with variant-specific TP/SL distances.

At inference, all RRR variants are scored and the best one is selected
per sample.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from fwbg_sdk.models import BaseModel, TrainingContext
from fwbg_sdk.registry import register_model


def _compute_binary_targets(
    df: pd.DataFrame,
    rrr: float,
    base_sl_atr: float,
    direction: str,
    atr_col: str = "_atr",
    max_bars: int = 50,
) -> np.ndarray:
    """Compute binary Win/Loss targets for a specific RRR variant.

    Simulates trades with TP = rrr * base_sl_atr * ATR, SL = base_sl_atr * ATR.
    Entry at next bar open (entry_delay=1). Returns 1 if TP hit first, 0 otherwise.
    """
    if atr_col not in df.columns:
        fallback = "vol_atr" if "vol_atr" in df.columns else None
        if fallback:
            atr_col = fallback
        else:
            raise ValueError(
                f"ATR column '{atr_col}' not found in DataFrame. "
                "Add the 'volatility' indicator to your pipeline config."
            )

    opens = df["O"].values
    highs = df["H"].values
    lows = df["L"].values
    atr = df[atr_col].values.astype(np.float64)
    n = len(df)
    targets = np.zeros(n, dtype=np.int32)

    for i in range(n - 1):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        tp_dist = rrr * base_sl_atr * atr_i
        sl_dist = base_sl_atr * atr_i
        entry_price = opens[i + 1]

        if direction == "long":
            for j in range(i + 1, min(i + 1 + max_bars, n)):
                if highs[j] - entry_price >= tp_dist:
                    targets[i] = 1
                    break
                if entry_price - lows[j] >= sl_dist:
                    break
        else:  # short
            for j in range(i + 1, min(i + 1 + max_bars, n)):
                if entry_price - lows[j] >= tp_dist:
                    targets[i] = 1
                    break
                if highs[j] - entry_price >= sl_dist:
                    break

    return targets


@register_model("xgboost_rrr")
class XGBoostRRRModel(BaseModel):
    """XGBoost classifier with RRR as an input feature."""

    name = "xgboost_rrr"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._rrr_variants: List[float] = []
        self._base_sl_atr: float = 2.0
        self._selected_rrr: Optional[np.ndarray] = None

    @property
    def selected_rrr(self) -> Optional[np.ndarray]:
        """Per-sample selected RRR from last predict_probability call."""
        return self._selected_rrr

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

        # Extract RRR-specific params
        self._rrr_variants = hyperparameters.pop("rrr_variants", [2.0, 3.0])
        self._base_sl_atr = hyperparameters.pop("base_sl_atr", 2.0)

        # Get full OHLC data for per-variant target computation
        fold_info = training_context.fold_information or {}
        train_df = fold_info.get("train_df")
        direction = training_context.direction or "long"

        self.progress.begin_stage("stacking", "Stacking dataset with per-variant targets")

        stacked_features = []
        stacked_targets = []
        for rrr in self._rrr_variants:
            df_copy = features.copy()
            df_copy["rrr"] = rrr

            if train_df is not None:
                variant_targets = _compute_binary_targets(
                    train_df, rrr, self._base_sl_atr, direction,
                )
            else:
                # Fallback: use passed targets (legacy / tests without train_df)
                variant_targets = targets.copy()

            stacked_features.append(df_copy)
            stacked_targets.append(variant_targets)

        X_stacked = pd.concat(stacked_features, ignore_index=True)
        y_stacked = np.concatenate(stacked_targets)

        # Handle sample weights — stack them too
        sample_weight = None
        if training_context.sample_weights is not None:
            sample_weight = np.tile(
                training_context.sample_weights, len(self._rrr_variants)
            )

        self.progress.complete_stage("stacking")
        self.logger.info(
            f"Stacked {len(features)} samples x {len(self._rrr_variants)} RRR variants "
            f"= {len(X_stacked)} rows"
        )

        # Prepare XGBoost params
        self.progress.begin_stage(
            "prepare_parameters", "Preparing XGBoost parameters"
        )
        params = hyperparameters.copy()
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", 0)
        params["n_jobs"] = get_xgboost_n_jobs()
        params.update(get_xgboost_params())
        self.progress.complete_stage("prepare_parameters")

        # Train
        self.progress.begin_stage(
            "fitting", "Fitting XGBoost model on stacked data"
        )
        self._model = XGBClassifier(**params)
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
            f"Trained: {len(y_stacked)} stacked samples, "
            f"{len(self._feature_names)} features (incl rrr), "
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
        from xgboost import XGBClassifier

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
        self._model = XGBClassifier(**cpu_params)
        self._model.fit(X, y, **fit_kwargs)
        self.progress.complete_stage("gpu_fallback")

    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        """Score all RRR variants and pick the best per sample."""
        n = len(features)
        best_probs = np.zeros((n, 2), dtype=np.float64)
        best_rrr = np.zeros(n, dtype=np.float64)
        best_win_prob = np.full(n, -1.0)

        win_idx = np.where(self._model.classes_ == 1)[0][0]

        for rrr in self._rrr_variants:
            df_copy = features.copy()
            df_copy["rrr"] = rrr
            probs = self._model.predict_proba(df_copy)

            better = probs[:, win_idx] > best_win_prob
            best_probs[better] = probs[better]
            best_rrr[better] = rrr
            best_win_prob[better] = probs[better, win_idx]

        self._selected_rrr = best_rrr
        return best_probs

    @property
    def _trained_classes_impl(self) -> np.ndarray:
        return self._model.classes_

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
        """Return per-trade TP/SL based on selected RRR."""
        if self._selected_rrr is None or atr is None:
            return None

        n = len(features)
        ptp = np.zeros((n, 2), dtype=np.float64)
        ptp[:, 0] = self._selected_rrr * self._base_sl_atr * atr  # TP
        ptp[:, 1] = self._base_sl_atr * atr  # SL
        return ptp

    @classmethod
    def get_required_indicators(cls) -> List[str]:
        return ["volatility"]

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
            "rrr_variants": [1.5, 2.0, 2.5, 3.0, 4.0],
            "base_sl_atr": 2.0,
        }
