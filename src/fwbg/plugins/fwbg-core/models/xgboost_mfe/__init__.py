"""XGBoost MFE Regression model plugin.

Predicts Maximum Favorable Excursion (how far a breakout runs) using
XGBRegressor instead of binary Win/Loss classification. MFE is normalized
by ATR for regime robustness.

SL is provided as an input feature (multiple SL variants are stacked),
letting the model learn which SL works best per setup. Each SL variant
gets its own MFE targets computed from OHLC data. At inference,
the SL variant with the best predicted MFE/SL ratio is selected.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from fwbg_sdk.registry import register_model

from fwbg.utils.model_bases import BaseStackedXGBoostModel


@register_model("xgboost_mfe")
class XGBoostMFEModel(BaseStackedXGBoostModel):
    """XGBoost regressor predicting MFE in ATR multiples."""

    name = "xgboost_mfe"
    version = "1.0.0"

    _variant_param_name = "sl_variants"
    _variant_feature_name = "sl_atr"
    _default_variants = [1.5, 2.0, 2.5, 3.0]

    def __init__(self) -> None:
        super().__init__()
        self._selected_sl_atr: Optional[np.ndarray] = None
        self._predicted_mfe: Optional[np.ndarray] = None
        self._classes = np.array([0, 1])  # pseudo-classes for pipeline compat

    @property
    def selected_sl_atr(self) -> Optional[np.ndarray]:
        """Per-sample selected SL (ATR mult) from last predict call."""
        return self._selected_sl_atr

    # ---- Hooks for BaseStackedXGBoostModel -------------------------------

    def _build_xgb(self, params: Dict[str, Any]):
        from xgboost import XGBRegressor

        return XGBRegressor(**params)

    def _extra_default_params(self) -> Dict[str, Any]:
        return {"objective": "reg:squarederror"}

    def _compute_variant_targets(
        self,
        train_df,
        variant: float,
        direction: str,
        fallback_targets: np.ndarray,
    ) -> np.ndarray:
        if train_df is not None:
            from fwbg.optimization.targets import compute_mfe_targets

            mfe_long, mfe_short = compute_mfe_targets(
                train_df, sl_atr=variant, max_bars=50,
            )
            return mfe_long if direction == "long" else mfe_short
        return fallback_targets.copy()

    def _post_process_stacked_targets(self, y_stacked: np.ndarray) -> np.ndarray:
        return np.maximum(y_stacked, 0.0)

    # ---- Inference --------------------------------------------------------

    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        """Score all SL variants, select best MFE/SL ratio per sample.

        Returns (n, 2) array where column 1 = predicted MFE in ATR.
        This is NOT a probability -- it is repurposed so the CT mechanism
        acts as an MFE threshold.
        """
        n = len(features)
        best_mfe = np.zeros(n, dtype=np.float64)
        best_ratio = np.full(n, -1.0)
        best_sl = np.full(n, self._variants[0])

        for sl in self._variants:
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

        probs = np.zeros((n, 2), dtype=np.float64)
        probs[:, 1] = best_mfe
        return probs

    @property
    def _trained_classes_impl(self) -> np.ndarray:
        return self._classes

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
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "sl_variants": [1.5, 2.0, 2.5, 3.0],
        }
