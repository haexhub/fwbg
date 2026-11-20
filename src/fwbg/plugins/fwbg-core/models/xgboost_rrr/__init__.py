"""XGBoost RRR-as-Feature model plugin.

Trains a single XGBClassifier on multiple RRR variants simultaneously.
RRR (reward-risk ratio = tp/sl) is added as an input feature, letting
the model learn which RRR works best for each market setup.

Each RRR variant gets its own binary Win/Loss targets computed from OHLC
data with variant-specific TP/SL distances.

At inference, all RRR variants are scored and the one with the highest
expected value per unit risk, ``p * rrr - (1 - p)``, is selected per sample.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from fwbg_sdk.registry import register_model

from fwbg.utils.model_bases import BaseStackedXGBoostModel


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
    Entry at next bar open (entry_delay=1). Returns 1.0 if TP is hit first, 0.0
    otherwise. A bar that touches both levels counts as a stop (the intrabar order
    is unknown; same convention as the simulation kernels).

    NaN marks a trade whose outcome is not known: neither level was reached and
    the ``max_bars`` window runs past the end of the frame (the last bar has no
    entry at all).  Training drops such rows instead of reading them as losses.
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
    targets = np.zeros(n, dtype=np.float64)
    if n:
        targets[n - 1] = np.nan

    for i in range(n - 1):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        tp_dist = rrr * base_sl_atr * atr_i
        sl_dist = base_sl_atr * atr_i
        entry_price = opens[i + 1]

        settled = False
        if direction == "long":
            for j in range(i + 1, min(i + 1 + max_bars, n)):
                if entry_price - lows[j] >= sl_dist:
                    settled = True
                    break
                if highs[j] - entry_price >= tp_dist:
                    targets[i] = 1.0
                    settled = True
                    break
        else:  # short
            for j in range(i + 1, min(i + 1 + max_bars, n)):
                if highs[j] - entry_price >= sl_dist:
                    settled = True
                    break
                if entry_price - lows[j] >= tp_dist:
                    targets[i] = 1.0
                    settled = True
                    break
        if not settled and i + 1 + max_bars > n:
            targets[i] = np.nan

    return targets


@register_model("xgboost_rrr")
class XGBoostRRRModel(BaseStackedXGBoostModel):
    """XGBoost classifier with RRR as an input feature."""

    name = "xgboost_rrr"
    version = "1.1.0"

    _variant_param_name = "rrr_variants"
    _variant_feature_name = "rrr"
    _default_variants = [2.0, 3.0]
    _needs_two_classes = True

    def __init__(self) -> None:
        super().__init__()
        self._base_sl_atr: float = 2.0
        self._selected_rrr: Optional[np.ndarray] = None

    @property
    def selected_rrr(self) -> Optional[np.ndarray]:
        """Per-sample selected RRR from last predict_probability call."""
        return self._selected_rrr

    # ---- Hooks for BaseStackedXGBoostModel -------------------------------

    def _on_variant_params_loaded(self, hyperparameters: Dict[str, Any]) -> None:
        self._base_sl_atr = hyperparameters.pop("base_sl_atr", 2.0)

    def _build_xgb(self, params: Dict[str, Any]):
        from xgboost import XGBClassifier

        return XGBClassifier(**params)

    def _post_process_stacked_targets(self, y_stacked: np.ndarray) -> np.ndarray:
        # Rows with unknown outcomes are gone by now: class labels are 0/1.
        return y_stacked.astype(np.int32)

    def _compute_variant_targets(
        self,
        train_df: Optional[pd.DataFrame],
        variant: float,
        direction: str,
        fallback_targets: np.ndarray,
    ) -> np.ndarray:
        if train_df is not None:
            return _compute_binary_targets(
                train_df,
                variant,
                self._base_sl_atr,
                direction,
            )
        return fallback_targets.copy()

    # ---- Inference --------------------------------------------------------

    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        """Score all RRR variants and pick the best expected value per sample.

        A higher RRR wins less often but pays more, so variants are ranked by
        ``p * rrr - (1 - p)`` (expected R per trade), not by P(win) alone.
        """
        n = len(features)
        best_probs = np.zeros((n, 2), dtype=np.float64)
        best_rrr = np.zeros(n, dtype=np.float64)
        best_ev = np.full(n, -np.inf)

        win_idx = np.where(self._model.classes_ == 1)[0][0]

        for rrr in self._variants:
            df_copy = features.copy()
            df_copy["rrr"] = rrr
            probs = self._model.predict_proba(df_copy)

            p_win = probs[:, win_idx]
            ev = p_win * rrr - (1.0 - p_win)
            better = ev > best_ev
            best_probs[better] = probs[better]
            best_rrr[better] = rrr
            best_ev[better] = ev[better]

        self._selected_rrr = best_rrr
        return best_probs

    @property
    def _trained_classes_impl(self) -> np.ndarray:
        return self._model.classes_

    def get_per_trade_params(
        self,
        features: pd.DataFrame,
        atr: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return per-trade TP/SL based on selected RRR.

        Rows whose ATR is NaN (warm-up) hold NaN distances: no trade there.
        """
        if self._selected_rrr is None or atr is None:
            return None

        n = len(features)
        ptp = np.zeros((n, 2), dtype=np.float64)
        ptp[:, 0] = self._selected_rrr * self._base_sl_atr * atr  # TP
        ptp[:, 1] = self._base_sl_atr * atr  # SL
        return ptp

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "rrr_variants": [1.5, 2.0, 2.5, 3.0, 4.0],
            "base_sl_atr": 2.0,
        }
