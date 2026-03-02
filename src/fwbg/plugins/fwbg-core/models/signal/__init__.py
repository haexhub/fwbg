"""Signal model plugin — uses pre-computed indicator columns as entry signals.

For rule-based strategies (e.g. ORB breakout + retest) where the entry
decision is deterministic. The pipeline still runs grid search over TP/SL,
but the model simply maps signal columns to probabilities (1.0 or 0.0).

Usage in strategy config:
    "model": {
        "type": "signal",
        "architecture": "long_short_separate",
        "hyperparameters": {
            "signal_column_long": "orb_s0_retest_bull",
            "signal_column_short": "orb_s0_retest_bear"
        }
    }

With ct: [0.5] in the grid, only bars where the signal fires (1.0 >= 0.5)
will trigger trades. Non-signal bars (0.0 < 0.5) are skipped.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from fwbg_sdk.models import BaseModel, TrainingContext
from fwbg_sdk.registry import register_model


@register_model("signal")
class SignalModel(BaseModel):
    """Model that reads pre-computed signal columns instead of training."""

    name = "signal"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._classes = np.array([0, 1])
        self._signal_col: str = ""
        self._start_hour: Optional[int] = None
        self._end_hour: Optional[int] = None

    def train(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        training_context: TrainingContext,
        **hyperparameters: Any,
    ) -> None:
        self._classes = np.array([0, 1])

        # Pick signal column based on direction from TrainingContext
        if training_context.direction == "long":
            self._signal_col = hyperparameters.get("signal_column_long", "")
        else:
            self._signal_col = hyperparameters.get("signal_column_short", "")

        # Fallback: use composed signal columns from signal_rules evaluator
        if not self._signal_col:
            composed = f"_composed_signal_{training_context.direction}"
            if composed in features.columns:
                self._signal_col = composed

        # Session hour filter (injected from indicator_overrides per asset class)
        self._start_hour = hyperparameters.get("signal_start_hour")
        self._end_hour = hyperparameters.get("signal_end_hour")

        self._fitted = True

    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        probs = np.zeros((len(features), 2), dtype=np.float64)
        if self._signal_col and self._signal_col in features.columns:
            probs[:, 1] = features[self._signal_col].fillna(0).clip(0, 1).values
        probs[:, 0] = 1.0 - probs[:, 1]

        # Hour window filter: zero out signals outside allowed hours
        if self._start_hour is not None and self._end_hour is not None:
            if isinstance(features.index, pd.DatetimeIndex):
                hours = features.index.hour
                if self._start_hour < self._end_hour:
                    in_session = (hours >= self._start_hour) & (hours < self._end_hour)
                else:
                    # Crosses midnight (e.g., ASX200: 23-06)
                    in_session = (hours >= self._start_hour) | (hours < self._end_hour)
                probs[~in_session, 1] = 0.0
                probs[~in_session, 0] = 1.0

        return probs

    @property
    def _trained_classes_impl(self) -> np.ndarray:
        return self._classes

    def _as_sklearn_estimator_impl(self) -> Any:
        return None

    @classmethod
    def get_reduced_hyperparameters(
        cls, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        return hyperparameters.copy()
