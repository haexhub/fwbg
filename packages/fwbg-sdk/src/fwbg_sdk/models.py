"""Base model plugin class with progress management and structured logging."""
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fwbg_sdk.base import BasePlugin, PluginPhase


@dataclass
class TrainingContext:
    """Everything a model might need for training."""
    sample_weights: Optional[np.ndarray] = None
    categorical_features: Optional[List[str]] = None
    validation_data: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    fold_information: Optional[dict] = None  # {"fold": 3, "total_folds": 8}
    direction: Optional[str] = None  # "long" or "short" — used by signal model


@dataclass
class TrainingStageInfo:
    """Information about a single training stage."""
    stage_name: str
    description: str
    started_at: float = 0.0
    completed_at: float = 0.0
    is_complete: bool = False

    @property
    def duration_seconds(self) -> float:
        if not self.is_complete:
            return time.monotonic() - self.started_at if self.started_at else 0.0
        return self.completed_at - self.started_at


class ModelProgressReporter:
    """Reports model training progress with timing and stage tracking.

    Integrates with fwbg's progress queue system and provides
    model-specific stage tracking, timing, and ETA estimation.

    Usage inside a model plugin:
        self.progress.begin_stage("fitting", "Fitting XGBoost model")
        model.fit(X, y)
        self.progress.complete_stage("fitting")
    """

    def __init__(self, model_name: str, logger: logging.Logger):
        self._model_name = model_name
        self._logger = logger
        self._stages: Dict[str, TrainingStageInfo] = {}
        self._stage_order: List[str] = []
        self._training_started_at: float = 0.0
        self._progress_callback: Optional[Callable[..., Any]] = None

    def set_progress_callback(
        self, callback: Optional[Callable[..., Any]]
    ) -> None:
        """Set external callback for progress updates."""
        self._progress_callback = callback

    def begin_training(self) -> None:
        self._training_started_at = time.monotonic()
        self._stages.clear()
        self._stage_order.clear()
        self._logger.debug(f"[{self._model_name}] Training started")

    def begin_stage(self, stage_name: str, description: str) -> None:
        info = TrainingStageInfo(
            stage_name=stage_name,
            description=description,
            started_at=time.monotonic(),
        )
        self._stages[stage_name] = info
        if stage_name not in self._stage_order:
            self._stage_order.append(stage_name)
        self._logger.debug(f"[{self._model_name}] {description}")
        if self._progress_callback:
            self._progress_callback(
                model_name=self._model_name,
                stage=stage_name,
                description=description,
                elapsed_seconds=self.get_elapsed_seconds(),
            )

    def complete_stage(self, stage_name: str) -> float:
        if stage_name not in self._stages:
            return 0.0
        info = self._stages[stage_name]
        info.completed_at = time.monotonic()
        info.is_complete = True
        duration = info.duration_seconds
        self._logger.debug(
            f"[{self._model_name}] '{stage_name}' completed in {duration:.2f}s"
        )
        return duration

    def complete_training(self) -> float:
        total = time.monotonic() - self._training_started_at
        self._logger.info(
            f"[{self._model_name}] Training completed in {total:.2f}s"
        )
        return total

    def get_elapsed_seconds(self) -> float:
        if self._training_started_at == 0.0:
            return 0.0
        return time.monotonic() - self._training_started_at

    def get_completed_stages(self) -> List[TrainingStageInfo]:
        return [
            self._stages[name] for name in self._stage_order
            if name in self._stages and self._stages[name].is_complete
        ]


class BaseModel(BasePlugin, ABC):
    """Base class for ML model plugins.

    Each model plugin owns the full training loop and decides
    about GPU handling, calibration, parameter reduction, etc.

    Built-in features:
    - Progress reporting via self.progress (ModelProgressReporter)
    - Structured logging via self.logger
    - Default calibration via sklearn (overridable)
    """

    phase = PluginPhase.MODEL
    stateful = True
    cacheable = False

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"fwbg.model.{self.name}")
        self.progress = ModelProgressReporter(self.name, self.logger)
        self._calibrated_model = None

    @abstractmethod
    def train(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        training_context: TrainingContext,
        **hyperparameters: Any,
    ) -> None:
        """Train the model. Full control over the training loop."""
        ...

    def _check_trained(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} model is not trained. Call train() first."
            )

    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities (n_samples, n_classes)."""
        self._check_trained()
        return self._predict_probability_impl(features)

    @abstractmethod
    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        """Implement class probability prediction. Called by predict_probability()."""
        ...

    @property
    def trained_classes(self) -> np.ndarray:
        """Return class labels learned during training."""
        self._check_trained()
        return self._trained_classes_impl

    @property
    @abstractmethod
    def _trained_classes_impl(self) -> np.ndarray:
        """Implement class labels property. Called by trained_classes."""
        ...

    def as_sklearn_estimator(self) -> Any:
        """Return sklearn-compatible estimator (for calibration etc.)."""
        self._check_trained()
        return self._as_sklearn_estimator_impl()

    @abstractmethod
    def _as_sklearn_estimator_impl(self) -> Any:
        """Implement sklearn estimator access. Called by as_sklearn_estimator()."""
        ...

    def calibrate(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        method: str = "isotonic",
    ) -> None:
        """Probability calibration. Default: sklearn CalibratedClassifierCV."""
        from sklearn.calibration import CalibratedClassifierCV

        self.progress.begin_stage("calibration", f"Calibrating probabilities ({method})")
        calibrated = CalibratedClassifierCV(
            self.as_sklearn_estimator(), method=method, cv=3
        )
        calibrated.fit(features, targets)
        self._calibrated_model = calibrated
        self.progress.complete_stage("calibration")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (argmax of predict_probability)."""
        probs = self.predict_probability(features)
        return self.trained_classes[np.argmax(probs, axis=1)]

    def predict_probability_calibrated(self, features: pd.DataFrame) -> np.ndarray:
        """Predict with calibration if available, otherwise raw."""
        if self._calibrated_model is not None:
            return self._calibrated_model.predict_proba(features)
        return self.predict_probability(features)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Feature importance dict. Not every model supports this."""
        return None

    @classmethod
    def get_reduced_hyperparameters(
        cls, hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return reduced-complexity hyperparameters for inner CV.

        Each model decides how to reduce its own complexity.
        Called by nested_cv when use_reduced_params=True.

        Default: returns hyperparameters unchanged (no reduction).
        Override in subclass for model-specific reduction logic.
        """
        return hyperparameters.copy()

    @property
    def is_trained(self) -> bool:
        return self._fitted
