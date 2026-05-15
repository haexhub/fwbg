# XGBoost Model Plugins Base Class — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate ~120 lines of duplicated infrastructure between `xgboost_rrr` and `xgboost_mfe` model plugins by introducing a shared `BaseStackedXGBoostModel` base class.

**Architecture:** Both plugins implement the same scaffolding: GPU/CPU fallback handler, stage-progress logging, variant stacking loop (per `sl_variant`), sample-weight tiling, and best-variant selection at inference. They differ only in (a) the target they regress (RRR feature vs MFE label) and (b) some defaults. Introduce a base class with template-method hooks for the target-specific bits.

**Tech Stack:** Python, xgboost, numpy, pandas, pytest.

---

## Background

Side-by-side diff of [xgboost_rrr/__init__.py](../../src/fwbg/plugins/fwbg-core/models/xgboost_rrr/__init__.py) and [xgboost_mfe/__init__.py](../../src/fwbg/plugins/fwbg-core/models/xgboost_mfe/__init__.py) shows:

- `_handle_gpu_fallback` is ~35 LOC, byte-identical
- variant stacking loop: ~40 LOC, structurally identical
- `fit_kwargs`/`sample_weight` tiling: identical
- Stage-progress reporting cadence: identical
- Best-variant inference path: nearly identical, differs only in what the model predicts

Only the actual "what is being predicted" differs:
- RRR uses risk-reward-ratio as an *input feature* (one regressor per SL variant)
- MFE regresses *maximum favourable excursion* as a *label*

---

## Success Criteria

1. New `BaseStackedXGBoostModel` class in a non-discoverable directory.
2. Both `xgboost_rrr` and `xgboost_mfe` shrink to <80 LOC each (subclass + target-specific hooks + manifest).
3. All tests in `tests/test_xgboost_rrr_model.py`, `tests/test_xgboost_mfe_model.py`, `tests/test_rrr_mfe_e2e.py`, and `tests/test_base_model_per_trade_params.py` pass.
4. End-to-end strategy run using xgboost_rrr or xgboost_mfe produces metrics within ±1e-6 of the pre-refactor baseline on a fixed-seed run.
5. The `sl_variants` default discrepancy noted in the code review (already fixed) stays consistent — both train() default and get_default_params() return the same list.

---

## Out of Scope

- Adding new model variants.
- Switching XGBoost API version.
- Hyperparameter optimization of the underlying booster.

---

## Task 1: Lock current behaviour with a regression test

**Files:**
- Test: `tests/test_xgboost_models_snapshot.py` (create)

**Step 1: Write deterministic-seed regression**

```python
"""Snapshot xgboost_rrr and xgboost_mfe predictions on fixed seed.

The refactor must not change predicted values for the same training data.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline.registry import get_registry


def _fixture():
    np.random.seed(42)
    n = 500
    X = pd.DataFrame(np.random.randn(n, 8), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.random.randn(n))
    weights = pd.Series(np.ones(n))
    return X, y, weights


@pytest.mark.parametrize("plugin_fqn", ["fwbg-core:xgboost_rrr", "fwbg-core:xgboost_mfe"])
def test_xgboost_predictions_deterministic(plugin_fqn):
    reg = get_registry()
    reg.auto_discover()
    cls = reg.get(plugin_fqn)
    model = cls()

    X, y, w = _fixture()
    hyper = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1, "random_state": 0}
    model.train(X, y, sample_weights=w, hyperparameters=hyper)
    preds = model.predict(X)

    # Save first 5 predictions as a stability check
    pred5 = preds[:5].round(8).tolist() if hasattr(preds, "round") else list(preds[:5])
    print(plugin_fqn, pred5)
    assert len(preds) == len(X)
```

**Step 2: Run, capture, harden into hard equality**

```bash
pytest tests/test_xgboost_models_snapshot.py -v -s
```

Copy printed values into an `EXPECTED` dict and re-assert. Commit as a hard equality test.

**Step 3: Commit**

```bash
git add tests/test_xgboost_models_snapshot.py
git commit -m "test: snapshot xgboost_rrr/mfe predictions for refactor lock"
```

---

## Task 2: Create the base class

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/models/_base/__init__.py`
- Create: `src/fwbg/plugins/fwbg-core/models/_base/stacked_xgb.py`

**Step 1: Confirm `_base` directory is skipped by plugin discovery**

```bash
grep -rn "_base\|underscore" src/fwbg/pipeline/registry.py
```

If not skipped, add a guard in `_discover_plugins_in_category` to skip dirs starting with `_`.

**Step 2: Write the base class**

```python
"""Shared scaffolding for stacked XGBoost model plugins.

Subclasses override `_build_target` (turn a training frame into the
regression target for a given sl_variant) and `_select_best_variant_at`
(pick which variant to use per-row at inference). Everything else — GPU
fallback, stage progress, sample-weight tiling, variant loop — lives here.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from fwbg_sdk import BaseModel


class BaseStackedXGBoostModel(BaseModel):
    """Train one XGBRegressor per `sl_variant`, blend at inference."""

    DEFAULT_SL_VARIANTS: List[float] = [1.5, 2.0, 2.5, 3.0]

    def __init__(self) -> None:
        super().__init__()
        self._models: Dict[float, xgb.XGBRegressor] = {}
        self._sl_variants: List[float] = list(self.DEFAULT_SL_VARIANTS)

    # ---- Hook methods (override in subclasses) --------------------------

    @abstractmethod
    def _build_target(self, X: pd.DataFrame, y: pd.Series, sl: float) -> np.ndarray:
        """Return the regression target for *sl* given the raw label/feature y."""

    @abstractmethod
    def _select_best_variant_at(self, predictions: Dict[float, np.ndarray]) -> np.ndarray:
        """Given per-variant predictions {sl: pred_array}, return the chosen prediction per row."""

    # ---- Shared train / predict ----------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_context=None,
    ) -> None:
        hyper = dict(hyperparameters or {})
        self._sl_variants = list(hyper.pop("sl_variants", self.DEFAULT_SL_VARIANTS))

        self._models.clear()
        for sl in self._sl_variants:
            target = self._build_target(X, y, sl)
            estimator = self._build_estimator(hyper)
            fit_kwargs: Dict[str, Any] = {}
            if sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights.values
            try:
                estimator.fit(X.values, target, **fit_kwargs)
            except xgb.core.XGBoostError:
                estimator = self._handle_gpu_fallback(estimator, hyper)
                estimator.fit(X.values, target, **fit_kwargs)
            self._models[sl] = estimator

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = {sl: model.predict(X.values) for sl, model in self._models.items()}
        return self._select_best_variant_at(preds)

    # ---- Common helpers ------------------------------------------------

    @staticmethod
    def _build_estimator(hyper: Dict[str, Any]) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=hyper.get("n_estimators", 100),
            max_depth=hyper.get("max_depth", 6),
            learning_rate=hyper.get("learning_rate", 0.1),
            tree_method=hyper.get("tree_method", "hist"),
            device=hyper.get("device", "cpu"),
            random_state=hyper.get("random_state", 0),
        )

    @staticmethod
    def _handle_gpu_fallback(estimator: xgb.XGBRegressor, hyper: Dict[str, Any]) -> xgb.XGBRegressor:
        hyper["device"] = "cpu"
        hyper["tree_method"] = "hist"
        return BaseStackedXGBoostModel._build_estimator(hyper)
```

**Step 3: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/models/_base/
git commit -m "feat: add BaseStackedXGBoostModel scaffolding"
```

---

## Task 3: Migrate xgboost_rrr

**Files:**
- Modify: `src/fwbg/plugins/fwbg-core/models/xgboost_rrr/__init__.py`

**Step 1: Subclass and override hooks**

```python
"""xgboost_rrr: stacked XGBoost regressor on RRR-augmented features."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from fwbg_sdk import register_model
from fwbg.plugins._base.stacked_xgb import BaseStackedXGBoostModel


@register_model("xgboost_rrr")
class XGBoostRRRModel(BaseStackedXGBoostModel):
    """RRR is used as a feature; target stays as-is per variant."""

    def _build_target(self, X: pd.DataFrame, y: pd.Series, sl: float) -> np.ndarray:
        return y.values  # unchanged — sl effect is in X via rrr feature

    def _select_best_variant_at(self, predictions: Dict[float, np.ndarray]) -> np.ndarray:
        # rrr: pick the variant with the highest expected return per row
        stacked = np.stack([predictions[sl] for sl in sorted(predictions)])
        best = stacked.max(axis=0)
        return best
```

NOTE: Preserve any RRR-specific feature engineering (e.g. "_rrr" column injection per variant) that lives in the current implementation — port it into `_build_target` or a `_prepare_features_for_variant` hook if needed. Check the diff carefully before deleting.

**Step 2: Run RRR test suites**

```bash
pytest tests/test_xgboost_rrr_model.py tests/test_xgboost_models_snapshot.py::test_xgboost_predictions_deterministic[fwbg-core:xgboost_rrr] -v
```
Expected: PASS.

**Step 3: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/models/xgboost_rrr/__init__.py
git commit -m "refactor: migrate xgboost_rrr to BaseStackedXGBoostModel"
```

---

## Task 4: Migrate xgboost_mfe

**Files:**
- Modify: `src/fwbg/plugins/fwbg-core/models/xgboost_mfe/__init__.py`

**Step 1: Subclass with MFE-specific target builder**

```python
"""xgboost_mfe: stacked XGBoost regressor on MFE labels per SL variant."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from fwbg_sdk import register_model
from fwbg.plugins._base.stacked_xgb import BaseStackedXGBoostModel
from fwbg.optimization.targets import compute_mfe_targets


@register_model("xgboost_mfe")
class XGBoostMFEModel(BaseStackedXGBoostModel):
    def _build_target(self, X: pd.DataFrame, y: pd.Series, sl: float) -> np.ndarray:
        # `y` here carries the OHLC frame the trainer attached; build per-variant target.
        # Cross-check the existing implementation for the exact contract!
        long_mfe, _ = compute_mfe_targets(y.attrs["df"], sl_atr=sl)
        return long_mfe

    def _select_best_variant_at(self, predictions: Dict[float, np.ndarray]) -> np.ndarray:
        stacked = np.stack([predictions[sl] for sl in sorted(predictions)])
        return stacked.max(axis=0)
```

NOTE: The `y.attrs["df"]` pattern above is illustrative. The current xgboost_mfe implementation has its own way of receiving the raw OHLC — read the existing code, match it.

**Step 2: Run MFE test suites**

```bash
pytest tests/test_xgboost_mfe_model.py tests/test_mfe_target_integration.py tests/test_rrr_mfe_e2e.py -v
```

**Step 3: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/models/xgboost_mfe/__init__.py
git commit -m "refactor: migrate xgboost_mfe to BaseStackedXGBoostModel"
```

---

## Task 5: End-to-end verification

**Step 1: Run the full xgboost-touching test set**

```bash
pytest tests/test_xgboost_rrr_model.py \
       tests/test_xgboost_mfe_model.py \
       tests/test_rrr_mfe_e2e.py \
       tests/test_base_model_per_trade_params.py \
       tests/test_mfe_target_integration.py -x
```

**Step 2: Run a fixed-seed strategy that uses xgboost_rrr or mfe**

```bash
fwbg --assets EURUSD --strategy-file strategies/configs/<an-xgb-strategy>.json
```
Confirm the run completes and the produced metrics file matches the pre-refactor baseline (commit hash of `feat/indicators-restructure-and-models` HEAD).

**Step 3: Verify LOC reduction**

```bash
wc -l src/fwbg/plugins/fwbg-core/models/{xgboost_rrr,xgboost_mfe}/__init__.py
```
Each should be <80 LOC.
