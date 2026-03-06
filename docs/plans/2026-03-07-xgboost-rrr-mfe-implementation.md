# XGBoost RRR & MFE Model Plugins — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement two new model plugins (`xgboost_rrr`, `xgboost_mfe`) that reframe the ML problem from binary classification to RRR-aware classification and MFE regression, plus a `per_trade_params` mechanism for dynamic per-trade TP/SL.

**Architecture:** New model plugins in `src/fwbg/plugins/fwbg-core/models/`, extending `BaseModel` from fwbg-sdk. Both use dataset stacking (multiple TP/SL variants with an extra feature column) and per-trade TP/SL overrides via `get_per_trade_params()`. The existing pipeline (grid search, nested CV, exit strategies) stays unchanged — these models integrate via the existing `predict_probability()` + CT mechanism.

**Tech Stack:** XGBoost (XGBClassifier for RRR, XGBRegressor for MFE), NumPy, Pandas, fwbg-sdk BaseModel

**Design doc:** `docs/plans/2026-03-07-xgboost-rrr-mfe-models.md`

---

## Task 1: Add `get_per_trade_params()` to BaseModel

**Files:**
- Modify: `packages/fwbg-sdk/src/fwbg_sdk/models.py:220` (after `get_feature_importance`)
- Test: `tests/test_base_model_per_trade_params.py`

**Step 1: Write the failing test**

Create `tests/test_base_model_per_trade_params.py`:

```python
"""Test that BaseModel exposes get_per_trade_params with default None."""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import BaseModel, TrainingContext


class DummyModel(BaseModel):
    name = "dummy"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._classes = np.array([0, 1])

    def train(self, features, targets, training_context, **hp):
        self._fitted = True

    def _predict_probability_impl(self, features):
        n = len(features)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    @property
    def _trained_classes_impl(self):
        return self._classes

    def _as_sklearn_estimator_impl(self):
        return None


class TestGetPerTradeParams:
    def test_default_returns_none(self):
        model = DummyModel()
        model.train(pd.DataFrame({"a": [1, 2]}), np.array([0, 1]), TrainingContext())
        result = model.get_per_trade_params(pd.DataFrame({"a": [1, 2]}))
        assert result is None

    def test_method_exists_on_base(self):
        assert hasattr(BaseModel, "get_per_trade_params")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_base_model_per_trade_params.py -v`
Expected: FAIL with `AttributeError: 'DummyModel' object has no attribute 'get_per_trade_params'`

**Step 3: Write minimal implementation**

In `packages/fwbg-sdk/src/fwbg_sdk/models.py`, add after `get_feature_importance()` (line ~222):

```python
def get_per_trade_params(
    self, features: pd.DataFrame, atr: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Return per-sample TP/SL overrides as absolute price distances.

    Models that select dynamic TP/SL per trade (e.g. xgboost_rrr, xgboost_mfe)
    override this. The returned array is used by _simulate_trades_core to
    override the global tp_dists/sl_dists per trade entry.

    Args:
        features: The same feature DataFrame passed to predict_probability().
        atr: Per-bar ATR values (absolute). Needed to convert ATR multiples
             to price distances.

    Returns:
        None (use global TP/SL) or ndarray of shape (n_samples, 2)
        where column 0 = TP distance, column 1 = SL distance.
    """
    return None
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_base_model_per_trade_params.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/fwbg-sdk/src/fwbg_sdk/models.py tests/test_base_model_per_trade_params.py
git commit -m "feat: add get_per_trade_params() to BaseModel with default None"
```

---

## Task 2: Wire `per_trade_params` into `_simulate_trades_core`

**Files:**
- Modify: `src/fwbg/optimization/targets.py:77-301` (`_simulate_trades_core` function)
- Test: `tests/test_per_trade_params_simulation.py`

**Step 1: Write the failing test**

Create `tests/test_per_trade_params_simulation.py`:

```python
"""Test that _simulate_trades_core uses per_trade_params when provided."""
import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.targets import _simulate_trades_core
from fwbg.core.context import SimulationContext


def _make_df(n=50):
    """Create a minimal OHLC DataFrame with uptrend for long signals."""
    np.random.seed(42)
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "O": base,
        "H": base + abs(np.random.randn(n)) * 0.3,
        "L": base - abs(np.random.randn(n)) * 0.3,
        "C": base + np.random.randn(n) * 0.1,
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    df["_regime"] = 7  # all directions allowed
    return df


def _make_probs(n, p_win=0.8):
    """Create mock probability array (n, 2) with high win probability."""
    probs = np.column_stack([np.full(n, 1 - p_win), np.full(n, p_win)])
    return probs


class TestPerTradeParams:
    def test_per_trade_params_overrides_tp_sl(self):
        df = _make_df(50)
        n = len(df)
        probs = _make_probs(n)

        ctx = SimulationContext.create_minimal(symbol="TEST", spread=0.5)

        # per_trade_params: huge TP (never hit), tiny SL (always hit)
        ptp = np.zeros((n, 2), dtype=np.float64)
        ptp[:, 0] = 1000.0  # TP distance — unreachable
        ptp[:, 1] = 0.001   # SL distance — instant stop

        result = _simulate_trades_core(
            df, probs, probs, 1, 1,
            ct_long=0.5, ct_short=0.5,
            tp=2, sl=1, ctx=ctx,
            per_trade_params=ptp,
        )
        trades = result["trades"]
        # With tiny SL, all trades should lose
        if trades:
            assert all(t["result"] == -1.0 for t in trades)

    def test_none_per_trade_params_uses_global(self):
        """When per_trade_params is None, behavior is unchanged."""
        df = _make_df(50)
        n = len(df)
        probs = _make_probs(n)

        ctx = SimulationContext.create_minimal(symbol="TEST", spread=0.5)

        result_without = _simulate_trades_core(
            df, probs, probs, 1, 1,
            ct_long=0.5, ct_short=0.5,
            tp=2, sl=1, ctx=ctx,
            per_trade_params=None,
        )
        result_default = _simulate_trades_core(
            df, probs, probs, 1, 1,
            ct_long=0.5, ct_short=0.5,
            tp=2, sl=1, ctx=ctx,
        )
        assert len(result_without["trades"]) == len(result_default["trades"])
```

**Context:** `SimulationContext.create_minimal` may not exist. Check `context.py` for a minimal constructor. If it doesn't exist, use:
```python
from fwbg.core.config import StrategyConfig, AssetConfig
ctx = SimulationContext.create(AssetConfig(symbol="TEST", spread=0.5), StrategyConfig())
```
Adjust the test accordingly based on what's available.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_per_trade_params_simulation.py -v`
Expected: FAIL with `TypeError: _simulate_trades_core() got an unexpected keyword argument 'per_trade_params'`

**Step 3: Add `per_trade_params` parameter to `_simulate_trades_core`**

In `src/fwbg/optimization/targets.py`, modify `_simulate_trades_core` signature (line 77):

```python
def _simulate_trades_core(
    df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    ct_long: float,
    ct_short: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    return_detailed: bool = False,
    timeout_bars: int = None,
    direction_filter: int = None,
    per_trade_params: Optional[np.ndarray] = None,  # NEW
) -> Dict[str, Any]:
```

Then inside the function, after computing `tp_dists` and `sl_dists` (around line 127), add the override logic:

```python
    # TP/SL-Distanzen vom Exit-Strategy-Plugin berechnen lassen
    tp_dists, sl_dists = _resolve_distances(df, tp, sl, ctx)

    # Per-trade TP/SL overrides from model (xgboost_rrr, xgboost_mfe)
    if per_trade_params is not None:
        tp_dists = per_trade_params[:, 0].copy()
        sl_dists = per_trade_params[:, 1].copy()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_per_trade_params_simulation.py -v`
Expected: PASS

**Step 5: Also update wrapper functions that call `_simulate_trades_core`**

Find all callers of `_simulate_trades_core` in `targets.py`:
- `simulate_trades_sequential` (~line 388)
- `simulate_trades_sequential_separate_ct` (~line 410)

Add `per_trade_params=None` parameter to both and pass through:

```python
def simulate_trades_sequential(
    df, probs_long, probs_short, long_win_idx, short_win_idx,
    ct, tp, sl, ctx, return_detailed=False, timeout_bars=None,
    per_trade_params=None,  # NEW
):
    return _simulate_trades_core(
        ...,
        per_trade_params=per_trade_params,
    )
```

Same for `simulate_trades_sequential_separate_ct`.

**Step 6: Update `evaluate_on_holdout` in `nested_cv.py`**

In `src/fwbg/optimization/nested_cv.py`, around lines 556-570, after getting probs, add per_trade_params retrieval:

```python
    # Per-trade TP/SL overrides from model
    per_trade_params_long = None
    per_trade_params_short = None
    if mod_long is not None and hasattr(mod_long, 'get_per_trade_params'):
        atr_col = "_atr" if "_atr" in holdout_df.columns else ("vol_atr" if "vol_atr" in holdout_df.columns else None)
        atr_vals = holdout_df[atr_col].values if atr_col else None
        ptp = mod_long.get_per_trade_params(holdout_df[features_long], atr=atr_vals)
        if ptp is not None:
            per_trade_params_long = ptp
    if mod_short is not None and hasattr(mod_short, 'get_per_trade_params'):
        atr_col = "_atr" if "_atr" in holdout_df.columns else ("vol_atr" if "vol_atr" in holdout_df.columns else None)
        atr_vals = holdout_df[atr_col].values if atr_col else None
        ptp = mod_short.get_per_trade_params(holdout_df[features_short], atr=atr_vals)
        if ptp is not None:
            per_trade_params_short = ptp

    # Merge: use long params for long entries, short params for short entries
    # For now, use long params (both models set params directionally)
    per_trade_params = per_trade_params_long  # will be None for standard models
```

Then pass `per_trade_params=per_trade_params` to the `simulate_trades_sequential` / `simulate_trades_sequential_separate_ct` calls.

**Step 7: Commit**

```bash
git add src/fwbg/optimization/targets.py src/fwbg/optimization/nested_cv.py tests/test_per_trade_params_simulation.py
git commit -m "feat: wire per_trade_params through simulation pipeline"
```

---

## Task 3: MFE Target Computation

Both models need MFE computation. This is a new function in `targets.py`.

**Files:**
- Modify: `src/fwbg/optimization/targets.py` (add `compute_mfe_targets` function)
- Test: `tests/test_mfe_targets.py`

**Step 1: Write the failing test**

Create `tests/test_mfe_targets.py`:

```python
"""Test MFE target computation."""
import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.targets import compute_mfe_targets


def _make_trending_df(n=100, direction="up"):
    """Create OHLC data with a clear trend for MFE testing."""
    np.random.seed(42)
    if direction == "up":
        base = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.2
    else:
        base = 200.0 - np.arange(n) * 0.5 + np.random.randn(n) * 0.2
    df = pd.DataFrame({
        "O": base,
        "H": base + abs(np.random.randn(n)) * 0.5,
        "L": base - abs(np.random.randn(n)) * 0.5,
        "C": base + np.random.randn(n) * 0.1,
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    # ATR column
    df["_atr"] = 1.0  # constant ATR for easy MFE calculation
    return df


class TestComputeMfeTargets:
    def test_returns_correct_shape(self):
        df = _make_trending_df(100)
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=20, spread=0.5
        )
        assert mfe_long.shape == (100,)
        assert mfe_short.shape == (100,)

    def test_mfe_non_negative(self):
        df = _make_trending_df(100)
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=20, spread=0.5
        )
        assert np.all(mfe_long >= 0.0)
        assert np.all(mfe_short >= 0.0)

    def test_mfe_in_atr_units(self):
        """MFE should be normalized by ATR."""
        df = _make_trending_df(100)
        df["_atr"] = 2.0  # double ATR → halve MFE values
        mfe_long_2, _ = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.5)
        df["_atr"] = 1.0
        mfe_long_1, _ = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.5)
        # With doubled ATR, MFE in ATR units should be roughly halved
        # (not exact due to SL also scaling with ATR)
        ratio = np.nanmean(mfe_long_2) / np.nanmean(mfe_long_1) if np.nanmean(mfe_long_1) > 0 else 0
        assert 0.3 < ratio < 0.8  # roughly half

    def test_uptrend_favors_long_mfe(self):
        """In an uptrend, long MFE should be higher than short MFE."""
        df = _make_trending_df(100, direction="up")
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=50, spread=0.1
        )
        assert np.nanmean(mfe_long) > np.nanmean(mfe_short)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mfe_targets.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_mfe_targets'`

**Step 3: Implement `compute_mfe_targets`**

Add to `src/fwbg/optimization/targets.py` (after `compute_targets_cached`, around line 517):

```python
def compute_mfe_targets(
    df: pd.DataFrame,
    sl_atr: float,
    max_bars: int = 50,
    spread: float = 0.0,
    atr_col: str = "_atr",
    timeout_bars: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Maximum Favorable Excursion in ATR multiples per bar.

    For each bar, simulates a hypothetical long and short trade with the
    given SL (in ATR multiples). Tracks the maximum favorable price movement
    before the trade is stopped out or times out. Returns MFE normalized
    by ATR.

    Args:
        df: DataFrame with OHLC and ATR column.
        sl_atr: Stop-loss in ATR multiples.
        max_bars: Maximum trade duration to consider.
        spread: Bid-ask spread.
        atr_col: Name of ATR column (default "_atr").
        timeout_bars: Optional trade timeout.

    Returns:
        (mfe_long, mfe_short): Arrays of shape (n,) with MFE in ATR multiples.
    """
    if atr_col not in df.columns:
        fallback = "vol_atr" if "vol_atr" in df.columns else None
        if fallback:
            atr_col = fallback
        else:
            raise ValueError(f"ATR column '{atr_col}' not found in DataFrame")

    closes = df["C"].values
    highs = df["H"].values
    lows = df["L"].values
    opens = df["O"].values
    atr = df[atr_col].values.astype(np.float64)
    n = len(df)
    effective_timeout = timeout_bars or max_bars

    mfe_long = np.zeros(n, dtype=np.float64)
    mfe_short = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        sl_dist = atr_i * sl_atr
        entry_price = opens[i + 1] if i + 1 < n else closes[i]  # entry_delay=1

        # Long MFE
        max_favorable = 0.0
        for j in range(i + 1, min(i + 1 + effective_timeout, n)):
            favorable = highs[j] - entry_price - spread
            max_favorable = max(max_favorable, favorable)
            adverse = entry_price - lows[j] + spread
            if adverse >= sl_dist:
                break
        mfe_long[i] = max_favorable / atr_i

        # Short MFE
        max_favorable = 0.0
        for j in range(i + 1, min(i + 1 + effective_timeout, n)):
            favorable = entry_price - lows[j] - spread
            max_favorable = max(max_favorable, favorable)
            adverse = highs[j] - entry_price + spread
            if adverse >= sl_dist:
                break
        mfe_short[i] = max_favorable / atr_i

    return mfe_long, mfe_short
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mfe_targets.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/fwbg/optimization/targets.py tests/test_mfe_targets.py
git commit -m "feat: add compute_mfe_targets for MFE regression targets"
```

---

## Task 4: Implement `xgboost_rrr` Model Plugin

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/models/xgboost_rrr/__init__.py`
- Modify: `src/fwbg/plugins/fwbg-core/models/__init__.py` (add import)
- Test: `tests/test_xgboost_rrr_model.py`

**Step 1: Write the failing test**

Create `tests/test_xgboost_rrr_model.py`:

```python
"""Tests for the xgboost_rrr model plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext


class TestXGBoostRRRModel:
    @pytest.fixture
    def model(self):
        from fwbg.core.registry import get_model
        model_class = get_model("xgboost_rrr")
        return model_class()

    @pytest.fixture
    def training_data(self):
        """Create training data with features and multi-RRR targets."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
        })
        # Binary targets: 60% win rate
        targets = (np.random.rand(n) > 0.4).astype(np.float64)
        return features, targets

    @pytest.fixture
    def atr_values(self):
        return np.full(200, 1.5)

    def test_registration(self):
        from fwbg.core.registry import get_model
        model_class = get_model("xgboost_rrr")
        assert model_class.name == "xgboost_rrr"

    def test_train_and_predict(self, model, training_data):
        features, targets = training_data
        ctx = TrainingContext(direction="long")
        model.train(
            features, targets, ctx,
            rrr_variants=[1.5, 2.0, 3.0],
            base_sl_atr=2.0,
        )
        assert model.is_trained

        probs = model.predict_probability(features)
        assert probs.shape == (len(features), 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_trained_classes(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), rrr_variants=[2.0, 3.0])
        assert 0 in model.trained_classes
        assert 1 in model.trained_classes

    def test_selected_rrr(self, model, training_data):
        features, targets = training_data
        model.train(
            features, targets, TrainingContext(),
            rrr_variants=[1.5, 2.0, 3.0],
        )
        model.predict_probability(features)
        assert model.selected_rrr is not None
        assert model.selected_rrr.shape == (len(features),)
        assert all(r in [1.5, 2.0, 3.0] for r in model.selected_rrr)

    def test_get_per_trade_params(self, model, training_data, atr_values):
        features, targets = training_data
        model.train(
            features, targets, TrainingContext(),
            rrr_variants=[1.5, 2.0, 3.0],
            base_sl_atr=2.0,
        )
        model.predict_probability(features)
        ptp = model.get_per_trade_params(features, atr=atr_values)
        assert ptp is not None
        assert ptp.shape == (len(features), 2)
        # TP distance = selected_rrr * base_sl_atr * atr
        # SL distance = base_sl_atr * atr
        assert np.all(ptp[:, 1] == 2.0 * 1.5)  # base_sl_atr * atr

    def test_get_per_trade_params_none_without_predict(self, model, training_data, atr_values):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), rrr_variants=[2.0])
        # Without calling predict first, should return None
        ptp = model.get_per_trade_params(features, atr=atr_values)
        assert ptp is None

    def test_feature_importance(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), rrr_variants=[2.0, 3.0])
        importance = model.get_feature_importance()
        assert importance is not None
        assert "rrr" in importance  # RRR should appear as a feature
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_xgboost_rrr_model.py -v`
Expected: FAIL with `ValueError: Unknown model: 'xgboost_rrr'`

**Step 3: Create the plugin**

Create directory and file `src/fwbg/plugins/fwbg-core/models/xgboost_rrr/__init__.py`:

```python
"""XGBoost RRR-as-Feature model plugin.

Trains a single XGBClassifier on multiple RRR variants simultaneously.
RRR (reward-risk ratio = tp/sl) is added as an input feature, letting
the model learn which RRR works best for each market setup.

At inference, all RRR variants are scored and the best one is selected
per sample.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from fwbg_sdk.models import BaseModel, TrainingContext
from fwbg_sdk.registry import register_model


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

        self.progress.begin_stage("stacking", "Stacking dataset for RRR variants")

        # Stack: duplicate dataset for each RRR variant, add rrr column
        stacked_features = []
        stacked_targets = []
        for rrr in self._rrr_variants:
            df_copy = features.copy()
            df_copy["rrr"] = rrr
            stacked_features.append(df_copy)
            stacked_targets.append(targets.copy())

        X_stacked = pd.concat(stacked_features, ignore_index=True)
        y_stacked = np.concatenate(stacked_targets)

        # Handle sample weights — stack them too
        sample_weight = None
        if training_context.sample_weights is not None:
            sample_weight = np.tile(training_context.sample_weights, len(self._rrr_variants))

        self.progress.complete_stage("stacking")
        self.logger.info(
            f"Stacked {len(features)} samples × {len(self._rrr_variants)} RRR variants "
            f"= {len(X_stacked)} rows"
        )

        # Prepare XGBoost params
        self.progress.begin_stage("prepare_parameters", "Preparing XGBoost parameters")
        params = hyperparameters.copy()
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", 0)
        params["n_jobs"] = get_xgboost_n_jobs()
        params.update(get_xgboost_params())
        self.progress.complete_stage("prepare_parameters")

        # Train
        self.progress.begin_stage("fitting", "Fitting XGBoost model on stacked data")
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
                self._handle_gpu_fallback(X_stacked, y_stacked, params, fit_kwargs)
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

    def _handle_gpu_fallback(self, X, y, params, fit_kwargs):
        from fwbg.utils.xgb_config import disable_gpu
        from xgboost import XGBClassifier

        self.progress.begin_stage("gpu_fallback", "GPU failed, falling back to CPU")
        self.logger.warning("CUDA error — falling back to CPU")
        disable_gpu()
        cpu_params = {k: v for k, v in params.items() if k not in ("device", "tree_method")}
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
        self, features: pd.DataFrame, atr: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Return per-trade TP/SL based on selected RRR."""
        if self._selected_rrr is None or atr is None:
            return None

        n = len(features)
        ptp = np.zeros((n, 2), dtype=np.float64)
        ptp[:, 0] = self._selected_rrr * self._base_sl_atr * atr  # TP distance
        ptp[:, 1] = self._base_sl_atr * atr                        # SL distance
        return ptp

    @classmethod
    def get_reduced_hyperparameters(cls, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        reduced = hyperparameters.copy()
        reduced["n_estimators"] = max(10, reduced.get("n_estimators", 100) // 2)
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
```

**Step 4: Register the plugin**

Modify `src/fwbg/plugins/fwbg-core/models/__init__.py`:

```python
"""Core model plugins."""
from . import xgboost  # noqa: F401
from . import xgboost_rrr  # noqa: F401
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_xgboost_rrr_model.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/models/xgboost_rrr/ src/fwbg/plugins/fwbg-core/models/__init__.py tests/test_xgboost_rrr_model.py
git commit -m "feat: add xgboost_rrr model plugin (RRR as feature)"
```

---

## Task 5: Implement `xgboost_mfe` Model Plugin

**Files:**
- Create: `src/fwbg/plugins/fwbg-core/models/xgboost_mfe/__init__.py`
- Modify: `src/fwbg/plugins/fwbg-core/models/__init__.py` (add import)
- Test: `tests/test_xgboost_mfe_model.py`

**Step 1: Write the failing test**

Create `tests/test_xgboost_mfe_model.py`:

```python
"""Tests for the xgboost_mfe model plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext


class TestXGBoostMFEModel:
    @pytest.fixture
    def model(self):
        from fwbg.core.registry import get_model
        model_class = get_model("xgboost_mfe")
        return model_class()

    @pytest.fixture
    def training_data(self):
        """Create training data with features and MFE targets."""
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
        })
        # Continuous MFE targets (in ATR multiples)
        targets = np.abs(np.random.randn(n)) * 2.0  # MFE values 0-6 ATR
        return features, targets

    @pytest.fixture
    def atr_values(self):
        return np.full(200, 1.5)

    def test_registration(self):
        from fwbg.core.registry import get_model
        model_class = get_model("xgboost_mfe")
        assert model_class.name == "xgboost_mfe"

    def test_train_and_predict(self, model, training_data):
        features, targets = training_data
        model.train(
            features, targets, TrainingContext(),
            sl_variants=[1.5, 2.0, 3.0],
        )
        assert model.is_trained

        probs = model.predict_probability(features)
        assert probs.shape == (len(features), 2)
        # Column 1 should be predicted MFE (non-negative)
        assert np.all(probs[:, 1] >= 0.0)

    def test_trained_classes(self, model, training_data):
        """MFE model uses pseudo-classes [0, 1] for pipeline compatibility."""
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[2.0])
        assert 0 in model.trained_classes
        assert 1 in model.trained_classes

    def test_predicted_mfe_reasonable(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[2.0])
        probs = model.predict_probability(features)
        predicted_mfe = probs[:, 1]
        # Should be in a reasonable range (0-10 ATR)
        assert np.all(predicted_mfe >= 0.0)
        assert np.all(predicted_mfe < 50.0)

    def test_get_per_trade_params(self, model, training_data, atr_values):
        features, targets = training_data
        model.train(
            features, targets, TrainingContext(),
            sl_variants=[1.5, 2.0, 3.0],
        )
        model.predict_probability(features)
        ptp = model.get_per_trade_params(features, atr=atr_values)
        assert ptp is not None
        assert ptp.shape == (len(features), 2)
        # TP = predicted_mfe * atr, SL = selected_sl_atr * atr
        assert np.all(ptp[:, 0] >= 0.0)  # TP distances
        assert np.all(ptp[:, 1] > 0.0)   # SL distances

    def test_selects_best_mfe_sl_ratio(self, model, training_data):
        """Model should select SL variant with best MFE/SL ratio."""
        features, targets = training_data
        model.train(
            features, targets, TrainingContext(),
            sl_variants=[1.0, 2.0, 5.0],
        )
        model.predict_probability(features)
        assert model.selected_sl_atr is not None
        assert model.selected_sl_atr.shape == (len(features),)
        assert all(s in [1.0, 2.0, 5.0] for s in model.selected_sl_atr)

    def test_feature_importance(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[2.0])
        importance = model.get_feature_importance()
        assert importance is not None
        assert "sl_atr" in importance
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_xgboost_mfe_model.py -v`
Expected: FAIL with `ValueError: Unknown model: 'xgboost_mfe'`

**Step 3: Create the plugin**

Create directory and file `src/fwbg/plugins/fwbg-core/models/xgboost_mfe/__init__.py`:

```python
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
            sample_weight = np.tile(training_context.sample_weights, len(self._sl_variants))

        self.progress.complete_stage("stacking")
        self.logger.info(
            f"Stacked {len(features)} samples × {len(self._sl_variants)} SL variants "
            f"= {len(X_stacked)} rows"
        )

        # Prepare XGBoost params
        self.progress.begin_stage("prepare_parameters", "Preparing XGBoost parameters")
        params = hyperparameters.copy()
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", 0)
        params.setdefault("objective", "reg:squarederror")
        params["n_jobs"] = get_xgboost_n_jobs()
        params.update(get_xgboost_params())
        self.progress.complete_stage("prepare_parameters")

        # Train regressor
        self.progress.begin_stage("fitting", "Fitting XGBRegressor on stacked data")
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
                self._handle_gpu_fallback(X_stacked, y_stacked, params, fit_kwargs)
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

    def _handle_gpu_fallback(self, X, y, params, fit_kwargs):
        from fwbg.utils.xgb_config import disable_gpu
        from xgboost import XGBRegressor

        self.progress.begin_stage("gpu_fallback", "GPU failed, falling back to CPU")
        disable_gpu()
        cpu_params = {k: v for k, v in params.items() if k not in ("device", "tree_method")}
        cpu_params["tree_method"] = "hist"
        cpu_params["device"] = "cpu"
        self._model = XGBRegressor(**cpu_params)
        self._model.fit(X, y, **fit_kwargs)
        self.progress.complete_stage("gpu_fallback")

    def _predict_probability_impl(self, features: pd.DataFrame) -> np.ndarray:
        """Score all SL variants, select best MFE/SL ratio per sample.

        Returns (n, 2) array where column 1 = predicted MFE in ATR.
        This is NOT a probability — it's repurposed so the CT mechanism
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
        self, features: pd.DataFrame, atr: Optional[np.ndarray] = None
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
    def get_reduced_hyperparameters(cls, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        reduced = hyperparameters.copy()
        reduced["n_estimators"] = max(10, reduced.get("n_estimators", 100) // 2)
        return reduced

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "sl_variants": [1.5, 2.0, 2.5, 3.0],
        }
```

**Step 4: Register the plugin**

Modify `src/fwbg/plugins/fwbg-core/models/__init__.py`:

```python
"""Core model plugins."""
from . import xgboost  # noqa: F401
from . import xgboost_rrr  # noqa: F401
from . import xgboost_mfe  # noqa: F401
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_xgboost_mfe_model.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/fwbg/plugins/fwbg-core/models/xgboost_mfe/ src/fwbg/plugins/fwbg-core/models/__init__.py tests/test_xgboost_mfe_model.py
git commit -m "feat: add xgboost_mfe model plugin (MFE regression)"
```

---

## Task 6: Handle Target Computation for New Model Types

The new models need different target computation. For `xgboost_rrr`, targets must be computed per RRR variant. For `xgboost_mfe`, MFE targets replace binary targets.

**Key insight:** The models handle their own target stacking internally during `train()`. The pipeline passes standard binary targets (from the exit strategy), and the models transform them as needed. For MFE, we need the pipeline to compute MFE targets instead of binary targets when the model type is `xgboost_mfe`.

**Files:**
- Modify: `src/fwbg/optimization/nested_cv.py:246-339` (`_evaluate_single_fold`)
- Modify: `src/fwbg/optimization/nested_cv.py:502-602` (`evaluate_on_holdout`)
- Test: `tests/test_mfe_target_integration.py`

**Step 1: Write the failing test**

Create `tests/test_mfe_target_integration.py`:

```python
"""Test that the pipeline computes MFE targets when model_type is xgboost_mfe."""
import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.targets import compute_mfe_targets


class TestMFETargetIntegration:
    def test_mfe_targets_compatible_with_train(self):
        """MFE targets can be used as training targets for xgboost_mfe."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "O": 100.0 + np.random.randn(n) * 0.5,
            "H": 101.0 + np.random.randn(n) * 0.5,
            "L": 99.0 + np.random.randn(n) * 0.5,
            "C": 100.0 + np.random.randn(n) * 0.5,
            "_atr": np.full(n, 1.0),
            "feat_1": np.random.randn(n),
        }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))

        mfe_long, mfe_short = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.5)

        from fwbg.core.registry import get_model
        model_class = get_model("xgboost_mfe")
        model = model_class()

        from fwbg_sdk.models import TrainingContext
        model.train(
            df[["feat_1"]], mfe_long, TrainingContext(direction="long"),
            sl_variants=[2.0],
        )
        assert model.is_trained

        probs = model.predict_probability(df[["feat_1"]])
        assert probs.shape == (n, 2)
        assert np.all(probs[:, 1] >= 0.0)
```

**Step 2: Run test**

Run: `python -m pytest tests/test_mfe_target_integration.py -v`
Expected: PASS (this tests integration, should work with code from Tasks 3+5)

**Step 3: Add model-type-aware target computation to `_evaluate_single_fold`**

In `src/fwbg/optimization/nested_cv.py`, inside `_evaluate_single_fold` (around line 277), add:

```python
    # For MFE models, compute MFE targets instead of binary targets
    if ctx.model_type == "xgboost_mfe":
        from fwbg.optimization.targets import compute_mfe_targets
        sl_variants = ctx.model_hyperparameters.get("sl_variants", [2.0])
        mfe_long, mfe_short = compute_mfe_targets(
            train_df, sl_atr=sl_variants[0], max_bars=ctx.max_trade_bars or 50,
            spread=ctx.spread,
        )
        # For feature selection and validation, we still need binary targets
        # (is the trade profitable at this TP/SL?)
        targets_long_binary, targets_short_binary, has_long, has_short = compute_targets(
            train_df, tp, sl, ctx, timeout_bars
        )
        # Use MFE for training, binary for validation
        targets_long = mfe_long
        targets_short = mfe_short
    else:
        # ... existing code ...
```

**Important:** This is a more complex integration. The exact changes depend on the structure of `_evaluate_single_fold`. The model's `train()` method needs MFE targets, but `evaluate_on_validation` still needs binary targets for CT optimization. Consider passing both target types.

**Alternative simpler approach:** Let the model internally compute MFE targets from the raw OHLC data. Pass the full DataFrame (not just features) to the model, and let it compute MFE in `train()`. This keeps the pipeline changes minimal.

For the simpler approach, modify `train_model()` in `nested_cv.py` to pass the full DataFrame as part of `TrainingContext`:

```python
# In train_model(), add to TrainingContext:
training_context = TrainingContext(
    sample_weights=sample_weight,
    direction=direction,
    fold_information={"full_df": train_df} if ctx.model_type in ("xgboost_mfe", "xgboost_rrr") else None,
)
```

Then in `XGBoostMFEModel.train()`, extract the full DataFrame from `training_context.fold_information` and compute MFE targets internally.

**Decision for implementor:** Choose the simpler approach (model computes MFE internally). This is cleaner because it keeps target computation logic inside the model plugin.

**Step 4: Commit**

```bash
git add src/fwbg/optimization/nested_cv.py tests/test_mfe_target_integration.py
git commit -m "feat: integrate MFE target computation into training pipeline"
```

---

## Task 7: End-to-End Integration Test

**Files:**
- Test: `tests/test_rrr_mfe_e2e.py`

**Step 1: Write end-to-end test**

Create `tests/test_rrr_mfe_e2e.py`:

```python
"""End-to-end tests for xgboost_rrr and xgboost_mfe models.

Tests the full flow: features → train → predict → per_trade_params → simulation.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext
from fwbg.optimization.targets import _simulate_trades_core


def _make_market_df(n=200):
    """Create realistic OHLC data with indicators."""
    np.random.seed(42)
    base = 100.0 + np.cumsum(np.random.randn(n) * 0.3)
    df = pd.DataFrame({
        "O": base,
        "H": base + abs(np.random.randn(n)) * 0.5,
        "L": base - abs(np.random.randn(n)) * 0.5,
        "C": base + np.random.randn(n) * 0.1,
        "_atr": np.full(n, 1.0),
        "_regime": np.full(n, 7, dtype=np.int8),
        "feat_momentum": np.random.randn(n),
        "feat_volatility": abs(np.random.randn(n)),
        "feat_trend": np.random.randn(n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="15min"))
    return df


class TestRRREndToEnd:
    def test_full_flow(self):
        from fwbg.core.registry import get_model

        df = _make_market_df(200)
        features = ["feat_momentum", "feat_volatility", "feat_trend"]
        targets = (np.random.rand(200) > 0.4).astype(np.float64)

        model = get_model("xgboost_rrr")()
        model.train(
            df[features], targets, TrainingContext(direction="long"),
            rrr_variants=[1.5, 2.0, 3.0], base_sl_atr=2.0,
        )

        probs = model.predict_probability(df[features])
        assert probs.shape == (200, 2)

        atr = df["_atr"].values
        ptp = model.get_per_trade_params(df[features], atr=atr)
        assert ptp is not None
        assert ptp.shape == (200, 2)
        # TP should be rrr * sl * atr
        # SL should be base_sl * atr = 2.0 * 1.0 = 2.0
        assert np.all(ptp[:, 1] == 2.0)


class TestMFEEndToEnd:
    def test_full_flow(self):
        from fwbg.core.registry import get_model
        from fwbg.optimization.targets import compute_mfe_targets

        df = _make_market_df(200)
        features = ["feat_momentum", "feat_volatility", "feat_trend"]

        mfe_long, _ = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.5)

        model = get_model("xgboost_mfe")()
        model.train(
            df[features], mfe_long, TrainingContext(direction="long"),
            sl_variants=[1.5, 2.0, 3.0],
        )

        probs = model.predict_probability(df[features])
        assert probs.shape == (200, 2)

        atr = df["_atr"].values
        ptp = model.get_per_trade_params(df[features], atr=atr)
        assert ptp is not None
        assert ptp.shape == (200, 2)
        # TP = predicted_mfe * atr, SL = selected_sl * atr
        assert np.all(ptp[:, 0] >= 0.0)
        assert np.all(ptp[:, 1] > 0.0)
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_rrr_mfe_e2e.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All existing tests still pass

**Step 4: Commit**

```bash
git add tests/test_rrr_mfe_e2e.py
git commit -m "test: add end-to-end tests for xgboost_rrr and xgboost_mfe"
```

---

## Task 8: Run Full Test Suite & Final Cleanup

**Step 1: Run all tests**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass (except the known pre-existing failure in `test_resource_manager.py`)

**Step 2: Check for lint issues**

Run: `python -m flake8 src/fwbg/plugins/fwbg-core/models/xgboost_rrr/ src/fwbg/plugins/fwbg-core/models/xgboost_mfe/ --max-line-length=120`
Expected: No errors

**Step 3: Final commit if any cleanup was needed**

```bash
git commit -m "chore: final cleanup for xgboost_rrr and xgboost_mfe plugins"
```

---

## Summary of All Files Changed/Created

| Action | File |
|--------|------|
| Modify | `packages/fwbg-sdk/src/fwbg_sdk/models.py` — add `get_per_trade_params()` |
| Modify | `src/fwbg/optimization/targets.py` — add `compute_mfe_targets()`, `per_trade_params` in `_simulate_trades_core` |
| Modify | `src/fwbg/optimization/nested_cv.py` — wire `per_trade_params` through `evaluate_on_holdout` |
| Modify | `src/fwbg/plugins/fwbg-core/models/__init__.py` — register new plugins |
| Create | `src/fwbg/plugins/fwbg-core/models/xgboost_rrr/__init__.py` |
| Create | `src/fwbg/plugins/fwbg-core/models/xgboost_mfe/__init__.py` |
| Create | `tests/test_base_model_per_trade_params.py` |
| Create | `tests/test_per_trade_params_simulation.py` |
| Create | `tests/test_mfe_targets.py` |
| Create | `tests/test_xgboost_rrr_model.py` |
| Create | `tests/test_xgboost_mfe_model.py` |
| Create | `tests/test_mfe_target_integration.py` |
| Create | `tests/test_rrr_mfe_e2e.py` |

## Notes for Implementor

1. **`SimulationContext.create_minimal`** may not exist — check `context.py` and adjust test fixtures to use whatever minimal constructor is available.
2. **Target stacking happens inside the model's `train()`** — the pipeline passes raw targets, the model stacks them internally.
3. **For xgboost_mfe in production**, the pipeline needs to call `compute_mfe_targets()` instead of `compute_targets()`. Task 6 outlines two approaches — prefer the simpler one (model computes MFE internally via `TrainingContext.fold_information`).
4. **GPU fallback** is copied from the existing XGBoost plugin — keep it consistent.
5. **`predict_probability()` semantic reuse**: For MFE model, column 1 = predicted MFE (not a probability). The CT mechanism still works because `probs[i, 1] >= ct` becomes `predicted_mfe >= mfe_threshold`.
