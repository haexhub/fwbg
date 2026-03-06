# XGBoost RRR-Feature & MFE-Regression Model Plugins

## Motivation

The current XGBoost classifier predicts binary Win/Loss on ~125 features with ~300 trades per fold. This leads to massive overfitting risk and doesn't meaningfully improve results over the signal-based model. Two new model types address this by reframing the ML problem:

1. **xgboost_rrr**: Let the model learn which RRR works best for each market setup
2. **xgboost_mfe**: Predict how far a breakout runs (regression) instead of binary classification

## Plugin Structure

```
src/fwbg/plugins/fwbg-core/models/
├── xgboost/          # existing — unchanged
├── xgboost_rrr/      # NEW
│   └── __init__.py
└── xgboost_mfe/      # NEW
    └── __init__.py
```

Both implement `BaseModel` and are selected via `"type": "xgboost_rrr"` or `"type": "xgboost_mfe"` in the strategy config.

---

## Model 1: xgboost_rrr — RRR as Feature

### Concept

Train a single XGBClassifier on multiple TP/SL combinations simultaneously, with RRR (`tp/sl`) as an additional input feature. The model learns: "given these market conditions AND this RRR, is the trade a win or loss?"

### Training

1. For each RRR variant in `rrr_variants` (e.g. `[1.5, 2.0, 2.5, 3.0, 4.0, 5.0]`), compute binary Win/Loss targets using the existing `compute_targets()` machinery.
2. Stack the dataset N times (once per RRR variant), adding `rrr` as an extra feature column.
3. Train one XGBClassifier on the stacked dataset. Features = indicator features + `rrr`.

### Inference

1. For each sample, score all RRR variants (same features, only `rrr` column varies).
2. Select the RRR with the highest Win probability.
3. Return this probability via `predict_probability()` → `(n_samples, 2)`.
4. Store the selected RRR per sample internally (`self._selected_rrr`).

### Config Example

```json
{
  "type": "xgboost_rrr",
  "architecture": "long_short_separate",
  "hyperparameters": {
    "rrr_variants": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    "base_sl_atr": 2.5
  }
}
```

- `rrr_variants`: RRR values to train on. TP is derived as `rrr * sl`.
- `base_sl_atr`: SL in ATR multiples (fixed across variants, TP varies).

### Grid Search Impact

- **No TP/SL grid** — the model selects the optimal RRR per trade.
- Grid contains only CT values and exit modifier variants.
- RRR variants are model parameters, not grid dimensions.

---

## Model 2: xgboost_mfe — MFE Regression

### Concept

Predict the Maximum Favorable Excursion (how far a breakout runs before reversal/stop) using XGBRegressor. MFE is normalized by ATR for regime robustness.

### Training

1. For each SL variant in `sl_variants` (e.g. `[1.5, 2.0, 2.5, 3.0]` in ATR), compute the MFE per trade — the maximum favorable price excursion before the trade is stopped out or times out.
2. Stack the dataset (once per SL variant), adding `sl_atr` as an extra feature column.
3. Train one XGBRegressor. Features = indicator features + `sl_atr`. Target = `mfe / atr` (continuous float).

### Inference

1. For each sample, score all SL variants.
2. Select the SL variant with the best `predicted_mfe / sl_atr` ratio (expected RRR).
3. Apply MFE threshold filter: only trade if `predicted_mfe >= mfe_threshold`.
4. Return predicted MFE via `predict_probability()` → `(n_samples, 2)`, column 1 = predicted MFE in ATR.
5. CT becomes the MFE threshold (e.g. `ct=1.5` means "trade only if predicted MFE >= 1.5 ATR").

### Config Example

```json
{
  "type": "xgboost_mfe",
  "architecture": "long_short_separate",
  "hyperparameters": {
    "sl_variants": [1.5, 2.0, 2.5, 3.0],
    "mfe_threshold": [1.0, 1.5, 2.0, 2.5]
  }
}
```

### TP Derivation

- TP is set dynamically per trade: `tp = predicted_mfe * atr_value`.
- SL is set from the selected variant: `sl = selected_sl_atr * atr_value`.

### Grid Search Impact

- **No TP/SL grid** — SL comes from the model's selection, TP from MFE prediction.
- Grid contains only MFE threshold values (mapped to CT) and exit modifier variants.

---

## Shared: per_trade_params Mechanism

Both models need dynamic TP/SL per trade. Currently exit strategies receive global TP/SL values.

### New BaseModel Method

```python
def get_per_trade_params(self, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return per-sample TP/SL overrides as absolute price distances.

    Returns:
        None (default, use global TP/SL) or ndarray of shape (n_samples, 2)
        where column 0 = TP distance, column 1 = SL distance (absolute prices).
    """
    return None
```

### Flow

1. Model predicts MFE/RRR in ATR multiples internally.
2. `get_per_trade_params()` converts to absolute price distances: `tp = predicted_atr_mult * atr_value`.
3. Exit strategy receives optional `per_trade_params` array.
4. If present, overrides the global TP/SL per trade.
5. Trade management (breakeven, trailing stop) still applies on top.

### Exit Strategy Compatibility

Works with any exit strategy (orb_based, atr_based, fixed, etc.):
- orb_based: range-based TP/SL is overridden, but trailing/BE logic is preserved.
- The override is at the absolute distance level, which is strategy-agnostic.

### Changes to Existing Code

| File | Change |
|------|--------|
| `packages/fwbg-sdk/src/fwbg_sdk/models.py` | Add `get_per_trade_params()` with default `None` |
| `packages/fwbg-sdk/src/fwbg_sdk/exit_strategies.py` | Add `per_trade_params` parameter |
| `src/fwbg/optimization/targets.py` | New `compute_mfe_targets()` function |
| `src/fwbg/optimization/nested_cv.py` | Pass `per_trade_params` through to simulation |
| `src/fwbg/simulation/numba_core.py` | Accept per-bar TP/SL arrays instead of scalars |

### What Does NOT Change

- Grid search logic (CT grid stays, semantics change for MFE model)
- Walk-forward splits (time-based, stacking happens after split)
- Feature selection (RRR/SL feature added before selection)
- Unified simulation flow

---

## Data Leakage Prevention

When stacking datasets for multiple RRR/SL variants:
- All variants of the same bar must be in the same fold (no cross-fold leakage).
- This is guaranteed by splitting temporally first, then stacking within each split.

## Training Data Size

- xgboost_rrr: 300 trades × 6 RRR variants = 1800 rows (but not independent)
- xgboost_mfe: 300 trades × 4 SL variants = 1200 rows (but not independent)
- XGBoost handles correlated rows well via bagging, but effective sample size is still ~300.
