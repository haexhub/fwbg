# Phase 7: Validation & Statistical Tests

## Purpose

The validation phase verifies the robustness of discovered strategies through walk-forward cross-validation and multiple statistical tests. Goal: Ensure that a discovered edge is not a random artifact or the result of overfitting.

---

## Walk-Forward Validation

FWBG uses **nested cross-validation** with expanding windows — the standard approach for time-series-based ML strategies.

### Fold Structure

```
Data:   |──────────────────────────────────────────|
        t=0                                      t=T

Fold 1: |====TRAIN====|==TEST==|
Fold 2: |========TRAIN========|==TEST==|
Fold 3: |============TRAIN============|==TEST==|
        ...
Fold N: |==================TRAIN==================|==TEST==|
```

Each fold has more training data than the previous one (expanding window). The test set is always in the future relative to training.

### Nested CV (Inner + Outer)

```
Outer Fold (Walk-Forward):
  ├── Train Split
  │   └── Inner CV (Grid Search):
  │       ├── Inner Fold 1: Train | Val
  │       ├── Inner Fold 2: Train | Val
  │       └── Inner Fold 3: Train | Val
  │       → Best TP/SL/CT combination
  └── Test Split
      → Evaluation with best combination
```

- **Outer Folds:** Walk-forward evaluation (expanding)
- **Inner Folds:** Grid search within each outer fold
- The best grid candidate from the inner CV is evaluated on the outer test split

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `folds` | `8` | Number of outer folds |
| `oos_size` | `4000` | Out-of-sample bars per fold |
| `n_inner_folds` | `3` | Number of inner folds for grid search |
| `embargo_bars` | `100` | Embargo bars between train and test (purging) |
| `sample_weights` | `false` | Trade-duration-based sample weights |
| `probability_calibration` | `false` | Probability calibration of predictions |
| `calibration_method` | `"isotonic"` | Calibration method ("isotonic" or "sigmoid") |

### Time-Series Purging (Embargo)

A gap of `embargo_bars` bars is inserted between train and test. This prevents information leakage from trades that span the train/test boundary.

```
|====TRAIN====|###EMBARGO###|==TEST==|
```

### Sample Weights

With `sample_weights: true`, trades are weighted by their duration. Longer trades receive higher weight since they contain more information. Trade durations are computed by the exit strategy via `return_durations=True`.

---

## Early Pruning

Two-phase grid search — reduces computation time for large grids.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `early_pruning.enabled` | `false` | Enable |
| `early_pruning.keep_ratio` | `0.5` | Top fraction of surviving combos |
| `early_pruning.min_survivors` | `10` | At least N combos survive |

**Process:**
1. **Phase 1 (Screening):** Evaluate all combos on inner fold 0
2. **Pruning:** Remove bottom half by PnL
3. **Phase 2 (Full Eval):** Evaluate only survivors on all inner folds

```json
"validation": {
  "early_pruning": {
    "enabled": true,
    "keep_ratio": 0.5,
    "min_survivors": 10
  }
}
```

---

## Statistical Tests

### 1. Monte Carlo Permutation Test

Tests whether the observed win rate is significantly better than chance:

- **1000 permutations** of trade results → null distribution
- **p-value < 0.05** → edge is statistically significant
- Additionally: **500 equity simulation paths** for bankruptcy rate

### 2. Deflated Sharpe Ratio (DSR)

*Bailey & López de Prado (2014)*

Corrects the observed Sharpe ratio for **multiple testing** — the more grid combinations tested, the more likely a high Sharpe is found by chance.

```
DSR = Φ((SR_obs - E[max(SR)]) / σ(SR))
```

- **E[max(SR)]** — Expected maximum Sharpe under the null hypothesis
- **σ(SR)** — Standard deviation of the Sharpe estimator (accounts for skewness/kurtosis)
- **DSR > 0.95** → Sharpe is significant even after multiple-testing correction

### 3. Probability of Backtest Overfitting (PBO)

*Bailey, Borwein, López de Prado, Zhu (2017)*

Measures the probability that the best in-sample strategy performs poorly out-of-sample.

**Method: Combinatorial Symmetric Cross-Validation (CSCV)**
- With 8 walk-forward folds: **C(8,4) = 70** possible IS/OOS splits
- For each split: Checks whether the best IS combo also ranks well OOS
- **PBO > 0.50** → Likely overfitting

### 4. Feature Stability

Analyzes the consistency of feature selection across all walk-forward folds (see [Phase 4: Feature Selection](4-feature-selection.md)).

---

## Significance Thresholds

| Metric | Good | Bad | Meaning |
|--------|------|-----|---------|
| p-value | < 0.05 | >= 0.05 | Edge is (not) due to chance |
| DSR | > 0.95 | < 0.50 | Sharpe does (not) survive multiple testing |
| PBO | < 0.20 | > 0.50 | Strategy is (likely) not overfitted |

---

## Result Interpretation

### Result Structure

```json
{
  "status": "significant",
  "overfitting": {
    "dsr": {
      "dsr": 0.982,
      "observed_sr": 1.85,
      "expected_max_sr": 2.51,
      "n_strategies": 144,
      "is_significant": true
    },
    "pbo": {
      "pbo": 0.12,
      "n_cscv_splits": 70,
      "is_overfit": false,
      "degradation": 0.88,
      "logit_mean": 1.45
    }
  }
}
```

### Status Values

| Status | Meaning |
|--------|---------|
| `significant` | Statistically significant edge found |
| `not_significant` | No edge (p-value >= 0.05) |
| `no_candidates` | No valid candidates after filtering |

---

## Live Bias Detection

During optimization, real-time bias checks are performed:

- **Mean Bias Ratio:** Measures deviation of fold performance from the mean
- **Extreme Folds:** Folds with unusually high/low performance
- **Win-Rate Consistency:** Standard deviation of win rate across folds
- **System-Wide Check:** At the end of the entire run across all assets

Detailed documentation: [Live Bias Detection](../LIVE_BIAS_DETECTION.md)

---

## Further Documentation

- [Robust Validation Guide](../ROBUST_VALIDATION_GUIDE.md) — Sample bias detection in detail
- [Live Bias Detection](../LIVE_BIAS_DETECTION.md) — Real-time bias checks
- [Strategy Configuration](../../strategies/README.md) — Validation parameters
