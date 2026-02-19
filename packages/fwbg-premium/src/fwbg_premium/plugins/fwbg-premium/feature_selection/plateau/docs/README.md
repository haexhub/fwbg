# Plateau Feature Selection

Feature selection algorithm that favors features lying on performance "plateaus" -- where neighboring parameter variants show similar importance -- to produce more robust and overfit-resistant feature sets.

## Concept

Standard importance-based feature selection can select features that are statistical flukes: a feature like `rsi_14` might show high importance while `rsi_12` and `rsi_16` show nearly zero. This isolated spike likely reflects overfitting to a specific parameter value rather than a genuinely useful signal. In contrast, a feature on a "plateau" -- where nearby parameter variants also show meaningful importance -- is far more likely to remain predictive on unseen data.

The Plateau selector addresses this by augmenting raw feature importance with a neighborhood stability analysis. For each feature, it identifies "neighbor" features that differ only in their numeric lookback parameter (e.g., `rsi_12` and `rsi_16` are neighbors of `rsi_14`; `macro_vix_chg_12h` and `macro_vix_chg_48h` are neighbors of `macro_vix_chg_24h`). It then computes a composite plateau score that rewards features whose neighbors have similar, consistently high importance.

The plugin also provides utility functions for parameter-level plateau analysis in grid search contexts, helping select TP/SL/CT parameter combinations that lie on stable performance plateaus rather than isolated peaks.

## Selection Algorithm

### Feature Plateau Selection

1. Train an XGBoost classifier to compute feature importances
2. Filter features below the `min_importance` threshold
3. For each surviving feature, find parameter-neighbor features using regex patterns:
   - `_Nh` suffix (hourly lookbacks, e.g., `chg_24h`)
   - `_Nd` suffix (daily lookbacks, e.g., `chg_5d`)
   - `_N_` mid-name (e.g., `sma_20_slope`)
   - `_N` final suffix (e.g., `rsi_14`, `ema_20`)
4. Compute neighbor-based scores:
   - **Stability**: `1 / (1 + CV)` where CV is the coefficient of variation of neighbor importances
   - **Plateau factor**: `1 / (1 + 0.5 * |importance - neighbor_mean| / neighbor_mean)` -- penalizes features far from their neighbor mean
   - **Plateau score**: `importance * (0.6 + 0.25 * stability + 0.15 * plateau_factor)`
5. A feature is marked as a plateau feature if `stability > 0.5` and `plateau_factor > 0.6`
6. Features without enough neighbors receive a `0.8x` importance penalty
7. Sort by plateau score (or raw importance if `prefer_plateau=False`) and apply `max_features` cap

### Neighbor Detection

Neighbor deltas are scaled by the magnitude of the lookback value:

| Value Range | Deltas Checked |
|-------------|----------------|
| 1-5 | +/-1, +/-2, +/-3 |
| 6-20 | +/-2, +/-4, +/-5 |
| 21-50 | +/-5, +/-10, +/-12 |
| 51+ | +/-10, +/-20, +/-24 |

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | `int` | `100` | Number of XGBoost trees for computing feature importances (min: 1, max: 10000) |
| `max_depth` | `int` | `5` | Maximum tree depth for the XGBoost importance model (min: 1, max: 50) |
| `min_importance` | `float` | `0.01` | Minimum feature importance threshold; features below this are excluded before plateau scoring (min: 0.0, max: 1.0) |
| `min_neighbors` | `int` | `1` | Minimum neighbor features required for plateau bonus (min: 0, max: 100) |
| `prefer_plateau` | `bool` | `true` | Sort by plateau score instead of raw importance (recommended for robustness) |
| `n_jobs` | `int` | `1` | Number of parallel threads for XGBoost training (min: 1, max: 128) |
| `max_features` | `int` | `None` | Optional hard cap on the number of selected features |

## Usage Notes

- Input data is automatically cleaned: `inf`/`-inf` values are replaced with `NaN`, and `NaN` is filled with `0`.
- If no features pass the `min_importance` threshold, the selector falls back to returning the top features by raw importance.
- The plateau approach is most effective when the feature set contains multiple lookback variants of the same indicator (e.g., RSI with periods 10, 12, 14, 16, 20).
- Features without numeric lookback parameters in their names (e.g., `macro_vix_vvix_ratio`) will not have neighbors detected and receive the `0.8x` penalty.
- Metadata includes raw importances, plateau scores, the list of plateau features, and the number of features with detected neighbors.
- The `calculate_param_plateau_score` and `select_best_plateau_candidate` utility functions are available for grid search parameter plateau analysis, operating on TP/SL/CT parameter spaces.
