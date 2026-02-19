# Fractional Differentiation Preprocessor

Preprocessor that applies fractional differentiation to price series, achieving stationarity while preserving long-range memory -- a technique from Lopez de Prado's "Advances in Financial Machine Learning."

## Concept

Financial time series present a fundamental dilemma for machine learning: raw prices are non-stationary (violating most ML model assumptions), but standard differentiation (returns, i.e., d=1) discards all memory of past price levels, throwing away valuable information. Fractional differentiation offers an elegant middle ground by allowing a non-integer differentiation order `d` between 0 and 1.

At `d=0`, the series is unchanged (full memory, non-stationary). At `d=1`, the series is fully differenced (stationary, no memory). The optimal `d` -- typically between 0.3 and 0.5 for financial data -- produces a series that is stationary (passes the ADF test) while retaining enough memory of past price levels to be useful for prediction. The transformation works by computing a weighted sum of past values using binomial-series weights that decay over time.

This plugin follows a strict fit/transform pattern to prevent lookahead bias. The `fit()` method learns the optimal `d` from training data only (via ADF test) and stores a history buffer. The `execute()` method applies the learned `d` to any data split (train, validation, or test). For validation/test data, the stored training history is prepended before transformation to eliminate NaN values at the start of the series without introducing any lookahead, since the history comes from temporally prior training data.

## Transformation

The fractional differentiation of order `d` for a series `X` at time `t` is:

```
X_d[t] = sum(w_k * X[t-k]) for k = 0, 1, ..., window_size
```

Where weights are computed recursively:

```
w_0 = 1
w_k = -w_{k-1} * (d - k + 1) / k
```

Weights below the `threshold` (default `1e-5`) are truncated. A `max_window` of 500 bars caps the convolution length for performance. The transformation is implemented via vectorized `np.convolve` for efficiency.

### Automatic d Optimization

When `auto_d=True` (default), the plugin searches for the minimum `d` that achieves stationarity:

1. Test `d` values from 0.1 to 1.0 in steps of 0.1
2. Apply fractional differentiation with each `d`
3. Run the Augmented Dickey-Fuller (ADF) test on the result
4. Return the smallest `d` where the p-value < 0.05
5. Fall back to `d=0.5` if no value achieves stationarity

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_d` | `bool` | `true` | Automatically find optimal d via ADF test on train data (recommended) |
| `default_d` | `float` | `0.4` | Fixed differentiation exponent when `auto_d` is disabled (min: 0.0, max: 1.0, step: 0.05) |
| `columns` | `list[string]` | `["O", "H", "L", "C"]` | DataFrame columns to apply fractional differentiation to |

## Usage Notes

- This plugin is **stateful** (`stateful=True`): `fit()` must be called on training data before `execute()` can be used. Calling `execute()` without prior `fit()` raises a `RuntimeError`.
- The plugin runs early in the preprocessing pipeline (`order=10`).
- For training data, NaN rows at the beginning (from the convolution warm-up period) are automatically trimmed.
- For validation/test data, the last `MAX_WINDOW` (500) rows of training data are prepended as history before transformation, then stripped from the result. This eliminates warm-up NaNs without lookahead.
- The plugin detects whether it is processing train or validation/test data by comparing the DataFrame's first index against the stored `train_end_idx_`.
- The learned `d` value is stored in `df.attrs["frac_diff_d"]` for downstream inspection.
- The `auto_d` search requires at least 100 non-NaN data points per test. Series shorter than this are skipped.
- This plugin requires `statsmodels` for the ADF test (only when `auto_d=True`).
- Not cacheable (`cacheable=False`) due to its stateful nature.
