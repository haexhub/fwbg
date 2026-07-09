# Plugin Spec — fractional_diff

**Kind**: preprocessing  •  **Version**: 2.0.0

## Capability

Applies López de Prado fractional differentiation to OHLC columns to make them stationary while preserving memory, with an optimal d learned from train data via ADF test.

## Summary

Preprocessor that fits an optimal fractional-differentiation exponent d on the training DataFrame's close series (via ADF test when auto_d is enabled, otherwise using default_d) and applies fractional differentiation to configured OHLC columns. Uses a sklearn fit/transform pattern: fit() stores d and the last MAX_WINDOW=500 rows of train data as history; execute() detects train vs. val/test by comparing the first index against the stored train_end_idx_, prepends the train history for val/test data to avoid initial NaNs, and drops leading NaNs for train data. Writes the effective d to df.attrs["frac_diff_d"].

## Inputs

- ctx.df: pandas DataFrame containing the columns configured via the columns param (default: O, H, L, C); for auto_d=True the C column must be present on the fit() DataFrame
- ctx (PipelineContext): pipeline context whose .df is read and updated

## Parameters

- `auto_d` (bool, default=True): Automatically find the optimal d via ADF test on the train close series during fit() (recommended). When False, default_d is used.
- `default_d` (float, default=0.4): Fixed differentiation exponent used when auto_d is disabled (0 = no transform, 1 = full diff, 0.3-0.5 typical for trading).
- `columns` (list[string], default=['O', 'H', 'L', 'C']): DataFrame columns to apply fractional differentiation to; only columns actually present in ctx.df during fit() are retained.

## Outputs

- ctx.df: DataFrame with configured OHLC columns replaced by their fractionally differentiated series (via vectorized convolution with weights truncated at threshold=1e-5 and window capped at MAX_WINDOW=500)
- df.attrs['frac_diff_d']: the fitted differentiation exponent d used for the transform
- Fitted state on the plugin instance: d_ (float), history_ (last <=500 train rows), train_end_idx_ (last train index), columns_ (subset of requested columns actually present)

## Acceptance Criteria

- AC-001: fit() populates self.columns_ with the intersection of the requested columns param and the columns present in ctx.df.
- AC-002: When auto_d=True, fit() sets self.d_ by calling _find_optimal_d on ctx.df['C'], which returns the smallest d in {0.1, 0.2, ..., 1.0} making the fractionally differentiated close stationary at p<0.05 via ADF, and falls back to 0.5 if none qualifies.
- AC-003: When auto_d=False, fit() sets self.d_ to default_d.
- AC-004: fit() stores self.history_ as the last min(MAX_WINDOW=500, len(df)) rows of ctx.df and self.train_end_idx_ as the last index of that DataFrame, and sets self._fitted = True.
- AC-005: If none of the requested columns are present in ctx.df during fit(), self.d_ is set to 0.0, self.history_ to None, and execute() returns ctx unchanged.
- AC-006: execute() raises RuntimeError when called before fit().
- AC-007: execute() returns ctx unchanged when self.columns_ is empty or self.d_ == 0.0.
- AC-008: For train data (ctx.df.index[0] <= self.train_end_idx_), execute() applies _frac_diff to each fitted column with the learned d and truncates the result to start at the first valid index of the first fitted column.
- AC-009: For val/test data (ctx.df.index[0] > self.train_end_idx_) with a stored history_, execute() concatenates history_ with ctx.df, transforms the combined frame, then returns only the rows matching the original ctx.df index (no NaN drop).
- AC-010: For val/test data without a stored history_, execute() falls back to transforming ctx.df in place without prepending history.
- AC-011: execute() sets df.attrs['frac_diff_d'] to self.d_ on the returned DataFrame, but only when the transform actually runs (self.columns_ is non-empty and self.d_ != 0.0). In the no-op branch (AC-007: self.columns_ is empty or self.d_ == 0.0) execute() returns ctx unchanged without setting df.attrs.
- AC-012: _frac_diff uses a vectorized np.convolve with weights whose |w|>1e-5 and window capped at MAX_WINDOW=500; d=0 returns the input series unchanged.

## Edge Cases

- ctx.df has none of the requested columns during fit(): d_=0.0, history_=None, execute() is a no-op returning ctx as-is.
- auto_d=True but ADF never produces p<0.05 for any d in the grid: d_ defaults to 0.5.
- auto_d=True and a candidate d yields fewer than 100 non-NaN rows or adfuller raises: that d is skipped and the search continues.
- d_ == 0.0 (either explicitly set via default_d=0 or via the no-columns branch): execute() returns ctx without modification and without setting df.attrs.
- Val/Test DataFrame index starts after train_end_idx_ but self.history_ is None: fallback path transforms without prepending, producing leading NaNs.
- Series shorter than MAX_WINDOW=500 during fit(): history_ contains only len(df) rows and the weight window is truncated to len(series).
- Requested columns param includes names not in ctx.df: those names are silently dropped from self.columns_.

## Assumptions

- ctx.df is a pandas DataFrame with a monotonically ordered index that is comparable across fit/execute calls (so index[0] <= train_end_idx_ correctly discriminates train vs. val/test slices).
- When auto_d=True, ctx.df contains a 'C' (close) column at fit time.
- statsmodels is available at runtime for the ADF-based auto_d search.
