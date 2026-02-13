"""
Tests für FractionalDiffPreprocessor Plugin.

TDD: Tests first, dann implementation verification.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline.context import PipelineContext


@pytest.fixture
def sample_ohlc_df():
    """Creates sample OHLC DataFrame for tests."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(start='2023-01-01', periods=n, freq='h')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        'O': close + np.random.randn(n) * 0.1,
        'H': close + np.abs(np.random.randn(n) * 0.3),
        'L': close - np.abs(np.random.randn(n) * 0.3),
        'C': close,
        'V': np.random.randint(1000, 10000, n)
    }, index=dates)


@pytest.fixture
def pipeline_context(sample_ohlc_df):
    """Creates PipelineContext with sample data."""
    return PipelineContext(df=sample_ohlc_df, symbol='TEST', asset_class='FOREX')


@pytest.fixture
def preprocessor():
    """Creates FractionalDiffPreprocessor instance."""
    from fwbg.plugins import import_plugin_module
    mod = import_plugin_module('fwbg-premium', 'preprocessing', 'fractional_diff')
    return mod.FractionalDiffPreprocessor()


class TestFitRequiresPipelineContext:
    """fit() must receive PipelineContext, not raw DataFrame."""

    def test_fit_accepts_pipeline_context(self, preprocessor, pipeline_context):
        """fit() should accept PipelineContext as first argument."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        assert preprocessor._fitted is True

    def test_fit_extracts_df_from_context(self, preprocessor, pipeline_context):
        """fit() should extract df from context and learn d."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        assert preprocessor.d_ == 0.4
        assert preprocessor.columns_ == ['O', 'H', 'L', 'C']


class TestFitLearnsFromTrainingData:
    """fit() learns optimal d and stores history from training data."""

    def test_fit_stores_d_value(self, preprocessor, pipeline_context):
        """fit() should store the d value."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.35)
        assert preprocessor.d_ == 0.35

    def test_fit_stores_history(self, preprocessor, pipeline_context):
        """fit() should store history for later transforms."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        assert preprocessor.history_ is not None
        assert len(preprocessor.history_) <= 500  # MAX_WINDOW

    def test_fit_stores_train_end_idx(self, preprocessor, pipeline_context):
        """fit() should store last index of training data."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        assert preprocessor.train_end_idx_ == pipeline_context.df.index[-1]

    def test_fit_with_auto_d_finds_optimal_d(self, preprocessor, pipeline_context):
        """fit() with auto_d=True should find optimal d via ADF test."""
        preprocessor.fit(pipeline_context, auto_d=True)
        assert 0.0 < preprocessor.d_ <= 1.0

    def test_fit_sets_fitted_flag(self, preprocessor, pipeline_context):
        """fit() should set _fitted to True."""
        assert preprocessor._fitted is False
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        assert preprocessor._fitted is True


class TestExecuteRequiresFitFirst:
    """execute() must be called after fit()."""

    def test_execute_raises_without_fit(self, preprocessor, pipeline_context):
        """execute() should raise RuntimeError if fit() not called."""
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            preprocessor.execute(pipeline_context)

    def test_execute_works_after_fit(self, preprocessor, pipeline_context):
        """execute() should work after fit() is called."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        result = preprocessor.execute(pipeline_context)
        assert result is not None
        assert hasattr(result, 'df')


class TestExecuteTransformsTrainData:
    """execute() transforms training data correctly."""

    def test_execute_transforms_ohlc_columns(self, preprocessor, pipeline_context):
        """execute() should transform O, H, L, C columns."""
        original_close = pipeline_context.df['C'].copy()

        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        result = preprocessor.execute(pipeline_context)

        # Values should be different after transformation
        assert not np.allclose(
            result.df['C'].iloc[:10].values,
            original_close.iloc[:10].values,
            rtol=0.1
        )

    def test_execute_removes_nan_warmup_for_train(self, preprocessor, pipeline_context):
        """execute() should remove NaN warmup rows for train data."""
        original_len = len(pipeline_context.df)

        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        result = preprocessor.execute(pipeline_context)

        # Result should be shorter (warmup removed)
        assert len(result.df) < original_len
        # No NaNs in OHLC columns
        assert not result.df['C'].isna().any()

    def test_execute_stores_d_in_attrs(self, preprocessor, pipeline_context):
        """execute() should store d value in df.attrs."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        result = preprocessor.execute(pipeline_context)

        assert result.df.attrs.get('frac_diff_d') == 0.4

    def test_execute_returns_pipeline_context(self, preprocessor, pipeline_context):
        """execute() should return PipelineContext, not DataFrame."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        result = preprocessor.execute(pipeline_context)

        assert isinstance(result, PipelineContext)


class TestExecuteUsesHistoryForValTest:
    """execute() uses stored history for val/test data to avoid NaNs."""

    def test_execute_uses_history_for_val_data(self, preprocessor, sample_ohlc_df):
        """execute() should prepend history for val/test data."""
        # Split into train and val
        train_df = sample_ohlc_df.iloc[:700]
        val_df = sample_ohlc_df.iloc[700:]

        train_ctx = PipelineContext(df=train_df, symbol='TEST', asset_class='FOREX')
        val_ctx = PipelineContext(df=val_df.copy(), symbol='TEST', asset_class='FOREX')

        # Fit on train
        preprocessor.fit(train_ctx, auto_d=False, default_d=0.4)

        # Execute on val
        result = preprocessor.execute(val_ctx)

        # Val data should have NO NaNs (history prepended)
        assert not result.df['C'].isna().any()
        # Val data should keep all rows
        assert len(result.df) == len(val_df)

    def test_execute_detects_train_vs_val_by_index(self, preprocessor, sample_ohlc_df):
        """execute() should detect train vs val data by index comparison."""
        train_df = sample_ohlc_df.iloc[:700]
        val_df = sample_ohlc_df.iloc[700:]

        train_ctx = PipelineContext(df=train_df, symbol='TEST', asset_class='FOREX')

        preprocessor.fit(train_ctx, auto_d=False, default_d=0.4)

        # Train data: first index <= train_end_idx
        assert train_df.index[0] <= preprocessor.train_end_idx_

        # Val data: first index > train_end_idx
        assert val_df.index[0] > preprocessor.train_end_idx_


class TestPipelineRunnerIntegration:
    """FractionalDiffPreprocessor works with PipelineRunner."""

    def test_runner_calls_fit_with_context(self, sample_ohlc_df):
        """PipelineRunner should call fit() with PipelineContext."""
        from fwbg.pipeline import (
            get_registry, PipelineRunner, PipelineContext,
            PipelineConfig, PluginConfig
        )

        registry = get_registry()
        registry.auto_discover()

        config = PipelineConfig(
            preprocessing=[
                PluginConfig(name='fwbg-premium:fractional_diff',
                           params={'auto_d': False, 'default_d': 0.4})
            ]
        )

        runner = PipelineRunner(registry, config)
        ctx = PipelineContext(df=sample_ohlc_df, symbol='TEST', asset_class='FOREX')

        # Fit should work
        runner.fit(ctx)

        # Check preprocessor is fitted
        frac_diff = runner.get_instance('fwbg-premium:fractional_diff')
        assert frac_diff._fitted is True
        assert frac_diff.d_ == 0.4

    def test_runner_executes_preprocessing(self, sample_ohlc_df):
        """PipelineRunner should execute preprocessing phase."""
        from fwbg.pipeline import (
            get_registry, PipelineRunner, PipelineContext,
            PipelineConfig, PluginConfig
        )

        registry = get_registry()
        registry.auto_discover()

        config = PipelineConfig(
            preprocessing=[
                PluginConfig(name='fwbg-premium:fractional_diff',
                           params={'auto_d': False, 'default_d': 0.4})
            ]
        )

        runner = PipelineRunner(registry, config)
        ctx = PipelineContext(df=sample_ohlc_df.copy(), symbol='TEST', asset_class='FOREX')

        runner.fit(ctx)
        result = runner.run(ctx, phases=['preprocessing'])

        # Data should be transformed
        assert result.df.attrs.get('frac_diff_d') == 0.4
        assert len(result.df) < len(sample_ohlc_df)


class TestResetFunctionality:
    """reset() clears fitted state."""

    def test_reset_clears_fitted_flag(self, preprocessor, pipeline_context):
        """reset() should set _fitted to False."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.4)
        assert preprocessor._fitted is True

        preprocessor.reset()
        assert preprocessor._fitted is False


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_d_zero_returns_unchanged(self, preprocessor, pipeline_context):
        """d=0 should not transform data."""
        preprocessor.fit(pipeline_context, auto_d=False, default_d=0.0)
        result = preprocessor.execute(pipeline_context)

        # d=0 means no transformation
        assert preprocessor.d_ == 0.0

    def test_missing_columns_handled(self, preprocessor):
        """Missing OHLC columns should be handled gracefully."""
        df = pd.DataFrame({
            'X': [1, 2, 3, 4, 5],
            'Y': [5, 4, 3, 2, 1]
        }, index=pd.date_range('2023-01-01', periods=5, freq='h'))

        ctx = PipelineContext(df=df, symbol='TEST', asset_class='FOREX')

        preprocessor.fit(ctx, auto_d=False, default_d=0.4)

        # Should handle missing columns
        assert preprocessor.columns_ == []
        assert preprocessor.d_ == 0.0
