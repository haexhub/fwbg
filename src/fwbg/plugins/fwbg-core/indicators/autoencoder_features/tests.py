"""Tests for autoencoder_features (PCA-based latent extraction) indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_ae = import_plugin_module("fwbg-core", "indicators", "autoencoder_features")
if _ae is None:
    pytest.skip(
        "fwbg-core autoencoder_features plugin not available",
        allow_module_level=True,
    )


@pytest.fixture
def indicator():
    return _ae.AutoencoderFeaturesIndicator()


@pytest.fixture
def df_with_features():
    """DataFrame with OHLCV + 20 synthetic indicator features."""
    n = 500
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.002,
            "L": close * 0.998,
            "C": close,
            "V": np.random.randint(100, 1000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    # Add synthetic indicator features (correlated with close)
    for i in range(20):
        df[f"feat_{i}"] = np.random.randn(n) + close * (0.01 * i)
    return df


@pytest.fixture
def df_ohlcv_only():
    """DataFrame with only OHLCV columns (no indicator features)."""
    n = 200
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.002,
            "L": close * 0.998,
            "C": close,
            "V": np.random.randint(100, 1000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )


@pytest.fixture
def df_few_features():
    """DataFrame with fewer features than default n_components."""
    n = 200
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.002,
            "L": close * 0.998,
            "C": close,
            "V": np.random.randint(100, 1000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    # Only 3 features (less than default n_components=8)
    for i in range(3):
        df[f"feat_{i}"] = np.random.randn(n) + close * (0.01 * i)
    return df


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    """Expected feature columns are produced."""

    def test_latent_columns_present(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        for i in range(8):
            assert f"ae_latent_{i}" in result.columns, f"Missing ae_latent_{i}"

    def test_reconstruction_error_present(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        assert "ae_reconstruction_error" in result.columns

    def test_explained_variance_present(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        assert "ae_explained_variance" in result.columns

    def test_preserves_original_columns(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns

    def test_preserves_input_features(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        for i in range(20):
            assert f"feat_{i}" in result.columns

    def test_returns_dataframe(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# No-lookahead tests
# ---------------------------------------------------------------------------


class TestNoLookahead:
    """Features must be shifted by 1 bar (no lookahead bias)."""

    def test_first_row_is_nan(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        for i in range(8):
            assert pd.isna(result[f"ae_latent_{i}"].iloc[0]), (
                f"ae_latent_{i} first row should be NaN"
            )
        assert pd.isna(result["ae_reconstruction_error"].iloc[0])
        assert pd.isna(result["ae_explained_variance"].iloc[0])

    def test_second_row_not_nan(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        assert not pd.isna(result["ae_latent_0"].iloc[1])
        assert not pd.isna(result["ae_reconstruction_error"].iloc[1])


# ---------------------------------------------------------------------------
# PCA behavior tests
# ---------------------------------------------------------------------------


class TestPCABehavior:
    """PCA correctly extracts latent features."""

    def test_latent_components_orthogonal(self, indicator, df_with_features):
        """Latent components should be uncorrelated (PCA property)."""
        result = indicator.compute(df_with_features)
        latent_cols = [f"ae_latent_{i}" for i in range(8)]
        latent = result[latent_cols].dropna()
        corr = latent.corr().to_numpy().copy()
        # Off-diagonal elements should be near zero
        np.fill_diagonal(corr, 0.0)
        max_corr = np.abs(corr).max()
        assert max_corr < 0.05, (
            f"Latent components should be uncorrelated, max off-diag corr={max_corr:.4f}"
        )

    def test_explained_variance_in_range(self, indicator, df_with_features):
        """Cumulative explained variance must be between 0 and 1."""
        result = indicator.compute(df_with_features)
        ev = result["ae_explained_variance"].dropna().iloc[0]
        assert 0.0 < ev <= 1.0, f"Explained variance out of range: {ev}"

    def test_reconstruction_error_nonnegative(self, indicator, df_with_features):
        result = indicator.compute(df_with_features)
        recon = result["ae_reconstruction_error"].dropna()
        assert (recon >= 0).all(), "Reconstruction error must be non-negative"

    def test_more_components_lower_reconstruction_error(
        self, indicator, df_with_features
    ):
        """More PCA components should yield lower mean reconstruction error."""
        result_few = indicator.compute(df_with_features.copy(), n_components=2)
        result_many = indicator.compute(df_with_features.copy(), n_components=10)
        mean_few = result_few["ae_reconstruction_error"].dropna().mean()
        mean_many = result_many["ae_reconstruction_error"].dropna().mean()
        assert mean_few > mean_many, (
            f"Fewer components should have higher reconstruction error: "
            f"2 components={mean_few:.4f}, 10 components={mean_many:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases are handled gracefully."""

    def test_no_indicator_features(self, indicator, df_ohlcv_only):
        """With only OHLCV, return df unchanged."""
        result = indicator.compute(df_ohlcv_only)
        assert list(result.columns) == list(df_ohlcv_only.columns)
        assert len(result) == len(df_ohlcv_only)

    def test_fewer_features_than_components(self, indicator, df_few_features):
        """n_components auto-reduces when fewer features available."""
        result = indicator.compute(df_few_features, n_components=8)
        # Should have 2 latent components (3 features - 1 = 2)
        assert "ae_latent_0" in result.columns
        assert "ae_latent_1" in result.columns
        assert "ae_latent_2" not in result.columns
        assert "ae_reconstruction_error" in result.columns

    def test_custom_n_components(self, indicator, df_with_features):
        """Custom n_components produces correct number of latent features."""
        result = indicator.compute(df_with_features, n_components=3)
        assert "ae_latent_0" in result.columns
        assert "ae_latent_1" in result.columns
        assert "ae_latent_2" in result.columns
        assert "ae_latent_3" not in result.columns

    def test_nan_handling(self, indicator, df_with_features):
        """NaN values in input features are handled via median imputation."""
        df = df_with_features.copy()
        # Sprinkle NaN across features
        rng = np.random.default_rng(42)
        for col in [f"feat_{i}" for i in range(20)]:
            nan_idx = rng.choice(len(df), size=20, replace=False)
            df.loc[df.index[nan_idx], col] = np.nan

        result = indicator.compute(df)
        # All ae_ columns should have values (except first row due to shift)
        for i in range(8):
            col = f"ae_latent_{i}"
            assert col in result.columns
            non_nan = result[col].iloc[1:]
            assert not non_nan.isna().any(), f"{col} has NaN after median imputation"

    def test_exclude_prefixes(self, indicator, df_with_features):
        """Columns matching exclude_prefixes are excluded from PCA input."""
        df = df_with_features.copy()
        # Add columns that should be excluded
        df["ae_existing_0"] = np.random.randn(len(df))
        df["target_col"] = np.random.randn(len(df))

        result = indicator.compute(
            df, exclude_prefixes=["ae_", "target_"]
        )
        # Should still produce ae_ features (from feat_ columns)
        assert "ae_latent_0" in result.columns

    def test_single_feature(self, indicator, df_ohlcv_only):
        """Single feature column: n_components reduces to 0 -> return unchanged."""
        df = df_ohlcv_only.copy()
        df["single_feat"] = np.random.randn(len(df))
        # 1 feature, min(n_components, n_features-1) = min(8, 0) = 0
        result = indicator.compute(df)
        # Should return df unchanged since effective_components < 1
        assert "ae_latent_0" not in result.columns


# ---------------------------------------------------------------------------
# Different n_components values
# ---------------------------------------------------------------------------


class TestDifferentComponents:
    """Various n_components settings work correctly."""

    @pytest.mark.parametrize("n_comp", [1, 2, 4, 12, 20])
    def test_n_components_variation(self, indicator, df_with_features, n_comp):
        result = indicator.compute(df_with_features, n_components=n_comp)
        effective = min(n_comp, 19)  # 20 features - 1 = 19 max
        for i in range(effective):
            assert f"ae_latent_{i}" in result.columns
        if effective < 20:
            assert f"ae_latent_{effective}" not in result.columns


# ---------------------------------------------------------------------------
# Plugin integration tests
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    """Plugin integrates correctly with the registry."""

    def test_plugin_importable(self):
        from fwbg.pipeline import get_registry

        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:autoencoder_features")
        assert plugin_cls is not None

    def test_benefits_from_stationary_false(self):
        from fwbg.pipeline import get_registry

        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:autoencoder_features")
        assert plugin_cls.benefits_from_stationary is False

    def test_default_params(self, indicator):
        params = indicator.get_default_params()
        assert "n_components" in params
        assert "exclude_prefixes" in params
        assert params["n_components"] == 8
        assert params["exclude_prefixes"] == ["ae_"]

    def test_name_and_version(self, indicator):
        assert indicator.name == "autoencoder_features"
        assert indicator.version == "1.0.0"

    def test_get_feature_columns(self, indicator):
        cols = indicator.get_feature_columns()
        assert "ae_latent_0" in cols
        assert "ae_latent_7" in cols
        assert "ae_reconstruction_error" in cols
        assert "ae_explained_variance" in cols
        assert len(cols) == 10  # 8 latent + recon_error + explained_var
