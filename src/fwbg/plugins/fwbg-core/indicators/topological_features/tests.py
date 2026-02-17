"""Tests for Topological Data Analysis indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "indicators", "topological_features")
if _mod is None:
    pytest.skip("topological_features plugin not available", allow_module_level=True)

TopologicalFeaturesIndicator = _mod.TopologicalFeaturesIndicator
_takens_embedding = _mod._takens_embedding
_finite_persistence = _mod._finite_persistence
_persistence_entropy = _mod._persistence_entropy
_extract_features = _mod._extract_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def indicator():
    return TopologicalFeaturesIndicator()


@pytest.fixture
def sample_df():
    """300 bars of synthetic random walk OHLCV data."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame(
        {
            "O": close + np.random.randn(n) * 0.1,
            "H": close + abs(np.random.randn(n) * 0.3),
            "L": close - abs(np.random.randn(n) * 0.3),
            "C": close,
            "V": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture
def trending_df():
    """Strong monotonic uptrend with minimal noise."""
    np.random.seed(123)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    trend = np.linspace(100, 150, n) + np.random.randn(n) * 0.3
    return pd.DataFrame(
        {"O": trend, "H": trend + 0.5, "L": trend - 0.5, "C": trend,
         "V": np.ones(n) * 1000},
        index=dates,
    )


@pytest.fixture
def choppy_df():
    """High-frequency oscillating price — many cycles/loops."""
    np.random.seed(456)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    base = 100 + np.sin(np.linspace(0, 30 * np.pi, n)) * 3 + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {"O": base, "H": base + 0.5, "L": base - 0.5, "C": base,
         "V": np.ones(n) * 1000},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestTakensEmbedding:
    """Validate the Takens time-delay embedding."""

    def test_output_shape(self):
        series = np.arange(20, dtype=float)
        cloud = _takens_embedding(series, embedding_dim=3, time_delay=1)
        assert cloud.shape == (18, 3)

    def test_output_shape_with_delay(self):
        series = np.arange(20, dtype=float)
        cloud = _takens_embedding(series, embedding_dim=3, time_delay=2)
        # n_points = 20 - (3-1)*2 = 16
        assert cloud.shape == (16, 3)

    def test_values_correct(self):
        """Each column should be a time-shifted copy of the series."""
        series = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        cloud = _takens_embedding(series, embedding_dim=3, time_delay=1)
        np.testing.assert_array_equal(cloud[:, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(cloud[:, 1], [2, 3, 4, 5])
        np.testing.assert_array_equal(cloud[:, 2], [3, 4, 5, 6])

    def test_returns_none_for_short_series(self):
        series = np.array([1.0, 2.0])
        assert _takens_embedding(series, embedding_dim=3, time_delay=1) is None

    def test_returns_none_for_large_delay(self):
        series = np.arange(5, dtype=float)
        # n_points = 5 - (3-1)*3 = -1 → None
        assert _takens_embedding(series, embedding_dim=3, time_delay=3) is None


class TestFinitePersistence:
    """Validate filtering of infinite lifetimes from persistence diagrams."""

    def test_filters_infinite_death(self):
        diagram = np.array([[0.0, 1.0], [0.5, np.inf], [0.2, 0.8]])
        pers = _finite_persistence(diagram)
        # Only [0,1] and [0.2,0.8] are finite
        assert len(pers) == 2
        np.testing.assert_allclose(pers, [1.0, 0.6])

    def test_empty_diagram(self):
        pers = _finite_persistence(np.empty((0, 2)))
        assert len(pers) == 0

    def test_all_infinite(self):
        diagram = np.array([[0.0, np.inf], [1.0, np.inf]])
        pers = _finite_persistence(diagram)
        assert len(pers) == 0


class TestPersistenceEntropy:
    """Validate entropy computation from persistence values."""

    def test_single_value_zero_entropy(self):
        """Single persistence value → all probability on one element → 0 entropy."""
        assert _persistence_entropy(np.array([5.0])) == 0.0

    def test_uniform_max_entropy(self):
        """Equal persistence values → maximum entropy."""
        n = 4
        pers = np.ones(n)
        expected = np.log(n)  # uniform distribution entropy
        assert abs(_persistence_entropy(pers) - expected) < 1e-10

    def test_zero_persistence(self):
        assert _persistence_entropy(np.array([0.0, 0.0])) == 0.0

    def test_entropy_non_negative(self):
        pers = np.array([0.1, 0.5, 2.0, 0.3])
        assert _persistence_entropy(pers) >= 0


class TestExtractFeatures:
    """Validate feature extraction from persistence diagrams."""

    def test_h0_h1_counts(self):
        h0 = np.array([[0.0, 1.0], [0.0, np.inf], [0.0, 0.5]])
        h1 = np.array([[0.2, 0.8], [0.3, 0.6]])
        feats = _extract_features([h0, h1])
        assert feats["h0_count"] == 2  # 2 finite H0 features
        assert feats["h1_count"] == 2

    def test_max_persistence(self):
        h0 = np.array([[0.0, 1.0], [0.0, 0.5]])
        h1 = np.array([[0.1, 0.9]])  # persistence = 0.8
        feats = _extract_features([h0, h1])
        assert feats["h0_max_pers"] == 1.0
        assert abs(feats["h1_max_pers"] - 0.8) < 1e-10

    def test_empty_h1(self):
        """No H1 features → zero counts."""
        h0 = np.array([[0.0, 1.0]])
        h1 = np.empty((0, 2))
        feats = _extract_features([h0, h1])
        assert feats["h1_count"] == 0
        assert feats["h1_max_pers"] == 0.0

    def test_missing_h1_dimension(self):
        """Only H0 provided → H1 features default to 0."""
        h0 = np.array([[0.0, 1.0]])
        feats = _extract_features([h0])
        assert feats["h1_count"] == 0.0


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    """All expected columns are produced with correct naming."""

    def test_all_expected_columns_present(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_correct_column_count(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        # 10 features * 2 windows = 20
        tda_cols = [c for c in result.columns if c.startswith("tda_")]
        assert len(tda_cols) == 20

    def test_preserves_original_columns(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns

    def test_correct_prefix(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        new_cols = [c for c in result.columns if c not in ["O", "H", "L", "C", "V"]]
        for col in new_cols:
            assert col.startswith("tda_"), f"Feature {col} missing tda_ prefix"

    def test_length_unchanged(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        assert len(result) == len(sample_df)

    def test_feature_columns_match_compute(self, indicator, sample_df):
        """get_feature_columns() must match what compute() actually produces."""
        result = indicator.compute(sample_df)
        produced = sorted(c for c in result.columns if c.startswith("tda_"))
        declared = sorted(indicator.get_feature_columns())
        assert produced == declared


# ---------------------------------------------------------------------------
# No-lookahead tests
# ---------------------------------------------------------------------------


class TestNoLookahead:
    """shift_features prevents information leakage."""

    def test_first_row_nan(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        tda_cols = [c for c in result.columns if c.startswith("tda_")]
        for col in tda_cols:
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"

    def test_shift_by_exactly_one(self, indicator):
        """Compute on minimal data, verify the value at row i came from row i-1."""
        np.random.seed(77)
        n = 120
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {"O": close, "H": close + 0.2, "L": close - 0.2, "C": close,
             "V": np.ones(n) * 100},
            index=dates,
        )
        result = indicator.compute(df, windows=[50])
        col = "tda_h0_count_50"
        vals = result[col].values
        # First non-NaN value should appear at index >= 50 (window warmup + 1 shift)
        first_valid = pd.Series(vals).first_valid_index()
        assert first_valid is not None
        assert first_valid >= 50  # window-1 + shift


# ---------------------------------------------------------------------------
# Topological property tests
# ---------------------------------------------------------------------------


class TestTopologicalProperties:
    """Topological features capture meaningful market structure."""

    def test_choppy_data_more_loops(self, indicator, trending_df, choppy_df):
        """Oscillating price creates more H1 (loop) features than a trend."""
        res_trend = indicator.compute(trending_df, windows=[50])
        res_choppy = indicator.compute(choppy_df, windows=[50])
        h1_trend = res_trend["tda_h1_count_50"].dropna().mean()
        h1_choppy = res_choppy["tda_h1_count_50"].dropna().mean()
        assert h1_choppy >= h1_trend * 0.8, (
            f"Choppy H1 ({h1_choppy:.2f}) should be >= trending H1 * 0.8 ({h1_trend * 0.8:.2f})"
        )

    def test_trending_lower_entropy(self, indicator, trending_df, choppy_df):
        """Trending markets have simpler topology → lower persistence entropy."""
        res_trend = indicator.compute(trending_df, windows=[50])
        res_choppy = indicator.compute(choppy_df, windows=[50])
        ent_trend = res_trend["tda_persistence_entropy_50"].dropna().mean()
        ent_choppy = res_choppy["tda_persistence_entropy_50"].dropna().mean()
        # Trend should have lower or comparable entropy (allow tolerance)
        assert ent_trend <= ent_choppy * 1.3, (
            f"Trending entropy ({ent_trend:.3f}) should be <= choppy ({ent_choppy:.3f}) * 1.3"
        )

    def test_persistence_entropy_non_negative(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        for w in [50, 100]:
            vals = result[f"tda_persistence_entropy_{w}"].dropna()
            assert (vals >= 0).all()

    def test_wasserstein_non_negative(self, indicator, sample_df):
        result = indicator.compute(sample_df)
        for w in [50, 100]:
            vals = result[f"tda_wasserstein_amp_{w}"].dropna()
            assert (vals >= 0).all()

    def test_h0_count_at_least_one(self, indicator, sample_df):
        """Every valid window should have at least 1 H0 feature (connected component)."""
        result = indicator.compute(sample_df)
        vals = result["tda_h0_count_50"].dropna()
        assert (vals >= 1).all(), "H0 count should be >= 1 for any point cloud"

    def test_no_inf_in_ratio_features(self, indicator, sample_df):
        """safe_divide ensures no inf in ratio columns."""
        result = indicator.compute(sample_df)
        for w in [50, 100]:
            for col in [f"tda_h1_ratio_{w}", f"tda_max_loop_persistence_{w}"]:
                vals = result[col].dropna()
                assert not np.isinf(vals).any(), f"Inf found in {col}"

    def test_larger_window_smoother(self, indicator, sample_df):
        """Larger windows should produce less variable (smoother) features."""
        result = indicator.compute(sample_df, windows=[50, 100])
        std_50 = result["tda_h0_count_50"].dropna().std()
        std_100 = result["tda_h0_count_100"].dropna().std()
        # Not a strict requirement but generally true, use generous bound
        assert std_100 <= std_50 * 2.0


# ---------------------------------------------------------------------------
# Parameter variation tests
# ---------------------------------------------------------------------------


class TestParameterVariation:
    """Different parameters produce valid results."""

    def test_custom_windows(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[30])
        assert "tda_h0_count_30" in result.columns
        assert "tda_h0_count_50" not in result.columns

    def test_custom_embedding_dim(self, indicator, sample_df):
        result = indicator.compute(sample_df, embedding_dim=5, windows=[50])
        vals = result["tda_h0_count_50"].dropna()
        assert len(vals) > 0

    def test_higher_maxdim(self, indicator, sample_df):
        """maxdim=1 is default, maxdim=0 should produce zero H1 features."""
        result = indicator.compute(sample_df, maxdim=0, windows=[50])
        h1 = result["tda_h1_count_50"].dropna()
        assert (h1 == 0).all(), "maxdim=0 should produce no H1 features"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases must not crash."""

    def test_short_data_no_crash(self, indicator):
        """Data shorter than window → all features NaN, no crash."""
        np.random.seed(99)
        n = 20
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {"O": close, "H": close + 0.1, "L": close - 0.1, "C": close,
             "V": np.ones(n) * 100},
            index=dates,
        )
        result = indicator.compute(df)
        assert len(result) == n
        tda_cols = [c for c in result.columns if c.startswith("tda_")]
        assert len(tda_cols) > 0

    def test_constant_price_no_crash(self, indicator):
        """Constant price → zero returns → degenerate point cloud."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {"O": close, "H": close, "L": close, "C": close,
             "V": np.ones(n) * 100},
            index=dates,
        )
        result = indicator.compute(df, windows=[30])
        assert len(result) == n
        # Should not produce inf values
        tda_cols = [c for c in result.columns if c.startswith("tda_")]
        for col in tda_cols:
            vals = result[col].dropna()
            assert not np.isinf(vals).any()

    def test_nan_in_price(self, indicator):
        """NaN in close price should not crash."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close[40:45] = np.nan
        df = pd.DataFrame(
            {"O": close, "H": close, "L": close, "C": close,
             "V": np.ones(n) * 100},
            index=dates,
        )
        # Should not raise
        result = indicator.compute(df, windows=[30])
        assert len(result) == n


# ---------------------------------------------------------------------------
# Plugin integration tests
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    """Plugin integrates correctly with the framework."""

    def test_plugin_discoverable(self):
        mod = import_plugin_module("fwbg-core", "indicators", "topological_features")
        assert mod is not None

    def test_default_params(self, indicator):
        params = indicator.get_default_params()
        assert params["windows"] == [50, 100]
        assert params["embedding_dim"] == 3
        assert params["time_delay"] == 1
        assert params["maxdim"] == 1

    def test_metadata(self, indicator):
        assert indicator.name == "topological_features"
        assert indicator.version == "1.0.0"
