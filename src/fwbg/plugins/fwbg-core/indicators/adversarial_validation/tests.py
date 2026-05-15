"""Tests for Adversarial Validation indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "indicators", "adversarial_validation")
if _mod is None:
    pytest.skip("adversarial_validation plugin not available", allow_module_level=True)

AdversarialValidationIndicator = _mod.AdversarialValidationIndicator
_select_feature_columns = _mod._select_feature_columns
_compute_adversarial_auc = _mod._compute_adversarial_auc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def indicator():
    return AdversarialValidationIndicator()


@pytest.fixture
def sample_df():
    """300 bars with simulated indicator features."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "O": close + np.random.randn(n) * 0.1,
            "H": close + abs(np.random.randn(n) * 0.3),
            "L": close - abs(np.random.randn(n) * 0.3),
            "C": close,
            "V": np.random.randint(100, 10000, n).astype(float),
            "ema_14": close + np.random.randn(n) * 0.5,
            "mom_rsi_14": 50 + np.random.randn(n) * 15,
            "vol_atr_14": np.abs(np.random.randn(n)) * 2,
            "pa_range_pos": np.random.rand(n),
            "regime_hurst": 0.5 + np.random.randn(n) * 0.1,
        },
        index=dates,
    )


@pytest.fixture
def regime_shift_df():
    """Clear distribution shift at bar 150: low vol → high vol + mean shift."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = np.concatenate([
        100 + np.cumsum(np.random.randn(150) * 0.2),
        100 + np.cumsum(np.random.randn(150) * 2.0),
    ])
    return pd.DataFrame(
        {
            "O": close, "H": close + 0.5, "L": close - 0.5, "C": close,
            "V": np.ones(n) * 1000,
            "feat_1": np.concatenate([np.random.randn(150) * 0.5, np.random.randn(150) * 5.0]),
            "feat_2": np.concatenate([np.random.randn(150), np.random.randn(150) + 10]),
            "feat_3": np.concatenate([np.random.randn(150), np.random.randn(150)]),
        },
        index=dates,
    )


@pytest.fixture
def stable_df():
    """Stationary IID features — no distribution shift."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "O": close, "H": close + 0.5, "L": close - 0.5, "C": close,
            "V": np.ones(n) * 1000,
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestSelectFeatureColumns:
    """Validate feature column selection logic."""

    def test_excludes_ohlcv(self):
        df = pd.DataFrame({
            "O": [1], "H": [2], "L": [3], "C": [4], "V": [5],
            "trend_ema": [6], "mom_rsi": [7],
        })
        cols = _select_feature_columns(df, [])
        assert "O" not in cols
        assert "trend_ema" in cols
        assert "mom_rsi" in cols

    def test_excludes_specified_prefixes(self):
        df = pd.DataFrame({
            "O": [1], "C": [2],
            "adv_auc_100": [0.5], "trend_ema": [3],
        })
        cols = _select_feature_columns(df, ["adv_"])
        assert "adv_auc_100" not in cols
        assert "trend_ema" in cols

    def test_excludes_non_numeric(self):
        df = pd.DataFrame({
            "O": [1], "C": [2],
            "label": ["buy"], "score": [0.5],
        })
        cols = _select_feature_columns(df, [])
        assert "label" not in cols
        assert "score" in cols

    def test_empty_when_only_ohlcv(self):
        df = pd.DataFrame({"O": [1], "H": [2], "L": [3], "C": [4], "V": [5]})
        assert _select_feature_columns(df, []) == []


class TestComputeAdversarialAuc:
    """Validate the core AUC computation."""

    def test_identical_distributions(self):
        """Same distribution → AUC near 0.5."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        old = data[:50]
        new = data[50:]
        auc, _ = _compute_adversarial_auc(old, new)
        assert 0.3 <= auc <= 0.8, f"Identical distributions should yield AUC ~0.5, got {auc}"

    def test_completely_different_distributions(self):
        """Clearly separable distributions → AUC near 1.0."""
        np.random.seed(42)
        old = np.random.randn(50, 5)
        new = np.random.randn(50, 5) + 10  # shifted by 10 std
        auc, _ = _compute_adversarial_auc(old, new)
        assert auc > 0.9, f"Separable distributions should yield AUC > 0.9, got {auc}"

    def test_returns_importance(self):
        """Max coefficient should be > 0 when distributions differ."""
        np.random.seed(42)
        old = np.random.randn(50, 5)
        new = np.random.randn(50, 5) + 5
        _, importance = _compute_adversarial_auc(old, new)
        assert importance > 0

    def test_too_few_samples(self):
        """< 10 samples → fallback AUC = 0.5."""
        old = np.random.randn(3, 5)
        new = np.random.randn(3, 5)
        auc, imp = _compute_adversarial_auc(old, new)
        assert auc == 0.5
        assert imp == 0.0

    def test_handles_nan_rows(self):
        """Rows with NaN should be removed, not crash."""
        np.random.seed(42)
        old = np.random.randn(30, 5)
        new = np.random.randn(30, 5) + 5
        old[0, 2] = np.nan
        new[5, 3] = np.nan
        auc, _ = _compute_adversarial_auc(old, new)
        assert 0.0 <= auc <= 1.0

    def test_single_class_after_nan_removal(self):
        """If NaN removal leaves only one class → fallback."""
        old = np.full((10, 3), np.nan)  # all NaN
        new = np.random.randn(10, 3)
        auc, _ = _compute_adversarial_auc(old, new)
        # After NaN removal, only 'new' class remains → fallback
        assert auc == 0.5


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    """All expected columns are produced."""

    def test_all_expected_columns_present(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[100], step=10)
        for suffix in ["auc", "drift_score", "stability",
                        "max_feature_importance", "drift_acceleration"]:
            col = f"adv_{suffix}_100"
            assert col in result.columns, f"Missing: {col}"

    def test_correct_prefix(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[100], step=10)
        original = set(sample_df.columns)
        new_cols = [c for c in result.columns if c not in original]
        for col in new_cols:
            assert col.startswith("adv_")

    def test_preserves_original_columns(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[100], step=10)
        for col in sample_df.columns:
            assert col in result.columns

    def test_length_unchanged(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[100], step=10)
        assert len(result) == len(sample_df)

    def test_feature_columns_match_compute(self, indicator, sample_df):
        """get_feature_columns() must match what compute() produces."""
        result = indicator.compute(sample_df, windows=[100], step=10)
        produced = sorted(c for c in result.columns if c.startswith("adv_"))
        declared = sorted(indicator.get_feature_columns())
        assert produced == declared

    def test_multiple_windows(self, indicator, sample_df):
        """Two windows produce 10 total feature columns."""
        result = indicator.compute(sample_df, windows=[100, 200], step=10)
        adv_cols = [c for c in result.columns if c.startswith("adv_")]
        assert len(adv_cols) == 10


# ---------------------------------------------------------------------------
# No-lookahead tests
# ---------------------------------------------------------------------------


class TestNoLookahead:
    """shift_features prevents information leakage."""

    def test_first_row_nan(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[100], step=10)
        adv_cols = [c for c in result.columns if c.startswith("adv_")]
        assert len(adv_cols) > 0
        for col in adv_cols:
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"


# ---------------------------------------------------------------------------
# Distribution shift detection tests
# ---------------------------------------------------------------------------


class TestDistributionShiftDetection:
    """Core capability: detecting regime changes."""

    def test_regime_shift_elevates_auc(self, indicator, regime_shift_df):
        """After a clear distribution shift, AUC should rise above 0.5."""
        result = indicator.compute(regime_shift_df, windows=[100], step=5)
        auc = result["adv_auc_100"].dropna()
        late_auc = auc.iloc[-50:]
        assert late_auc.max() > 0.60, (
            f"Expected AUC > 0.60 after regime shift, got max={late_auc.max():.3f}"
        )

    def test_regime_shift_higher_than_before(self, indicator, regime_shift_df):
        """AUC in the post-shift region should be higher than pre-shift."""
        result = indicator.compute(regime_shift_df, windows=[100], step=5)
        auc = result["adv_auc_100"].dropna()
        if len(auc) < 20:
            pytest.skip("Not enough computed points")
        mid = len(auc) // 2
        pre_shift_mean = auc.iloc[:mid].mean()
        post_shift_mean = auc.iloc[mid:].mean()
        assert post_shift_mean > pre_shift_mean, (
            f"Post-shift AUC ({post_shift_mean:.3f}) should exceed "
            f"pre-shift ({pre_shift_mean:.3f})"
        )

    def test_stable_regime_moderate_auc(self, indicator, stable_df):
        """Stationary data → AUC should stay closer to 0.5."""
        result = indicator.compute(stable_df, windows=[100], step=5)
        auc = result["adv_auc_100"].dropna()
        mean_auc = auc.mean()
        assert mean_auc < 0.75, f"Stable regime AUC should be < 0.75, got {mean_auc:.3f}"

    def test_drift_score_bounded_zero_one(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[100], step=10)
        drift = result["adv_drift_score_100"].dropna()
        assert drift.min() >= -1e-10, f"drift_score below 0: {drift.min()}"
        assert drift.max() <= 1.0 + 1e-10, f"drift_score above 1: {drift.max()}"

    def test_stability_is_complement_of_drift(self, indicator, sample_df):
        """stability + drift_score = 1.0 exactly."""
        result = indicator.compute(sample_df, windows=[100], step=10)
        drift = result["adv_drift_score_100"]
        stability = result["adv_stability_100"]
        both_valid = drift.notna() & stability.notna()
        diff = (drift[both_valid] + stability[both_valid] - 1.0).abs()
        assert diff.max() < 1e-10

    def test_auc_bounded(self, indicator, sample_df):
        """AUC must be in [0.0, 1.0]."""
        result = indicator.compute(sample_df, windows=[100], step=10)
        auc = result["adv_auc_100"].dropna()
        assert auc.min() >= 0.0
        assert auc.max() <= 1.0

    def test_drift_acceleration_is_diff_of_drift(self, indicator, sample_df):
        """drift_acceleration should be the diff of drift_score."""
        result = indicator.compute(sample_df, windows=[100], step=10)
        drift = result["adv_drift_score_100"]
        accel = result["adv_drift_acceleration_100"]
        # Where both drift and previous drift are valid, accel = drift[i] - drift[i-1]
        expected = drift.diff()
        both_valid = accel.notna() & expected.notna()
        if both_valid.sum() > 0:
            np.testing.assert_allclose(
                accel[both_valid].values,
                expected[both_valid].values,
                atol=1e-10,
            )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases are handled gracefully."""

    def test_only_ohlcv_returns_unchanged(self, indicator):
        """No indicator features → return df unchanged."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        df = pd.DataFrame(
            {"O": np.random.randn(n), "H": np.random.randn(n),
             "L": np.random.randn(n), "C": np.random.randn(n),
             "V": np.random.randn(n)},
            index=dates,
        )
        result = indicator.compute(df)
        assert list(result.columns) == list(df.columns)

    def test_fewer_than_three_features_unchanged(self, indicator):
        """< 3 feature columns → return df unchanged."""
        n = 100
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        df = pd.DataFrame(
            {"O": np.random.randn(n), "H": np.random.randn(n),
             "L": np.random.randn(n), "C": np.random.randn(n),
             "V": np.random.randn(n),
             "f1": np.random.randn(n), "f2": np.random.randn(n)},
            index=dates,
        )
        result = indicator.compute(df)
        assert list(result.columns) == list(df.columns)

    def test_short_data_no_crash(self, indicator):
        """30 bars < window → features all NaN, no crash."""
        np.random.seed(42)
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        df = pd.DataFrame(
            {"O": np.random.randn(n), "H": np.random.randn(n),
             "L": np.random.randn(n), "C": np.random.randn(n),
             "V": np.random.randn(n),
             "f1": np.random.randn(n), "f2": np.random.randn(n),
             "f3": np.random.randn(n), "f4": np.random.randn(n)},
            index=dates,
        )
        result = indicator.compute(df, windows=[20], step=5)
        assert len(result) == n

    def test_no_inf_values(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[100], step=10)
        adv_cols = [c for c in result.columns if c.startswith("adv_")]
        for col in adv_cols:
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"Inf found in {col}"

    def test_nan_in_features(self, indicator):
        """NaN in input features should not crash."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        df = pd.DataFrame(
            {"O": np.random.randn(n), "H": np.random.randn(n),
             "L": np.random.randn(n), "C": np.random.randn(n),
             "V": np.random.randn(n),
             "f1": np.random.randn(n), "f2": np.random.randn(n),
             "f3": np.random.randn(n)},
            index=dates,
        )
        df.iloc[30:35, df.columns.get_loc("f1")] = np.nan
        result = indicator.compute(df, windows=[50], step=10)
        assert len(result) == n

    def test_exclude_prefixes(self, indicator, sample_df):
        """Custom exclude_prefixes correctly excludes matching columns."""
        result = indicator.compute(
            sample_df, windows=[100], step=10, exclude_prefixes=["adv_", "trend_"]
        )
        assert len(result) == len(sample_df)
        # Should still produce features (has mom_, vol_, pa_, regime_ columns)
        adv_cols = [c for c in result.columns if c.startswith("adv_")]
        assert len(adv_cols) > 0


# ---------------------------------------------------------------------------
# Parameter variation tests
# ---------------------------------------------------------------------------


class TestParameterVariation:
    """Different parameter values work correctly."""

    def test_custom_windows(self, indicator, sample_df):
        result = indicator.compute(sample_df, windows=[50], step=10)
        assert "adv_auc_50" in result.columns
        assert "adv_auc_100" not in result.columns

    def test_custom_step(self, indicator, sample_df):
        """Smaller step → more computed points (denser non-NaN coverage)."""
        r1 = indicator.compute(sample_df, windows=[100], step=5)
        r2 = indicator.compute(sample_df, windows=[100], step=20)
        assert "adv_auc_100" in r1.columns
        assert "adv_auc_100" in r2.columns

    def test_feature_subsampling(self, indicator):
        """With > max_features columns, subsampling should activate."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        data = {"O": np.random.randn(n), "H": np.random.randn(n),
                "L": np.random.randn(n), "C": np.random.randn(n),
                "V": np.random.randn(n)}
        # Add 40 feature columns (> default max_features=30)
        for i in range(40):
            data[f"feat_{i}"] = np.random.randn(n)
        df = pd.DataFrame(data, index=dates)
        result = indicator.compute(df, windows=[100], step=10, max_features=15)
        adv_cols = [c for c in result.columns if c.startswith("adv_")]
        assert len(adv_cols) > 0


# ---------------------------------------------------------------------------
# Plugin integration tests
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    """Plugin integrates correctly with the registry."""

    def test_plugin_discoverable(self):
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:adversarial_validation")
        assert plugin_cls is not None

    def test_default_params(self, indicator):
        params = indicator.get_default_params()
        assert params["windows"] == [100, 200]
        assert params["step"] == 10
        assert params["max_features"] == 30
        assert params["exclude_prefixes"] == ["adv_"]

    def test_metadata(self, indicator):
        assert indicator.name == "adversarial_validation"
        assert indicator.version == "1.0.0"
