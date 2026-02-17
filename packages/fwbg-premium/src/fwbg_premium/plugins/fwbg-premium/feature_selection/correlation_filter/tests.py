"""Tests for correlation_filter feature selector."""
import numpy as np
import pandas as pd
import pytest

from . import CorrelationFilter


@pytest.fixture
def selector():
    return CorrelationFilter()


@pytest.fixture
def correlated_data():
    """Create dataset with known correlation structure."""
    rng = np.random.default_rng(42)
    n = 200

    base_signal = rng.standard_normal(n)
    noise1 = rng.standard_normal(n) * 0.1
    noise2 = rng.standard_normal(n) * 0.1

    return pd.DataFrame({
        "important_macro_vix": base_signal,                    # base signal
        "redundant_macro_vvix": base_signal + noise1,          # ~0.99 corr with vix
        "redundant_macro_skew": base_signal + noise2,          # ~0.99 corr with vix
        "independent_trend_ema": rng.standard_normal(n),       # independent
        "independent_vol_rv": rng.standard_normal(n),          # independent
        "semi_correlated": base_signal * 0.6 + rng.standard_normal(n) * 0.8,  # ~0.6 corr
    })


@pytest.fixture
def targets():
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], size=200)


def test_removes_highly_correlated(selector, correlated_data, targets):
    """Should remove features correlated >0.7 with already-kept features."""
    selected, meta = selector.select_features(
        correlated_data, targets, max_correlation=0.7
    )

    # First feature (most important) should always be kept
    assert "important_macro_vix" in selected

    # Redundant features (>0.9 corr) should be dropped
    assert "redundant_macro_vvix" not in selected
    assert "redundant_macro_skew" not in selected

    # Independent features should be kept
    assert "independent_trend_ema" in selected
    assert "independent_vol_rv" in selected

    # Semi-correlated (~0.6) should be kept at 0.7 threshold
    assert "semi_correlated" in selected

    assert meta["n_dropped"] == 2


def test_preserves_order(selector, correlated_data, targets):
    """Output should maintain input order (importance ranking)."""
    selected, _ = selector.select_features(
        correlated_data, targets, max_correlation=0.7
    )
    input_order = list(correlated_data.columns)
    for i in range(len(selected) - 1):
        assert input_order.index(selected[i]) < input_order.index(selected[i + 1])


def test_max_features_limit(selector, targets):
    """Should respect max_features even if all features are uncorrelated."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((200, 10)),
                     columns=[f"feat_{i}" for i in range(10)])

    selected, meta = selector.select_features(
        X, targets, max_features=5, max_correlation=0.99
    )
    assert len(selected) == 5
    assert selected == [f"feat_{i}" for i in range(5)]


def test_single_feature(selector, targets):
    """Should handle single feature gracefully."""
    X = pd.DataFrame({"only_one": np.random.randn(200)})
    selected, meta = selector.select_features(X, targets)
    assert selected == ["only_one"]
    assert meta["n_dropped"] == 0


def test_empty_input(selector, targets):
    """Should handle empty input gracefully."""
    X = pd.DataFrame()
    selected, meta = selector.select_features(X, targets)
    assert selected == []
    assert meta["n_dropped"] == 0


def test_strict_threshold(selector, correlated_data, targets):
    """With very strict threshold, more features should be dropped."""
    selected, meta = selector.select_features(
        correlated_data, targets, max_correlation=0.3
    )
    # At 0.3, semi_correlated (~0.6) should also be dropped
    assert len(selected) < 6
    assert meta["n_dropped"] > 2


def test_metadata_has_drop_reasons(selector, correlated_data, targets):
    """Metadata should explain why each feature was dropped."""
    _, meta = selector.select_features(
        correlated_data, targets, max_correlation=0.7
    )
    assert "drop_reasons" in meta
    for dropped_feat in meta["dropped"]:
        assert dropped_feat in meta["drop_reasons"]
        reason = meta["drop_reasons"][dropped_feat]
        assert "r=" in reason  # should contain correlation value
