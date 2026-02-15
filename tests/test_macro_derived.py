"""
Tests for config-driven macro derived features.

Verifies compute_macro_derived() correctly applies subtract/ratio operations.
"""
import numpy as np
import pandas as pd
import pytest


class TestComputeMacroDerived:
    """compute_macro_derived() applies subtract/ratio ops from config."""

    @pytest.fixture
    def macro_df(self):
        n = 100
        return pd.DataFrame({
            "macro_tnx": np.full(n, 4.0),
            "macro_irx": np.full(n, 1.5),
            "macro_fvx": np.full(n, 3.0),
            "macro_vix": np.full(n, 20.0),
            "macro_vvix": np.full(n, 100.0),
            "macro_spx": np.full(n, 5000.0),
            "macro_tlt": np.full(n, 100.0),
            "macro_hyg": np.full(n, 80.0),
            "macro_lqd": np.full(n, 110.0),
            "macro_russell": np.full(n, 2000.0),
            "macro_xlk": np.full(n, 200.0),
            "macro_xlu": np.full(n, 70.0),
        })

    def test_subtract_operation(self, macro_df):
        from fwbg.data.config import compute_macro_derived
        result = compute_macro_derived(macro_df)
        assert "macro_yield_curve_10y_3m" in result.columns
        expected = 4.0 - 1.5
        np.testing.assert_allclose(result["macro_yield_curve_10y_3m"].values, expected)

    def test_ratio_operation(self, macro_df):
        from fwbg.data.config import compute_macro_derived
        result = compute_macro_derived(macro_df)
        assert "macro_vix_vvix_ratio" in result.columns
        expected = 20.0 / (100.0 + 1e-10)
        np.testing.assert_allclose(result["macro_vix_vvix_ratio"].values, expected, rtol=1e-6)

    def test_all_derived_features_created(self, macro_df):
        from fwbg.data.config import compute_macro_derived, MACRO_DERIVED_FEATURES
        result = compute_macro_derived(macro_df)
        for entry in MACRO_DERIVED_FEATURES:
            assert entry["name"] in result.columns, f"Missing derived feature: {entry['name']}"

    def test_missing_columns_skipped(self):
        from fwbg.data.config import compute_macro_derived
        df = pd.DataFrame({"macro_tnx": [4.0, 4.1]})
        result = compute_macro_derived(df)
        assert "macro_yield_curve_10y_3m" not in result.columns

    def test_empty_dataframe(self):
        from fwbg.data.config import compute_macro_derived
        df = pd.DataFrame()
        result = compute_macro_derived(df)
        assert isinstance(result, pd.DataFrame)

    def test_custom_config(self):
        from fwbg.data.config import compute_macro_derived
        df = pd.DataFrame({"col_a": [10.0], "col_b": [3.0]})
        custom = [{"name": "custom_diff", "op": "subtract", "a": "col_a", "b": "col_b"}]
        result = compute_macro_derived(df, config=custom)
        assert "custom_diff" in result.columns
        assert result["custom_diff"].iloc[0] == 7.0
