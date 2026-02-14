"""
Tests for config-driven macro derived features.

Verifies:
1. MACRO_DERIVED_FEATURES config defines subtract/ratio operations
2. compute_macro_derived() applies operations correctly
3. Missing columns are silently skipped
4. INTEREST_RATES and INTEREST_RATE_DIFFS configs exist
5. compute_interest_rates() loads rate data and computes diffs
6. process.py no longer has hardcoded macro formulas
"""
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Test: Config structure
# =============================================================================

class TestMacroDerivedConfig:
    """MACRO_DERIVED_FEATURES must define operations declaratively."""

    def test_config_exists(self):
        """MACRO_DERIVED_FEATURES must be importable from data.config."""
        from fwbg.data.config import MACRO_DERIVED_FEATURES
        assert isinstance(MACRO_DERIVED_FEATURES, list)
        assert len(MACRO_DERIVED_FEATURES) > 0

    def test_config_has_required_keys(self):
        """Each entry must have name, op, a, b."""
        from fwbg.data.config import MACRO_DERIVED_FEATURES
        for entry in MACRO_DERIVED_FEATURES:
            assert "name" in entry, f"Missing 'name' in {entry}"
            assert "op" in entry, f"Missing 'op' in {entry}"
            assert "a" in entry, f"Missing 'a' in {entry}"
            assert "b" in entry, f"Missing 'b' in {entry}"

    def test_config_operations_are_valid(self):
        """Operations must be 'subtract' or 'ratio'."""
        from fwbg.data.config import MACRO_DERIVED_FEATURES
        valid_ops = {"subtract", "ratio"}
        for entry in MACRO_DERIVED_FEATURES:
            assert entry["op"] in valid_ops, f"Invalid op '{entry['op']}' in {entry}"

    def test_yield_curve_in_config(self):
        """Yield curve features must be defined in config."""
        from fwbg.data.config import MACRO_DERIVED_FEATURES
        names = {e["name"] for e in MACRO_DERIVED_FEATURES}
        assert "macro_yield_curve_10y_3m" in names
        assert "macro_yield_curve_10y_5y" in names

    def test_ratios_in_config(self):
        """Key ratios must be defined in config."""
        from fwbg.data.config import MACRO_DERIVED_FEATURES
        names = {e["name"] for e in MACRO_DERIVED_FEATURES}
        assert "macro_vix_vvix_ratio" in names
        assert "macro_risk_ratio_spx_tlt" in names
        assert "macro_credit_spread_proxy" in names
        assert "macro_smallcap_ratio" in names
        assert "macro_tech_defensive_ratio" in names

    def test_interest_rate_config_exists(self):
        """INTEREST_RATES config must exist."""
        from fwbg.data.config import INTEREST_RATES
        assert isinstance(INTEREST_RATES, list)
        assert len(INTEREST_RATES) >= 2  # at least FED and ECB

    def test_interest_rate_diff_config_exists(self):
        """INTEREST_RATE_DIFFS config must exist."""
        from fwbg.data.config import INTEREST_RATE_DIFFS
        assert isinstance(INTEREST_RATE_DIFFS, list)
        assert len(INTEREST_RATE_DIFFS) >= 1  # at least USD-EUR


# =============================================================================
# Test: compute_macro_derived()
# =============================================================================

class TestComputeMacroDerived:
    """compute_macro_derived() must apply subtract/ratio ops from config."""

    @pytest.fixture
    def macro_df(self):
        """DataFrame with macro columns."""
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
        """Subtract op: result = a - b."""
        from fwbg.data.config import compute_macro_derived
        result = compute_macro_derived(macro_df)

        assert "macro_yield_curve_10y_3m" in result.columns
        expected = 4.0 - 1.5  # tnx - irx
        np.testing.assert_allclose(result["macro_yield_curve_10y_3m"].values, expected)

    def test_ratio_operation(self, macro_df):
        """Ratio op: result = a / (b + 1e-10)."""
        from fwbg.data.config import compute_macro_derived
        result = compute_macro_derived(macro_df)

        assert "macro_vix_vvix_ratio" in result.columns
        expected = 20.0 / (100.0 + 1e-10)
        np.testing.assert_allclose(result["macro_vix_vvix_ratio"].values, expected, rtol=1e-6)

    def test_all_derived_features_created(self, macro_df):
        """All features from config should be created when columns exist."""
        from fwbg.data.config import compute_macro_derived, MACRO_DERIVED_FEATURES
        result = compute_macro_derived(macro_df)

        for entry in MACRO_DERIVED_FEATURES:
            assert entry["name"] in result.columns, f"Missing derived feature: {entry['name']}"

    def test_missing_columns_skipped(self):
        """Missing source columns should be silently skipped."""
        from fwbg.data.config import compute_macro_derived

        df = pd.DataFrame({"macro_tnx": [4.0, 4.1]})  # irx missing
        result = compute_macro_derived(df)

        # yield_curve_10y_3m requires both tnx and irx - should be skipped
        assert "macro_yield_curve_10y_3m" not in result.columns

    def test_empty_dataframe(self):
        """Empty DataFrame should not crash."""
        from fwbg.data.config import compute_macro_derived
        df = pd.DataFrame()
        result = compute_macro_derived(df)
        assert isinstance(result, pd.DataFrame)

    def test_custom_config(self):
        """Custom config can override defaults."""
        from fwbg.data.config import compute_macro_derived

        df = pd.DataFrame({"col_a": [10.0], "col_b": [3.0]})
        custom = [{"name": "custom_diff", "op": "subtract", "a": "col_a", "b": "col_b"}]
        result = compute_macro_derived(df, config=custom)

        assert "custom_diff" in result.columns
        assert result["custom_diff"].iloc[0] == 7.0


# =============================================================================
# Test: No hardcoded formulas in process.py
# =============================================================================

class TestNoHardcodedMacroFormulas:
    """process.py must not contain hardcoded macro derived feature formulas."""

    def test_no_hardcoded_yield_curve(self):
        """process.py must not compute yield curves inline."""
        import inspect
        from fwbg.optimization import process
        source = inspect.getsource(process)

        assert 'macro_yield_curve_10y_3m' not in source, (
            "process.py still has hardcoded yield curve computation"
        )

    def test_no_hardcoded_vix_ratio(self):
        """process.py must not compute VIX/VVIX ratio inline."""
        import inspect
        from fwbg.optimization import process
        source = inspect.getsource(process)

        assert 'macro_vix_vvix_ratio' not in source, (
            "process.py still has hardcoded VIX/VVIX ratio"
        )

    def test_no_hardcoded_risk_ratio(self):
        """process.py must not compute risk ratios inline."""
        import inspect
        from fwbg.optimization import process
        source = inspect.getsource(process)

        assert 'macro_risk_ratio_spx_tlt' not in source
        assert 'macro_credit_spread_proxy' not in source
        assert 'macro_smallcap_ratio' not in source
        assert 'macro_tech_defensive_ratio' not in source

    def test_no_hardcoded_rate_diff(self):
        """process.py must not compute rate diffs inline."""
        import inspect
        from fwbg.optimization import process
        source = inspect.getsource(process)

        assert 'macro_rate_diff_usd_eur' not in source

    def test_uses_compute_macro_derived(self):
        """process.py must import compute_macro_derived."""
        import inspect
        from fwbg.optimization import process
        source = inspect.getsource(process)

        assert 'compute_macro_derived' in source, (
            "process.py should use compute_macro_derived()"
        )
