"""
Tests for the dynamics indicator plugin.

This plugin computes rate-of-change (derivatives) of other indicators:
- Changes in RSI, ATR, ADX, BB-width, MACD, Stochastic over 4h/8h/24h windows
- Percent changes for the same
- Lag features (shifted values)
- Acceleration (2nd derivative) of key indicators and price

Purpose: captures momentum-of-momentum - e.g. is RSI rising or falling?
A rising RSI with positive acceleration signals an accelerating trend.

Column names (from get_feature_columns):
  dyn_rsi14_chg_4h / _8h / _24h   - absolute RSI change
  dyn_rsi14_pct_4h / ...           - percent RSI change
  dyn_atr_chg_4h / ...             - percent ATR (vol_atr_pct_14) change
  dyn_bbwidth_chg_4h / ...         - percent BB-width change
  dyn_adx_chg_4h / ...             - absolute ADX change
  dyn_macd_chg_4h / _8h            - MACD change
  dyn_stoch_chg_4h / _8h           - Stochastic change
  lag_rsi14_4h / _8h / _24h        - lagged RSI
  lag_atr_4h / _8h / _24h          - lagged ATR
  lag_adx_4h / _8h                 - lagged ADX
  lag_price_chg_4h/8h/24h/48h      - lagged price pct change
  accel_rsi / accel_atr / accel_adx / accel_price  - 2nd derivatives

NOTE: The plugin calls shift_features() which shifts all outputs by 1 bar to
prevent lookahead bias. Tests account for this where relevant.
"""
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_dyn = import_plugin_module("fwbg-premium", "indicators", "dynamics")
if _dyn is None:
    pytest.skip("dynamics plugin not available", allow_module_level=True)


def _find_indicator_class(module):
    import inspect
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and not inspect.isabstract(obj)
            and hasattr(obj, "compute")
            and hasattr(obj, "get_feature_columns")
        ):
            return obj
    raise RuntimeError(f"Could not find indicator class in {module}")


def _make_ohlc_with_stub_indicators(
    n=600,
    rsi_value=50.0,
    atr_pct_value=0.01,
    adx_value=25.0,
    seed=42,
):
    """
    Build an OHLC DataFrame with stub pre-computed indicator columns
    that DynamicsIndicators depends on (read via df.get()).

    Exact column names used by the plugin (from __init__.py):
      mom_rsi_14      - RSI
      trend_adx_14    - ADX
      vol_atr_pct_14  - ATR as pct of close
      vol_bb_wband_20 - Bollinger band width
      mom_stoch_k_14  - Stochastic %K
      trend_macd      - MACD normalised
    """
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    df = pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.005,
            "L": close * 0.995,
            "C": close,
        },
        index=pd.date_range("2022-01-03", periods=n, freq="h"),
    )

    # Exact column names the plugin reads (df.get / df.columns checks)
    df["mom_rsi_14"] = rsi_value
    df["trend_adx_14"] = adx_value
    df["vol_atr_pct_14"] = atr_pct_value
    df["vol_bb_wband_20"] = 0.02
    df["mom_stoch_k_14"] = 50.0
    df["trend_macd"] = 0.0

    return df


def _get_ind():
    return _find_indicator_class(_dyn)()


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestPluginAttributes:
    def test_class_name(self):
        cls = _find_indicator_class(_dyn)
        assert cls.__name__ == "DynamicsIndicators"

    def test_feature_columns_declared(self):
        cols = _get_ind().get_feature_columns()
        assert len(cols) > 0, "No feature columns declared"

    def test_feature_columns_contains_rsi_change(self):
        cols = _get_ind().get_feature_columns()
        rsi_chg = [c for c in cols if "rsi" in c and "chg" in c]
        assert len(rsi_chg) >= 3, f"Expected >=3 RSI-change columns, got {rsi_chg}"

    def test_feature_columns_contains_acceleration(self):
        cols = _get_ind().get_feature_columns()
        accel = [c for c in cols if "accel" in c]
        assert len(accel) >= 4, f"Expected >=4 acceleration columns, got {accel}"

    def test_feature_columns_contains_lag(self):
        cols = _get_ind().get_feature_columns()
        lags = [c for c in cols if c.startswith("lag_")]
        assert len(lags) >= 4, f"Expected >=4 lag columns, got {lags}"

    def test_compute_returns_dataframe(self):
        df = _make_ohlc_with_stub_indicators()
        result = _get_ind().compute(df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_preserves_original_columns(self):
        df = _make_ohlc_with_stub_indicators()
        result = _get_ind().compute(df)
        for col in df.columns:
            assert col in result.columns, f"Original column {col!r} missing from result"

    def test_all_declared_features_present(self):
        df = _make_ohlc_with_stub_indicators()
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Declared feature {col!r} not in result"

    def test_no_inf_values(self):
        df = _make_ohlc_with_stub_indicators()
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                bad = result[col].isin([float("inf"), float("-inf")])
                assert not bad.any(), f"{col} contains inf"

    def test_first_row_nan(self):
        """All feature columns should be NaN in row 0 (shift_features adds a leading NaN)."""
        df = _make_ohlc_with_stub_indicators()
        result = _get_ind().compute(df)
        for col in _get_ind().get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN"


# ---------------------------------------------------------------------------
# Constant-indicator sanity checks
# ---------------------------------------------------------------------------


class TestConstantIndicators:
    """When upstream indicators are constant, all change/acceleration features should be ~0."""

    def test_rsi_changes_zero_when_constant_rsi(self):
        df = _make_ohlc_with_stub_indicators(rsi_value=50.0)
        result = _get_ind().compute(df)
        change_cols = [c for c in result.columns if "rsi" in c and "chg" in c]
        assert change_cols, "No RSI change columns found in result"
        for col in change_cols:
            valid = result[col].dropna()
            assert (valid.abs() < 1e-8).all(), (
                f"{col} should be exactly 0 for constant RSI, got max={valid.abs().max():.6e}"
            )

    def test_rsi_pct_changes_zero_when_constant_rsi(self):
        df = _make_ohlc_with_stub_indicators(rsi_value=50.0)
        result = _get_ind().compute(df)
        pct_cols = [c for c in result.columns if "rsi" in c and "pct" in c]
        for col in pct_cols:
            valid = result[col].dropna()
            assert (valid.abs() < 1e-8).all(), (
                f"{col} should be 0 for constant RSI, got max={valid.abs().max():.6e}"
            )

    def test_adx_changes_zero_when_constant_adx(self):
        df = _make_ohlc_with_stub_indicators(adx_value=25.0)
        result = _get_ind().compute(df)
        change_cols = [c for c in result.columns if "adx" in c and "chg" in c]
        assert change_cols, "No ADX change columns found in result"
        for col in change_cols:
            valid = result[col].dropna()
            assert (valid.abs() < 1e-8).all(), (
                f"{col} should be 0 for constant ADX, got max={valid.abs().max():.6e}"
            )

    def test_atr_changes_zero_when_constant_atr(self):
        df = _make_ohlc_with_stub_indicators(atr_pct_value=0.01)
        result = _get_ind().compute(df)
        change_cols = [c for c in result.columns if "atr" in c and "chg" in c]
        assert change_cols, "No ATR change columns found in result"
        for col in change_cols:
            valid = result[col].dropna()
            assert (valid.abs() < 1e-8).all(), (
                f"{col} should be 0 for constant ATR, got max={valid.abs().max():.6e}"
            )

    def test_acceleration_zero_when_all_constant(self):
        df = _make_ohlc_with_stub_indicators()
        result = _get_ind().compute(df)
        accel_cols = [c for c in result.columns if c.startswith("accel_") and "rsi" in c]
        for col in accel_cols:
            valid = result[col].dropna()
            assert (valid.abs() < 1e-8).all(), (
                f"{col} should be 0 for constant inputs, got max={valid.abs().max():.6e}"
            )


# ---------------------------------------------------------------------------
# Rising-indicator checks
# ---------------------------------------------------------------------------


class TestRisingIndicators:
    """When RSI rises linearly, change features should be consistently positive."""

    def _make_rising_rsi_df(self, n=600):
        df = _make_ohlc_with_stub_indicators(n=n)
        df["mom_rsi_14"] = np.linspace(30.0, 70.0, n)
        return df

    def test_rsi_absolute_changes_positive(self):
        """dyn_rsi14_chg_* should be positive when RSI is monotonically rising."""
        df = self._make_rising_rsi_df()
        result = _get_ind().compute(df)
        for col in ["dyn_rsi14_chg_4h", "dyn_rsi14_chg_8h", "dyn_rsi14_chg_24h"]:
            if col not in result.columns:
                continue
            # Skip early NaN rows; check the stable tail
            valid = result[col].dropna().iloc[50:]
            assert (valid > 0).all(), (
                f"{col} should be positive when RSI rises linearly, got {valid.describe()}"
            )

    def test_rsi_pct_changes_positive(self):
        df = self._make_rising_rsi_df()
        result = _get_ind().compute(df)
        for col in ["dyn_rsi14_pct_4h", "dyn_rsi14_pct_8h", "dyn_rsi14_pct_24h"]:
            if col not in result.columns:
                continue
            valid = result[col].dropna().iloc[50:]
            assert (valid > 0).all(), (
                f"{col} should be positive when RSI rises, got {valid.describe()}"
            )

    def test_lag_rsi_lower_than_current(self):
        """Lagged RSI should be < current RSI in an uptrend."""
        df = self._make_rising_rsi_df()
        result = _get_ind().compute(df)
        if "lag_rsi14_4h" in result.columns:
            tail = result[["mom_rsi_14", "lag_rsi14_4h"]].dropna().iloc[50:]
            # lag is 4h behind current -> lag < current RSI
            assert (tail["lag_rsi14_4h"] < tail["mom_rsi_14"]).all(), (
                "lag_rsi14_4h should be < current RSI in uptrend"
            )


# ---------------------------------------------------------------------------
# Acceleration tests
# ---------------------------------------------------------------------------


class TestAcceleration:
    """Acceleration (2nd derivative) tests."""

    def test_acceleration_near_zero_for_linear_rsi(self):
        """Linear RSI -> constant change rate -> acceleration = 0."""
        df = _make_ohlc_with_stub_indicators()
        df["mom_rsi_14"] = np.linspace(30.0, 70.0, len(df))
        result = _get_ind().compute(df)
        if "accel_rsi" in result.columns:
            valid = result["accel_rsi"].dropna()
            # Linear changes -> accel should be 0 (within floating-point noise)
            assert (valid.abs() < 1e-6).all(), (
                f"accel_rsi should be ~0 for linear RSI, got max={valid.abs().max():.2e}"
            )

    def test_acceleration_positive_for_accelerating_rsi(self):
        """Quadratic RSI (accelerating) -> non-negative accel_rsi."""
        n = 600
        df = _make_ohlc_with_stub_indicators(n=n)
        # Quadratic: change rate grows over time
        t = np.arange(n, dtype=float)
        df["mom_rsi_14"] = 30.0 + (t / n) ** 2 * 40.0
        result = _get_ind().compute(df)
        if "accel_rsi" in result.columns:
            valid = result["accel_rsi"].dropna().iloc[50:]
            assert (valid >= 0).all(), (
                f"accel_rsi should be >= 0 for quadratic RSI, got min={valid.min():.4f}"
            )

    def test_accel_price_non_constant(self):
        """With random-walk prices, accel_price should not be uniformly zero."""
        df = _make_ohlc_with_stub_indicators(seed=99)
        result = _get_ind().compute(df)
        if "accel_price" in result.columns:
            valid = result["accel_price"].dropna()
            assert valid.std() > 0, "accel_price should vary for random-walk prices"
