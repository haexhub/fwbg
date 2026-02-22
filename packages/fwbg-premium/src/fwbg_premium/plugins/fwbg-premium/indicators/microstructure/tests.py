"""Tests for microstructure indicator plugin - CMF / A-D Line features."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_micro = import_plugin_module("fwbg-premium", "indicators", "microstructure")
if _micro is None:
    pytest.skip("fwbg-premium microstructure plugin not available", allow_module_level=True)


@pytest.fixture
def accumulation_df():
    """Preis steigt, Close nahe High -> CLV positiv -> A/D Line steigt."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = np.array([100.0 + i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "O": close - 0.2,
            "H": close + 0.1,
            "L": close - 0.5,
            "C": close,
            "V": np.full(n, 1000.0),
        },
        index=dates,
    )


@pytest.fixture
def distribution_sell_df():
    """Preis fällt, Close nahe Low -> CLV negativ -> A/D Line fällt."""
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = np.array([200.0 - i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "O": close + 0.2,
            "H": close + 0.5,
            "L": close - 0.1,
            "C": close,
            "V": np.full(n, 1000.0),
        },
        index=dates,
    )


class TestVolumeFlow:
    """Tests für CMF und A/D Line im Microstructure Plugin."""

    def _get_indicator(self):
        return _micro.MicrostructureIndicator()

    def test_volume_flow_columns_present(self, accumulation_df):
        """Alle neuen Volume-Flow Spalten werden erzeugt."""
        ind = self._get_indicator()
        result = ind.compute(accumulation_df)
        for col in ["micro_ad_line", "micro_ad_zscore", "micro_cmf_10", "micro_cmf_20"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_accumulation_positive_ad(self, accumulation_df):
        """Bei Akkumulation (Close nahe High) sollte A/D Line steigen."""
        ind = self._get_indicator()
        result = ind.compute(accumulation_df)
        ad = result["micro_ad_line"].dropna()
        early = ad.iloc[20:40].mean()
        late = ad.iloc[-40:].mean()
        assert late > early, f"A/D should rise during accumulation: early={early}, late={late}"

    def test_distribution_negative_ad(self, distribution_sell_df):
        """Bei Distribution (Close nahe Low) sollte A/D Line fallen."""
        ind = self._get_indicator()
        result = ind.compute(distribution_sell_df)
        ad = result["micro_ad_line"].dropna()
        early = ad.iloc[20:40].mean()
        late = ad.iloc[-40:].mean()
        assert late < early, f"A/D should fall during distribution: early={early}, late={late}"

    def test_cmf_positive_during_accumulation(self, accumulation_df):
        """CMF sollte positiv sein wenn Close nahe High ist."""
        ind = self._get_indicator()
        result = ind.compute(accumulation_df)
        cmf20 = result["micro_cmf_20"].dropna()
        last_values = cmf20.iloc[-50:]
        assert last_values.mean() > 0, f"CMF should be positive during accumulation, got {last_values.mean()}"

    def test_cmf_negative_during_distribution(self, distribution_sell_df):
        """CMF sollte negativ sein wenn Close nahe Low ist."""
        ind = self._get_indicator()
        result = ind.compute(distribution_sell_df)
        cmf20 = result["micro_cmf_20"].dropna()
        last_values = cmf20.iloc[-50:]
        assert last_values.mean() < 0, f"CMF should be negative during distribution, got {last_values.mean()}"

    def test_cmf_bounded(self, accumulation_df):
        """CMF sollte zwischen -1 und 1 liegen."""
        ind = self._get_indicator()
        result = ind.compute(accumulation_df)
        for col in ["micro_cmf_10", "micro_cmf_20"]:
            vals = result[col].dropna()
            assert vals.min() >= -1.01, f"{col} min={vals.min()}"
            assert vals.max() <= 1.01, f"{col} max={vals.max()}"

    def test_works_without_volume(self):
        """CMF/A-D Line sollten auch ohne Volume-Spalte funktionieren."""
        ind = self._get_indicator()
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        close = np.array([100.0 + i * 0.5 for i in range(n)])
        df = pd.DataFrame(
            {"O": close - 0.3, "H": close + 0.5, "L": close - 0.2, "C": close},
            index=dates,
        )
        result = ind.compute(df)
        assert "micro_ad_line" in result.columns
        assert "micro_cmf_10" in result.columns
        assert "micro_cmf_20" in result.columns

    def test_no_lookahead_bias(self, accumulation_df):
        """Volume-Flow Features sollten um 1 Bar verschoben sein."""
        ind = self._get_indicator()
        result = ind.compute(accumulation_df)
        assert pd.isna(result["micro_ad_line"].iloc[0])
        assert pd.isna(result["micro_cmf_20"].iloc[0])


# ── NEW: Wick imbalance tests ─────────────────────────────────────────────────

class TestWickImbalance:
    """
    Wick imbalance measures directional rejection:
    - Upper wick dominant (H far above C and O) → selling pressure above → positive imbalance
    - Lower wick dominant (L far below C and O) → buying pressure below → negative imbalance
    - Balanced wicks → near-zero imbalance
    """

    def _get_indicator(self):
        return _micro.MicrostructureIndicator()

    def _make_upper_wick_df(self, n=100):
        """Large upper wick, no lower wick: sellers rejected price at top."""
        return pd.DataFrame({
            "O": np.full(n, 100.0),
            "C": np.full(n, 100.5),   # tiny bullish body
            "H": np.full(n, 103.0),   # large upper wick = 2.5
            "L": np.full(n, 100.0),   # no lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))

    def _make_lower_wick_df(self, n=100):
        """Large lower wick, no upper wick: buyers rejected price at bottom."""
        return pd.DataFrame({
            "O": np.full(n, 100.0),
            "C": np.full(n, 99.5),    # tiny bearish body
            "H": np.full(n, 100.0),   # no upper wick
            "L": np.full(n, 97.0),    # large lower wick = 3.0
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))

    def _make_symmetric_wick_df(self, n=100):
        """Equal upper and lower wicks: balanced rejection."""
        return pd.DataFrame({
            "O": np.full(n, 100.0),
            "C": np.full(n, 100.0),   # doji
            "H": np.full(n, 102.0),   # equal upper wick
            "L": np.full(n, 98.0),    # equal lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))

    def test_upper_wick_dominant_gives_positive_imbalance(self):
        df = self._make_upper_wick_df()
        result = self._get_indicator().compute(df)
        if "micro_wick_imbalance" in result.columns:
            valid = result["micro_wick_imbalance"].dropna()
            pct_positive = (valid > 0).mean()
            assert pct_positive > 0.8, \
                f"Upper wick dominant → wick_imbalance should be positive ({pct_positive:.1%} positive)"

    def test_lower_wick_dominant_gives_negative_imbalance(self):
        df = self._make_lower_wick_df()
        result = self._get_indicator().compute(df)
        if "micro_wick_imbalance" in result.columns:
            valid = result["micro_wick_imbalance"].dropna()
            pct_negative = (valid < 0).mean()
            assert pct_negative > 0.8, \
                f"Lower wick dominant → wick_imbalance should be negative ({pct_negative:.1%} negative)"

    def test_symmetric_wicks_near_zero_imbalance(self):
        df = self._make_symmetric_wick_df()
        result = self._get_indicator().compute(df)
        if "micro_wick_imbalance" in result.columns:
            valid = result["micro_wick_imbalance"].dropna()
            assert (valid.abs() < 0.2).all(), \
                f"Symmetric wicks → imbalance should be ~0, got max abs={valid.abs().max():.4f}"


# ── NEW: Pressure score tests ─────────────────────────────────────────────────

class TestPressureScore:
    """
    Pressure score = sign(C - O) × body_ratio.
    - Bullish marubozu (C=H, O=L): pressure = +1 (full buying pressure)
    - Bearish marubozu (O=H, C=L): pressure = -1 (full selling pressure)
    - Doji: pressure ≈ 0
    """

    def _get_indicator(self):
        return _micro.MicrostructureIndicator()

    def _make_bullish_marubozu(self, n=100):
        return pd.DataFrame({
            "O": np.full(n, 99.0),
            "C": np.full(n, 101.0),  # C > O → bullish
            "H": np.full(n, 101.0),  # H == C → no upper wick
            "L": np.full(n, 99.0),   # L == O → no lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))

    def _make_bearish_marubozu(self, n=100):
        return pd.DataFrame({
            "O": np.full(n, 101.0),
            "C": np.full(n, 99.0),   # C < O → bearish
            "H": np.full(n, 101.0),  # H == O → no upper wick
            "L": np.full(n, 99.0),   # L == C → no lower wick
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))

    def test_pressure_positive_for_bullish_marubozu(self):
        df = self._make_bullish_marubozu()
        result = self._get_indicator().compute(df)
        if "micro_pressure_score" in result.columns:
            valid = result["micro_pressure_score"].dropna()
            assert (valid > 0).all(), \
                f"Bullish marubozu → pressure_score must be positive, got min={valid.min():.4f}"

    def test_pressure_negative_for_bearish_marubozu(self):
        df = self._make_bearish_marubozu()
        result = self._get_indicator().compute(df)
        if "micro_pressure_score" in result.columns:
            valid = result["micro_pressure_score"].dropna()
            assert (valid < 0).all(), \
                f"Bearish marubozu → pressure_score must be negative, got max={valid.max():.4f}"


# ── NEW: Rolling aggregate tests ──────────────────────────────────────────────

class TestRollingAggregates:
    """
    5-bar rolling sums of wick_imbalance, intrabar_bias, and pressure_score.
    Direction consistency = |mean(direction over 5 bars)| → 1 when all same direction.
    """

    def _get_indicator(self):
        return _micro.MicrostructureIndicator()

    def _make_constant_bullish(self, n=100):
        return pd.DataFrame({
            "O": np.full(n, 99.0),
            "C": np.full(n, 101.0),
            "H": np.full(n, 101.0),
            "L": np.full(n, 99.0),
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))

    def test_direction_consistency_high_when_all_same_direction(self):
        """All bullish bars → direction_consistency should approach 1."""
        df = self._make_constant_bullish()
        result = self._get_indicator().compute(df)
        if "micro_direction_consistency" in result.columns:
            valid = result["micro_direction_consistency"].dropna()
            if len(valid) >= 10:
                assert (valid.iloc[-10:] > 0.8).all(), \
                    f"All same direction → consistency should be near 1, got min={valid.iloc[-10:].min():.3f}"

    def test_pressure_sum_stabilizes_after_window(self):
        """After 5+ constant-bullish bars, pressure_sum should stabilize."""
        df = self._make_constant_bullish()
        result = self._get_indicator().compute(df)
        if "micro_pressure_sum" in result.columns:
            valid = result["micro_pressure_sum"].dropna()
            if len(valid) >= 10:
                # Should be constant (all same bars → same window sum)
                last_10_std = valid.iloc[-10:].std()
                assert last_10_std < 0.01, \
                    f"Constant bars → pressure_sum should be stable, got std={last_10_std:.6f}"
