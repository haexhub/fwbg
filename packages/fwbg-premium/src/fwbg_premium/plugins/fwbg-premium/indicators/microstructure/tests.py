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
        return _micro.MicrostructureIndicators()

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
