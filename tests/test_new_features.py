"""
Tests für neue Feature-Ergänzungen:
- Auto-Korrelation (distribution plugin)
- CMF / Accumulation-Distribution Line (microstructure plugin)
- Rolling Beta (risk plugin)

Verwendet synthetische Daten mit bekannten Eigenschaften.
"""
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def trending_df():
    """Linearer Aufwärtstrend — bekannte Auto-Korrelation > 0."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = np.array([100.0 + i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "O": close - 0.3,
            "H": close + 0.5,
            "L": close - 0.5,
            "C": close,
            "V": np.random.default_rng(42).integers(500, 2000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def mean_reverting_df():
    """Oszillierender Preis — bekannte Auto-Korrelation < 0."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    # Preis oszilliert um 100 mit alternierenden +/- Schritten
    rng = np.random.default_rng(42)
    base = 100.0
    prices = [base]
    for i in range(1, n):
        # Starke Mean-Reversion: immer Richtung umkehren + noise
        step = -np.sign(prices[-1] - base) * 1.0 + rng.normal(0, 0.3)
        prices.append(prices[-1] + step)
    close = np.array(prices)
    return pd.DataFrame(
        {
            "O": close - 0.2,
            "H": close + 0.3,
            "L": close - 0.3,
            "C": close,
            "V": rng.integers(500, 2000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def correlated_asset_df():
    """Asset das stark mit 'Benchmark' korreliert — für Beta-Tests."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)

    # SPX als Benchmark
    spx_returns = rng.normal(0.0005, 0.01, n)
    spx = 4000.0 * np.cumprod(1 + spx_returns)

    # Asset mit Beta ~2.0 zum SPX
    asset_returns = 2.0 * spx_returns + rng.normal(0, 0.005, n)
    close = 100.0 * np.cumprod(1 + asset_returns)

    return pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.002,
            "L": close * 0.998,
            "C": close,
            "V": rng.integers(500, 2000, n).astype(float),
            "macro_spx": spx,
        },
        index=dates,
    )


@pytest.fixture
def accumulation_df():
    """Preis steigt, Close nahe High → CLV positiv → A/D Line steigt.

    CLV = (2C - H - L) / (H - L)
    Mit H=C+0.1, L=C-0.5: CLV = (0.4) / (0.6) = +0.667
    """
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = np.array([100.0 + i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "O": close - 0.2,
            "H": close + 0.1,  # Close nahe am High
            "L": close - 0.5,  # Low weit unter Close
            "C": close,
            "V": np.full(n, 1000.0),
        },
        index=dates,
    )


@pytest.fixture
def distribution_sell_df():
    """Preis fällt, Close nahe Low → CLV negativ → A/D Line fällt.

    CLV = (2C - H - L) / (H - L)
    Mit H=C+0.5, L=C-0.1: CLV = (-0.4) / (0.6) = -0.667
    """
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = np.array([200.0 - i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "O": close + 0.2,
            "H": close + 0.5,  # High weit über Close
            "L": close - 0.1,  # Close nahe am Low
            "C": close,
            "V": np.full(n, 1000.0),
        },
        index=dates,
    )


# =============================================================================
# AUTO-KORRELATION TESTS
# =============================================================================


class TestAutoCorrelation:
    """Tests für Auto-Korrelation Features im Distribution Plugin."""

    def _get_indicator(self):
        from fwbg.core.registry import INDICATOR_REGISTRY
        from fwbg.plugins import import_plugin_module

        if "distribution" not in INDICATOR_REGISTRY:
            import_plugin_module("fwbg-premium", "indicators", "distribution")
        return INDICATOR_REGISTRY["distribution"]()

    def test_autocorr_columns_present(self, trending_df):
        """Alle Auto-Korrelation Spalten werden erzeugt."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        for lag in [1, 5, 10, 20]:
            assert f"dist_autocorr_{lag}" in result.columns
        assert "dist_autocorr_1_change" in result.columns

    def test_trending_positive_autocorrelation(self, trending_df):
        """Trending-Daten sollten positive Auto-Korrelation auf Lag 1 haben."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        # Shift berücksichtigen: Features sind um 1 verschoben
        autocorr_1 = result["dist_autocorr_1"].dropna()
        # Die letzten Werte sollten positiv sein (trending)
        last_values = autocorr_1.iloc[-50:]
        mean_ac = last_values.mean()
        assert mean_ac > 0, f"Expected positive autocorr for trend, got {mean_ac}"

    def test_mean_reverting_negative_autocorrelation(self, mean_reverting_df):
        """Mean-Reverting-Daten sollten negative Auto-Korrelation auf Lag 1 haben."""
        ind = self._get_indicator()
        result = ind.compute(mean_reverting_df)
        autocorr_1 = result["dist_autocorr_1"].dropna()
        last_values = autocorr_1.iloc[-50:]
        mean_ac = last_values.mean()
        assert mean_ac < 0, f"Expected negative autocorr for mean-reversion, got {mean_ac}"

    def test_autocorr_bounded(self, trending_df):
        """Auto-Korrelation sollte zwischen -1 und 1 liegen."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        for lag in [1, 5, 10, 20]:
            col = result[f"dist_autocorr_{lag}"].dropna()
            assert col.min() >= -1.01, f"Lag {lag}: min={col.min()}"
            assert col.max() <= 1.01, f"Lag {lag}: max={col.max()}"

    def test_autocorr_change_detects_regime_shift(self):
        """Auto-Korrelation Change sollte Regime-Wechsel erkennen."""
        ind = self._get_indicator()
        n = 400
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        rng = np.random.default_rng(42)

        # Erste Hälfte: trending, zweite Hälfte: mean-reverting
        prices = [100.0]
        for i in range(1, n):
            if i < 200:
                step = 0.5 + rng.normal(0, 0.1)  # trending
            else:
                step = -np.sign(prices[-1] - 200) * 1.0 + rng.normal(0, 0.2)
                # mean-reverting
            prices.append(prices[-1] + step)

        df = pd.DataFrame(
            {
                "O": np.array(prices) - 0.2,
                "H": np.array(prices) + 0.3,
                "L": np.array(prices) - 0.3,
                "C": np.array(prices),
                "V": rng.integers(500, 2000, n).astype(float),
            },
            index=dates,
        )

        result = ind.compute(df)
        change_col = result["dist_autocorr_1_change"].dropna()
        # Nach dem Regime-Shift sollte sich die Change-Spalte bewegen
        assert not change_col.empty
        assert change_col.std() > 0, "Change should vary during regime shift"

    def test_no_lookahead_bias(self, trending_df):
        """Features sollten um 1 Bar verschoben sein (kein Lookahead)."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        # Erste Zeile sollte NaN sein (shift by 1)
        assert pd.isna(result["dist_autocorr_1"].iloc[0])


# =============================================================================
# CMF / ACCUMULATION-DISTRIBUTION LINE TESTS
# =============================================================================


class TestVolumeFlow:
    """Tests für CMF und A/D Line im Microstructure Plugin."""

    def _get_indicator(self):
        from fwbg.core.registry import INDICATOR_REGISTRY
        from fwbg.plugins import import_plugin_module

        if "microstructure" not in INDICATOR_REGISTRY:
            import_plugin_module("fwbg-premium", "indicators", "microstructure")
        return INDICATOR_REGISTRY["microstructure"]()

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
        # Spätere Werte sollten höher sein als frühere
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
            {
                "O": close - 0.3,
                "H": close + 0.5,
                "L": close - 0.2,
                "C": close,
            },
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


# =============================================================================
# ROLLING BETA TESTS
# =============================================================================


class TestRollingBeta:
    """Tests für Rolling Beta im Risk Plugin."""

    def _get_indicator(self):
        from fwbg.core.registry import INDICATOR_REGISTRY
        from fwbg.plugins import import_plugin_module

        if "risk" not in INDICATOR_REGISTRY:
            import_plugin_module("fwbg-premium", "indicators", "risk")
        return INDICATOR_REGISTRY["risk"]()

    def test_beta_columns_present(self, correlated_asset_df):
        """Beta-Spalten werden erzeugt wenn macro_spx vorhanden."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        assert "beta_spx_50" in result.columns
        assert "beta_spx_100" in result.columns

    def test_beta_approximately_correct(self, correlated_asset_df):
        """Beta ~2.0 für ein Asset das mit Beta=2.0 zum SPX konstruiert wurde."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        beta_100 = result["beta_spx_100"].dropna()
        last_values = beta_100.iloc[-50:]
        mean_beta = last_values.mean()
        # Beta sollte ungefähr 2.0 sein (±0.5 wegen Noise)
        assert 1.2 < mean_beta < 3.0, f"Expected beta ~2.0, got {mean_beta}"

    def test_no_beta_without_spx(self, trending_df):
        """Ohne macro_spx sollte kein Beta berechnet werden (keine Fehler)."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        # beta_spx Spalten sollten nicht im DataFrame sein
        beta_cols = [c for c in result.columns if c.startswith("beta_spx")]
        assert len(beta_cols) == 0, f"Unexpected beta columns without SPX: {beta_cols}"

    def test_beta_no_lookahead(self, correlated_asset_df):
        """Beta-Features sollten um 1 Bar verschoben sein."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        assert pd.isna(result["beta_spx_50"].iloc[0])
        assert pd.isna(result["beta_spx_100"].iloc[0])

    def test_beta_sign_positive_for_correlated(self, correlated_asset_df):
        """Beta sollte positiv sein für positiv korrelierte Assets."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        beta_50 = result["beta_spx_50"].dropna()
        last_values = beta_50.iloc[-50:]
        assert last_values.mean() > 0, f"Beta should be positive, got {last_values.mean()}"
