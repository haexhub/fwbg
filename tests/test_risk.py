"""
Tests für Risk/Tail-Risk Features.

Testet:
- Drawdown State Features
- CVaR (Conditional Value at Risk)
- Vol-of-Vol (Volatility of Volatility)
- Crash Probability Proxy
- Correlation Features
"""
import numpy as np
import pandas as pd
import pytest

from optimizer.indicators.risk import (
    compute_drawdown_features,
    compute_cvar_features,
    compute_vol_of_vol_features,
    compute_crash_probability_features,
    compute_correlation_features,
)


# === FIXTURES ===

@pytest.fixture
def sample_ohlc():
    """Erstellt Sample OHLC-Daten."""
    np.random.seed(42)
    n = 300
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_price = close * (1 + np.random.randn(n) * 0.002)

    df = pd.DataFrame({
        'O': open_price,
        'H': high,
        'L': low,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    return df


@pytest.fixture
def drawdown_ohlc():
    """Erstellt OHLC mit deutlichem Drawdown."""
    n = 200
    # Erst hoch, dann runter
    close = np.concatenate([
        np.linspace(100, 120, 100),  # Anstieg auf 120
        np.linspace(120, 100, 100),  # Zurück auf 100
    ])

    df = pd.DataFrame({
        'O': close,
        'H': close * 1.01,
        'L': close * 0.99,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    return df


@pytest.fixture
def volatile_ohlc():
    """Erstellt volatile OHLC-Daten."""
    np.random.seed(42)
    n = 300
    # Hohe Volatilität
    returns = np.random.randn(n) * 0.03  # 3% statt 1%
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.015))
    low = close * (1 - np.abs(np.random.randn(n) * 0.015))

    df = pd.DataFrame({
        'O': close,
        'H': high,
        'L': low,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    return df


@pytest.fixture
def ohlc_with_macro():
    """Erstellt OHLC mit Makro-Daten (SPX, VIX)."""
    np.random.seed(42)
    n = 300
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))

    # SPX korreliert mit Asset
    spx_returns = returns * 0.8 + np.random.randn(n) * 0.005
    spx = 4000 * np.exp(np.cumsum(spx_returns))

    # VIX negativ korreliert - use pandas Series for pct_change
    close_series = pd.Series(close)
    vix = 20 - close_series.pct_change().rolling(20).std() * 1000 + np.random.randn(n) * 2
    vix = np.clip(vix.values, 10, 40)

    df = pd.DataFrame({
        'O': close,
        'H': close * 1.01,
        'L': close * 0.99,
        'C': close,
        'macro_spx': spx,
        'macro_vix': vix,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    return df


# === TESTS FÜR COMPUTE_DRAWDOWN_FEATURES ===

class TestComputeDrawdownFeatures:
    """Tests für compute_drawdown_features()."""

    def test_adds_dd_pct_columns(self, sample_ohlc):
        """Sollte DD Prozent Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_drawdown_features(df)

        assert "risk_dd_pct_50" in result.columns
        assert "risk_dd_pct_100" in result.columns
        assert "risk_dd_pct_200" in result.columns

    def test_adds_dd_ratio_columns(self, sample_ohlc):
        """Sollte DD Ratio Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_drawdown_features(df)

        assert "risk_dd_ratio_50" in result.columns
        assert "risk_dd_ratio_100" in result.columns
        assert "risk_dd_ratio_200" in result.columns

    def test_adds_bars_since_peak_columns(self, sample_ohlc):
        """Sollte Bars Since Peak Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_drawdown_features(df)

        assert "risk_bars_since_peak" in result.columns
        assert "risk_bars_since_peak_log" in result.columns

    def test_adds_recovery_ratio_column(self, sample_ohlc):
        """Sollte Recovery Ratio Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_drawdown_features(df)

        assert "risk_recovery_ratio" in result.columns

    def test_dd_pct_is_negative_or_zero(self, sample_ohlc):
        """Drawdown Prozent sollte <= 0 sein."""
        df = sample_ohlc.copy()
        result = compute_drawdown_features(df)

        dd_pct = result["risk_dd_pct_100"].dropna()
        assert all(v <= 0 for v in dd_pct)

    def test_drawdown_detected_correctly(self, drawdown_ohlc):
        """Drawdown sollte korrekt erkannt werden."""
        df = drawdown_ohlc.copy()
        result = compute_drawdown_features(df)

        # Am Ende (bei 100) sollte DD von 120 = -16.7%
        final_dd = result["risk_dd_pct_100"].iloc[-1]
        assert final_dd < -0.1, f"Expected DD < -10%, got {final_dd}"

    def test_recovery_ratio_between_0_and_1(self, sample_ohlc):
        """Recovery Ratio sollte zwischen 0 und 1 sein."""
        df = sample_ohlc.copy()
        result = compute_drawdown_features(df)

        recovery = result["risk_recovery_ratio"].dropna()
        assert all(0 <= v <= 1 for v in recovery)

    def test_bars_since_peak_increases_in_drawdown(self, drawdown_ohlc):
        """Bars since peak sollte im Drawdown steigen."""
        df = drawdown_ohlc.copy()
        result = compute_drawdown_features(df)

        bars = result["risk_bars_since_peak"].dropna()
        # Im zweiten Teil (Drawdown) sollte der Counter steigen
        second_half = bars.iloc[len(bars)//2:]
        assert second_half.iloc[-1] > second_half.iloc[0]


# === TESTS FÜR COMPUTE_CVAR_FEATURES ===

class TestComputeCvarFeatures:
    """Tests für compute_cvar_features()."""

    def test_adds_var_columns(self, sample_ohlc):
        """Sollte VaR Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_cvar_features(df)

        assert "risk_var_5_50" in result.columns
        assert "risk_var_5_100" in result.columns
        assert "risk_var_1_50" in result.columns
        assert "risk_var_1_100" in result.columns

    def test_adds_cvar_columns(self, sample_ohlc):
        """Sollte CVaR Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_cvar_features(df)

        assert "risk_cvar_5_50" in result.columns
        assert "risk_cvar_5_100" in result.columns
        assert "risk_cvar_1_50" in result.columns
        assert "risk_cvar_1_100" in result.columns

    def test_adds_tail_ratio_column(self, sample_ohlc):
        """Sollte Tail Ratio Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_cvar_features(df)

        assert "risk_cvar_tail_ratio" in result.columns

    def test_adds_cvar_change_column(self, sample_ohlc):
        """Sollte CVaR Change Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_cvar_features(df)

        assert "risk_cvar_5_change" in result.columns

    def test_var_is_negative(self, sample_ohlc):
        """VaR (Verlust-Quantil) sollte negativ sein."""
        df = sample_ohlc.copy()
        result = compute_cvar_features(df)

        var_5 = result["risk_var_5_100"].dropna()
        assert var_5.mean() < 0, "VaR should be negative on average"

    def test_cvar_worse_than_var(self, sample_ohlc):
        """CVaR sollte <= VaR sein (schlimmere Verluste im Tail)."""
        df = sample_ohlc.copy()
        result = compute_cvar_features(df)

        var_5 = result["risk_var_5_100"].dropna()
        cvar_5 = result["risk_cvar_5_100"].dropna()

        # Align indices
        common_idx = var_5.index.intersection(cvar_5.index)
        assert (cvar_5.loc[common_idx] <= var_5.loc[common_idx]).mean() > 0.9

    def test_volatile_has_worse_cvar(self, sample_ohlc, volatile_ohlc):
        """Volatile Serie sollte schlechteren CVaR haben."""
        normal = compute_cvar_features(sample_ohlc.copy())
        volatile = compute_cvar_features(volatile_ohlc.copy())

        normal_cvar = normal["risk_cvar_5_100"].dropna().mean()
        volatile_cvar = volatile["risk_cvar_5_100"].dropna().mean()

        assert volatile_cvar < normal_cvar, "Volatile should have worse CVaR"


# === TESTS FÜR COMPUTE_VOL_OF_VOL_FEATURES ===

class TestComputeVolOfVolFeatures:
    """Tests für compute_vol_of_vol_features()."""

    def test_adds_vol_of_vol_columns(self, sample_ohlc):
        """Sollte Vol-of-Vol Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_vol_of_vol_features(df)

        assert "risk_vol_of_vol_20" in result.columns
        assert "risk_vol_of_vol_50" in result.columns
        assert "risk_vol_of_vol_100" in result.columns

    def test_adds_zscore_column(self, sample_ohlc):
        """Sollte Z-Score Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_vol_of_vol_features(df)

        assert "risk_vol_of_vol_zscore" in result.columns

    def test_adds_trend_column(self, sample_ohlc):
        """Sollte Trend Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_vol_of_vol_features(df)

        assert "risk_vol_of_vol_trend" in result.columns

    def test_vol_of_vol_is_non_negative(self, sample_ohlc):
        """Vol-of-Vol sollte nicht-negativ sein."""
        df = sample_ohlc.copy()
        result = compute_vol_of_vol_features(df)

        vov = result["risk_vol_of_vol_50"].dropna()
        assert all(v >= 0 for v in vov)


# === TESTS FÜR COMPUTE_CRASH_PROBABILITY_FEATURES ===

class TestComputeCrashProbabilityFeatures:
    """Tests für compute_crash_probability_features()."""

    def test_adds_crash_probability_column(self, sample_ohlc):
        """Sollte Crash Probability Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_crash_probability_features(df)

        assert "risk_crash_probability" in result.columns

    def test_adds_crash_prob_change_column(self, sample_ohlc):
        """Sollte Crash Prob Change Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_crash_probability_features(df)

        assert "risk_crash_prob_change" in result.columns

    def test_adds_crash_regime_column(self, sample_ohlc):
        """Sollte Crash Regime Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_crash_probability_features(df)

        assert "risk_crash_regime" in result.columns

    def test_crash_probability_between_0_and_1(self, sample_ohlc):
        """Crash Probability sollte zwischen 0 und 1 sein."""
        df = sample_ohlc.copy()
        # Füge benötigte Features hinzu
        df = compute_cvar_features(df)
        df = compute_vol_of_vol_features(df)

        result = compute_crash_probability_features(df)
        crash_prob = result["risk_crash_probability"].dropna()

        assert all(0 <= v <= 1 for v in crash_prob)

    def test_crash_regime_is_binary(self, sample_ohlc):
        """Crash Regime sollte 0 oder 1 sein."""
        df = sample_ohlc.copy()
        result = compute_crash_probability_features(df)

        regime = result["risk_crash_regime"].dropna()
        assert all(v in [0, 1] for v in regime)

    def test_handles_missing_components(self, sample_ohlc):
        """Sollte fehlende Komponenten behandeln."""
        df = sample_ohlc.copy()
        # Keine Vorberechnung der Features
        result = compute_crash_probability_features(df)

        # Sollte ohne Fehler laufen
        assert "risk_crash_probability" in result.columns


# === TESTS FÜR COMPUTE_CORRELATION_FEATURES ===

class TestComputeCorrelationFeatures:
    """Tests für compute_correlation_features()."""

    def test_adds_correlation_columns_with_spx(self, ohlc_with_macro):
        """Sollte Korrelations-Spalten hinzufügen wenn SPX verfügbar."""
        df = ohlc_with_macro.copy()
        result = compute_correlation_features(df)

        assert "corr_spx_20" in result.columns
        assert "corr_spx_50" in result.columns
        assert "corr_spx_100" in result.columns

    def test_adds_stability_columns(self, ohlc_with_macro):
        """Sollte Stability Spalten hinzufügen."""
        df = ohlc_with_macro.copy()
        result = compute_correlation_features(df)

        assert "corr_spx_stability" in result.columns
        assert "corr_spx_decoupling" in result.columns

    def test_adds_lead_lag_columns(self, ohlc_with_macro):
        """Sollte Lead-Lag Spalten hinzufügen."""
        df = ohlc_with_macro.copy()
        result = compute_correlation_features(df)

        assert "lead_lag_spx" in result.columns

    def test_adds_vix_columns(self, ohlc_with_macro):
        """Sollte VIX Spalten hinzufügen wenn VIX verfügbar."""
        df = ohlc_with_macro.copy()
        result = compute_correlation_features(df)

        assert "corr_vix_20" in result.columns
        assert "corr_vix_50" in result.columns
        assert "lead_lag_vix" in result.columns
        assert "vix_lead_signal" in result.columns

    def test_correlation_between_minus1_and_1(self, ohlc_with_macro):
        """Korrelation sollte zwischen -1 und 1 sein."""
        df = ohlc_with_macro.copy()
        result = compute_correlation_features(df)

        corr = result["corr_spx_50"].dropna()
        assert all(-1 <= v <= 1 for v in corr)

    def test_decoupling_is_non_negative(self, ohlc_with_macro):
        """Decoupling sollte nicht-negativ sein (absolute Änderung)."""
        df = ohlc_with_macro.copy()
        result = compute_correlation_features(df)

        decoupling = result["corr_spx_decoupling"].dropna()
        assert all(v >= 0 for v in decoupling)

    def test_handles_missing_macro_data(self, sample_ohlc):
        """Sollte fehlende Makro-Daten behandeln."""
        df = sample_ohlc.copy()
        # Keine macro_spx oder macro_vix
        result = compute_correlation_features(df)

        # Sollte ohne Fehler laufen, aber keine Korrelationsspalten haben
        assert "corr_spx_50" not in result.columns
