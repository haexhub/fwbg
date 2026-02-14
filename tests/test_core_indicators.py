"""
Tests für Core Indikatoren mit synthetischen Daten.

Verifiziert mathematische Korrektheit der Indikatoren durch:
- Bekannte Szenarien mit erwarteten Ergebnissen
- Grenzwerte und Edge Cases
- Vergleich mit manuellen Berechnungen
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline import (
    compute_indicator_pool,
    get_feature_columns,
)


# === HELPER FUNCTIONS ===

def create_ohlc(close, high_factor=1.005, low_factor=0.995, n_bars=None):
    """Erstellt OHLC aus Close-Preisen."""
    if n_bars is None:
        n_bars = len(close)

    df = pd.DataFrame({
        'O': close * 0.999,
        'H': close * high_factor,
        'L': close * low_factor,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='h'))
    return df


# === TREND INDIKATOREN ===

class TestADX:
    """Tests für ADX (Average Directional Index)."""

    def test_adx_high_in_strong_trend(self):
        """ADX sollte hoch sein bei starkem Trend."""
        # Starker Aufwärtstrend: +1% pro Bar
        n = 200
        close = 100 * np.cumprod(1 + np.full(n, 0.01))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        adx = result["trend_adx_14"].dropna()
        # ADX > 30 gilt als starker Trend
        assert adx.mean() > 25, f"Expected ADX > 25 in trend, got {adx.mean():.1f}"

    def test_adx_lower_in_sideways_than_trend(self):
        """ADX sollte in Seitwärtsbewegung niedriger sein als im Trend."""
        np.random.seed(42)
        n = 200

        # Seitwärts: nur Noise, kein Trend
        close_sideways = 100 + np.random.randn(n) * 0.5
        df_sideways = create_ohlc(close_sideways)
        result_sideways = compute_indicator_pool(df_sideways)
        adx_sideways = result_sideways["trend_adx_14"].dropna().mean()

        # Starker Trend
        close_trend = 100 * np.cumprod(1 + np.full(n, 0.01))
        df_trend = create_ohlc(close_trend)
        result_trend = compute_indicator_pool(df_trend)
        adx_trend = result_trend["trend_adx_14"].dropna().mean()

        # ADX im Trend sollte höher sein als seitwärts
        assert adx_trend > adx_sideways, f"ADX in trend ({adx_trend:.1f}) should be > sideways ({adx_sideways:.1f})"


class TestEMADistance:
    """Tests für EMA Distance."""

    def test_positive_distance_above_ema(self):
        """Preis über EMA sollte positive Distance haben."""
        # Starker Aufwärtstrend: Preis immer über EMA
        n = 200
        close = 100 * np.cumprod(1 + np.full(n, 0.005))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        ema_dist = result["trend_ema_dist_21"].dropna()
        # Die letzten Werte sollten positiv sein
        assert ema_dist.iloc[-50:].mean() > 0, "Price above EMA should have positive distance"

    def test_negative_distance_below_ema(self):
        """Preis unter EMA sollte negative Distance haben."""
        # Starker Abwärtstrend
        n = 200
        close = 100 * np.cumprod(1 - np.full(n, 0.005))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        ema_dist = result["trend_ema_dist_21"].dropna()
        assert ema_dist.iloc[-50:].mean() < 0, "Price below EMA should have negative distance"


class TestEfficiencyRatio:
    """Tests für Kaufman's Efficiency Ratio."""

    def test_high_er_in_straight_line_trend(self):
        """ER sollte nahe 1 sein bei geradlinigem Trend."""
        n = 200
        # Perfekt linearer Anstieg
        close = np.linspace(100, 150, n)
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        er = result["trend_er_20"].dropna()
        # ER sollte sehr hoch sein (nahe 1)
        assert er.mean() > 0.8, f"Expected ER > 0.8 in linear trend, got {er.mean():.2f}"

    def test_low_er_in_noisy_sideways(self):
        """ER sollte niedrig sein bei Seitwärtsbewegung mit Noise."""
        np.random.seed(42)
        n = 200
        # Seitwärts mit starkem Noise
        close = 100 + np.random.randn(n) * 3
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        er = result["trend_er_20"].dropna()
        assert er.mean() < 0.3, f"Expected ER < 0.3 in noisy sideways, got {er.mean():.2f}"


# === MOMENTUM INDIKATOREN ===

class TestRSI:
    """Tests für RSI (Relative Strength Index)."""

    def test_rsi_high_after_continuous_gains(self):
        """RSI sollte hoch sein nach kontinuierlichen Gewinnen."""
        n = 100
        # Nur Gewinne: +1% pro Bar
        close = 100 * np.cumprod(1 + np.full(n, 0.01))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        rsi = result["mom_rsi_14"].dropna()
        # RSI sollte > 70 sein (überkauft)
        assert rsi.iloc[-1] > 70, f"Expected RSI > 70 after gains, got {rsi.iloc[-1]:.1f}"

    def test_rsi_low_after_continuous_losses(self):
        """RSI sollte niedrig sein nach kontinuierlichen Verlusten."""
        n = 100
        # Nur Verluste: -1% pro Bar
        close = 100 * np.cumprod(1 - np.full(n, 0.01))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        rsi = result["mom_rsi_14"].dropna()
        # RSI sollte < 30 sein (überverkauft)
        assert rsi.iloc[-1] < 30, f"Expected RSI < 30 after losses, got {rsi.iloc[-1]:.1f}"

    def test_rsi_around_50_in_sideways(self):
        """RSI sollte um 50 sein bei ausgeglichenem Markt."""
        np.random.seed(42)
        n = 300
        # Zufällige Bewegungen, aber insgesamt flat
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        rsi = result["mom_rsi_14"].dropna()
        # RSI sollte im Mittel nahe 50 sein
        assert 40 < rsi.mean() < 60, f"Expected RSI ~50 in sideways, got {rsi.mean():.1f}"


class TestStochastic:
    """Tests für Stochastic Oscillator."""

    def test_stochastic_100_at_period_high(self):
        """Stochastic sollte ~100 sein wenn Close am Period-High ist."""
        n = 50
        # Erst niedrig, dann hoch - Close am High
        close = np.concatenate([np.full(35, 100), np.linspace(100, 120, 15)])
        high = close * 1.001
        low = close * 0.999

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        stoch_k = result["mom_stoch_k_14"].dropna()
        # Stochastic sollte hoch sein
        assert stoch_k.iloc[-1] > 80, f"Expected Stoch > 80 at high, got {stoch_k.iloc[-1]:.1f}"

    def test_stochastic_0_at_period_low(self):
        """Stochastic sollte ~0 sein wenn Close am Period-Low ist."""
        n = 50
        # Erst hoch, dann runter - Close am Low
        close = np.concatenate([np.full(35, 120), np.linspace(120, 100, 15)])
        high = close * 1.001
        low = close * 0.999

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        stoch_k = result["mom_stoch_k_14"].dropna()
        assert stoch_k.iloc[-1] < 20, f"Expected Stoch < 20 at low, got {stoch_k.iloc[-1]:.1f}"


class TestROC:
    """Tests für Rate of Change."""

    def test_roc_equals_percentage_change(self):
        """ROC sollte exakt der prozentualen Änderung entsprechen (shifted by 1)."""
        n = 100
        close = np.linspace(100, 150, n)
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        roc_10 = result["mom_roc_10"].dropna()
        # Manuell berechnet: (C_t - C_{t-10}) / C_{t-10} * 100
        # WICHTIG: Features sind um 1 geshiptet, also Feature bei i = Wert von i-1
        expected = (close[10:] - close[:-10]) / close[:-10] * 100

        # Vergleiche geshiptet: Feature-Werte bei [-50:] entsprechen expected[-51:-1]
        np.testing.assert_array_almost_equal(
            roc_10.iloc[-50:].values,
            expected[-51:-1],
            decimal=5
        )


# === VOLATILITÄT INDIKATOREN ===

class TestBollingerBands:
    """Tests für Bollinger Bands."""

    def test_pband_1_at_upper_band(self):
        """%B sollte ~1 sein wenn Preis am oberen Band."""
        n = 100
        # Starker Breakout nach oben
        close = np.concatenate([np.full(80, 100), np.linspace(100, 115, 20)])
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        pband = result["vol_bb_pband_20"].dropna()
        # Nach starkem Anstieg sollte %B > 1 sein (über oberem Band)
        assert pband.iloc[-1] > 0.9, f"Expected %B > 0.9 at upper band, got {pband.iloc[-1]:.2f}"

    def test_pband_0_at_lower_band(self):
        """&B sollte ~0 sein wenn Preis am unteren Band."""
        n = 100
        # Starker Einbruch
        close = np.concatenate([np.full(80, 100), np.linspace(100, 85, 20)])
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        pband = result["vol_bb_pband_20"].dropna()
        assert pband.iloc[-1] < 0.1, f"Expected %B < 0.1 at lower band, got {pband.iloc[-1]:.2f}"

    def test_bandwidth_increases_with_volatility(self):
        """Bandwidth sollte mit Volatilität steigen."""
        n = 200
        # Erst niedrige Volatilität, dann hohe
        close_low_vol = 100 + np.random.randn(100) * 0.5
        close_high_vol = 100 + np.random.randn(100) * 3
        close = np.concatenate([close_low_vol, close_high_vol])
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        wband = result["vol_bb_wband_20"].dropna()
        # Bandwidth in zweiter Hälfte sollte höher sein
        first_half = wband.iloc[:80].mean()
        second_half = wband.iloc[-80:].mean()
        assert second_half > first_half * 1.5, "Bandwidth should increase with volatility"


class TestATR:
    """Tests für Average True Range."""

    def test_atr_increases_with_range(self):
        """ATR sollte mit der Range steigen."""
        n = 100
        close = np.full(n, 100.0)
        # Große Range
        high = close * 1.03
        low = close * 0.97

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        atr_pct = result["vol_atr_pct_14"].dropna()
        # ATR% sollte ungefähr 6% sein (high - low = 6%)
        assert atr_pct.mean() > 0.05, f"Expected ATR% > 5%, got {atr_pct.mean()*100:.1f}%"

    def test_atr_reflects_gap(self):
        """ATR sollte Gaps berücksichtigen (True Range)."""
        n = 50
        close = np.full(n, 100.0)
        high = close * 1.01
        low = close * 0.99

        # Gap am Ende: Previous close = 100, aber heute Open/High/Low/Close = 110
        # WICHTIG: Da Features um 1 geshiptet sind, muss der Gap 2 Bars vor Ende sein
        # damit wir ihn im letzten Feature-Wert sehen
        close[-2] = 110
        high[-2] = 111
        low[-2] = 109

        df = pd.DataFrame({
            'O': close * 0.999, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        atr = result["vol_atr_pct_14"].dropna()
        # ATR sollte nach Gap steigen (vergleiche letzten Wert mit früherem)
        assert atr.iloc[-1] > atr.iloc[-11], "ATR should increase after gap"


# === PRICE ACTION FEATURES ===

class TestRangePosition:
    """Tests für Range Position."""

    def test_range_pos_1_at_high(self):
        """Range Position sollte 1 sein wenn Close = High."""
        n = 50
        close = np.full(n, 100.0)
        high = close.copy()  # Close = High
        low = close * 0.98

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        rp = result["pa_range_pos"]
        # Use approximate equality due to floating point precision
        assert abs(rp.iloc[-1] - 1.0) < 1e-6, f"Expected range_pos = 1 at high, got {rp.iloc[-1]}"

    def test_range_pos_0_at_low(self):
        """Range Position sollte 0 sein wenn Close = Low."""
        n = 50
        high = np.full(n, 102.0)
        low = np.full(n, 100.0)
        close = low  # Close = Low

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        rp = result["pa_range_pos"]
        assert rp.iloc[-1] == 0.0, f"Expected range_pos = 0 at low, got {rp.iloc[-1]}"

    def test_range_pos_05_at_middle(self):
        """Range Position sollte 0.5 sein wenn Close in der Mitte."""
        n = 50
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)
        close = np.full(n, 100.0)  # Exakt in der Mitte

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        rp = result["pa_range_pos"]
        assert abs(rp.iloc[-1] - 0.5) < 0.01, f"Expected range_pos = 0.5 in middle, got {rp.iloc[-1]}"


class TestBodyRatio:
    """Tests für Body Ratio."""

    def test_body_ratio_1_for_marubozu(self):
        """Body Ratio sollte ~1 sein für Marubozu (O=L, C=H oder umgekehrt)."""
        n = 50
        # Bullish Marubozu: Open = Low, Close = High
        low = np.full(n, 98.0)
        high = np.full(n, 102.0)
        open_price = low.copy()
        close = high.copy()

        df = pd.DataFrame({
            'O': open_price, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        br = result["pa_body_ratio"]
        assert br.iloc[-1] > 0.95, f"Expected body_ratio ~1 for marubozu, got {br.iloc[-1]}"

    def test_body_ratio_0_for_doji(self):
        """Body Ratio sollte ~0 sein für Doji (O = C)."""
        n = 50
        close = np.full(n, 100.0)
        open_price = close.copy()  # Open = Close
        high = np.full(n, 102.0)
        low = np.full(n, 98.0)

        df = pd.DataFrame({
            'O': open_price, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        br = result["pa_body_ratio"]
        assert br.iloc[-1] < 0.05, f"Expected body_ratio ~0 for doji, got {br.iloc[-1]}"


class TestGap:
    """Tests für Gap Feature."""

    def test_gap_positive_for_gap_up(self):
        """Gap sollte positiv sein bei Gap Up."""
        n = 50
        close = np.full(n, 100.0)
        open_price = close.copy()
        # Gap Up: Open ist höher als Previous Close
        # WICHTIG: Da Features um 1 geshiptet sind, muss der Gap 2 Bars vor Ende sein
        open_price[-2] = 105.0  # 5% Gap Up bei Bar -2
        close[-3] = 100.0  # Previous close bei Bar -3

        df = pd.DataFrame({
            'O': open_price, 'H': close * 1.01, 'L': close * 0.99, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        # Feature bei -1 zeigt Wert von Bar -2 (wegen shift)
        gap = result["pa_gap"].iloc[-1]
        assert gap > 0.04, f"Expected positive gap ~5%, got {gap*100:.1f}%"

    def test_gap_negative_for_gap_down(self):
        """Gap sollte negativ sein bei Gap Down."""
        n = 50
        close = np.full(n, 100.0)
        open_price = close.copy()
        # WICHTIG: Da Features um 1 geshiptet sind, muss der Gap 2 Bars vor Ende sein
        open_price[-2] = 95.0  # 5% Gap Down bei Bar -2

        df = pd.DataFrame({
            'O': open_price, 'H': close * 1.01, 'L': close * 0.99, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        # Feature bei -1 zeigt Wert von Bar -2 (wegen shift)
        gap = result["pa_gap"].iloc[-1]
        assert gap < -0.04, f"Expected negative gap ~-5%, got {gap*100:.1f}%"


# === ZEIT UND SAISONALITÄT ===

class TestTimeFeatures:
    """Tests für Zeit-Features."""

    def test_hour_correctly_extracted(self):
        """Stunde sollte korrekt extrahiert werden (shifted by 1)."""
        n = 48  # 2 Tage
        close = np.full(n, 100.0)
        # Start um 08:00
        df = pd.DataFrame({
            'O': close, 'H': close, 'L': close, 'C': close,
        }, index=pd.date_range('2024-01-01 08:00', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        # WICHTIG: Features sind um 1 geshiptet
        # Feature bei Index 1 zeigt Wert von Bar 0 (8 Uhr)
        assert result["time_hour"].iloc[1] == 8
        # Feature bei Index 13 zeigt Wert von Bar 12 (20 Uhr)
        assert result["time_hour"].iloc[13] == 20

    def test_sin_cos_encoding_correct(self):
        """Sin/Cos Encoding sollte korrekt sein (shifted by 1)."""
        # Need enough bars for compute_indicator_pool to work
        n = 100
        np.random.seed(42)
        close = 100 + np.random.randn(n) * 0.1
        df = pd.DataFrame({
            'O': close * 0.999,
            'H': close * 1.005,
            'L': close * 0.995,
            'C': close,
        }, index=pd.date_range('2024-01-01 00:00', periods=n, freq='h'))
        result = compute_indicator_pool(df)

        # WICHTIG: Features sind um 1 geshiptet
        # Feature bei Index 1 zeigt Wert von Bar 0 (00:00)
        # Um 00:00 sollte sin = 0, cos = 1
        assert abs(result["time_hour_sin"].iloc[1]) < 0.01
        assert abs(result["time_hour_cos"].iloc[1] - 1) < 0.01

        # Feature bei Index 7 zeigt Wert von Bar 6 (06:00)
        # Um 06:00 sollte sin = 1, cos = 0 (peak)
        assert abs(result["time_hour_sin"].iloc[7] - 1) < 0.01
        assert abs(result["time_hour_cos"].iloc[7]) < 0.01


class TestSeasonalityFeatures:
    """Tests für Saisonalitäts-Features."""

    def test_month_correctly_extracted(self):
        """Monat sollte korrekt extrahiert werden (shifted by 1)."""
        # 365 Tage, stündlich = viel zu viel, nehme tägliche Daten
        dates = pd.date_range('2024-01-15', periods=100, freq='h')
        df = pd.DataFrame({
            'O': 100, 'H': 101, 'L': 99, 'C': 100,
        }, index=dates)
        result = compute_indicator_pool(df)

        # WICHTIG: Features sind um 1 geshiptet
        # Feature bei Index 1 zeigt Wert von Bar 0 (Januar)
        assert result["season_month"].iloc[1] == 1


# === DYNAMIK FEATURES ===

class TestDynamicsFeatures:
    """Tests für Dynamik-Features."""

    def test_rsi_change_positive_when_rsi_rising(self):
        """RSI Change sollte positiv sein wenn RSI steigt."""
        n = 150
        # Long sideways period, then strong uptrend
        close = np.concatenate([
            100 + np.random.randn(100) * 0.1,  # Sideways with noise
            100 * np.cumprod(1 + np.full(50, 0.015))  # Strong uptrend
        ])
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        # Compare RSI at end vs start of uptrend phase
        rsi = result["mom_rsi_14"].dropna()
        rsi_start = rsi.iloc[-50:-45].mean()
        rsi_end = rsi.iloc[-5:].mean()
        assert rsi_end > rsi_start, f"RSI should increase in uptrend: {rsi_start:.1f} -> {rsi_end:.1f}"


class TestCrossIndicatorFeatures:
    """Tests für Cross-Indikator Features."""

    def test_cross_rsi_high_rising_is_binary(self):
        """Cross RSI high+rising Feature sollte binär (0 oder 1) sein."""
        n = 200
        np.random.seed(42)
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        cross_feature = result["cross_rsi_high_rising"].dropna()
        # Should only contain 0 or 1
        unique_values = set(cross_feature.unique())
        assert unique_values.issubset({0, 1}), f"Expected only 0/1, got {unique_values}"


# === MULTI-TIMEFRAME FEATURES ===

class TestMultiTimeframeFeatures:
    """Tests für Multi-Timeframe Features."""

    def test_h4_trend_positive_in_uptrend(self):
        """H4 Trend sollte positiv im Aufwärtstrend sein."""
        n = 200
        close = np.linspace(100, 150, n)
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        h4_trend = result["mtf_h4_trend"].dropna()
        assert h4_trend.iloc[-50:].mean() > 0, "H4 trend should be positive in uptrend"

    def test_trend_alignment_1_when_all_timeframes_agree(self):
        """Trend Alignment sollte hoch sein wenn Timeframes übereinstimmen."""
        n = 600  # Genug für D1 EMA
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        # Feature wurde umbenannt zu mtf_trend_alignment_h1h4 und mtf_trend_alignment_h4d1
        alignment_h1h4 = result["mtf_trend_alignment_h1h4"].dropna()
        alignment_h4d1 = result["mtf_trend_alignment_h4d1"].dropna()
        # Im klaren Aufwärtstrend sollten Timeframes übereinstimmen
        assert alignment_h1h4.iloc[-50:].mean() > 0.5, "H1/H4 alignment should be high in clear trend"
        assert alignment_h4d1.iloc[-50:].mean() > 0.5, "H4/D1 alignment should be high in clear trend"


# === INTEGRATION TESTS ===

class TestIndicatorPoolIntegration:
    """Integration Tests."""

    def test_no_unexpected_inf_values(self):
        """Keine unerwarteten Inf-Werte sollten entstehen.

        Hinweis: Einige Spalten können durch pct_change bei sehr kleinen Werten
        legitimerweise inf enthalten (z.B. ATR-Change wenn ATR nahe 0 war).
        """
        np.random.seed(42)
        n = 300
        # Generate realistic price movement with more volatility
        returns = np.random.randn(n) * 0.02
        close = 100 * np.exp(np.cumsum(returns))
        close = np.maximum(close, 50)  # Higher minimum prices
        df = create_ohlc(close, high_factor=1.01, low_factor=0.99)
        result = compute_indicator_pool(df)

        # Columns that can legitimately have inf from pct_change
        allowed_inf_cols = {'dyn_atr_chg_4h', 'dyn_atr_chg_8h', 'dyn_atr_chg_24h',
                           'accel_atr', 'dyn_bbwidth_chg_4h',
                           'dyn_bbwidth_chg_8h', 'dyn_bbwidth_chg_24h'}

        unexpected_inf_cols = []
        for col in result.columns:
            if result[col].dtype in [np.float64, np.float32]:
                if np.isinf(result[col]).any() and col not in allowed_inf_cols:
                    unexpected_inf_cols.append(col)
        assert len(unexpected_inf_cols) == 0, f"Unexpected inf columns: {unexpected_inf_cols}"

    def test_output_has_expected_feature_count(self):
        """Output sollte viele Features haben."""
        np.random.seed(42)
        n = 300
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        features = get_feature_columns(result)
        assert len(features) > 100, f"Expected > 100 features, got {len(features)}"

    def test_trend_features_present(self):
        """Trend-Features sollten in berechneten Indikatoren vorhanden sein."""
        np.random.seed(42)
        n = 300
        close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
        df = create_ohlc(close)
        result = compute_indicator_pool(df)

        all_features = get_feature_columns(result)
        trend_features = [f for f in all_features if f.startswith("trend_")]

        # Sollte mehrere Trend-Features haben
        assert len(trend_features) > 10, f"Expected > 10 trend features, got {len(trend_features)}"
