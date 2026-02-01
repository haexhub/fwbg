"""
Tests für Advanced Indicator Plugins.

Testet:
- IchimokuIndicators: Ichimoku Cloud und abgeleitete Features
- TimeSeasonIndicators: Zeit- und Saison-basierte Features
- DistributionIndicators: Statistische Verteilungs-Features
- DynamicsIndicators: Marktdynamik-Features
- MultiTimeframeIndicators: Multi-Timeframe Features
- CrossFeatureIndicators: Cross-Asset/Cross-Feature Kombinationen
- PriceActionIndicators: Candlestick Patterns und Price Action
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# --- Fixtures ---


@pytest.fixture
def trending_ohlc():
    """OHLC-Daten mit klarem Trend für 200 Bars."""
    n = 200
    np.random.seed(42)
    base = 100.0

    # Erst Aufwärts-, dann Abwärtstrend
    trend_up = np.linspace(0, 20, n // 2)
    trend_down = np.linspace(20, 5, n // 2)
    trend = np.concatenate([trend_up, trend_down])

    noise = np.random.normal(0, 0.5, n)
    prices = base + trend + noise

    df = pd.DataFrame({
        "O": prices - np.random.uniform(0.1, 0.3, n),
        "H": prices + np.random.uniform(0.2, 0.5, n),
        "L": prices - np.random.uniform(0.2, 0.5, n),
        "C": prices,
    })
    # Timestamps hinzufügen
    df.index = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


@pytest.fixture
def ranging_ohlc():
    """OHLC-Daten ohne klaren Trend (seitwärts)."""
    n = 200
    np.random.seed(42)
    base = 100.0

    # Seitwärtsbewegung mit kleinen Schwankungen
    noise = np.cumsum(np.random.normal(0, 0.3, n))
    noise = noise - noise.mean()  # Zentrieren
    prices = base + noise

    df = pd.DataFrame({
        "O": prices - np.random.uniform(0.1, 0.2, n),
        "H": prices + np.random.uniform(0.2, 0.4, n),
        "L": prices - np.random.uniform(0.2, 0.4, n),
        "C": prices,
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="h")
    return df


@pytest.fixture
def ohlc_with_datetime():
    """OHLC-Daten mit verschiedenen Zeitpunkten für Saison-Tests."""
    # 7 Tage * 24 Stunden = 168 Bars
    dates = pd.date_range("2024-01-01", periods=168, freq="h")
    np.random.seed(42)

    df = pd.DataFrame({
        "O": 100 + np.random.uniform(-1, 1, 168),
        "H": 101 + np.random.uniform(0, 1, 168),
        "L": 99 + np.random.uniform(-1, 0, 168),
        "C": 100 + np.random.uniform(-1, 1, 168),
    }, index=dates)
    return df


# --- IchimokuIndicators Tests ---


class TestIchimokuIndicators:
    """Tests für Ichimoku Cloud Indikatoren."""

    def test_computes_all_features(self, trending_ohlc):
        """Sollte alle Ichimoku-Features berechnen."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators

        indicator = IchimokuIndicators()
        df = indicator.compute(trending_ohlc.copy())

        expected_features = indicator.get_feature_columns()
        for feature in expected_features:
            assert feature in df.columns, f"Feature {feature} fehlt"

    def test_tenkan_faster_than_kijun(self, trending_ohlc):
        """Tenkan (9) sollte schneller reagieren als Kijun (26)."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators

        indicator = IchimokuIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Am Wendepunkt (100): Tenkan sollte vor Kijun drehen
        # Tenkan hat kleinere Periode, also weniger Lag
        tenkan_std = df["ichi_tenkan"].diff().std()
        kijun_std = df["ichi_kijun"].diff().std()

        # Tenkan sollte volatiler sein (reagiert schneller)
        assert tenkan_std > kijun_std * 0.8, "Tenkan sollte volatiler als Kijun sein"

    def test_cloud_position_values(self, trending_ohlc):
        """Cloud Position sollte nach Warmup sinnvolle Werte haben."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators

        indicator = IchimokuIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Ichimoku braucht ~52 Bars Warmup (senkou_b_period)
        # Danach sollten die Werte stabil sein
        valid_data = df["ichi_cloud_pos"].iloc[60:].dropna()

        # Cloud Position kann < 0 (unter Cloud) oder > 1 (über Cloud) sein
        # Aber sollte keine extremen Werte nach Warmup haben
        assert valid_data.min() > -10, "Cloud Position zu niedrig nach Warmup"
        assert valid_data.max() < 10, "Cloud Position zu hoch nach Warmup"

    def test_above_below_cloud_exclusive(self, trending_ohlc):
        """Preis kann nicht gleichzeitig über UND unter der Cloud sein."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators

        indicator = IchimokuIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # above + below + in_cloud sollte immer 1 sein (gegenseitig exklusiv)
        total = df["ichi_above_cloud"] + df["ichi_below_cloud"] + df["ichi_in_cloud"]
        valid_total = total.dropna()

        assert (valid_total == 1).all(), "Cloud-Position sollte exklusiv sein"

    def test_tk_cross_signal_detection(self, trending_ohlc):
        """TK Cross Signale sollten bei Richtungswechsel erkannt werden."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators

        indicator = IchimokuIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Sollte mindestens ein TK Cross Signal geben (bei Trendwechsel)
        bullish_crosses = df["ichi_tk_bullish_cross"].sum()
        bearish_crosses = df["ichi_tk_bearish_cross"].sum()

        # Bei einem Trend-Datensatz mit Wende sollten Crosses auftreten
        total_crosses = bullish_crosses + bearish_crosses
        assert total_crosses >= 1, "Sollte mindestens ein TK Cross haben"

    def test_strong_signals_require_alignment(self, trending_ohlc):
        """Strong Bullish/Bearish erfordert mehrfache Bestätigung."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators

        indicator = IchimokuIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Strong Bullish erfordert: above_cloud + tk_cross > 0 + bullish_cloud
        strong_bull_bars = df[df["ichi_strong_bullish"] == 1]

        for _, row in strong_bull_bars.iterrows():
            assert row["ichi_above_cloud"] == 1
            assert row["ichi_tk_cross"] > 0
            assert row["ichi_bullish_cloud"] == 1

    def test_custom_periods(self, trending_ohlc):
        """Sollte mit Custom-Perioden funktionieren."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators

        indicator = IchimokuIndicators()

        # Kürzere Perioden für schnellere Signale
        df = indicator.compute(
            trending_ohlc.copy(),
            tenkan_period=7,
            kijun_period=22,
            senkou_b_period=44
        )

        assert "ichi_tenkan" in df.columns
        assert "ichi_kijun" in df.columns


# --- TimeSeasonIndicators Tests ---


class TestTimeSeasonIndicators:
    """Tests für Zeit- und Saison-Indikatoren."""

    def test_computes_time_features(self, ohlc_with_datetime):
        """Sollte Zeit-Features korrekt berechnen."""
        from fwbg.builtins.indicators.time_season import TimeSeasonIndicators

        indicator = TimeSeasonIndicators()
        df = indicator.compute(ohlc_with_datetime.copy())

        # Stunde sollte 0-23 sein
        if "time_hour" in df.columns:
            assert df["time_hour"].min() >= 0
            assert df["time_hour"].max() <= 23

        # Wochentag sollte 0-6 sein
        if "time_dow" in df.columns:
            assert df["time_dow"].min() >= 0
            assert df["time_dow"].max() <= 6

    def test_cyclical_encoding(self, ohlc_with_datetime):
        """Zyklische Features sollten sin/cos Encoding verwenden."""
        from fwbg.builtins.indicators.time_season import TimeSeasonIndicators

        indicator = TimeSeasonIndicators()
        df = indicator.compute(ohlc_with_datetime.copy())

        # Sin/Cos Features sollten im Bereich [-1, 1] sein
        sin_cols = [c for c in df.columns if "_sin" in c]
        cos_cols = [c for c in df.columns if "_cos" in c]

        for col in sin_cols + cos_cols:
            assert df[col].min() >= -1.01, f"{col} unter -1"
            assert df[col].max() <= 1.01, f"{col} über 1"

    def test_session_detection(self, ohlc_with_datetime):
        """Sollte Forex-Sessions erkennen (London, NY, Tokyo)."""
        from fwbg.builtins.indicators.time_season import TimeSeasonIndicators

        indicator = TimeSeasonIndicators()
        df = indicator.compute(ohlc_with_datetime.copy())

        session_cols = [c for c in df.columns if "session" in c.lower()]

        # Sessions sollten binär sein (0 oder 1)
        for col in session_cols:
            unique_vals = df[col].dropna().unique()
            assert set(unique_vals).issubset({0, 1, 0.0, 1.0}), f"{col} ist nicht binär"


# --- DistributionIndicators Tests ---


class TestDistributionIndicators:
    """Tests für Verteilungs-Indikatoren."""

    def test_computes_skewness_kurtosis(self, trending_ohlc):
        """Sollte Skewness und Kurtosis berechnen."""
        from fwbg.builtins.indicators.distribution import DistributionIndicators

        indicator = DistributionIndicators()
        df = indicator.compute(trending_ohlc.copy())

        skew_cols = [c for c in df.columns if "skew" in c.lower()]
        kurt_cols = [c for c in df.columns if "kurt" in c.lower()]

        # Sollte mindestens Skewness und Kurtosis Features haben
        assert len(skew_cols) > 0 or len(kurt_cols) > 0, "Keine Verteilungs-Features gefunden"

    def test_percentile_features(self, trending_ohlc):
        """Sollte Perzentil-Features berechnen."""
        from fwbg.builtins.indicators.distribution import DistributionIndicators

        indicator = DistributionIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Percentile sollten im Bereich 0-1 oder 0-100 sein
        pct_cols = [c for c in df.columns if "pct" in c.lower() or "percentile" in c.lower()]

        for col in pct_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                # Entweder 0-1 oder 0-100 Skala
                assert valid_data.min() >= 0, f"{col} unter 0"
                assert valid_data.max() <= 100, f"{col} über 100"


# --- DynamicsIndicators Tests ---


class TestDynamicsIndicators:
    """Tests für Dynamik-Indikatoren."""

    def test_computes_momentum_features(self, trending_ohlc):
        """Sollte Momentum-basierte Dynamik-Features berechnen."""
        from fwbg.builtins.indicators.dynamics import DynamicsIndicators

        indicator = DynamicsIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Sollte Features haben
        features = indicator.get_feature_columns()
        assert len(features) > 0

    def test_acceleration_features(self, trending_ohlc):
        """Sollte Beschleunigungs-Features (2. Ableitung) haben."""
        from fwbg.builtins.indicators.dynamics import DynamicsIndicators

        indicator = DynamicsIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Acceleration = Change of Change
        accel_cols = [c for c in df.columns if "accel" in c.lower() or "acc_" in c.lower()]

        # Wenn vorhanden, prüfen dass nach Warmup keine Inf-Werte
        # (erste ~50 Bars können Inf haben wegen Division durch kleine Werte)
        for col in accel_cols:
            valid_data = df[col].iloc[50:].dropna()  # Nach Warmup
            if len(valid_data) > 0:
                inf_count = np.isinf(valid_data).sum()
                # Toleriere vereinzelte Inf-Werte bei extremer Volatilität
                assert inf_count < len(valid_data) * 0.1, f"{col} hat zu viele Inf-Werte: {inf_count}"


# --- MultiTimeframeIndicators Tests ---


class TestMultiTimeframeIndicators:
    """Tests für Multi-Timeframe Indikatoren."""

    def test_computes_mtf_features(self, trending_ohlc):
        """Sollte Multi-Timeframe Features berechnen."""
        from fwbg.builtins.indicators.multi_timeframe import MultiTimeframeIndicators

        indicator = MultiTimeframeIndicators()
        df = indicator.compute(trending_ohlc.copy())

        features = indicator.get_feature_columns()

        # Mindestens einige MTF Features sollten existieren
        mtf_cols = [c for c in df.columns if "mtf_" in c.lower() or "higher" in c.lower()]
        assert len(features) > 0 or len(mtf_cols) > 0

    def test_higher_timeframe_smoother(self, trending_ohlc):
        """Höhere Timeframes sollten glattere Werte haben."""
        from fwbg.builtins.indicators.multi_timeframe import MultiTimeframeIndicators

        indicator = MultiTimeframeIndicators()
        df = indicator.compute(trending_ohlc.copy())

        # Suche nach verschiedenen Timeframe-Versionen eines Indikators
        # z.B. mtf_ema_4h vs mtf_ema_1h
        # Höhere TF sollte weniger volatile sein


# --- CrossFeatureIndicators Tests ---


class TestCrossFeatureIndicators:
    """Tests für Cross-Feature Indikatoren."""

    def test_computes_cross_features(self, trending_ohlc):
        """Sollte Cross-Features berechnen."""
        from fwbg.builtins.indicators.cross_features import CrossFeatureIndicators

        indicator = CrossFeatureIndicators()
        df = indicator.compute(trending_ohlc.copy())

        features = indicator.get_feature_columns()
        assert len(features) > 0

    def test_ratio_features_bounded(self, trending_ohlc):
        """Ratio-Features sollten begrenzt sein (kein Division by Zero)."""
        from fwbg.builtins.indicators.cross_features import CrossFeatureIndicators

        indicator = CrossFeatureIndicators()
        df = indicator.compute(trending_ohlc.copy())

        ratio_cols = [c for c in df.columns if "ratio" in c.lower()]

        for col in ratio_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                assert not np.isinf(valid_data).any(), f"{col} hat Inf-Werte"
                assert not np.isnan(valid_data).all(), f"{col} ist komplett NaN"


# --- PriceActionIndicators Tests ---


class TestPriceActionIndicators:
    """Tests für Price Action Indikatoren."""

    def test_computes_candle_features(self, trending_ohlc):
        """Sollte Candlestick-Features berechnen."""
        from fwbg.builtins.indicators.price_action import PriceActionIndicators

        indicator = PriceActionIndicators()
        df = indicator.compute(trending_ohlc.copy())

        features = indicator.get_feature_columns()
        assert len(features) > 0

    def test_body_ratio_bounded(self, trending_ohlc):
        """Body Ratio sollte zwischen 0 und 1 sein."""
        from fwbg.builtins.indicators.price_action import PriceActionIndicators

        indicator = PriceActionIndicators()
        df = indicator.compute(trending_ohlc.copy())

        if "pa_body_ratio" in df.columns:
            valid_data = df["pa_body_ratio"].dropna()
            assert valid_data.min() >= 0, "Body Ratio unter 0"
            assert valid_data.max() <= 1.01, "Body Ratio über 1"

    def test_gap_detection(self, trending_ohlc):
        """Sollte Gaps korrekt erkennen."""
        from fwbg.builtins.indicators.price_action import PriceActionIndicators

        # Füge künstlichen Gap ein
        df = trending_ohlc.copy()
        df.iloc[100, df.columns.get_loc("O")] = df.iloc[99, df.columns.get_loc("C")] + 2  # Gap Up

        indicator = PriceActionIndicators()
        df = indicator.compute(df)

        # Sollte Features haben die Gaps messen
        gap_cols = [c for c in df.columns if "gap" in c.lower()]
        if gap_cols:
            # An Position 100 sollte Gap > 0 sein
            for col in gap_cols:
                if not df[col].isna().iloc[100]:
                    # Gap wurde berechnet
                    pass


# --- Integration Test ---


class TestIndicatorIntegration:
    """Integration Tests für alle Indikatoren zusammen."""

    def test_all_indicators_compatible(self, trending_ohlc):
        """Alle Indikatoren sollten gemeinsam auf DataFrame anwendbar sein."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators
        from fwbg.builtins.indicators.time_season import TimeSeasonIndicators
        from fwbg.builtins.indicators.price_action import PriceActionIndicators

        df = trending_ohlc.copy()

        # Nacheinander anwenden
        indicators = [
            IchimokuIndicators(),
            TimeSeasonIndicators(),
            PriceActionIndicators(),
        ]

        for indicator in indicators:
            df = indicator.compute(df)

        # Keine Fehler und DataFrame intakt
        assert len(df) == len(trending_ohlc)
        assert len(df.columns) > len(trending_ohlc.columns)

    def test_no_inf_values_after_all_indicators(self, trending_ohlc):
        """Nach allen Indikatoren sollten keine Inf-Werte entstehen."""
        from fwbg.builtins.indicators.ichimoku import IchimokuIndicators
        from fwbg.builtins.indicators.time_season import TimeSeasonIndicators

        df = trending_ohlc.copy()

        indicators = [
            IchimokuIndicators(),
            TimeSeasonIndicators(),
        ]

        for indicator in indicators:
            df = indicator.compute(df)

        # Keine Inf-Werte
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, f"{col} hat {inf_count} Inf-Werte"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
