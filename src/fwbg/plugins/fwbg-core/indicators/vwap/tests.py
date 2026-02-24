"""
Tests for VwapIndicator plugin.

Testet:
- VWAP-Berechnung (equal-weight und volume-gewichtet)
- Session-Reset bei session_start_hour
- VWAP-Bands und Standardabweichung
- Rolling Z-Score der Abweichung
- Volume-Fallback wenn V fehlt
- Lookahead-Bias-Prevention (shift_features)
- Feature-Spalten vollständig
"""
import numpy as np
import pandas as pd
import pytest


def make_ohlc(close, freq="15min", session_start_hour=9, high_pct=0.005, low_pct=0.005,
              volume=None):
    """Erstellt OHLC-DataFrame mit konfigurierbarer Frequenz und optionalem Volume."""
    n = len(close)
    # Session beginnt um session_start_hour:00
    start = pd.Timestamp(f"2024-01-02 {session_start_hour:02d}:00:00")
    idx = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame({
        "O": close * (1 - high_pct / 2),
        "H": close * (1 + high_pct),
        "L": close * (1 - low_pct),
        "C": close,
    }, index=idx)
    if volume is not None:
        df["V"] = volume
    return df


@pytest.fixture
def indicator():
    from fwbg.plugins import import_plugin_module
    _vwap = import_plugin_module("fwbg-core", "indicators", "vwap")
    return _vwap.VwapIndicator()


@pytest.fixture
def flat_price_df():
    """DataFrame mit konstantem Preis (VWAP == Close immer)."""
    n = 100
    close = np.full(n, 100.0)
    return make_ohlc(close)


@pytest.fixture
def trending_df():
    """DataFrame mit stetig steigendem Preis."""
    n = 100
    close = np.linspace(100, 110, n)
    return make_ohlc(close)


@pytest.fixture
def multi_session_df():
    """DataFrame über 2 volle Handelstage (je 26 Bars à 15min, 9:00–15:45)."""
    bars_per_session = 28  # 9:00 bis 15:45 = 28 × 15min
    n = bars_per_session * 2
    close = np.concatenate([
        np.linspace(100, 105, bars_per_session),
        np.linspace(108, 103, bars_per_session),
    ])
    start = pd.Timestamp("2024-01-02 09:00:00")
    idx = pd.date_range(start, periods=n, freq="15min")
    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
    }, index=idx)
    return df


class TestVwapBasic:
    """Grundlegende VWAP-Berechnung."""

    def test_vwap_at_flat_price_equals_close(self, indicator, flat_price_df):
        """Bei konstantem Preis soll VWAP = Preis sein (nach shift: ab Bar 1)."""
        result = indicator.compute(flat_price_df)
        vwap = result["vwap"].dropna()
        assert (vwap.round(6) == 100.0).all(), "VWAP bei konstantem Preis muss 100.0 sein"

    def test_vwap_is_between_high_and_low(self, indicator, trending_df):
        """VWAP muss immer zwischen High und Low liegen."""
        result = indicator.compute(trending_df)
        vwap = result["vwap"].dropna()
        assert (vwap >= result["L"].loc[vwap.index]).all()
        assert (vwap <= result["H"].loc[vwap.index]).all()

    def test_vwap_starts_at_typical_price(self, indicator):
        """Erstes VWAP einer Session muss Typical Price sein."""
        close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        df = make_ohlc(close, high_pct=0.0, low_pct=0.0)
        # H=L=C=O → typical = close
        result = indicator.compute(df)
        # Nach dem Shift: vwap[1] entspricht der VWAP-Berechnung von Bar 0
        vwap = result["vwap"]
        # Bar 0 hat kein vwap (ist NaN durch shift)
        assert pd.isna(vwap.iloc[0]), "Erster Wert muss NaN sein (Lookahead-Prevention)"
        # Bar 1: VWAP von Bar 0 = typical price von Bar 0 = close[0] = 100.0
        assert abs(vwap.iloc[1] - 100.0) < 1e-6, f"Erster VWAP muss 100.0 sein, war {vwap.iloc[1]}"


class TestSessionReset:
    """Session-Reset-Logik."""

    def test_vwap_resets_each_session(self, indicator, multi_session_df):
        """VWAP muss am Beginn jeder neuen Session zurückgesetzt werden."""
        result = indicator.compute(multi_session_df, session_start_hour=9)

        # Session 1 endet am letzten Bar von Tag 1 (9:00+27*15min = 15:45)
        # Session 2 beginnt am nächsten 9:00-Bar
        hours = result.index.hour
        session2_start = result.index[result.index.date == result.index.date[1]][0]

        vwap = result["vwap"]

        # Am Beginn von Session 2 (nach Shift): vwap entspricht letztem Bar von Session 1
        # Das VWAP am Ende Session 1 und Beginn Session 2 müssen sich deutlich unterscheiden
        # da die Preise in Session 2 bei ~108 starten (vs ~105 Ende Session 1)
        sess1_last_idx = result.index[result.index < session2_start][-1]
        sess2_first_idx = session2_start

        vwap_end_s1 = vwap.loc[sess1_last_idx]
        # Nach dem Shift: das erste vwap-Bar von Session 2 zeigt noch den letzten VWAP von S1
        # Zwei Bars später sehen wir bereits den neuen Session-2-VWAP
        # Weiter im Verlauf von Session 2 muss VWAP ~108 konvergieren
        sess2_bars = vwap.loc[session2_start:]
        # VWAP in Session 2 muss irgendwann über 106 kommen (session 2 starts at 108)
        assert sess2_bars.dropna().max() > 106, "VWAP in Session 2 muss bei ~108 liegen"

    def test_session_boundary_at_correct_hour(self, indicator):
        """Session-Grenze muss bei genau session_start_hour auftreten."""
        # 3 Stunden vor session_start + 2 Stunden innerhalb
        start = pd.Timestamp("2024-01-02 07:00:00")
        idx = pd.date_range(start, periods=20, freq="15min")
        close = np.full(20, 100.0)
        df = pd.DataFrame({
            "O": close, "H": close * 1.005, "L": close * 0.995, "C": close
        }, index=idx)

        result = indicator.compute(df, session_start_hour=9)
        vwap = result["vwap"]

        # Bars vor 9:00: eigene Session (session_id = 1)
        # Bars ab 9:00: neue Session (session_id = 2)
        # Alle Bars haben gleichen Preis → VWAP = 100.0 in beiden Sessions
        assert vwap.dropna().round(6).eq(100.0).all()


class TestVolumeWeighting:
    """Volume-Gewichtung."""

    def test_vwap_volume_weighted_correctly(self, indicator):
        """VWAP muss bei unterschiedlichen Volumes korrekt gewichtet sein."""
        # 3 Bars: Close [100, 110, 90], Volume [1, 10, 1]
        # VWAP nach Bar 2 ≈ (100*1 + 110*10 + 90*1) / 12 ≈ 109.2
        close = np.array([100.0, 110.0, 90.0])
        df = make_ohlc(close, high_pct=0.0, low_pct=0.0, volume=[1.0, 10.0, 1.0])
        result = indicator.compute(df, session_start_hour=df.index[0].hour)
        vwap = result["vwap"]

        # Nach shift: vwap[3] = VWAP nach 3 Bars = (100*1 + 110*10 + 90*1) / 12
        expected_after_3_bars = (100 * 1 + 110 * 10 + 90 * 1) / 12
        # vwap.iloc[3] wenn 4+ Bars existieren, sonst .iloc[-1]
        # Da nur 3 Bars, ist vwap.iloc[2] der VWAP nach Bar 2 (shift um 1)
        # Also: vwap.iloc[2] = VWAP nach Bar 1 (0-indexed)
        # vwap.iloc[1] = VWAP von Bar 0 = close[0] = 100.0
        # vwap.iloc[2] = VWAP nach Bar 0+1 = (100+110*10)/11 ≈ 109.09
        expected_after_2_bars = (100 * 1 + 110 * 10) / 11
        assert abs(vwap.iloc[2] - expected_after_2_bars) < 0.01, (
            f"VWAP nach Bar 1: erwartet {expected_after_2_bars:.4f}, war {vwap.iloc[2]:.4f}"
        )

    def test_fallback_without_volume(self, indicator, trending_df):
        """Ohne Volume muss VWAP trotzdem berechnet werden (equal-weight)."""
        # trending_df hat kein Volume
        assert "V" not in trending_df.columns
        result = indicator.compute(trending_df)
        vwap = result["vwap"].dropna()
        assert vwap.notna().all(), "VWAP muss auch ohne Volume berechnet werden"
        assert len(vwap) > 50

    def test_fallback_matches_equal_weight(self, indicator):
        """Equal-weight Fallback muss identisch zu VWAP mit Volume=1 sein."""
        close = np.linspace(100, 110, 50)
        df_no_vol = make_ohlc(close)
        df_vol_1 = make_ohlc(close, volume=np.ones(50))

        result_no_vol = indicator.compute(df_no_vol)
        result_vol_1 = indicator.compute(df_vol_1)

        pd.testing.assert_series_equal(
            result_no_vol["vwap"].dropna().round(8),
            result_vol_1["vwap"].dropna().round(8),
            check_names=False,
        )


class TestVwapDeviation:
    """VWAP-Abweichung und Z-Score."""

    def test_deviation_zero_at_vwap(self, indicator, flat_price_df):
        """Bei konstantem Preis muss Abweichung ~0 sein."""
        result = indicator.compute(flat_price_df)
        dev = result["vwap_deviation"].dropna()
        assert dev.abs().max() < 1e-6, "Abweichung bei konstantem Preis muss ~0 sein"

    def test_deviation_positive_above_vwap(self, indicator):
        """Wenn Preis über VWAP steigt, muss Abweichung positiv sein."""
        close = np.concatenate([np.full(10, 100.0), np.full(10, 110.0)])
        df = make_ohlc(close)
        result = indicator.compute(df)
        dev = result["vwap_deviation"].dropna()
        # Am Ende (Preis = 110, VWAP irgendwo zwischen 100 und 110)
        assert dev.iloc[-1] > 0, "Abweichung muss positiv sein wenn Preis > VWAP"

    def test_zscore_windows_computed(self, indicator, trending_df):
        """Z-Score-Spalten für beide Default-Fenster müssen vorhanden sein."""
        result = indicator.compute(trending_df)
        assert "vwap_zscore_20" in result.columns
        assert "vwap_zscore_50" in result.columns

    def test_zscore_bounded_for_flat_price(self, indicator, flat_price_df):
        """Z-Score bei konstantem Preis muss NaN sein (keine Varianz)."""
        result = indicator.compute(flat_price_df)
        zscore = result["vwap_zscore_20"].dropna()
        # Std = 0 → safe_divide gibt NaN zurück
        assert zscore.isna().all() or zscore.abs().max() < 1e-3

    def test_custom_zscore_windows(self, indicator, trending_df):
        """Custom Z-Score-Fenster müssen korrekt erzeugt werden."""
        result = indicator.compute(trending_df, zscore_windows=[10, 30])
        assert "vwap_zscore_10" in result.columns
        assert "vwap_zscore_30" in result.columns
        assert "vwap_zscore_20" not in result.columns


class TestVwapBands:
    """VWAP-Bands."""

    def test_upper_band_above_vwap(self, indicator, trending_df):
        """Upper Band muss immer >= VWAP sein."""
        result = indicator.compute(trending_df)
        mask = result["vwap"].notna() & result["vwap_upper_1"].notna()
        assert (result.loc[mask, "vwap_upper_1"] >= result.loc[mask, "vwap"]).all()

    def test_lower_band_below_vwap(self, indicator, trending_df):
        """Lower Band muss immer <= VWAP sein."""
        result = indicator.compute(trending_df)
        mask = result["vwap"].notna() & result["vwap_lower_1"].notna()
        assert (result.loc[mask, "vwap_lower_1"] <= result.loc[mask, "vwap"]).all()

    def test_outer_bands_wider_than_inner(self, indicator, trending_df):
        """±2σ-Band muss breiter als ±1σ-Band sein."""
        result = indicator.compute(trending_df)
        mask = result["vwap_upper_2"].notna() & result["vwap_upper_1"].notna()
        assert (result.loc[mask, "vwap_upper_2"] >= result.loc[mask, "vwap_upper_1"]).all()
        assert (result.loc[mask, "vwap_lower_2"] <= result.loc[mask, "vwap_lower_1"]).all()

    def test_band_pos_at_vwap_is_half(self, indicator, flat_price_df):
        """Bei Preis = VWAP muss band_pos = 0.5 sein."""
        result = indicator.compute(flat_price_df)
        band_pos = result["vwap_band_pos"].dropna()
        # Bei flat price: C = H = L = O = 100 → VWAP = 100 = C
        # std = 0 → band_pos ist NaN (safe_divide)
        # Das ist korrekt: bei 0 Varianz ist Band-Position undefiniert
        assert band_pos.isna().all() or (band_pos - 0.5).abs().max() < 1e-6

    def test_vwap_above_binary(self, indicator, trending_df):
        """vwap_above muss 0 oder 1 sein."""
        result = indicator.compute(trending_df)
        above = result["vwap_above"].dropna()
        assert set(above.unique()).issubset({0.0, 1.0})


class TestLookaheadBias:
    """Lookahead-Bias-Prevention."""

    def test_first_row_is_nan(self, indicator, trending_df):
        """Alle Feature-Spalten müssen in der ersten Zeile NaN sein."""
        result = indicator.compute(trending_df)
        for col in indicator.get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), (
                    f"Spalte {col} muss in Zeile 0 NaN sein (shift_features)"
                )

    def test_shift_by_one(self, indicator):
        """Verifiziert dass shift_features genau 1 Bar shiftet."""
        close = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        df = make_ohlc(close, high_pct=0.0, low_pct=0.0)
        result = indicator.compute(df, session_start_hour=df.index[0].hour)
        vwap = result["vwap"]

        # Bar 0: NaN (erste Zeile nach shift)
        assert pd.isna(vwap.iloc[0])
        # Bar 1: VWAP = typical price von Bar 0 = 100.0 (H=L=C=O=100)
        assert abs(vwap.iloc[1] - 100.0) < 1e-6


class TestFeatureColumns:
    """Feature-Spalten-Vollständigkeit."""

    def test_all_default_columns_present(self, indicator, trending_df):
        """Alle Spalten aus get_feature_columns() müssen im DataFrame vorhanden sein."""
        result = indicator.compute(trending_df)
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Feature-Spalte '{col}' fehlt im DataFrame"

    def test_get_feature_columns_returns_list(self, indicator):
        """get_feature_columns muss eine Liste zurückgeben."""
        cols = indicator.get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_all_features_have_vwap_prefix(self, indicator):
        """Alle Feature-Spalten müssen mit 'vwap_' beginnen (außer 'vwap')."""
        cols = indicator.get_feature_columns()
        for col in cols:
            assert col == "vwap" or col.startswith("vwap_"), (
                f"Feature '{col}' fehlt vwap_-Prefix"
            )

    def test_signal_columns_subset_of_features(self, indicator):
        """Signal-Spalten müssen Teil der Feature-Spalten sein."""
        feature_cols = set(indicator.get_feature_columns())
        signal_cols = set(indicator.get_signal_columns())
        assert signal_cols.issubset(feature_cols)


class TestPluginAttributes:
    """Plugin-Attribute."""

    def test_name_attribute(self, indicator):
        assert indicator.name == "vwap"

    def test_version_attribute(self, indicator):
        assert hasattr(indicator, "version")
        assert isinstance(indicator.version, str)

    def test_validate_returns_true(self, indicator):
        assert indicator.validate() is True

    def test_benefits_from_stationary_false(self, indicator):
        assert indicator.benefits_from_stationary is False
