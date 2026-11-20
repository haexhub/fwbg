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


def make_ohlc(
    close,
    freq="15min",
    session_start_hour=9,
    high_pct=0.005,
    low_pct=0.005,
    volume=None,
):
    """Erstellt OHLC-DataFrame mit konfigurierbarer Frequenz und optionalem Volume."""
    n = len(close)
    # Session beginnt um session_start_hour:00
    start = pd.Timestamp(f"2024-01-02 {session_start_hour:02d}:00:00")
    idx = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame(
        {
            "O": close * (1 - high_pct / 2),
            "H": close * (1 + high_pct),
            "L": close * (1 - low_pct),
            "C": close,
        },
        index=idx,
    )
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
    """DataFrame über 2 volle Handelstage mit Session-Reset bei 09:00.

    Session 1: Jan 2 09:00 – Jan 3 08:45 (96 bars, price 100→105)
    Session 2: Jan 3 09:00 – 15:45 (28 bars, price 108→103)
    """
    bars_session1 = 96  # 24h at 15min = full day until next 09:00
    bars_session2 = 28  # 7h trading in session 2
    n = bars_session1 + bars_session2
    close = np.concatenate(
        [
            np.linspace(100, 105, bars_session1),
            np.linspace(108, 103, bars_session2),
        ]
    )
    start = pd.Timestamp("2024-01-02 09:00:00")
    idx = pd.date_range(start, periods=n, freq="15min")
    df = pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.005,
            "L": close * 0.995,
            "C": close,
        },
        index=idx,
    )
    return df


class TestVwapBasic:
    """Grundlegende VWAP-Berechnung."""

    def test_vwap_at_flat_price_equals_close(self, indicator, flat_price_df):
        """Bei konstantem Preis soll VWAP = Preis sein (nach shift: ab Bar 1)."""
        result = indicator.compute(flat_price_df)
        vwap = result["vwap"].dropna()
        assert (vwap.round(6) == 100.0).all(), (
            "VWAP bei konstantem Preis muss 100.0 sein"
        )

    def test_vwap_within_session_range(self, indicator, trending_df):
        """VWAP muss zwischen Session-cummin(L) und Session-cummax(H) liegen."""
        result = indicator.compute(trending_df)
        vwap = result["vwap"].dropna()
        # VWAP is a cumulative weighted average of typical prices.
        # Each tp is between L and H, so VWAP stays within session's running H/L envelope.
        cum_h = result["H"].expanding().max().loc[vwap.index]
        cum_l = result["L"].expanding().min().loc[vwap.index]
        assert (vwap >= cum_l).all(), "VWAP below session cumulative low"
        assert (vwap <= cum_h).all(), "VWAP above session cumulative high"

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
        assert abs(vwap.iloc[1] - 100.0) < 1e-6, (
            f"Erster VWAP muss 100.0 sein, war {vwap.iloc[1]}"
        )


class TestSessionReset:
    """Session-Reset-Logik."""

    def test_vwap_resets_each_session(self, indicator, multi_session_df):
        """VWAP muss am Beginn jeder neuen Session zurückgesetzt werden."""
        result = indicator.compute(multi_session_df, session_start_hour=9)

        # Session 2 starts at 09:00 Jan 3
        session2_start = pd.Timestamp("2024-01-03 09:00:00")
        vwap = result["vwap"]

        # Session 2 prices start at ~108, VWAP should converge there
        sess2_bars = vwap.loc[session2_start:]
        assert sess2_bars.dropna().max() > 106, "VWAP in Session 2 muss bei ~108 liegen"

        # VWAP just before session 2 (end of session 1) should be ~102-103
        sess1_last_vwap = vwap.loc[:session2_start].iloc[-2]  # -2: before shift effect
        assert sess1_last_vwap < 106, (
            f"Session 1 VWAP end should be <106 (prices 100-105), got {sess1_last_vwap}"
        )

    def test_session_boundary_at_correct_hour(self, indicator):
        """Session-Grenze muss bei genau session_start_hour auftreten."""
        # 3 Stunden vor session_start + 2 Stunden innerhalb
        start = pd.Timestamp("2024-01-02 07:00:00")
        idx = pd.date_range(start, periods=20, freq="15min")
        close = np.full(20, 100.0)
        df = pd.DataFrame(
            {"O": close, "H": close * 1.005, "L": close * 0.995, "C": close}, index=idx
        )

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

        # Da nur 3 Bars, ist vwap.iloc[2] der VWAP nach Bar 1 (shift um 1)
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
        assert (
            result.loc[mask, "vwap_upper_2"] >= result.loc[mask, "vwap_upper_1"]
        ).all()
        assert (
            result.loc[mask, "vwap_lower_2"] <= result.loc[mask, "vwap_lower_1"]
        ).all()

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


# -- regression tests: band std, session resets, param-aware columns ----------


def _flat_bars(close, index, volume=None):
    """Bars with H = L = C = O, so the typical price equals the close."""
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({"O": close, "H": close, "L": close, "C": close}, index=index)
    if volume is not None:
        df["V"] = np.asarray(volume, dtype=float)
    return df


def _reference_bands(tp, volume):
    """Volume-weighted mean and std of all typical prices up to each bar."""
    vwap, std = [], []
    for k in range(1, len(tp) + 1):
        t, w = np.asarray(tp[:k], float), np.asarray(volume[:k], float)
        mean = (w * t).sum() / w.sum()
        vwap.append(mean)
        std.append(np.sqrt((w * (t - mean) ** 2).sum() / w.sum()))
    return np.array(vwap), np.array(std)


class TestBandStd:
    def test_band_width_is_the_volume_weighted_std_of_the_session(self, indicator):
        tp = [100.0, 110.0, 90.0, 104.0, 97.0, 108.0]
        vol = [1.0, 10.0, 1.0, 5.0, 2.0, 7.0]
        index = pd.date_range("2024-01-02 09:00", periods=len(tp), freq="15min")
        out = indicator.compute(_flat_bars(tp, index, vol), band_multipliers=[1.0, 2.0])
        ref_vwap, ref_std = _reference_bands(tp, vol)
        # features are shifted by one bar: row k+1 holds the value of bar k
        np.testing.assert_allclose(
            out["vwap"].iloc[1:].values, ref_vwap[:-1], rtol=1e-9
        )
        np.testing.assert_allclose(
            (out["vwap_upper_1"] - out["vwap"]).iloc[1:].values,
            ref_std[:-1],
            rtol=1e-6,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            (out["vwap"] - out["vwap_lower_2"]).iloc[1:].values,
            2 * ref_std[:-1],
            rtol=1e-6,
            atol=1e-9,
        )

    def test_band_std_without_volume_is_the_plain_std(self, indicator):
        tp = [100.0, 101.0, 103.0, 99.0, 102.0]
        index = pd.date_range("2024-01-02 09:00", periods=len(tp), freq="15min")
        out = indicator.compute(_flat_bars(tp, index))
        _, ref_std = _reference_bands(tp, np.ones(len(tp)))
        np.testing.assert_allclose(
            (out["vwap_upper_1"] - out["vwap"]).iloc[1:].values,
            ref_std[:-1],
            rtol=1e-6,
            atol=1e-9,
        )

    def test_band_std_is_stable_at_high_price_levels(self, indicator):
        tp = np.array([60000.0, 60010.0, 59990.0, 60005.0, 59995.0])
        index = pd.date_range("2024-01-02 09:00", periods=len(tp), freq="15min")
        out = indicator.compute(_flat_bars(tp, index))
        _, ref_std = _reference_bands(tp, np.ones(len(tp)))
        np.testing.assert_allclose(
            (out["vwap_upper_1"] - out["vwap"]).iloc[1:].values,
            ref_std[:-1],
            rtol=1e-6,
            atol=1e-6,
        )


class TestSessionResetTimezone:
    def _ny_sessions(self, days, price_of_day):
        """Three 15-minute bars from 09:30 New York on each day, naive UTC index."""
        frames = []
        for day, price in zip(days, price_of_day):
            local = pd.date_range(
                f"{day} 09:30", periods=3, freq="15min", tz="America/New_York"
            )
            frames.append(
                _flat_bars([price] * 3, local.tz_convert("UTC").tz_localize(None))
            )
        return pd.concat(frames)

    def test_resets_each_new_york_morning_across_the_dst_change(self, indicator):
        # 2024-03-08 is EST (14:30 UTC open), 2024-03-11 is EDT (13:30 UTC open)
        df = self._ny_sessions(["2024-03-08", "2024-03-11"], [100.0, 200.0])
        out = indicator.compute(df, session_start_hour=9, session_tz="America/New_York")
        # row 4 holds the VWAP after the first bar of day 2: 200, not the 150 of a
        # running VWAP that never reset
        assert out["vwap"].iloc[4] == pytest.approx(200.0)
        assert out["vwap"].iloc[3] == pytest.approx(100.0)

    def test_reset_when_no_bar_exists_in_the_start_hour(self, indicator):
        days = ["2024-01-02", "2024-01-03"]
        index = pd.DatetimeIndex(
            [pd.Timestamp(f"{d} {h:02d}:00") for d in days for h in (10, 11, 12)]
        )
        df = _flat_bars([100.0] * 3 + [200.0] * 3, index)
        out = indicator.compute(df, session_start_hour=9)
        assert out["vwap"].iloc[4] == pytest.approx(200.0)

    def test_tz_aware_index_is_converted(self, indicator):
        local = pd.date_range(
            "2024-03-08 09:30", periods=3, freq="15min", tz="America/New_York"
        )
        local2 = pd.date_range(
            "2024-03-11 09:30", periods=3, freq="15min", tz="America/New_York"
        )
        df = _flat_bars([100.0] * 3 + [200.0] * 3, local.append(local2))
        out = indicator.compute(df, session_start_hour=9, session_tz="America/New_York")
        assert out["vwap"].iloc[4] == pytest.approx(200.0)

    def test_daily_bars_reset_on_every_bar(self, indicator):
        close = np.array([100.0, 110.0, 90.0, 120.0])
        df = _flat_bars(close, pd.bdate_range("2024-01-02", periods=4))
        out = indicator.compute(df)
        # one-bar sessions: VWAP is the bar's own typical price (shifted by one bar)
        np.testing.assert_allclose(out["vwap"].iloc[1:].values, close[:-1])
        assert out["vwap_deviation"].iloc[1:].abs().max() < 1e-12


class TestFeatureColumnsFollowParams:
    def test_custom_params_columns_match_compute(self, indicator, trending_df):
        params = {"zscore_windows": [10, 30], "band_multipliers": [1.5, 3]}
        out = indicator.compute(trending_df, **params)
        cols = indicator.get_feature_columns(params)
        assert set(cols) == {c for c in out.columns if c.startswith("vwap")}
        assert "vwap_upper_1_5" in cols and "vwap_lower_3" in cols
        assert "vwap_zscore_20" not in cols

    def test_defaults_unchanged(self, indicator):
        assert indicator.get_feature_columns() == indicator.get_feature_columns({})
        assert "vwap_zscore_50" in indicator.get_feature_columns()
