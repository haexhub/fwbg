"""
Structure Indicator Plugin.

Enthält:
- FFT (Fast Fourier Transform) Features
- Path Efficiency / Fractal Dimension
- Convexity Features (EMA 2. Ableitung)
- Event Features (Time-Since-Event)
- VWAP Features

FFT Interpretation:
- Hohe spektrale Entropie = Rauschen (schwierig zu traden)
- Niedrige spektrale Entropie = klare Zyklen (vorhersagbar)
- High Low-Freq Ratio = langfristige Trends dominieren
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator


def _bars_since_event(event_series: pd.Series) -> pd.Series:
    """
    Berechnet Bars seit dem letzten Event (True/1).

    Args:
        event_series: Series mit 1 wo Event auftritt, 0 sonst

    Returns:
        Series mit Anzahl Bars seit letztem Event
    """
    event_groups = event_series.cumsum()
    result = event_series.groupby(event_groups).cumcount()

    first_event_idx = event_series.idxmax() if event_series.any() else None
    if first_event_idx is not None:
        result.loc[:first_event_idx] = np.nan

    return result


def _compute_fft_features(close: np.ndarray, window: int) -> dict:
    """
    Berechnet FFT-basierte Features für einen Rolling-Window.

    Extrahiert aus dem Frequenzspektrum:
    - Dominante Frequenz (Hauptzyklus)
    - Spektrale Energie (Gesamtstärke der Zyklen)
    - Spektrale Entropie (Verteilung der Energie über Frequenzen)
    - Low-Freq Ratio (Anteil langfristiger Trends)
    """
    n = len(close)

    dominant_freq = np.full(n, np.nan)
    dominant_power = np.full(n, np.nan)
    spectral_energy = np.full(n, np.nan)
    spectral_entropy = np.full(n, np.nan)
    low_freq_ratio = np.full(n, np.nan)

    for i in range(window, n):
        segment = close[i - window:i]
        segment_detrended = segment - np.mean(segment)

        # Hanning Window (reduziert Spectral Leakage)
        windowed = segment_detrended * np.hanning(window)

        fft_result = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(window)

        # Power Spectrum
        power = np.abs(fft_result) ** 2

        # Ignoriere DC-Komponente
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        if len(power_no_dc) == 0 or np.sum(power_no_dc) < 1e-10:
            continue

        # Dominante Frequenz
        dom_idx = np.argmax(power_no_dc)
        dominant_freq[i] = freqs_no_dc[dom_idx]
        dominant_power[i] = power_no_dc[dom_idx] / (np.sum(power_no_dc) + 1e-10)

        # Spektrale Energie (normalisiert)
        spectral_energy[i] = np.log1p(np.sum(power_no_dc))

        # Spektrale Entropie
        power_norm = power_no_dc / (np.sum(power_no_dc) + 1e-10)
        power_norm = power_norm[power_norm > 1e-10]
        spectral_entropy[i] = -np.sum(power_norm * np.log(power_norm))

        # Low-Frequency Ratio
        cutoff = len(power_no_dc) // 4
        if cutoff > 0:
            low_freq_ratio[i] = np.sum(power_no_dc[:cutoff]) / (np.sum(power_no_dc) + 1e-10)

    return {
        "dom_freq": dominant_freq,
        "dom_power": dominant_power,
        "energy": spectral_energy,
        "entropy": spectral_entropy,
        "lowfreq": low_freq_ratio,
    }


@register_indicator("structure")
class StructureIndicators(BaseIndicator):
    """
    Structure-bezogene Features.

    Features:
    - FFT Features (64, 128, 256 Fenster)
    - Path Efficiency (10, 20, 50, 100)
    - Fractal Dimension Proxy
    - Convexity (EMA 2. Ableitung)
    - Event Features (Bars seit High/Low/Cross)
    - VWAP-ähnliche Features
    """

    group = "structure"

    def compute(
        self,
        df: pd.DataFrame,
        fft_windows: List[int] = None,
        path_windows: List[int] = None,
        convexity_periods: List[int] = None,
        event_periods: List[int] = None,
        vwap_windows: List[int] = None,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Structure-Features.

        Args:
            df: DataFrame mit OHLC-Daten
            fft_windows: Fenstergrößen für FFT (default: [64, 128, 256])
            path_windows: Fenstergrößen für Path Efficiency (default: [10, 20, 50, 100])
            convexity_periods: Perioden für Convexity (default: [21, 50])
            event_periods: Perioden für Event Features (default: [20, 50])
            vwap_windows: Fenstergrößen für VWAP (default: [20, 50, 100])

        Returns:
            DataFrame mit Structure-Features
        """
        if fft_windows is None:
            fft_windows = [64, 128, 256]
        if path_windows is None:
            path_windows = [10, 20, 50, 100]
        if convexity_periods is None:
            convexity_periods = [21, 50]
        if event_periods is None:
            event_periods = [20, 50]
        if vwap_windows is None:
            vwap_windows = [20, 50, 100]

        features = {}
        close = df["C"].values
        close_series = df["C"]

        # === FFT Features ===
        for window in fft_windows:
            if len(close) < window * 2:
                continue
            fft_features = _compute_fft_features(close, window)
            suffix = f"_{window}"
            features[f"fft_dom_freq{suffix}"] = fft_features["dom_freq"]
            features[f"fft_dom_power{suffix}"] = fft_features["dom_power"]
            features[f"fft_energy{suffix}"] = fft_features["energy"]
            features[f"fft_entropy{suffix}"] = fft_features["entropy"]
            features[f"fft_lowfreq{suffix}"] = fft_features["lowfreq"]

        # === Path Efficiency / Fractal Dimension ===
        for window in path_windows:
            net_change = abs(close_series - close_series.shift(window))
            abs_changes = abs(close_series.diff())
            path_length = abs_changes.rolling(window).sum()

            pe = net_change / (path_length + 1e-10)
            features[f"path_efficiency_{window}"] = pe
            features[f"fractal_dim_{window}"] = 1 + (1 - pe)

        # Path Efficiency Änderung
        if 20 in path_windows:
            pe_20 = features["path_efficiency_20"]
            features["path_efficiency_20_chg"] = pe_20 - pe_20.shift(10)
        if 50 in path_windows:
            pe_50 = features["path_efficiency_50"]
            features["path_efficiency_50_chg"] = pe_50 - pe_50.shift(20)

        # === Convexity Features ===
        for period in convexity_periods:
            ema = ta.trend.ema_indicator(df["C"], window=period)
            ema_slope = ema.diff()
            ema_convexity = ema_slope.diff()

            convex = ema_convexity / (df["C"] + 1e-10) * 1000
            features[f"convex_ema_{period}"] = convex
            features[f"convex_ema_{period}_smooth"] = convex.rolling(5).mean()

        if 21 in convexity_periods and 50 in convexity_periods:
            features["convex_divergence"] = features["convex_ema_21"] - features["convex_ema_50"]

        if 21 in convexity_periods:
            convex_21 = features["convex_ema_21_smooth"]
            convex_std = convex_21.rolling(100).std()
            features["convex_zscore"] = (convex_21 - convex_21.rolling(100).mean()) / (convex_std + 1e-10)

        # === Event Features ===
        for period in event_periods:
            rolling_high = df["H"].rolling(period).max()
            rolling_low = df["L"].rolling(period).min()

            is_new_high = (df["H"] >= rolling_high).astype(int)
            is_new_low = (df["L"] <= rolling_low).astype(int)

            bars_since_high = _bars_since_event(is_new_high)
            bars_since_low = _bars_since_event(is_new_low)

            features[f"event_bars_since_high_{period}"] = bars_since_high
            features[f"event_bars_since_low_{period}"] = bars_since_low
            features[f"event_bars_since_high_{period}_log"] = np.log1p(bars_since_high)
            features[f"event_bars_since_low_{period}_log"] = np.log1p(bars_since_low)

        # EMA-Cross Event
        ema_8 = ta.trend.ema_indicator(df["C"], window=8)
        ema_21 = ta.trend.ema_indicator(df["C"], window=21)
        ema_cross = ((ema_8 > ema_21) != (ema_8.shift(1) > ema_21.shift(1))).astype(int)
        bars_ema_cross = _bars_since_event(ema_cross)
        features["event_bars_since_ema_cross"] = bars_ema_cross
        features["event_bars_since_ema_cross_log"] = np.log1p(bars_ema_cross)

        # RSI Extremwert Event
        rsi = ta.momentum.rsi(df["C"], window=14)
        rsi_extreme = ((rsi > 70) | (rsi < 30)).astype(int)
        bars_rsi = _bars_since_event(rsi_extreme)
        features["event_bars_since_rsi_extreme"] = bars_rsi
        features["event_bars_since_rsi_extreme_log"] = np.log1p(bars_rsi)

        # Volatilitäts-Spike Event
        atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)
        atr_mean = atr.rolling(50).mean()
        vol_spike = (atr > 2 * atr_mean).astype(int)
        bars_vol = _bars_since_event(vol_spike)
        features["event_bars_since_vol_spike"] = bars_vol
        features["event_bars_since_vol_spike_log"] = np.log1p(bars_vol)

        # === VWAP-ähnliche Features ===
        tp = (df["H"] + df["L"] + df["C"]) / 3

        for window in vwap_windows:
            vwap = tp.rolling(window).mean()
            features[f"structure_vwap_dist_{window}"] = (df["C"] - vwap) / vwap

        vwap_50 = tp.rolling(50).mean()
        above_vwap = (df["C"] > vwap_50).astype(int)
        features["structure_vwap_time_above"] = above_vwap.rolling(20).mean()

        vwap_cross = (above_vwap != above_vwap.shift(1)).astype(int)
        features["structure_bars_since_vwap_cross"] = _bars_since_event(vwap_cross)

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = pd.DataFrame(features, index=df.index)
        for col in features_df.columns:
            features_df[col] = features_df[col].shift(1)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            # FFT Features
            "fft_dom_freq_64", "fft_dom_power_64", "fft_energy_64",
            "fft_entropy_64", "fft_lowfreq_64",
            "fft_dom_freq_128", "fft_dom_power_128", "fft_energy_128",
            "fft_entropy_128", "fft_lowfreq_128",
            "fft_dom_freq_256", "fft_dom_power_256", "fft_energy_256",
            "fft_entropy_256", "fft_lowfreq_256",
            # Path Efficiency
            "path_efficiency_10", "path_efficiency_20",
            "path_efficiency_50", "path_efficiency_100",
            "fractal_dim_10", "fractal_dim_20",
            "fractal_dim_50", "fractal_dim_100",
            "path_efficiency_20_chg", "path_efficiency_50_chg",
            # Convexity
            "convex_ema_21", "convex_ema_21_smooth",
            "convex_ema_50", "convex_ema_50_smooth",
            "convex_divergence", "convex_zscore",
            # Event Features
            "event_bars_since_high_20", "event_bars_since_high_20_log",
            "event_bars_since_low_20", "event_bars_since_low_20_log",
            "event_bars_since_high_50", "event_bars_since_high_50_log",
            "event_bars_since_low_50", "event_bars_since_low_50_log",
            "event_bars_since_ema_cross", "event_bars_since_ema_cross_log",
            "event_bars_since_rsi_extreme", "event_bars_since_rsi_extreme_log",
            "event_bars_since_vol_spike", "event_bars_since_vol_spike_log",
            # VWAP Features
            "structure_vwap_dist_20", "structure_vwap_dist_50", "structure_vwap_dist_100",
            "structure_vwap_time_above", "structure_bars_since_vwap_cross",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "fft_windows": [64, 128, 256],
            "path_windows": [10, 20, 50, 100],
            "convexity_periods": [21, 50],
            "event_periods": [20, 50],
            "vwap_windows": [20, 50, 100],
        }


__all__ = ["StructureIndicators"]
