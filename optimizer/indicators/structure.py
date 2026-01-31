"""
Struktur-bezogene Features.

Enthält:
- FFT (Fast Fourier Transform) Features
- Path Efficiency / Fractal Dimension
- Convexity Features (EMA 2. Ableitung)
- Event Features (Time-Since-Event)
- VWAP Features
"""
import numpy as np
import pandas as pd
import ta


def _bars_since_event(event_series):
    """
    Berechnet Bars seit dem letzten Event (True/1).

    Args:
        event_series: Series mit 1 wo Event auftritt, 0 sonst

    Returns:
        Series mit Anzahl Bars seit letztem Event
    """
    # Erstelle Gruppen bei jedem Event
    event_groups = event_series.cumsum()

    # Zähle innerhalb jeder Gruppe
    result = event_series.groupby(event_groups).cumcount()

    # Am Anfang (vor erstem Event) auf NaN setzen
    first_event_idx = event_series.idxmax() if event_series.any() else None
    if first_event_idx is not None:
        result.loc[:first_event_idx] = np.nan

    return result


def compute_fft_features(df, window: int):
    """
    Berechnet FFT-basierte Features für einen Rolling-Window.

    Extrahiert aus dem Frequenzspektrum:
    - Dominante Frequenz (Hauptzyklus)
    - Spektrale Energie (Gesamtstärke der Zyklen)
    - Spektrale Entropie (Verteilung der Energie über Frequenzen)

    Args:
        df: DataFrame mit Close-Preisen
        window: Fenstergröße für FFT (sollte Potenz von 2 sein)

    Returns:
        DataFrame mit zusätzlichen FFT-Features
    """
    close = df["C"].values
    n = len(close)

    # Initialisiere Arrays für Features
    dominant_freq = np.full(n, np.nan)
    dominant_power = np.full(n, np.nan)
    spectral_energy = np.full(n, np.nan)
    spectral_entropy = np.full(n, np.nan)
    low_freq_ratio = np.full(n, np.nan)

    for i in range(window, n):
        # Fenster extrahieren und detrenden (Mittelwert entfernen)
        segment = close[i - window:i]
        segment_detrended = segment - np.mean(segment)

        # Hanning Window anwenden (reduziert Spectral Leakage)
        windowed = segment_detrended * np.hanning(window)

        # FFT berechnen
        fft_result = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(window)

        # Power Spectrum (Magnitude squared)
        power = np.abs(fft_result) ** 2

        # Ignoriere DC-Komponente (Index 0)
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]

        if len(power_no_dc) == 0 or np.sum(power_no_dc) < 1e-10:
            continue

        # 1. Dominante Frequenz (Index der maximalen Power)
        dom_idx = np.argmax(power_no_dc)
        dominant_freq[i] = freqs_no_dc[dom_idx]
        dominant_power[i] = power_no_dc[dom_idx] / (np.sum(power_no_dc) + 1e-10)

        # 2. Spektrale Energie (normalisiert)
        spectral_energy[i] = np.log1p(np.sum(power_no_dc))

        # 3. Spektrale Entropie (Maß für Verteilung der Energie)
        # Hohe Entropie = gleichmäßig verteilt (Rauschen)
        # Niedrige Entropie = konzentriert auf wenige Frequenzen (klare Zyklen)
        power_norm = power_no_dc / (np.sum(power_no_dc) + 1e-10)
        power_norm = power_norm[power_norm > 1e-10]  # Vermeide log(0)
        spectral_entropy[i] = -np.sum(power_norm * np.log(power_norm))

        # 4. Low-Frequency Ratio (Anteil der Energie in niedrigen Frequenzen)
        # Niedrige Frequenzen = langfristige Trends
        cutoff = len(power_no_dc) // 4
        if cutoff > 0:
            low_freq_ratio[i] = np.sum(power_no_dc[:cutoff]) / (np.sum(power_no_dc) + 1e-10)

    # Features zum DataFrame hinzufügen
    suffix = f"_{window}"
    df[f"fft_dom_freq{suffix}"] = dominant_freq
    df[f"fft_dom_power{suffix}"] = dominant_power
    df[f"fft_energy{suffix}"] = spectral_energy
    df[f"fft_entropy{suffix}"] = spectral_entropy
    df[f"fft_lowfreq{suffix}"] = low_freq_ratio

    return df


def compute_event_features(df):
    """
    Berechnet Time-Since-Event Features.

    Bars seit wichtigen Ereignissen - die KI lernt:
    "Ein Ausbruch nach 100 Bars Seitwärtsphase ist stärker als einer nach 5 Bars."
    """
    # Rolling High/Low für verschiedene Perioden
    for period in [20, 50]:
        rolling_high = df["H"].rolling(period).max()
        rolling_low = df["L"].rolling(period).min()

        # Ist aktueller Preis ein neues High/Low?
        is_new_high = (df["H"] >= rolling_high).astype(int)
        is_new_low = (df["L"] <= rolling_low).astype(int)

        # Bars seit letztem High/Low
        df[f"event_bars_since_high_{period}"] = _bars_since_event(is_new_high)
        df[f"event_bars_since_low_{period}"] = _bars_since_event(is_new_low)

    # Bars seit EMA-Cross (EMA8 kreuzt EMA21)
    ema_8 = ta.trend.ema_indicator(df["C"], window=8)
    ema_21 = ta.trend.ema_indicator(df["C"], window=21)
    ema_cross = ((ema_8 > ema_21) != (ema_8.shift(1) > ema_21.shift(1))).astype(int)
    df["event_bars_since_ema_cross"] = _bars_since_event(ema_cross)

    # Bars seit RSI Extremwert (>70 oder <30)
    rsi = df.get("mom_rsi_14", ta.momentum.rsi(df["C"], window=14))
    rsi_extreme = ((rsi > 70) | (rsi < 30)).astype(int)
    df["event_bars_since_rsi_extreme"] = _bars_since_event(rsi_extreme)

    # Bars seit Volatilitäts-Spike (ATR > 2x Durchschnitt)
    atr = df.get("_atr", ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14))
    atr_mean = atr.rolling(50).mean()
    vol_spike = (atr > 2 * atr_mean).astype(int)
    df["event_bars_since_vol_spike"] = _bars_since_event(vol_spike)

    # Log-Transformation für bessere Skalierung bei großen Werten
    for col in df.columns:
        if col.startswith("event_bars_since_"):
            df[col + "_log"] = np.log1p(df[col])

    return df


def compute_path_efficiency(df):
    """
    Berechnet Path Efficiency / Fractal Dimension Proxy.

    Misst wie "gerade" eine Preisbewegung ist:
    - PE = |Netto-Bewegung| / Summe(|Einzelbewegungen|)
    - 1.0 = perfekt gerade Linie (stark trending)
    - 0.0 = viel Hin und Her (Range/Noise)

    Ähnlich wie Efficiency Ratio, aber robuster berechnet.
    """
    close = df["C"]

    for window in [10, 20, 50, 100]:
        # Netto-Bewegung (Luftlinie von Start zu Ende)
        net_change = abs(close - close.shift(window))

        # Summe aller absoluten Einzelbewegungen (Pfadlänge)
        abs_changes = abs(close.diff())
        path_length = abs_changes.rolling(window).sum()

        # Path Efficiency (0-1)
        pe = net_change / (path_length + 1e-10)
        df[f"path_efficiency_{window}"] = pe

        # Fractal Dimension Proxy: D = 1 + (1 - PE)
        # D ≈ 1: Trending, D ≈ 2: Range/Random
        df[f"fractal_dim_{window}"] = 1 + (1 - pe)

    # Änderung der Path Efficiency (Regime-Shift)
    df["path_efficiency_20_chg"] = df["path_efficiency_20"] - df["path_efficiency_20"].shift(10)
    df["path_efficiency_50_chg"] = df["path_efficiency_50"] - df["path_efficiency_50"].shift(20)

    return df


def compute_convexity_features(df):
    """
    Berechnet Convexity Features (zweite Ableitung des EMA).

    Interpretation:
    - Positiv (konvex): Trend beschleunigt sich (parabolisch)
    - Negativ (konkav): Trend verlangsamt sich (Abflachung)
    - Nahe 0: Linearer Trend

    Frühwarnsystem für parabolische Tops/Bottoms.
    """
    for period in [21, 50]:
        ema = ta.trend.ema_indicator(df["C"], window=period)

        # Erste Ableitung (Geschwindigkeit/Slope)
        ema_slope = ema.diff()

        # Zweite Ableitung (Beschleunigung/Convexity)
        ema_convexity = ema_slope.diff()

        # Normalisiere durch Preis für Vergleichbarkeit
        df[f"convex_ema_{period}"] = ema_convexity / (df["C"] + 1e-10) * 1000

        # Geglättete Version (weniger Noise)
        df[f"convex_ema_{period}_smooth"] = df[f"convex_ema_{period}"].rolling(5).mean()

    # Convexity-Divergenz (kurzfristig vs langfristig)
    df["convex_divergence"] = df["convex_ema_21"] - df["convex_ema_50"]

    # Extremwerte-Indikator (potenzielle Wendepunkte)
    convex_21 = df["convex_ema_21_smooth"]
    convex_std = convex_21.rolling(100).std()
    df["convex_zscore"] = (convex_21 - convex_21.rolling(100).mean()) / (convex_std + 1e-10)

    return df


def compute_vwap_features(df):
    """
    Berechnet VWAP-ähnliche Features (ohne echtes Volume).

    Da wir kein Volume haben, nutzen wir Typical Price als Proxy.
    VWAP ist ein wichtiger Referenzpunkt für institutionelle Trader.
    """
    # Typical Price = (H + L + C) / 3
    tp = (df["H"] + df["L"] + df["C"]) / 3

    # Rolling VWAP-Proxy (ohne Volume-Gewichtung)
    for window in [20, 50, 100]:
        vwap = tp.rolling(window).mean()
        df[f"structure_vwap_dist_{window}"] = (df["C"] - vwap) / vwap

    # Zeit oberhalb/unterhalb VWAP (Acceptance vs Rejection)
    vwap_50 = tp.rolling(50).mean()
    above_vwap = (df["C"] > vwap_50).astype(int)

    # Rolling Ratio: Wie viel Zeit über VWAP?
    df["structure_vwap_time_above"] = above_vwap.rolling(20).mean()

    # VWAP Cross (kürzlich gekreuzt?)
    vwap_cross = (above_vwap != above_vwap.shift(1)).astype(int)
    df["structure_bars_since_vwap_cross"] = _bars_since_event(vwap_cross)

    return df
