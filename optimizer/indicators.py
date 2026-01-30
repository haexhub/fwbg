"""
Technische Indikatoren für den Optimizer
"""
import numpy as np
import pandas as pd
import ta
from statsmodels.tsa.stattools import adfuller


def get_frac_diff_weights(d: float, size: int, threshold: float = 1e-4) -> np.ndarray:
    """
    Berechnet die Gewichte für Fractional Differentiation.

    Args:
        d: Differenzierungsordnung (0 < d < 1)
        size: Maximale Anzahl der Gewichte
        threshold: Abbruchschwelle für kleine Gewichte (höher = kürzeres Lookback)

    Returns:
        Array mit Gewichten (absteigend)
    """
    weights = [1.0]
    for k in range(1, size):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.array(weights[::-1])


def frac_diff(series: np.ndarray, d: float, threshold: float = 1e-4) -> np.ndarray:
    """
    Wendet Fractional Differentiation auf eine Zeitreihe an.

    Args:
        series: Input-Zeitreihe
        d: Differenzierungsordnung (0 < d < 1)
        threshold: Abbruchschwelle für Gewichte (höher = weniger NaN)

    Returns:
        Fractionally differenzierte Serie (mit NaN am Anfang)
    """
    weights = get_frac_diff_weights(d, len(series), threshold)
    width = len(weights)

    result = np.full(len(series), np.nan)
    for i in range(width - 1, len(series)):
        result[i] = np.dot(weights, series[i - width + 1:i + 1])

    return result


def find_min_d_for_stationarity(series: np.ndarray,
                                 max_d: float = 1.0,
                                 p_value_threshold: float = 0.05,
                                 step: float = 0.05) -> float:
    """
    Findet den minimalen d-Wert der Stationarität erreicht (ADF-Test).

    Args:
        series: Input-Zeitreihe (ohne NaN)
        max_d: Maximaler d-Wert zum Testen
        p_value_threshold: p-Wert Schwelle für Stationarität
        step: Schrittgröße für d-Suche

    Returns:
        Optimaler d-Wert
    """
    # Entferne NaN für die Suche
    series_clean = series[~np.isnan(series)]
    if len(series_clean) < 100:
        return 0.5  # Default wenn zu wenig Daten

    for d in np.arange(step, max_d + step, step):
        diff_series = frac_diff(series_clean, d)
        # Entferne NaN vom Anfang
        diff_clean = diff_series[~np.isnan(diff_series)]

        if len(diff_clean) < 50:
            continue

        try:
            adf_result = adfuller(diff_clean, maxlag=1)
            p_value = adf_result[1]

            if p_value < p_value_threshold:
                return round(d, 2)
        except Exception:
            continue

    return max_d  # Fallback auf volle Differenzierung


def apply_frac_diff_preprocessing(df, use_auto_d=True, default_d=0.4):
    """
    Wendet Fractional Differentiation als Preprocessing auf OHLC-Daten an.
    Transformiert die Preise zu stationären Werten bei Erhalt von Memory.

    Nach López de Prado: Diese transformierten Daten sollten für alle
    weiteren Indikator-Berechnungen verwendet werden.

    Args:
        df: DataFrame mit OHLC Daten
        use_auto_d: Automatische d-Optimierung via ADF-Test
        default_d: Standard d-Wert wenn auto_d=False

    Returns:
        Tuple (DataFrame mit transformierten OHLC, d-Wert)
    """
    # Speichere Original-Close für spätere Referenz
    df["_original_close"] = df["C"].copy()

    # Finde optimalen d-Wert basierend auf Close-Preis
    if use_auto_d:
        sample_size = min(5000, len(df))
        sample = df["C"].values[-sample_size:]
        d = find_min_d_for_stationarity(sample)
    else:
        d = default_d

    # Transformiere OHLC
    for col in ["O", "H", "L", "C"]:
        if col in df.columns:
            df[col] = frac_diff(df[col].values, d)

    # Entferne NaN-Zeilen vom Anfang (Warmup-Periode)
    first_valid = df["C"].first_valid_index()
    if first_valid is not None:
        df = df.loc[first_valid:]

    return df, d


def apply_log_returns_preprocessing(df):
    """
    Transformiert OHLC-Daten zu Log-Returns.

    Log-Returns sind approximativ normalverteilt und additiv über Zeit,
    was für ML-Modelle oft besser funktioniert.

    Args:
        df: DataFrame mit OHLC Daten

    Returns:
        DataFrame mit transformierten OHLC
    """
    # Speichere Original-Close für spätere Referenz
    df["_original_close"] = df["C"].copy()

    # Berechne Log-Returns für OHLC
    for col in ["O", "H", "L", "C"]:
        if col in df.columns:
            df[col] = np.log(df[col] / df[col].shift(1))

    # Erste Zeile hat NaN durch shift
    df = df.iloc[1:]

    return df


def apply_normalize_preprocessing(df, window=100):
    """
    Normalisiert OHLC-Daten mit Rolling Z-Score.

    Args:
        df: DataFrame mit OHLC Daten
        window: Rolling-Window für Mean/Std Berechnung

    Returns:
        DataFrame mit normalisierten OHLC
    """
    # Speichere Original-Close für spätere Referenz
    if "_original_close" not in df.columns:
        df["_original_close"] = df["C"].copy()

    # Z-Score Normalisierung für OHLC
    for col in ["O", "H", "L", "C"]:
        if col in df.columns:
            rolling_mean = df[col].rolling(window).mean()
            rolling_std = df[col].rolling(window).std()
            df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-10)

    # Entferne Warmup-Periode
    df = df.iloc[window:]

    return df


def apply_preprocessing(df, preprocessing_params):
    """
    Wendet alle konfigurierten Preprocessing-Schritte auf OHLC-Daten an.

    Reihenfolge:
    1. Fractional Differentiation (falls aktiviert)
    2. Log-Returns (falls aktiviert)
    3. Normalisierung (falls aktiviert)

    Args:
        df: DataFrame mit OHLC Daten
        preprocessing_params: PreprocessingParams Objekt

    Returns:
        Tuple (DataFrame, preprocessing_info dict)
    """
    info = {"applied": [], "frac_diff_d": None}

    # 1. Fractional Differentiation
    if preprocessing_params.fractional_differentiation:
        df, d = apply_frac_diff_preprocessing(
            df,
            use_auto_d=preprocessing_params.frac_diff_auto_d,
            default_d=preprocessing_params.frac_diff_default_d
        )
        info["applied"].append("fractional_differentiation")
        info["frac_diff_d"] = d

    # 2. Log-Returns
    if preprocessing_params.log_returns:
        df = apply_log_returns_preprocessing(df)
        info["applied"].append("log_returns")

    # 3. Normalisierung
    if preprocessing_params.normalize:
        df = apply_normalize_preprocessing(df, preprocessing_params.normalize_window)
        info["applied"].append(f"normalize_{preprocessing_params.normalize_window}")

    return df, info


def _compute_fft_features(df, window: int):
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


def compute_indicator_pool(df):
    """
    Berechnet erweiterten Indikator-Pool für echte Feature Selection.
    ATR wird separat berechnet (nur für TP/SL Sizing, nicht als Feature).
    """
    # ATR separat für Sizing (nicht im Feature-Pool)
    df["_atr"] = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)

    # === TREND INDIKATOREN ===
    for period in [7, 14, 21]:
        df[f"trend_adx_{period}"] = ta.trend.adx(df["H"], df["L"], df["C"], window=period)

    for period in [8, 21, 50, 100, 200]:
        ema = ta.trend.ema_indicator(df["C"], window=period)
        df[f"trend_ema_dist_{period}"] = (df["C"] - ema) / df["C"]

    for period in [20, 50, 200]:
        sma = ta.trend.sma_indicator(df["C"], window=period)
        df[f"trend_sma_dist_{period}"] = (df["C"] - sma) / df["C"]

    macd = ta.trend.MACD(df["C"])
    df["trend_macd"] = macd.macd_diff() / df["C"]
    df["trend_macd_signal"] = macd.macd_signal() / df["C"]

    for period in [14, 20]:
        df[f"trend_cci_{period}"] = ta.trend.cci(df["H"], df["L"], df["C"], window=period)

    aroon = ta.trend.AroonIndicator(df["H"], df["L"], window=25)
    df["trend_aroon_up"] = aroon.aroon_up()
    df["trend_aroon_down"] = aroon.aroon_down()

    # === KAUFMAN'S EFFICIENCY RATIO ===
    # ER = |Change| / Sum(|Daily Changes|)
    # ER nahe 1 = starker Trend, ER nahe 0 = Seitwärtsbewegung/Noise
    for period in [10, 20, 50]:
        # Netto-Bewegung (Richtung)
        change = abs(df["C"] - df["C"].shift(period))
        # Summe aller absoluten Tagesbewegungen (Volatilität/Noise)
        volatility = abs(df["C"].diff()).rolling(period).sum()
        # Efficiency Ratio
        df[f"trend_er_{period}"] = change / (volatility + 1e-10)

    # ER Change (Momentum des ER selbst)
    df["trend_er_10_chg"] = df["trend_er_10"] - df["trend_er_10"].shift(5)
    df["trend_er_20_chg"] = df["trend_er_20"] - df["trend_er_20"].shift(10)

    # === MOMENTUM INDIKATOREN ===
    for period in [7, 14, 21]:
        df[f"mom_rsi_{period}"] = ta.momentum.rsi(df["C"], window=period)

    for period in [14, 21]:
        stoch = ta.momentum.StochasticOscillator(df["H"], df["L"], df["C"], window=period)
        df[f"mom_stoch_k_{period}"] = stoch.stoch()
        df[f"mom_stoch_d_{period}"] = stoch.stoch_signal()

    for period in [14, 21]:
        df[f"mom_williams_{period}"] = ta.momentum.williams_r(df["H"], df["L"], df["C"], lbp=period)

    df["mom_uo"] = ta.momentum.ultimate_oscillator(df["H"], df["L"], df["C"])

    for period in [5, 10, 20]:
        df[f"mom_roc_{period}"] = ta.momentum.roc(df["C"], window=period)

    # === VOLATILITÄT INDIKATOREN ===
    for period in [20]:
        bb = ta.volatility.BollingerBands(df["C"], window=period)
        df[f"vol_bb_pband_{period}"] = bb.bollinger_pband()
        df[f"vol_bb_wband_{period}"] = bb.bollinger_wband()

    kc = ta.volatility.KeltnerChannel(df["H"], df["L"], df["C"])
    df["vol_kc_pband"] = kc.keltner_channel_pband()
    df["vol_kc_wband"] = kc.keltner_channel_wband()

    dc = ta.volatility.DonchianChannel(df["H"], df["L"], df["C"])
    df["vol_dc_pband"] = dc.donchian_channel_pband()
    df["vol_dc_wband"] = dc.donchian_channel_wband()

    for period in [7, 14, 21]:
        atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)
        df[f"vol_atr_pct_{period}"] = atr / df["C"]

    df["vol_atr"] = df["_atr"]

    # === DISTRIBUTION FEATURES (Skewness & Kurtosis) ===
    # Rolling Returns für Verteilungs-Analyse
    returns = df["C"].pct_change()

    for period in [20, 50, 100]:
        # Rolling Skewness: Asymmetrie der Return-Verteilung
        # Positiv = mehr positive Ausreisser, Negativ = mehr negative Ausreisser
        df[f"dist_skew_{period}"] = returns.rolling(period).skew()

        # Rolling Kurtosis: Schwere der Tails
        # Hoch = Fat Tails (mehr Extremereignisse), Niedrig = dünne Tails
        df[f"dist_kurt_{period}"] = returns.rolling(period).kurt()

    # Normalisierte Versionen (z-Score über längere Periode)
    for period in [20, 50]:
        skew_col = f"dist_skew_{period}"
        kurt_col = f"dist_kurt_{period}"

        # Z-Score der Skewness/Kurtosis relativ zur Historie
        df[f"dist_skew_{period}_z"] = (
            (df[skew_col] - df[skew_col].rolling(200).mean()) /
            (df[skew_col].rolling(200).std() + 1e-10)
        )
        df[f"dist_kurt_{period}_z"] = (
            (df[kurt_col] - df[kurt_col].rolling(200).mean()) /
            (df[kurt_col].rolling(200).std() + 1e-10)
        )

    # === FFT (FAST FOURIER TRANSFORMATION) FEATURES ===
    # Extrahiert dominante Frequenzen/Zyklen aus der Preisbewegung
    for window in [64, 128, 256]:
        df = _compute_fft_features(df, window)

    # === VOLUME INDIKATOREN ===
    if "V" in df.columns or "Volume" in df.columns:
        vol_col = "V" if "V" in df.columns else "Volume"
        obv = ta.volume.on_balance_volume(df["C"], df[vol_col])
        df["vol_obv_change"] = obv.pct_change(periods=5)
        df["vol_mfi"] = ta.volume.money_flow_index(df["H"], df["L"], df["C"], df[vol_col])

    # === PRICE ACTION FEATURES ===
    df["pa_range_pos"] = (df["C"] - df["L"]) / (df["H"] - df["L"] + 1e-10)
    df["pa_hh"] = (df["H"] > df["H"].shift(1)).astype(int).rolling(5).sum()
    df["pa_ll"] = (df["L"] < df["L"].shift(1)).astype(int).rolling(5).sum()
    df["pa_body_ratio"] = abs(df["C"] - df["O"]) / (df["H"] - df["L"] + 1e-10)
    df["pa_gap"] = (df["O"] - df["C"].shift(1)) / df["C"].shift(1)

    # === ICHIMOKU CLOUD ===
    ichimoku = ta.trend.IchimokuIndicator(df["H"], df["L"])
    df["ichi_tenkan"] = ichimoku.ichimoku_conversion_line()
    df["ichi_kijun"] = ichimoku.ichimoku_base_line()
    df["ichi_senkou_a"] = ichimoku.ichimoku_a()
    df["ichi_senkou_b"] = ichimoku.ichimoku_b()

    cloud_top = df[["ichi_senkou_a", "ichi_senkou_b"]].max(axis=1)
    cloud_bottom = df[["ichi_senkou_a", "ichi_senkou_b"]].min(axis=1)
    cloud_thickness = (cloud_top - cloud_bottom) / df["C"]
    df["ichi_cloud_pos"] = (df["C"] - cloud_bottom) / (cloud_top - cloud_bottom + 1e-10)
    df["ichi_cloud_thick"] = cloud_thickness
    df["ichi_tk_cross"] = (df["ichi_tenkan"] - df["ichi_kijun"]) / df["C"]
    df["ichi_price_kijun"] = (df["C"] - df["ichi_kijun"]) / df["C"]

    # === ZEIT FEATURES ===
    df["time_hour"] = df.index.hour
    df["time_day"] = df.index.dayofweek
    df["time_hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["time_hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    # === SAISONALITÄT FEATURES ===
    df["season_month"] = df.index.month
    df["season_month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["season_month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["season_quarter"] = df.index.quarter
    df["season_quarter_sin"] = np.sin(2 * np.pi * df.index.quarter / 4)
    df["season_quarter_cos"] = np.cos(2 * np.pi * df.index.quarter / 4)
    df["season_week"] = df.index.isocalendar().week.astype(int)
    df["season_week_sin"] = np.sin(2 * np.pi * df["season_week"] / 52)
    df["season_week_cos"] = np.cos(2 * np.pi * df["season_week"] / 52)
    df["season_dayofmonth"] = df.index.day
    df["season_dayofmonth_sin"] = np.sin(2 * np.pi * df.index.day / 31)
    df["season_dayofmonth_cos"] = np.cos(2 * np.pi * df.index.day / 31)

    # === DYNAMIK FEATURES ===
    for lookback in [4, 8, 24]:
        df[f"dyn_rsi14_chg_{lookback}h"] = df["mom_rsi_14"] - df["mom_rsi_14"].shift(lookback)
        df[f"dyn_rsi14_pct_{lookback}h"] = df["mom_rsi_14"].pct_change(lookback) * 100

    for lookback in [4, 8, 24]:
        df[f"dyn_atr_chg_{lookback}h"] = df["vol_atr_pct_14"].pct_change(lookback) * 100
        df[f"dyn_bbwidth_chg_{lookback}h"] = df["vol_bb_wband_20"].pct_change(lookback) * 100

    for lookback in [4, 8, 24]:
        df[f"dyn_adx_chg_{lookback}h"] = df["trend_adx_14"] - df["trend_adx_14"].shift(lookback)

    for lookback in [4, 8]:
        df[f"dyn_macd_chg_{lookback}h"] = df["trend_macd"] - df["trend_macd"].shift(lookback)

    for lookback in [4, 8]:
        df[f"dyn_stoch_chg_{lookback}h"] = df["mom_stoch_k_14"] - df["mom_stoch_k_14"].shift(lookback)

    # === LAG FEATURES ===
    for lag in [4, 8, 24]:
        df[f"lag_rsi14_{lag}h"] = df["mom_rsi_14"].shift(lag)

    for lag in [4, 8, 24]:
        df[f"lag_atr_{lag}h"] = df["vol_atr_pct_14"].shift(lag)

    for lag in [4, 8]:
        df[f"lag_adx_{lag}h"] = df["trend_adx_14"].shift(lag)

    for lag in [4, 8, 24, 48]:
        df[f"lag_price_chg_{lag}h"] = (df["C"] - df["C"].shift(lag)) / df["C"].shift(lag) * 100

    # === MOMENTUM BESCHLEUNIGUNG ===
    df["accel_rsi"] = df["dyn_rsi14_chg_4h"] - df["dyn_rsi14_chg_4h"].shift(4)
    df["accel_atr"] = df["dyn_atr_chg_4h"] - df["dyn_atr_chg_4h"].shift(4)

    # === CROSS-INDIKATOR FEATURES ===
    df["cross_rsi_high_rising"] = ((df["mom_rsi_14"] > 70) & (df["dyn_rsi14_chg_4h"] > 0)).astype(int)
    df["cross_rsi_low_falling"] = ((df["mom_rsi_14"] < 30) & (df["dyn_rsi14_chg_4h"] < 0)).astype(int)
    df["cross_vol_trend"] = df["dyn_atr_chg_4h"] * df["trend_adx_14"] / 100

    # === MULTI-TIMEFRAME FEATURES ===
    h4_high = df["H"].rolling(4).max()
    h4_low = df["L"].rolling(4).min()
    h4_close = df["C"]
    h4_open = df["O"].shift(3)

    df["mtf_h4_trend"] = (h4_close - h4_open) / (h4_high - h4_low + 1e-10)
    df["mtf_h4_range_pos"] = (df["C"] - h4_low) / (h4_high - h4_low + 1e-10)

    h4_ema_20 = ta.trend.ema_indicator(df["C"], window=20*4)
    h4_ema_50 = ta.trend.ema_indicator(df["C"], window=50*4)
    df["mtf_h4_ema20_dist"] = (df["C"] - h4_ema_20) / df["C"]
    df["mtf_h4_ema50_dist"] = (df["C"] - h4_ema_50) / df["C"]

    h1_trend = df["trend_ema_dist_21"]
    h4_trend = df["mtf_h4_ema20_dist"]
    df["mtf_trend_alignment"] = (np.sign(h1_trend) == np.sign(h4_trend)).astype(int)

    df["mtf_h4_adx"] = ta.trend.adx(h4_high, h4_low, df["C"], window=14)
    df["mtf_h4_rsi"] = ta.momentum.rsi(df["C"], window=14*4)
    df["mtf_h4_atr_pct"] = ta.volatility.average_true_range(h4_high, h4_low, df["C"], window=14) / df["C"]
    df["mtf_vol_ratio"] = df["vol_atr_pct_14"] / (df["mtf_h4_atr_pct"] + 1e-10)

    d1_high = df["H"].rolling(24).max()
    d1_low = df["L"].rolling(24).min()
    df["mtf_d1_range_pos"] = (df["C"] - d1_low) / (d1_high - d1_low + 1e-10)

    d1_ema_20 = ta.trend.ema_indicator(df["C"], window=20*24)
    df["mtf_d1_ema20_dist"] = (df["C"] - d1_ema_20) / df["C"]

    d1_trend = df["mtf_d1_ema20_dist"]
    df["mtf_consensus"] = (
        (np.sign(h1_trend) == np.sign(h4_trend)) &
        (np.sign(h4_trend) == np.sign(d1_trend))
    ).astype(int)

    # === REGIME FEATURES (Markt-Charakter) ===
    # Hurst-Exponent: H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random
    # Verwende Original-Close falls Frac-Diff aktiv (sonst sind Preise transformiert)
    close_for_hurst = df["_original_close"].values if "_original_close" in df.columns else df["C"].values

    # Mehrere Fenstergrößen für unterschiedliche Zeitskalen
    for window in [100, 200, 500]:
        hurst_values = compute_rolling_hurst(close_for_hurst, window=window, step=10)
        df[f"regime_hurst_{window}"] = hurst_values

    # Hurst-Änderung (Regime-Shift Detektion)
    df["regime_hurst_100_chg"] = df["regime_hurst_100"] - df["regime_hurst_100"].shift(24)
    df["regime_hurst_200_chg"] = df["regime_hurst_200"] - df["regime_hurst_200"].shift(48)

    # Hurst Divergenz zwischen Zeitskalen (kurzfristig vs langfristig)
    df["regime_hurst_divergence"] = df["regime_hurst_100"] - df["regime_hurst_500"]

    return df


def get_feature_columns(df):
    """Gibt alle Feature-Spalten zurück (ohne interne Spalten wie _atr)."""
    exclude = ["O", "H", "L", "C", "V", "Volume", "_atr", "_regime_ok"]
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def filter_features_by_group(all_features, group_name):
    """
    Filtert Features nach einer Feature-Gruppe aus FEATURE_GROUPS.

    Args:
        all_features: Liste aller verfügbaren Features
        group_name: Name der Gruppe (z.B. "trend", "momentum", "trend_momentum")

    Returns:
        Liste der Features die zur Gruppe gehören
    """
    from .config import FEATURE_GROUPS

    if group_name not in FEATURE_GROUPS:
        return all_features  # Fallback: alle Features

    group = FEATURE_GROUPS[group_name]
    prefixes = group["prefixes"]

    filtered = []
    for feat in all_features:
        for prefix in prefixes:
            if feat.startswith(prefix):
                filtered.append(feat)
                break

    return filtered


def compute_hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    Berechnet den Hurst-Exponenten mittels R/S (Rescaled Range) Analyse.

    Interpretation:
    - H > 0.5: Trending/Persistent (gute Bedingungen für Trend-Following)
    - H = 0.5: Random Walk (schwierig zu traden)
    - H < 0.5: Mean-Reverting (gute Bedingungen für Mean-Reversion)

    Args:
        series: Preis-Zeitreihe
        max_lag: Maximale Lag-Größe für R/S Analyse

    Returns:
        Hurst-Exponent (0-1)
    """
    if len(series) < max_lag * 2:
        return 0.5  # Default bei zu wenig Daten

    # Verwende Log-Returns für bessere Skalierung
    returns = np.diff(np.log(series + 1e-10))
    returns = returns[~np.isnan(returns)]

    if len(returns) < max_lag:
        return 0.5

    lags = range(10, min(max_lag, len(returns) // 4))
    rs_values = []
    lag_values = []

    for lag in lags:
        # Teile Serie in Subseries
        n_subseries = len(returns) // lag
        if n_subseries < 2:
            continue

        rs_lag = []
        for i in range(n_subseries):
            subseries = returns[i * lag:(i + 1) * lag]
            if len(subseries) < 2:
                continue

            # Berechne kumulative Abweichung vom Mittelwert
            mean_val = np.mean(subseries)
            cumdev = np.cumsum(subseries - mean_val)

            # Range und Standardabweichung
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(subseries, ddof=1)

            if s > 1e-10:
                rs_lag.append(r / s)

        if rs_lag:
            rs_values.append(np.mean(rs_lag))
            lag_values.append(lag)

    if len(lag_values) < 3:
        return 0.5

    # Log-Log Regression für Hurst
    log_lags = np.log(lag_values)
    log_rs = np.log(rs_values)

    # Lineare Regression
    slope, _ = np.polyfit(log_lags, log_rs, 1)

    # Hurst = Slope, begrenzt auf [0, 1]
    return float(np.clip(slope, 0.0, 1.0))


def compute_rolling_hurst(series: np.ndarray, window: int = 100, step: int = 10) -> np.ndarray:
    """
    Berechnet Rolling Hurst-Exponent.

    Args:
        series: Preis-Zeitreihe
        window: Fenstergröße für Hurst-Berechnung
        step: Schrittgröße für effizientere Berechnung

    Returns:
        Array mit Hurst-Werten (NaN am Anfang)
    """
    result = np.full(len(series), np.nan)

    for i in range(window, len(series), step):
        window_data = series[i - window:i]
        h = compute_hurst_exponent(window_data, max_lag=min(50, window // 4))

        # Fülle alle Werte bis zum nächsten Step
        end_idx = min(i + step, len(series))
        result[i:end_idx] = h

    # Forward-fill für Lücken
    for i in range(1, len(result)):
        if np.isnan(result[i]) and not np.isnan(result[i - 1]):
            result[i] = result[i - 1]

    return result


def compute_regime_filter(df, regime_params=None):
    """
    Berechnet Regime-Filter basierend auf konfigurierbaren Bedingungen.

    Args:
        df: DataFrame mit Indikatoren
        regime_params: Optional RegimeFilterParams mit Konfiguration

    Returns:
        Boolean Series (True = Trading erlaubt)
    """
    # Default: Kein Filter aktiv (alle Trades erlaubt)
    adx_min = 0
    vix_max = None
    hurst_min = None
    hurst_max = None

    if regime_params is not None:
        adx_min = regime_params.adx_min if regime_params.adx_enabled else 0
        if regime_params.vix_enabled:
            vix_max = regime_params.vix_max
        if regime_params.hurst_enabled:
            hurst_min = regime_params.hurst_min
            hurst_max = regime_params.hurst_max

    # ADX Filter (adx_min=0 bedeutet kein Filter)
    if adx_min > 0:
        adx_14 = df.get("trend_adx_14", ta.trend.adx(df["H"], df["L"], df["C"], window=14))
        regime_ok = adx_14 >= adx_min
    else:
        # Kein ADX-Filter - alle Bars erlaubt
        regime_ok = pd.Series(True, index=df.index)

    # VIX Filter (nur wenn explizit konfiguriert)
    if vix_max is not None and "sent_vix" in df.columns:
        vix_ok = df["sent_vix"] < vix_max
        regime_ok = regime_ok & vix_ok

    # Hurst Filter
    if hurst_min is not None or hurst_max is not None:
        if "_hurst" not in df.columns:
            # Berechne Hurst wenn nicht vorhanden
            close_values = df["C"].values if "_original_close" not in df.columns else df["_original_close"].values
            df["_hurst"] = compute_rolling_hurst(close_values, window=100, step=10)

        if hurst_min is not None:
            regime_ok = regime_ok & (df["_hurst"] >= hurst_min)
        if hurst_max is not None:
            regime_ok = regime_ok & (df["_hurst"] <= hurst_max)

    return regime_ok
