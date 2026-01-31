"""
Regime-Detection Features.

Enthält:
- Hurst-Exponent (Trending vs Mean-Reverting)
- Regime-Filter
"""
import numpy as np
import pandas as pd
import ta


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


def compute_regime_features(df):
    """
    Berechnet Regime-bezogene Features.

    Args:
        df: DataFrame mit OHLC-Daten

    Returns:
        DataFrame mit zusätzlichen Regime-Features
    """
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
