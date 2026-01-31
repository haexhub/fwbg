"""
Core Indicator Computations.

Enthält die Haupt-Funktion compute_indicator_pool() sowie
alle Standard-Indikatoren (Trend, Momentum, Volatility, etc.).
"""
import numpy as np
import pandas as pd
import ta

from .regime import compute_rolling_hurst, compute_regime_features
from .structure import (
    compute_fft_features,
    compute_event_features,
    compute_path_efficiency,
    compute_convexity_features,
    compute_vwap_features,
)
from .risk import (
    compute_drawdown_features,
    compute_cvar_features,
    compute_vol_of_vol_features,
    compute_crash_probability_features,
    compute_correlation_features,
)


def compute_indicator_pool(df, symbol: str = None):
    """
    Berechnet erweiterten Indikator-Pool für echte Feature Selection.
    ATR wird separat berechnet (nur für TP/SL Sizing, nicht als Feature).

    Args:
        df: DataFrame mit OHLC-Daten
        symbol: Optional - Symbol für Progress-UI Updates (z.B. "EURUSD")

    Returns:
        DataFrame mit allen berechneten Features
    """
    import time
    from ..progress import report_phase

    def _phase(msg):
        """Meldet Phase an Progress-UI falls Symbol angegeben."""
        if symbol:
            report_phase(symbol, f"Indikatoren: {msg}")

    t_start = time.time()
    n_rows = len(df)

    # ATR separat für Sizing (nicht im Feature-Pool)
    df["_atr"] = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)

    # === TREND INDIKATOREN ===
    _phase("Trend (ADX, EMA, MACD)")
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
    for period in [10, 20, 50]:
        change = abs(df["C"] - df["C"].shift(period))
        volatility = abs(df["C"].diff()).rolling(period).sum()
        df[f"trend_er_{period}"] = change / (volatility + 1e-10)

    df["trend_er_10_chg"] = df["trend_er_10"] - df["trend_er_10"].shift(5)
    df["trend_er_20_chg"] = df["trend_er_20"] - df["trend_er_20"].shift(10)

    # === MOMENTUM INDIKATOREN ===
    _phase("Momentum (RSI, Stochastic)")
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
    _phase("Volatilität (Bollinger, ATR)")
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
    returns = df["C"].pct_change()

    for period in [20, 50, 100]:
        df[f"dist_skew_{period}"] = returns.rolling(period).skew()
        df[f"dist_kurt_{period}"] = returns.rolling(period).kurt()

    for period in [20, 50]:
        skew_col = f"dist_skew_{period}"
        kurt_col = f"dist_kurt_{period}"
        df[f"dist_skew_{period}_z"] = (
            (df[skew_col] - df[skew_col].rolling(200).mean()) /
            (df[skew_col].rolling(200).std() + 1e-10)
        )
        df[f"dist_kurt_{period}_z"] = (
            (df[kurt_col] - df[kurt_col].rolling(200).mean()) /
            (df[kurt_col].rolling(200).std() + 1e-10)
        )

    # === FFT FEATURES ===
    _phase("FFT (64/128/256)")
    for window in [64, 128, 256]:
        df = compute_fft_features(df, window)

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
    _phase("Multi-Timeframe (H4, D1)")
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

    # === REGIME FEATURES ===
    _phase("Regime (Hurst)")
    df = compute_regime_features(df)

    # === EVENT FEATURES ===
    _phase("Event-Features")
    df = compute_event_features(df)

    # === PATH EFFICIENCY / FRACTAL DIMENSION ===
    _phase("Path-Efficiency")
    df = compute_path_efficiency(df)

    # === CONVEXITY FEATURES ===
    _phase("Convexity")
    df = compute_convexity_features(df)

    # === CORRELATION FEATURES ===
    _phase("Correlation (SPX, VIX)")
    df = compute_correlation_features(df)

    # === RISK FEATURES ===
    _phase("Risk (Drawdown, CVaR)")
    df = compute_drawdown_features(df)
    df = compute_cvar_features(df)
    df = compute_vol_of_vol_features(df)
    df = compute_vwap_features(df)
    df = compute_crash_probability_features(df)

    elapsed = time.time() - t_start
    _phase(f"Fertig ({len(df.columns)} Features)")

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
    from ..config import FEATURE_GROUPS

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
