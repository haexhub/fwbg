"""
Risk/Tail-Risk Features.

Enthält:
- Drawdown State Features
- CVaR (Conditional Value at Risk / Expected Shortfall)
- Vol-of-Vol (Volatility of Volatility)
- Crash Probability Proxy
- Correlation Features
"""
import numpy as np
import pandas as pd
import ta

from .structure import _bars_since_event


def compute_drawdown_features(df):
    """
    Berechnet Drawdown State Features.

    Extrem wichtig für Risk Management:
    - Wie weit sind wir vom Peak entfernt?
    - Wie lange dauert der Drawdown schon?
    """
    close = df["C"]

    # Rolling Maximum (Peak) über verschiedene Fenster
    for window in [50, 100, 200]:
        rolling_max = close.rolling(window, min_periods=1).max()

        # Aktueller Drawdown in %
        dd_pct = (close - rolling_max) / rolling_max
        df[f"risk_dd_pct_{window}"] = dd_pct

        # Drawdown Ratio: Aktueller DD / Max DD des Fensters
        # 0 = kein DD, 1 = am Maximum DD
        rolling_min_dd = dd_pct.rolling(window, min_periods=1).min()
        df[f"risk_dd_ratio_{window}"] = dd_pct / (rolling_min_dd - 1e-10)

    # Time since peak (Bars seit letztem High)
    rolling_max_200 = close.rolling(200, min_periods=1).max()
    is_at_peak = (close >= rolling_max_200).astype(int)
    df["risk_bars_since_peak"] = _bars_since_event(is_at_peak)
    df["risk_bars_since_peak_log"] = np.log1p(df["risk_bars_since_peak"])

    # Recovery Ratio: Wie viel vom letzten DD haben wir erholt?
    # Nützlich um zu erkennen ob wir in einer Erholung sind
    dd_200 = df["risk_dd_pct_200"]
    rolling_min_close = close.rolling(50, min_periods=1).min()
    recovery = (close - rolling_min_close) / (rolling_max_200 - rolling_min_close + 1e-10)
    df["risk_recovery_ratio"] = recovery.clip(0, 1)

    return df


def compute_cvar_features(df):
    """
    Berechnet Rolling CVaR (Conditional Value at Risk / Expected Shortfall).

    CVaR = Erwarteter Verlust gegeben dass wir im Tail sind.
    Besser als VaR weil es die Größe der Tail-Verluste berücksichtigt.
    """
    returns = df["C"].pct_change()

    for window in [50, 100]:
        for percentile in [5, 1]:  # 95% und 99% CVaR
            # Rolling VaR (Value at Risk)
            var = returns.rolling(window).quantile(percentile / 100)
            df[f"risk_var_{percentile}_{window}"] = var

            # CVaR = Durchschnitt aller Returns unter dem VaR
            # Approximation: Durchschnitt der schlechtesten X%
            def calc_cvar(x, pct=percentile):
                if len(x) < 10:
                    return np.nan
                threshold = np.percentile(x, pct)
                tail_returns = x[x <= threshold]
                return tail_returns.mean() if len(tail_returns) > 0 else threshold

            cvar = returns.rolling(window).apply(calc_cvar, raw=True)
            df[f"risk_cvar_{percentile}_{window}"] = cvar

    # CVaR Ratio: CVaR1 / CVaR5 - wie viel schlimmer sind extreme Events?
    df["risk_cvar_tail_ratio"] = df["risk_cvar_1_100"] / (df["risk_cvar_5_100"] + 1e-10)

    # CVaR Change: Verschlechtert sich das Tail-Risk?
    df["risk_cvar_5_change"] = df["risk_cvar_5_100"] - df["risk_cvar_5_100"].shift(20)

    return df


def compute_vol_of_vol_features(df):
    """
    Berechnet Volatility-of-Volatility Features.

    Vol-of-Vol = Wie stabil ist die Volatilität selbst?
    Hohe Vol-of-Vol = Regime-Unsicherheit = Vorsicht geboten.
    """
    # ATR Change als Basis
    atr = df.get("_atr", ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14))
    atr_change = atr.pct_change()

    # Rolling Std der ATR-Changes = Vol-of-Vol
    for window in [20, 50, 100]:
        df[f"risk_vol_of_vol_{window}"] = atr_change.rolling(window).std()

    # Vol-of-Vol Z-Score (relativ zur eigenen Historie)
    vov_100 = df["risk_vol_of_vol_100"]
    vov_mean = vov_100.rolling(200).mean()
    vov_std = vov_100.rolling(200).std()
    df["risk_vol_of_vol_zscore"] = (vov_100 - vov_mean) / (vov_std + 1e-10)

    # Vol-of-Vol Regime: Ist Vol-of-Vol steigend oder fallend?
    df["risk_vol_of_vol_trend"] = df["risk_vol_of_vol_50"] - df["risk_vol_of_vol_50"].shift(10)

    return df


def compute_crash_probability_features(df):
    """
    Berechnet Crash Probability Proxy.

    Kombiniert mehrere Warnsignale:
    - Hohe Kurtosis (Fat Tails)
    - Steigende Vol-of-Vol
    - Correlation Decoupling
    - Extreme CVaR

    Ein hoher Wert bedeutet erhöhte Crash-Wahrscheinlichkeit.
    """
    # Stelle sicher dass benötigte Features existieren
    has_kurt = "dist_kurt_50" in df.columns
    has_vov = "risk_vol_of_vol_zscore" in df.columns
    has_decoupling = "corr_spx_decoupling" in df.columns
    has_cvar = "risk_cvar_5_100" in df.columns

    # Initialisiere Crash Score
    crash_score = pd.Series(0.0, index=df.index)
    n_components = 0

    if has_kurt:
        # Hohe Kurtosis = mehr Extremereignisse
        kurt_zscore = df["dist_kurt_50_z"] if "dist_kurt_50_z" in df.columns else (
            (df["dist_kurt_50"] - df["dist_kurt_50"].rolling(100).mean()) /
            (df["dist_kurt_50"].rolling(100).std() + 1e-10)
        )
        crash_score += kurt_zscore.clip(0, 3) / 3  # 0-1, nur positive (Fat Tails)
        n_components += 1

    if has_vov:
        # Hohe Vol-of-Vol = instabiles Regime
        vov_contrib = df["risk_vol_of_vol_zscore"].clip(0, 3) / 3
        crash_score += vov_contrib
        n_components += 1

    if has_decoupling:
        # Decoupling = Korrelationsbruch = Warnsignal
        decoupling_norm = df["corr_spx_decoupling"] / (df["corr_spx_decoupling"].rolling(100).max() + 1e-10)
        crash_score += decoupling_norm.clip(0, 1)
        n_components += 1

    if has_cvar:
        # Extreme CVaR = Tail-Risk steigt
        cvar_zscore = (df["risk_cvar_5_100"] - df["risk_cvar_5_100"].rolling(100).mean()) / (
            df["risk_cvar_5_100"].rolling(100).std() + 1e-10
        )
        # Negativer CVaR ist schlecht, also invertieren
        crash_score += (-cvar_zscore).clip(0, 3) / 3
        n_components += 1

    # Normalisiere auf 0-1
    if n_components > 0:
        df["risk_crash_probability"] = crash_score / n_components
    else:
        df["risk_crash_probability"] = 0.0

    # Crash Probability Change (steigt das Risiko?)
    df["risk_crash_prob_change"] = df["risk_crash_probability"] - df["risk_crash_probability"].shift(10)

    # Crash Regime (binär: erhöhtes Risiko ja/nein)
    df["risk_crash_regime"] = (df["risk_crash_probability"] > 0.6).astype(int)

    return df


def compute_correlation_features(df):
    """
    Berechnet Korrelations-Features.

    1. Correlation Stability: Rollierende Korrelation zu Benchmark
       - Decoupling (plötzlicher Korrelationsbruch) = Vorbote für Volatilität

    2. Lead-Lag Momentum: VIX/Benchmark führt oft vor Asset
       - Momentum-Differenz zeigt Divergenzen
    """
    close = df["C"]

    # Prüfe ob Makro-Daten verfügbar sind
    has_spx = "macro_spx" in df.columns
    has_vix = "macro_vix" in df.columns or "sent_vix" in df.columns

    if has_spx:
        spx = df["macro_spx"]

        # Rolling Correlation mit SPX
        for window in [20, 50, 100]:
            corr = close.rolling(window).corr(spx)
            df[f"corr_spx_{window}"] = corr

        # Correlation Stability (Änderung der Korrelation)
        df["corr_spx_stability"] = df["corr_spx_50"] - df["corr_spx_50"].shift(10)

        # Decoupling Indicator (absolute Änderung)
        df["corr_spx_decoupling"] = abs(df["corr_spx_stability"])

        # Momentum Differenz (Lead-Lag)
        asset_mom = close.pct_change(20)
        spx_mom = spx.pct_change(20)
        df["lead_lag_spx"] = asset_mom - spx_mom

    vix_col = "macro_vix" if "macro_vix" in df.columns else ("sent_vix" if "sent_vix" in df.columns else None)
    if vix_col:
        vix = df[vix_col]

        # VIX-Asset Korrelation (normalerweise negativ für Aktien)
        for window in [20, 50]:
            df[f"corr_vix_{window}"] = close.rolling(window).corr(vix)

        # VIX Lead-Lag: VIX-Änderung vs. Asset-Änderung
        # Positive Werte = VIX steigt schneller als Asset fällt
        vix_change = vix.pct_change(10)
        asset_change = close.pct_change(10)
        df["lead_lag_vix"] = vix_change + asset_change  # VIX steigt normalerweise wenn Asset fällt

        # VIX führt oft - prüfe ob VIX vor 5 Bars signalisiert hat
        df["vix_lead_signal"] = vix_change.shift(5) * 100  # VIX Änderung vor 5 Bars

    return df
