"""
Risk Indicator Plugin.

Enthält:
- Drawdown State Features
- CVaR (Conditional Value at Risk / Expected Shortfall)
- Vol-of-Vol (Volatility of Volatility)
- Crash Probability Proxy
- Correlation Features

Risk-Management ist entscheidend:
- Drawdown-Tiefe und -Dauer zeigen Stress
- CVaR misst Tail-Risk (schlimmer als VaR)
- Vol-of-Vol zeigt Regime-Unsicherheit
- Crash-Probability kombiniert Warnsignale
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator


def _bars_since_event(event_series: pd.Series) -> pd.Series:
    """Berechnet Bars seit dem letzten Event (True/1)."""
    event_groups = event_series.cumsum()
    result = event_series.groupby(event_groups).cumcount()

    first_event_idx = event_series.idxmax() if event_series.any() else None
    if first_event_idx is not None:
        result.loc[:first_event_idx] = np.nan

    return result


def _compute_rolling_cvar(
    returns: pd.Series,
    window: int,
    percentile: int
) -> pd.Series:
    """
    Berechnet Rolling CVaR (Conditional Value at Risk).

    CVaR = Erwarteter Verlust gegeben dass wir im Tail sind.
    """
    def calc_cvar(x):
        if len(x) < 10:
            return np.nan
        threshold = np.percentile(x, percentile)
        tail_returns = x[x <= threshold]
        return tail_returns.mean() if len(tail_returns) > 0 else threshold

    return returns.rolling(window).apply(calc_cvar, raw=True)


@register_indicator("risk")
class RiskIndicators(BaseIndicator):
    """
    Risk/Tail-Risk Features.

    Features:
    - Drawdown State (50, 100, 200)
    - VaR und CVaR (5%, 1%)
    - Vol-of-Vol (20, 50, 100)
    - Crash Probability Proxy
    - Correlation Features (SPX, VIX - falls verfügbar)
    """

    group = "risk"

    def compute(
        self,
        df: pd.DataFrame,
        dd_windows: List[int] = None,
        cvar_windows: List[int] = None,
        cvar_percentiles: List[int] = None,
        vov_windows: List[int] = None,
        compute_correlations: bool = True,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Risk-Features.

        Args:
            df: DataFrame mit OHLC-Daten
            dd_windows: Fenster für Drawdown (default: [50, 100, 200])
            cvar_windows: Fenster für CVaR (default: [50, 100])
            cvar_percentiles: Perzentile für CVaR (default: [5, 1])
            vov_windows: Fenster für Vol-of-Vol (default: [20, 50, 100])
            compute_correlations: Berechne Korrelationen falls Makro-Daten vorhanden

        Returns:
            DataFrame mit Risk-Features
        """
        if dd_windows is None:
            dd_windows = [50, 100, 200]
        if cvar_windows is None:
            cvar_windows = [50, 100]
        if cvar_percentiles is None:
            cvar_percentiles = [5, 1]
        if vov_windows is None:
            vov_windows = [20, 50, 100]

        features = {}
        close = df["C"]
        returns = close.pct_change()

        # === Drawdown Features ===
        for window in dd_windows:
            rolling_max = close.rolling(window, min_periods=1).max()
            dd_pct = (close - rolling_max) / rolling_max
            features[f"risk_dd_pct_{window}"] = dd_pct

            rolling_min_dd = dd_pct.rolling(window, min_periods=1).min()
            features[f"risk_dd_ratio_{window}"] = dd_pct / (rolling_min_dd - 1e-10)

        # Time since peak
        if 200 in dd_windows:
            rolling_max_200 = close.rolling(200, min_periods=1).max()
            is_at_peak = (close >= rolling_max_200).astype(int)
            bars_since_peak = _bars_since_event(is_at_peak)
            features["risk_bars_since_peak"] = bars_since_peak
            features["risk_bars_since_peak_log"] = np.log1p(bars_since_peak)

            rolling_min_close = close.rolling(50, min_periods=1).min()
            recovery = (close - rolling_min_close) / (rolling_max_200 - rolling_min_close + 1e-10)
            features["risk_recovery_ratio"] = recovery.clip(0, 1)

        # === CVaR Features ===
        for window in cvar_windows:
            for percentile in cvar_percentiles:
                var = returns.rolling(window).quantile(percentile / 100)
                features[f"risk_var_{percentile}_{window}"] = var
                features[f"risk_cvar_{percentile}_{window}"] = _compute_rolling_cvar(returns, window, percentile)

        if 100 in cvar_windows and 1 in cvar_percentiles and 5 in cvar_percentiles:
            cvar_1_100 = features["risk_cvar_1_100"]
            cvar_5_100 = features["risk_cvar_5_100"]
            features["risk_cvar_tail_ratio"] = cvar_1_100 / (cvar_5_100 + 1e-10)
            features["risk_cvar_5_change"] = cvar_5_100 - cvar_5_100.shift(20)

        # === Vol-of-Vol Features ===
        atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)
        atr_change = atr.pct_change()

        for window in vov_windows:
            features[f"risk_vol_of_vol_{window}"] = atr_change.rolling(window).std()

        if 100 in vov_windows:
            vov_100 = features["risk_vol_of_vol_100"]
            vov_mean = vov_100.rolling(200).mean()
            vov_std = vov_100.rolling(200).std()
            features["risk_vol_of_vol_zscore"] = (vov_100 - vov_mean) / (vov_std + 1e-10)

        if 50 in vov_windows:
            vov_50 = features["risk_vol_of_vol_50"]
            features["risk_vol_of_vol_trend"] = vov_50 - vov_50.shift(10)

        # Concat features first, then compute crash prob and correlations
        df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)

        # === Crash Probability ===
        self._compute_crash_probability(df)

        # === Correlation Features ===
        if compute_correlations:
            self._compute_correlation_features(df)

        return df

    def _compute_crash_probability(self, df: pd.DataFrame) -> None:
        """
        Berechnet Crash Probability Proxy.

        Kombiniert mehrere Warnsignale:
        - Hohe Kurtosis (Fat Tails)
        - Steigende Vol-of-Vol
        - Extreme CVaR
        """
        crash_score = pd.Series(0.0, index=df.index)
        n_components = 0

        # Kurtosis-basiert
        has_kurt = "dist_kurt_50" in df.columns
        if has_kurt:
            kurt_zscore = df.get(
                "dist_kurt_50_z",
                (df["dist_kurt_50"] - df["dist_kurt_50"].rolling(100).mean()) /
                (df["dist_kurt_50"].rolling(100).std() + 1e-10)
            )
            crash_score += kurt_zscore.clip(0, 3) / 3
            n_components += 1

        # Vol-of-Vol basiert
        if "risk_vol_of_vol_zscore" in df.columns:
            vov_contrib = df["risk_vol_of_vol_zscore"].clip(0, 3) / 3
            crash_score += vov_contrib
            n_components += 1

        # Correlation Decoupling basiert
        if "corr_spx_decoupling" in df.columns:
            decoupling_norm = df["corr_spx_decoupling"] / (
                df["corr_spx_decoupling"].rolling(100).max() + 1e-10
            )
            crash_score += decoupling_norm.clip(0, 1)
            n_components += 1

        # CVaR basiert
        if "risk_cvar_5_100" in df.columns:
            cvar_zscore = (
                df["risk_cvar_5_100"] - df["risk_cvar_5_100"].rolling(100).mean()
            ) / (df["risk_cvar_5_100"].rolling(100).std() + 1e-10)
            crash_score += (-cvar_zscore).clip(0, 3) / 3
            n_components += 1

        if n_components > 0:
            df["risk_crash_probability"] = crash_score / n_components
        else:
            df["risk_crash_probability"] = 0.0

        df["risk_crash_prob_change"] = (
            df["risk_crash_probability"] - df["risk_crash_probability"].shift(10)
        )
        df["risk_crash_regime"] = (df["risk_crash_probability"] > 0.6).astype(int)

    def _compute_correlation_features(self, df: pd.DataFrame) -> None:
        """
        Berechnet Korrelations-Features falls Makro-Daten verfügbar.
        """
        close = df["C"]

        # SPX Korrelation
        has_spx = "macro_spx" in df.columns
        if has_spx:
            spx = df["macro_spx"]

            for window in [20, 50, 100]:
                corr = close.rolling(window).corr(spx)
                df[f"corr_spx_{window}"] = corr

            df["corr_spx_stability"] = df["corr_spx_50"] - df["corr_spx_50"].shift(10)
            df["corr_spx_decoupling"] = abs(df["corr_spx_stability"])

            # Momentum Differenz (Lead-Lag)
            asset_mom = close.pct_change(20)
            spx_mom = spx.pct_change(20)
            df["lead_lag_spx"] = asset_mom - spx_mom

        # VIX Korrelation
        vix_col = (
            "macro_vix" if "macro_vix" in df.columns
            else ("sent_vix" if "sent_vix" in df.columns else None)
        )
        if vix_col:
            vix = df[vix_col]

            for window in [20, 50]:
                df[f"corr_vix_{window}"] = close.rolling(window).corr(vix)

            vix_change = vix.pct_change(10)
            asset_change = close.pct_change(10)
            df["lead_lag_vix"] = vix_change + asset_change

            df["vix_lead_signal"] = vix_change.shift(5) * 100

    def get_feature_columns(self) -> List[str]:
        return [
            # Drawdown
            "risk_dd_pct_50", "risk_dd_pct_100", "risk_dd_pct_200",
            "risk_dd_ratio_50", "risk_dd_ratio_100", "risk_dd_ratio_200",
            "risk_bars_since_peak", "risk_bars_since_peak_log",
            "risk_recovery_ratio",
            # VaR/CVaR
            "risk_var_5_50", "risk_var_1_50",
            "risk_var_5_100", "risk_var_1_100",
            "risk_cvar_5_50", "risk_cvar_1_50",
            "risk_cvar_5_100", "risk_cvar_1_100",
            "risk_cvar_tail_ratio", "risk_cvar_5_change",
            # Vol-of-Vol
            "risk_vol_of_vol_20", "risk_vol_of_vol_50", "risk_vol_of_vol_100",
            "risk_vol_of_vol_zscore", "risk_vol_of_vol_trend",
            # Crash Probability
            "risk_crash_probability", "risk_crash_prob_change", "risk_crash_regime",
            # Correlations (optional, nur wenn Makro-Daten vorhanden)
            "corr_spx_20", "corr_spx_50", "corr_spx_100",
            "corr_spx_stability", "corr_spx_decoupling", "lead_lag_spx",
            "corr_vix_20", "corr_vix_50", "lead_lag_vix", "vix_lead_signal",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "dd_windows": [50, 100, 200],
            "cvar_windows": [50, 100],
            "cvar_percentiles": [5, 1],
            "vov_windows": [20, 50, 100],
            "compute_correlations": True,
        }


__all__ = ["RiskIndicators"]
