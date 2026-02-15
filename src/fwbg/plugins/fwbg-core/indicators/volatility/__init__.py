"""
Volatility Indicator Plugin.

Enthält: ATR, Bollinger Bands, Keltner Channel, Donchian Channel,
Garman-Klass, Parkinson, Yang-Zhang Volatility Estimators.
"""
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features, safe_divide, EPSILON
from fwbg.core import register_indicator


def _garman_klass(high: pd.Series, low: pd.Series, open_: pd.Series,
                  close: pd.Series, window: int) -> pd.Series:
    """
    Garman-Klass Volatility Estimator.

    Nutzt OHLC-Daten für effizientere Vol-Schätzung als Close-only.
    σ²_GK = 0.5 * ln(H/L)² - (2ln2 - 1) * ln(C/O)²
    """
    log_hl = np.log(high / (low + EPSILON))
    log_co = np.log(close / (open_ + EPSILON))
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return np.sqrt(gk.rolling(window, min_periods=max(1, window // 2)).mean().clip(lower=0))


def _parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Parkinson Volatility Estimator.

    Nutzt High-Low Range für bessere Vol-Schätzung.
    σ²_P = ln(H/L)² / (4 * ln(2))
    """
    log_hl_sq = np.log(high / (low + EPSILON)) ** 2
    return np.sqrt(log_hl_sq.rolling(window, min_periods=max(1, window // 2)).mean()
                   / (4 * np.log(2)))


def _yang_zhang(high: pd.Series, low: pd.Series, open_: pd.Series,
                close: pd.Series, window: int) -> pd.Series:
    """
    Yang-Zhang Volatility Estimator.

    Kombiniert Overnight, Close-to-Close und Rogers-Satchell Varianz.
    Robustester OHLC-basierter Schätzer.
    σ²_YZ = σ²_O + k*σ²_C + (1-k)*σ²_RS
    """
    # Overnight variance: ln(O_t / C_{t-1})
    log_oc = np.log(open_ / (close.shift(1) + EPSILON))
    var_o = log_oc.rolling(window, min_periods=max(1, window // 2)).var()

    # Close-to-close variance: ln(C_t / C_{t-1})
    log_cc = np.log(close / (close.shift(1) + EPSILON))
    var_c = log_cc.rolling(window, min_periods=max(1, window // 2)).var()

    # Rogers-Satchell variance
    log_ho = np.log(high / (open_ + EPSILON))
    log_hc = np.log(high / (close + EPSILON))
    log_lo = np.log(low / (open_ + EPSILON))
    log_lc = np.log(low / (close + EPSILON))
    rs = log_ho * log_hc + log_lo * log_lc
    var_rs = rs.rolling(window, min_periods=max(1, window // 2)).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = var_o + k * var_c + (1 - k) * var_rs
    return np.sqrt(yz.clip(lower=0))


@register_indicator("volatility")
class VolatilityIndicators(BaseIndicator):
    """
    Volatilitäts-Indikatoren für Trading-Strategien.

    Features:
    - ATR (7, 14, 21 Perioden) - als Prozent vom Preis
    - Bollinger Bands (20 Perioden) - pband, wband
    - Keltner Channel - pband, wband
    - Donchian Channel - pband, wband
    - Garman-Klass Volatility (effizientere OHLC-basierte Vol)
    - Parkinson Volatility (High-Low basierte Vol)
    - Yang-Zhang Volatility (robusteste OHLC-Vol)
    """

    # Required attributes
    name = "volatility"
    version = "3.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        atr_periods: List[int] = None,
        bb_period: int = 20,
        vol_est_windows: List[int] = None,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet alle Volatilitäts-Indikatoren.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C)
            atr_periods: ATR-Perioden (default: [7, 14, 21])
            bb_period: Bollinger Bands Periode (default: 20)
            vol_est_windows: Fenster für OHLC-Vol-Schätzer (default: [20, 50])
        """
        if atr_periods is None:
            atr_periods = [7, 14, 21]
        if vol_est_windows is None:
            vol_est_windows = [20, 50]

        features = {}

        # ATR (für interne Nutzung und als Feature)
        atr_14 = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )
        features["_atr"] = atr_14
        features["vol_atr"] = atr_14

        # ATR als Prozent vom Preis
        for period in atr_periods:
            atr = ta.volatility.average_true_range(
                df["H"], df["L"], df["C"], window=period
            )
            features[f"vol_atr_pct_{period}"] = safe_divide(atr, df["C"])

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["C"], window=bb_period)
        features[f"vol_bb_pband_{bb_period}"] = bb.bollinger_pband()
        features[f"vol_bb_wband_{bb_period}"] = bb.bollinger_wband()

        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(df["H"], df["L"], df["C"])
        features["vol_kc_pband"] = kc.keltner_channel_pband()
        features["vol_kc_wband"] = kc.keltner_channel_wband()

        # Donchian Channel
        dc = ta.volatility.DonchianChannel(df["H"], df["L"], df["C"])
        features["vol_dc_pband"] = dc.donchian_channel_pband()
        features["vol_dc_wband"] = dc.donchian_channel_wband()

        # === OHLC Volatility Estimators ===
        for window in vol_est_windows:
            features[f"vol_gk_{window}"] = _garman_klass(
                df["H"], df["L"], df["O"], df["C"], window
            )
            features[f"vol_parkinson_{window}"] = _parkinson(
                df["H"], df["L"], window
            )
            features[f"vol_yz_{window}"] = _yang_zhang(
                df["H"], df["L"], df["O"], df["C"], window
            )

        # Vol-Estimator Ratio: Yang-Zhang / ATR% (Divergenz = Regime-Shift)
        if 20 in vol_est_windows:
            atr_pct_14 = features.get("vol_atr_pct_14", safe_divide(atr_14, df["C"]))
            features["vol_yz_atr_ratio"] = safe_divide(
                features["vol_yz_20"], atr_pct_14
            )

        # === Volatility Compression (Percentile Ranking) ===
        compression_lookback = params.get("compression_lookback", 100)
        for period in atr_periods:
            key = f"vol_atr_pct_{period}"
            if key in features:
                features[f"{key}_rank"] = features[key].rolling(
                    compression_lookback, min_periods=compression_lookback // 2
                ).rank(pct=True)

        bb_wband_key = f"vol_bb_wband_{bb_period}"
        if bb_wband_key in features:
            features[f"{bb_wband_key}_rank"] = features[bb_wband_key].rolling(
                compression_lookback, min_periods=compression_lookback // 2
            ).rank(pct=True)

        # Compression flag: ATR14 and BB Width both below 20th percentile
        atr_rank = features.get("vol_atr_pct_14_rank")
        bb_rank = features.get(f"{bb_wband_key}_rank")
        if atr_rank is not None and bb_rank is not None:
            features["vol_compression"] = (
                (atr_rank < 0.2) & (bb_rank < 0.2)
            ).astype(float)

        # === Realized Vol vs Implied Vol (VIX) ===
        rv_window = params.get("rv_window", 20)
        log_returns = np.log(df["C"] / (df["C"].shift(1) + EPSILON))
        # Annualized realized vol (24h bars * 252 trading days)
        rv = log_returns.rolling(rv_window * 24, min_periods=rv_window * 12).std()
        rv_annualized = rv * np.sqrt(252 * 24) * 100  # as percentage
        features[f"vol_rv_{rv_window}"] = rv_annualized

        if "macro_vix" in df.columns:
            vix = df["macro_vix"]
            features["vol_rv_iv_spread"] = rv_annualized - vix
            features["vol_rv_iv_ratio"] = safe_divide(rv_annualized, vix)

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        """Gibt Liste aller Volatilitäts-Feature-Spalten zurück."""
        return [
            # ATR
            "vol_atr",
            "vol_atr_pct_7", "vol_atr_pct_14", "vol_atr_pct_21",
            # Bollinger
            "vol_bb_pband_20", "vol_bb_wband_20",
            # Keltner
            "vol_kc_pband", "vol_kc_wband",
            # Donchian
            "vol_dc_pband", "vol_dc_wband",
            # OHLC Volatility Estimators
            "vol_gk_20", "vol_gk_50",
            "vol_parkinson_20", "vol_parkinson_50",
            "vol_yz_20", "vol_yz_50",
            "vol_yz_atr_ratio",
            # Compression (Percentile Rankings)
            "vol_atr_pct_7_rank", "vol_atr_pct_14_rank", "vol_atr_pct_21_rank",
            "vol_bb_wband_20_rank",
            "vol_compression",
            # Realized vs Implied Vol
            "vol_rv_20",
            "vol_rv_iv_spread",
            "vol_rv_iv_ratio",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "atr_periods": [7, 14, 21],
            "bb_period": 20,
            "vol_est_windows": [20, 50],
        }


__all__ = ["VolatilityIndicators"]
