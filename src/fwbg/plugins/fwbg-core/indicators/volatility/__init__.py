"""
Volatility Indicator Plugin.

Enthält: ATR, Bollinger Bands, Keltner Channel, Donchian Channel,
Garman-Klass, Parkinson, Yang-Zhang Volatility Estimators.
"""

import math
from typing import List
import numpy as np
import pandas as pd
import ta

from fwbg_sdk import (
    BaseIndicator,
    register_indicator,
    shift_features,
    safe_divide,
    EPSILON,
)
from fwbg.utils.indicator_math import wilder_atr
from fwbg.data.resample import bar_timing


def _garman_klass(
    high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series, window: int
) -> pd.Series:
    """
    Garman-Klass Volatility Estimator.

    Nutzt OHLC-Daten für effizientere Vol-Schätzung als Close-only.
    σ²_GK = 0.5 * ln(H/L)² - (2ln2 - 1) * ln(C/O)²
    """
    log_hl = np.log(high / (low + EPSILON))
    log_co = np.log(close / (open_ + EPSILON))
    gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return np.sqrt(
        gk.rolling(window, min_periods=max(1, window // 2)).mean().clip(lower=0)
    )


def _parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Parkinson Volatility Estimator.

    Nutzt High-Low Range für bessere Vol-Schätzung.
    σ²_P = ln(H/L)² / (4 * ln(2))
    """
    log_hl_sq = np.log(high / (low + EPSILON)) ** 2
    return np.sqrt(
        log_hl_sq.rolling(window, min_periods=max(1, window // 2)).mean()
        / (4 * np.log(2))
    )


def _yang_zhang(
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """
    Yang-Zhang volatility estimator (Yang and Zhang, 2000).

    sigma^2_YZ = sigma^2_overnight + k * sigma^2_open-close + (1 - k) * sigma^2_RS

    - overnight: sample variance of ln(O_t / C_{t-1})
    - open-close: sample variance of ln(C_t / O_t)
    - RS: mean of the Rogers-Satchell term
    - k = 0.34 / (1.34 + (n + 1) / (n - 1))

    The open-to-close term (not close-to-close) keeps the overnight move out of
    the second component, so it is counted once.
    """
    min_periods = max(2, window // 2)

    # Overnight variance: ln(O_t / C_{t-1})
    log_oc = np.log(open_ / (close.shift(1) + EPSILON))
    var_o = log_oc.rolling(window, min_periods=min_periods).var()

    # Open-to-close variance: ln(C_t / O_t)
    log_co = np.log(close / (open_ + EPSILON))
    var_c = log_co.rolling(window, min_periods=min_periods).var()

    # Rogers-Satchell variance
    log_ho = np.log(high / (open_ + EPSILON))
    log_hc = np.log(high / (close + EPSILON))
    log_lo = np.log(low / (open_ + EPSILON))
    log_lc = np.log(low / (close + EPSILON))
    rs = log_ho * log_hc + log_lo * log_lc
    var_rs = rs.rolling(window, min_periods=min_periods).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = var_o + k * var_c + (1 - k) * var_rs
    return np.sqrt(yz.clip(lower=0))


def _infer_bars_per_day(df: pd.DataFrame) -> tuple[float, int]:
    """Return an annualisation scale and the first row where it is known.

    Use the nominal spacing on daily or longer data. For intraday data, fix
    the median of the first five closed dates after the initial partial date.
    A week of observations avoids relying on a single short session. Callers
    can provide bars_per_day from a known schedule instead.
    """
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        return 1.0, 0
    spacing, ready = bar_timing(df)
    if spacing is None:
        return 1.0, len(df)
    if spacing >= pd.Timedelta(days=1):
        return 1.0 / max(1, round(spacing / pd.Timedelta(days=1) * 5 / 7)), ready
    dates = index.normalize()
    starts = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    if len(starts) < 6:
        return 1.0, len(df)
    counts = np.diff(starts[:6])
    return float(np.median(counts)), max(ready, int(starts[5]))


@register_indicator("volatility")
class VolatilityIndicators(BaseIndicator):
    """
    Volatilitäts-Indikatoren für Trading-Strategien.

    Features:
    - ATR (7, 14, 21 Perioden) - als Prozent vom Preis
    - Bollinger Bands (20 Perioden) - pband, wband
    - Keltner Channel (original 1960 SMA/range version) - pband, wband
    - Donchian Channel - pband, wband
    - Garman-Klass Volatility (effizientere OHLC-basierte Vol)
    - Parkinson Volatility (High-Low basierte Vol)
    - Yang-Zhang Volatility (robusteste OHLC-Vol)
    """

    # Required attributes
    name = "volatility"
    version = "3.1.0"

    def compute(
        self,
        df: pd.DataFrame,
        atr_periods: List[int] = None,
        bb_period: int = 20,
        vol_est_windows: List[int] = None,
        **params,
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
        atr_14 = wilder_atr(df["H"], df["L"], df["C"], window=14)
        features["_atr"] = atr_14
        features["vol_atr"] = atr_14

        # ATR als Prozent vom Preis
        for period in atr_periods:
            atr = wilder_atr(df["H"], df["L"], df["C"], window=period)
            features[f"vol_atr_pct_{period}"] = safe_divide(atr, df["C"])

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["C"], window=bb_period)
        features[f"vol_bb_pband_{bb_period}"] = bb.bollinger_pband()
        features[f"vol_bb_wband_{bb_period}"] = bb.bollinger_wband()

        # Keltner Channel: the original 1960 construction (ta's default
        # original_version=True), a 10-bar SMA of the typical price with a
        # range-based envelope, not the EMA(20) +/- 2 ATR(10) variant.
        kc = ta.volatility.KeltnerChannel(
            df["H"], df["L"], df["C"], original_version=True
        )
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
            features[f"vol_parkinson_{window}"] = _parkinson(df["H"], df["L"], window)
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
                features[f"{key}_rank"] = (
                    features[key]
                    .rolling(
                        compression_lookback, min_periods=compression_lookback // 2
                    )
                    .rank(pct=True)
                )

        bb_wband_key = f"vol_bb_wband_{bb_period}"
        if bb_wband_key in features:
            features[f"{bb_wband_key}_rank"] = (
                features[bb_wband_key]
                .rolling(compression_lookback, min_periods=compression_lookback // 2)
                .rank(pct=True)
            )

        # Compression flag: ATR14 and BB Width both below 20th percentile
        atr_rank = features.get("vol_atr_pct_14_rank")
        bb_rank = features.get(f"{bb_wband_key}_rank")
        if atr_rank is not None and bb_rank is not None:
            # NaN while either rank is still warming up (a comparison against
            # NaN is False and would read as "not compressed").
            features["vol_compression"] = (
                ((atr_rank < 0.2) & (bb_rank < 0.2))
                .astype(float)
                .where(atr_rank.notna() & bb_rank.notna())
            )

        # === Realized Vol vs Implied Vol (VIX) ===
        rv_window = params.get("rv_window", 20)
        # rv_window counts days; bars_per_day converts it to bars and annualises
        # (24 = hourly bars around the clock, 1 = daily bars). Without an explicit
        # value it is read from the index spacing.
        bars_per_day = params.get("bars_per_day")
        rv_ready = 0
        if bars_per_day is None:
            bars_per_day, rv_ready = _infer_bars_per_day(df)
        if (
            isinstance(bars_per_day, bool)
            or not isinstance(bars_per_day, (int, float))
            or not math.isfinite(bars_per_day)
            or bars_per_day <= 0
        ):
            raise ValueError(
                f"volatility: bars_per_day must be a positive number, not {bars_per_day!r}"
            )
        rv_bars = max(2, int(round(rv_window * bars_per_day)))
        log_returns = np.log(df["C"] / (df["C"].shift(1) + EPSILON))
        rv = log_returns.rolling(rv_bars, min_periods=max(2, rv_bars // 2)).std()
        rv_annualized = rv * np.sqrt(252 * bars_per_day) * 100  # as percentage
        # Row rv_ready can use estimates known as soon as that bar opens.
        rv_annualized.iloc[: max(0, rv_ready - 1)] = np.nan
        features[f"vol_rv_{rv_window}"] = rv_annualized

        if "macro_vix" in df.columns:
            vix = df["macro_vix"]
            features["vol_rv_iv_spread"] = rv_annualized - vix
            features["vol_rv_iv_ratio"] = safe_divide(rv_annualized, vix)

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        """Feature columns compute() produces for ``params`` (defaults when None).

        ``vol_rv_iv_spread`` / ``vol_rv_iv_ratio`` are only produced when the
        frame has a ``macro_vix`` column; they are always advertised.
        """
        params = params or {}
        atr_periods = params.get("atr_periods") or [7, 14, 21]
        bb_period = params.get("bb_period") or 20
        windows = params.get("vol_est_windows") or [20, 50]
        rv_window = params.get("rv_window") or 20

        cols = ["vol_atr"]
        cols += [f"vol_atr_pct_{p}" for p in atr_periods]
        cols += [f"vol_bb_pband_{bb_period}", f"vol_bb_wband_{bb_period}"]
        cols += ["vol_kc_pband", "vol_kc_wband", "vol_dc_pband", "vol_dc_wband"]
        for w in windows:
            cols += [f"vol_gk_{w}", f"vol_parkinson_{w}", f"vol_yz_{w}"]
        if 20 in windows:
            cols.append("vol_yz_atr_ratio")
        cols += [f"vol_atr_pct_{p}_rank" for p in atr_periods]
        cols.append(f"vol_bb_wband_{bb_period}_rank")
        if 14 in atr_periods:
            cols.append("vol_compression")
        cols += [f"vol_rv_{rv_window}", "vol_rv_iv_spread", "vol_rv_iv_ratio"]
        return cols

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "atr_periods": [7, 14, 21],
            "bb_period": 20,
            "vol_est_windows": [20, 50],
            "bars_per_day": None,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "atr_periods": {
                "type": "list[int]",
                "default": [7, 14, 21],
                "description": "Periods for ATR (Average True Range) calculation, expressed as percentage of price. ATR measures market volatility by decomposing the entire range of a bar. Shorter periods capture recent volatility spikes, longer periods give a smoother baseline.",
                "min": 2,
                "max": 500,
            },
            "bb_period": {
                "type": "int",
                "default": 20,
                "description": "Lookback period for Bollinger Bands. Defines the SMA center and standard deviation envelope width. The classic 20-period setting corresponds roughly to one trading month. Also used for Bollinger %B (price position) and bandwidth (volatility squeeze) features.",
                "min": 5,
                "max": 500,
                "step": 1,
            },
            "vol_est_windows": {
                "type": "list[int]",
                "default": [20, 50],
                "description": "Rolling window sizes for OHLC-based volatility estimators (Garman-Klass, Parkinson, Yang-Zhang). These estimators are more statistically efficient than close-only volatility. Shorter windows react faster to regime changes, longer windows provide more stable estimates.",
                "min": 5,
                "max": 500,
            },
            "bars_per_day": {
                "type": "float",
                "default": None,
                "description": "Bars per trading day for the realised volatility vol_rv_20 (20 days): 24 for hourly bars around the clock, 7 for hourly bars of a 7 hour session, 96 for M15, 1 for daily bars. Sets the rolling window in bars and the annualisation. Empty: inferred from the bar spacing of the index.",
                "min": 0.1,
                "max": 1440,
            },
        }


__all__ = ["VolatilityIndicators"]
