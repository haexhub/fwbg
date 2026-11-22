"""Wilder ATR, RSI and ADX with a NaN warm-up, shared by all core callers.

The ``ta`` library seeds the first ``window - 1`` ATR rows with 0.0 and starts
its RSI averages at an artificial zero change. Both look like valid readings
("zero volatility", a real RSI) to every consumer: percent ranks, sizing,
ATR based exits, entry filters. These helpers return NaN until a full window
exists, so a caller either handles the NaN or refuses to act on it.

Import path for other packages::

    from fwbg.utils.indicator_math import wilder_adx, wilder_atr, wilder_rsi
"""

import numpy as np
import pandas as pd
import ta


def wilder_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Wilder ATR with a NaN warm-up.

    ``ta`` seeds the first ``window - 1`` rows with 0.0. The values from the
    first full window on are unchanged; fewer than ``window`` bars give all NaN.
    """
    if len(close) < window:
        return pd.Series(np.nan, index=close.index)
    atr = ta.volatility.average_true_range(high, low, close, window=window)
    atr.iloc[: window - 1] = np.nan
    return atr


def wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Wilder's RSI with a NaN warm-up.

    The average gain and loss are seeded with the simple mean of the first
    ``window`` price changes and then smoothed with alpha = 1 / window, so the
    first value is at index ``window`` (``window + 1`` bars) and everything
    before it is NaN. ``ta`` instead starts the average at an artificial zero
    change on bar 0 and emits values from ``window - 1`` on.

    With no movement at all (average gain and loss both 0, a flat series) the
    RSI is 50: neither side dominates. Only gains give 100, only losses 0.
    """
    delta = close.diff()
    averages = []
    for part in (delta.clip(lower=0.0), (-delta).clip(lower=0.0)):
        seeded = part.copy()
        seeded.iloc[: window + 1] = np.nan
        if len(part) > window:
            seeded.iloc[window] = part.iloc[1 : window + 1].mean()
        averages.append(seeded.ewm(alpha=1.0 / window, adjust=False).mean())
    avg_gain, avg_loss = averages
    rs = avg_gain / avg_loss.where(avg_loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss > 0, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi.where(avg_gain.notna() & avg_loss.notna())


def wilder_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Wilder ADX with a NaN warm-up.

    The first value needs ``2 * period`` bars (smoothed +DI/-DI from bar
    ``period``, then the mean of ``period`` DX values), so rows
    ``0 .. 2 * period - 2`` are NaN. The ``ta`` library fills them with 0.0 and
    raises on shorter input; a NaN anywhere in its input also turns every later
    value into NaN. Bars with a NaN high, low or close are therefore left out of
    the recursion (the missing bar is bridged, not interpolated) and stay NaN in
    the result.
    """
    if period < 2:
        raise ValueError(f"ADX period must be >= 2, got {period}")

    out = np.full(len(close), np.nan)
    valid = (high.notna() & low.notna() & close.notna()).to_numpy()
    if valid.sum() < 2 * period:
        return pd.Series(out, index=close.index)

    h, lo, c = high[valid], low[valid], close[valid]
    adx = ta.trend.adx(h, lo, c, window=period).to_numpy().copy()
    adx[: 2 * period - 1] = np.nan
    out[valid] = adx
    return pd.Series(out, index=close.index)
