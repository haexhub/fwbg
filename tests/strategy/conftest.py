"""
Shared helpers for strategy pipeline integration tests.

Each strategy has its own [strategyName].test.py alongside the JSON config.
These helpers provide realistic synthetic OHLCV data and scenario generators
that allow testing whether the pipeline correctly detects trading patterns.
"""
import numpy as np
import pandas as pd


# ── Base data generators ──────────────────────────────────────────────────────

def make_m15_ohlcv(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Realistic M15 OHLC + Volume data.
    n=6000 ≈ 62 days of 15-min bars.
    Starts on a Monday so weekly features have a valid anchor.
    """
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.0008
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * 0.0001
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * spread
    low  = np.minimum(open_, close) - np.abs(np.random.randn(n)) * spread
    volume = np.random.randint(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": volume},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="15min"),
    )


def make_h1_ohlcv(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Realistic H1 OHLC + Volume data.
    n=5000 ≈ 208 days of hourly bars.
    """
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.001
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * 0.0001
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * spread
    low  = np.minimum(open_, close) - np.abs(np.random.randn(n)) * spread
    volume = np.random.randint(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": volume},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="h"),
    )


def make_m1_ohlcv(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Realistic M1 OHLC + Volume data.
    n=10000 ≈ 17 days of 1-min bars (6.5h/day trading).
    Starts on a Monday so session features have a valid anchor.
    """
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.0002
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * 0.00005
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * spread
    low  = np.minimum(open_, close) - np.abs(np.random.randn(n)) * spread
    volume = np.random.randint(100, 2000, n).astype(float)
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": volume},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="min"),
    )


def make_m5_ohlcv(n: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Realistic M5 OHLC + Volume data.
    n=8000 ≈ 28 days of 5-min bars.
    Starts on a Monday so session features have a valid anchor.
    """
    np.random.seed(seed)
    returns = np.random.randn(n) * 0.0004
    close = 10000 * np.exp(np.cumsum(returns))
    spread = close * 0.00008
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * spread
    low  = np.minimum(open_, close) - np.abs(np.random.randn(n)) * spread
    volume = np.random.randint(200, 3000, n).astype(float)
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": volume},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="5min"),
    )


# ── Scenario generators ───────────────────────────────────────────────────────

def make_liquidity_sweep_scenario(n_base: int = 200, n_post: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Bullish liquidity sweep scenario:
    - n_base bars of uptrend forming a clear swing low
    - 1 sweep bar: wick dips below swing low, then closes back above → bull sweep
    - n_post bars of recovery after the sweep

    The sweep bar should trigger lsw_bull_sweep_active = 1 in the next bar.
    """
    np.random.seed(seed)
    # Phase 1: consolidation building a swing low around 100
    n = n_base + 1 + n_post
    close = np.concatenate([
        100 + np.cumsum(np.random.randn(n_base) * 0.3),   # random walk near 100
        [101.0],                                             # recovery after sweep
        101 + np.cumsum(np.random.randn(n_post) * 0.2),    # uptrend after sweep
    ])
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low  = np.minimum(open_, close) * 0.998

    # Force sweep bar: wick below recent low (swing low ≈ 99.5), close above
    swing_low = close[:n_base].min()
    sweep_idx = n_base
    open_[sweep_idx] = swing_low + 0.5  # open above swing low
    close[sweep_idx] = swing_low + 0.3  # close above swing low (bullish close)
    low[sweep_idx]   = swing_low - 0.5  # wick below swing low (the sweep)
    high[sweep_idx]  = swing_low + 1.0

    idx = pd.date_range("2022-01-03 08:00", periods=n, freq="h")
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": np.full(n, 1000.0)},
        index=idx,
    ), sweep_idx


def make_orb_breakout_scenario(
    session_hour: int = 8,
    range_bars: int = 2,
    n_pre: int = 50,
    n_post: int = 30,
    breakout_direction: str = "up",
    seed: int = 42,
) -> pd.DataFrame:
    """
    ORB breakout scenario on M15 data:
    - n_pre bars of context
    - range_bars bars define the Opening Range (fixed H/L at 101/99)
    - breakout_direction bars: price closes decisively above/below range
    - n_post bars of follow-through

    The breakout bar should trigger orb_breakout_up=1 (or _down=1).
    """
    np.random.seed(seed)
    orb_high, orb_low = 101.0, 99.0
    n_total = n_pre + range_bars + 1 + n_post

    # Start on a Monday at midnight so weekly features work
    start = pd.Timestamp("2022-01-03 00:00")
    idx = pd.date_range(start, periods=n_total, freq="15min")

    close = np.full(n_total, 100.0)
    open_ = np.full(n_total, 100.0)
    high  = np.full(n_total, 100.5)
    low   = np.full(n_total, 99.5)

    # Set session hour bars for ORB
    orb_start = n_pre
    for i in range(range_bars):
        close[orb_start + i] = 100.0
        open_[orb_start + i] = 100.0
        high[orb_start + i]  = orb_high
        low[orb_start + i]   = orb_low

    # Breakout bar
    bo_idx = orb_start + range_bars
    if breakout_direction == "up":
        close[bo_idx] = orb_high + 0.5
        open_[bo_idx] = orb_high + 0.1
        high[bo_idx]  = orb_high + 1.0
        low[bo_idx]   = orb_high - 0.1
    else:
        close[bo_idx] = orb_low - 0.5
        open_[bo_idx] = orb_low - 0.1
        high[bo_idx]  = orb_low + 0.1
        low[bo_idx]   = orb_low - 1.0

    df = pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": np.full(n_total, 1000.0)},
        index=idx,
    )
    return df, bo_idx


def make_fvg_scenario(n_pre: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Fair Value Gap (FVG) scenario:
    - 3-candle imbalance: bar[i].High < bar[i+2].Low → bullish FVG
    - The FVG should trigger fvg_bull_active=1 in subsequent bars

    Pattern:
      bar[i]:   H=100, L=98 (narrow bar)
      bar[i+1]: H=103, L=99 (impulsive bar)
      bar[i+2]: H=105, L=101 → L > H[i] → gap [100, 101] = bullish FVG

    Returns (df, fvg_bar_idx) where fvg_bar_idx is bar i+2 (the bar that creates the FVG).
    """
    np.random.seed(seed)
    n = n_pre + 10
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high  = close + 0.5
    low   = close - 0.5

    # Plant FVG at bar n_pre
    i = n_pre - 2
    high[i]   = 100.0; low[i]   = 98.0
    open_[i]  = 98.5;  close[i] = 99.5

    high[i+1]  = 103.0; low[i+1]  = 99.0
    open_[i+1] = 99.5;  close[i+1] = 102.5

    high[i+2]  = 105.0; low[i+2]  = 101.0  # L[i+2] > H[i] → FVG
    open_[i+2] = 101.5; close[i+2] = 104.0

    idx = pd.date_range("2022-01-03 08:00", periods=n, freq="h")
    return pd.DataFrame(
        {"O": open_, "H": high, "L": low, "C": close, "V": np.full(n, 1000.0)},
        index=idx,
    ), i + 2  # FVG becomes active after bar i+2
