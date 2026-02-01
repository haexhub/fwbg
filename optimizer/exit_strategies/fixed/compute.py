"""
Compute-Funktionen für Fixed Exit Strategy.

Verwendet die bestehenden Numba-optimierten Funktionen aus simulation.py.
"""
from typing import Tuple
import numpy as np
import pandas as pd

from ...simulation import compute_targets_numba


def compute_targets_fixed(
    df: pd.DataFrame,
    tp: int,
    sl: int,
    ctx: "SimulationContext",
    timeout_bars: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Win/Loss Targets mit fixen TP/SL-Werten.

    TP und SL sind Multiplikatoren des Spreads.
    Beispiel: tp=30 bei spread=0.0001 -> TP = 30 Pips = 0.003

    Args:
        df: DataFrame mit OHLC-Daten (Spalten: O, H, L, C)
        tp: Take-Profit als Spread-Multiplikator
        sl: Stop-Loss als Spread-Multiplikator
        ctx: SimulationContext mit spread, max_trade_bars, etc.
        timeout_bars: Optional - Trade schließen nach X Bars

    Returns:
        (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
    """
    # OHLC-Arrays extrahieren
    opn_v = df["O"].values.astype(np.float64)
    cls_v = df["C"].values.astype(np.float64)
    hgh_v = df["H"].values.astype(np.float64)
    low_v = df["L"].values.astype(np.float64)

    # Distanzen berechnen (Spread * Multiplikator)
    tp_distance = ctx.spread * tp
    sl_distance = ctx.spread * sl
    slippage = ctx.spread * 0.5

    # max_bars: Wie weit maximal simuliert wird
    max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)

    # timeout_bars: Wann Trade geschlossen wird (0 = kein Timeout)
    timeout_val = timeout_bars if timeout_bars else 0

    return compute_targets_numba(
        opn_v, cls_v, hgh_v, low_v,
        tp_distance, sl_distance, ctx.spread, slippage,
        max_bars, timeout_val
    )
