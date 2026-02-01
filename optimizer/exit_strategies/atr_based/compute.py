"""
Compute-Funktionen für ATR-basierte Exit Strategy.

Berechnet dynamische TP/SL basierend auf ATR bei Trade-Eröffnung.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from numba import njit

from ...simulation import _simulate_trade_numba


@njit(cache=True, parallel=False)
def compute_targets_atr_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_values: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    spread: float,
    slippage: float,
    min_tp_distance: float,
    min_sl_distance: float,
    max_bars: int,
    timeout_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Win/Loss Targets mit ATR-basierten TP/SL.

    TP/SL werden pro Trade basierend auf ATR bei Entry berechnet:
    - tp_distance = max(atr[entry_idx] * tp_mult, min_tp_distance)
    - sl_distance = max(atr[entry_idx] * sl_mult, min_sl_distance)

    Args:
        opens, closes, highs, lows: OHLC-Arrays
        atr_values: ATR-Werte pro Bar
        tp_mult: ATR-Multiplikator für Take-Profit
        sl_mult: ATR-Multiplikator für Stop-Loss
        spread: Bid-Ask Spread
        slippage: Slippage-Kosten
        min_tp_distance: Mindest-TP in Preiseinheiten (Spread-Schutz)
        min_sl_distance: Mindest-SL in Preiseinheiten
        max_bars: Maximale Simulationslänge
        timeout_bars: Trade schließen nach X Bars (0 = kein Timeout)

    Returns:
        (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
    """
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        # ATR bei Entry (= Bar nach Signal)
        entry_idx = i + 1
        if entry_idx >= n:
            continue

        # ATR bei Signal-Bar verwenden (da Entry bei nächster Bar)
        atr_at_signal = atr_values[i]

        # Dynamische TP/SL berechnen mit Mindest-Werten
        tp_distance = max(atr_at_signal * tp_mult, min_tp_distance)
        sl_distance = max(atr_at_signal * sl_mult, min_sl_distance)

        # Long Trade simulieren
        result_long, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        # Short Trade simulieren
        result_short, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short


def compute_targets_atr(
    df: pd.DataFrame,
    tp_mult: float,
    sl_mult: float,
    ctx: "SimulationContext",
    atr_period: int = 14,
    min_tp_pips: int = 10,
    min_sl_pips: int = 15,
    timeout_bars: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Win/Loss Targets mit ATR-basierten TP/SL.

    High-Level Wrapper für die Numba-Funktion.

    Args:
        df: DataFrame mit OHLC-Daten und _atr Spalte
        tp_mult: ATR-Multiplikator für Take-Profit
        sl_mult: ATR-Multiplikator für Stop-Loss
        ctx: SimulationContext mit spread, max_trade_bars, etc.
        atr_period: ATR-Periode (für Fallback-Berechnung)
        min_tp_pips: Mindest-TP in Pips (Spread-Schutz)
        min_sl_pips: Mindest-SL in Pips
        timeout_bars: Optional - Trade schließen nach X Bars

    Returns:
        (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
    """
    # OHLC-Arrays extrahieren
    opn_v = df["O"].values.astype(np.float64)
    cls_v = df["C"].values.astype(np.float64)
    hgh_v = df["H"].values.astype(np.float64)
    low_v = df["L"].values.astype(np.float64)

    # ATR-Array - verwende vorberechnete Spalte oder berechne
    if "_atr" in df.columns:
        atr_v = df["_atr"].values.astype(np.float64)
    elif "vol_atr" in df.columns:
        atr_v = df["vol_atr"].values.astype(np.float64)
    else:
        # Fallback: ATR berechnen
        import ta
        atr_series = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=atr_period
        )
        atr_v = atr_series.values.astype(np.float64)

    # NaN durch 0 ersetzen (am Anfang der Serie)
    atr_v = np.nan_to_num(atr_v, nan=0.0)

    # Mindest-Distanzen in Preiseinheiten
    min_tp_distance = ctx.spread * min_tp_pips
    min_sl_distance = ctx.spread * min_sl_pips

    slippage = ctx.spread * 0.5

    # max_bars: Wie weit maximal simuliert wird
    max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)

    # timeout_bars: Wann Trade geschlossen wird (0 = kein Timeout)
    timeout_val = timeout_bars if timeout_bars else 0

    return compute_targets_atr_numba(
        opn_v, cls_v, hgh_v, low_v,
        atr_v, tp_mult, sl_mult,
        ctx.spread, slippage,
        min_tp_distance, min_sl_distance,
        max_bars, timeout_val
    )
