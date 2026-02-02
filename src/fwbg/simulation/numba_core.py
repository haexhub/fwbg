"""
Numba-optimierte Kernfunktionen für Trade-Simulation.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _simulate_trade_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    direction: int,
    tp_distance: float,
    sl_distance: float,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
) -> tuple:
    """
    Numba-optimierte Trade-Simulation (Kern-Loop).

    Args:
        opens, closes, highs, lows: OHLC-Arrays
        idx: Signal-Index (Entry bei idx+1)
        direction: 1=Long, -1=Short
        tp_distance: Take-Profit Distanz in Preiseinheiten
        sl_distance: Stop-Loss Distanz in Preiseinheiten
        spread: Bid-Ask Spread
        slippage: Slippage-Kosten
        max_bars: Maximale Simulation-Länge
        timeout_bars: Trade schließen nach X Bars (0 = kein Timeout)

    Returns:
        (result, exit_idx, exit_price, exit_reason)
        result: 1.0=Win, -1.0=Loss, 0.0=Kein Ergebnis
        exit_reason: 0=TP, 1=SL, 2=Timeout, -1=Kein Exit
    """
    entry_idx = idx + 1
    n = len(closes)

    if entry_idx >= n:
        return 0.0, -1, 0.0, -1

    # Entry-Preis
    entry_price = opens[entry_idx]

    # TP/SL-Levels berechnen
    if direction == 1:  # Long
        entry = entry_price + spread + slippage
        tp = entry + tp_distance - slippage
        sl = entry - sl_distance - slippage
    else:  # Short
        entry = entry_price - spread - slippage
        tp = entry - tp_distance + slippage
        sl = entry + sl_distance + slippage

    # Maximale Simulation-Länge
    end_idx = min(entry_idx + max_bars, n)

    # Simulation-Loop
    for j in range(entry_idx, end_idx):
        if direction == 1:  # Long
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:  # Short
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

        if tp_hit and sl_hit:
            # Beide im selben Bar - konservativ: Loss
            return -1.0, j, sl, 1

        if tp_hit:
            return 1.0, j, tp, 0

        if sl_hit:
            return -1.0, j, sl, 1

    # Kein TP/SL erreicht - Timeout-Handling
    if timeout_bars > 0:
        timeout_idx = min(entry_idx + timeout_bars - 1, n - 1)
        if timeout_idx >= entry_idx:
            exit_price = closes[timeout_idx]

            # PnL berechnen
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price

            result = 1.0 if pnl > 0 else -1.0
            return result, timeout_idx, exit_price, 2

    # Kein Exit
    return 0.0, -1, 0.0, -1


@njit(cache=True, parallel=True)
def compute_targets_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_distance: float,
    sl_distance: float,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
) -> tuple:
    """
    Berechnet Long/Short Targets für alle Bars.

    PARALLELISIERT: Nutzt alle verfügbaren CPU-Kerne für 2-4x Speedup.

    Args:
        opens, closes, highs, lows: OHLC-Arrays
        tp_distance: Take-Profit Distanz
        sl_distance: Stop-Loss Distanz
        spread: Bid-Ask Spread
        slippage: Slippage-Kosten
        max_bars: Maximale Simulation-Länge
        timeout_bars: Trade schließen nach X Bars (0 = kein Timeout)

    Returns:
        (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
    """
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)

    # Parallelisierte Schleife - jeder Bar unabhängig
    for i in prange(n - 1):
        # Long Trade
        result_long, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage, max_bars, timeout_bars
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        # Short Trade
        result_short, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage, max_bars, timeout_bars
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short
