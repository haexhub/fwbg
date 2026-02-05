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
    # WICHTIG: Slippage wirkt IMMER gegen den Trader:
    # - Entry-Slippage: schlechterer Einstieg
    # - Exit-Slippage: wird bei PnL-Berechnung berücksichtigt, nicht bei Level
    # Die TP/SL-Levels sind die TRIGGER-Levels, der tatsächliche Exit-Preis
    # wäre schlechter durch Slippage (bereits in entry eingerechnet für Netto-PnL)
    if direction == 1:  # Long
        entry = entry_price + spread + slippage  # Kaufe teurer
        tp = entry + tp_distance  # TP-Level (Trigger)
        sl = entry - sl_distance  # SL-Level (Trigger)
    else:  # Short
        entry = entry_price - spread - slippage  # Verkaufe billiger
        tp = entry - tp_distance  # TP-Level (Trigger)
        sl = entry + sl_distance  # SL-Level (Trigger)

    # Maximale Simulation-Länge
    end_idx = min(entry_idx + max_bars, n)

    # Timeout-Index berechnen (falls aktiviert)
    # WICHTIG: Timeout wird INNERHALB des Loops geprüft, nicht erst danach!
    timeout_idx = -1
    if timeout_bars > 0:
        timeout_idx = min(entry_idx + timeout_bars - 1, n - 1)

    # Simulation-Loop
    for j in range(entry_idx, end_idx):
        # Timeout-Check ZUERST (Timeout hat Priorität über TP/SL nach timeout_bars)
        if timeout_idx > 0 and j >= timeout_idx:
            exit_price = closes[j]
            # PnL berechnen (Exit-Slippage bereits in entry eingerechnet)
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            result = 1.0 if pnl > 0 else -1.0
            return result, j, exit_price, 2  # exit_reason=2 (Timeout)

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

    # Kein Exit (weder TP/SL noch Timeout innerhalb max_bars)
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
