"""
ATR Trailing Stop Exit Modifier.

Erweitert eine Basis-Exit-Strategie (z.B. atr_based) mit Trailing-Stop-
und Breakeven-Stop-Logik. Kann optional zu bestehenden Pipelines hinzugefügt
werden, ohne die Basis-Exit-Strategie zu verändern.

Konfiguration in der Strategy-JSON:
    {
      "exit_strategy": "atr_based",
      "exit_params": "atr_intraday",
      "exit_modifier": "trailing_stop",
      "exit_modifier_params": {
        "breakeven_trigger": 0.5,
        "trail_atr_mult": 0.5
      }
    }

Breakeven Stop:
    Sobald der Preis `breakeven_trigger × tp_distance` in die Gewinnzone läuft,
    wird der SL auf den Entry-Preis gezogen. Einmal profitable Trades können
    nicht mehr als Verlust enden.

Trailing Stop:
    Nach Breakeven-Aktivierung folgt der SL dem besten erreichten Preis mit
    `trail_atr_mult × ATR` Abstand.

Kombinierbar:
    breakeven_trigger=0.5, trail_atr_mult=0.5  → beides aktiv (empfohlen)
    breakeven_trigger=0.5, trail_atr_mult=0.0  → nur Breakeven
    breakeven_trigger=0.0, trail_atr_mult=0.5  → nur Trailing ab Entry
"""
from typing import Tuple

import numpy as np
from numba import njit

from fwbg_sdk import BaseExitModifier, register_exit_modifier


# ---------------------------------------------------------------------------
# Numba kernel — single-trade simulation mit Breakeven- und Trailing-Stop
# ---------------------------------------------------------------------------

@njit(cache=True)
def _simulate_trade_trailing_numba(
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
    breakeven_trigger: float,
    trail_distance: float,
    trail_tp_dist: float,
) -> tuple:
    """
    Single-trade simulation mit Breakeven- und Trailing-Stop.

    Args:
        breakeven_trigger: Bruchteil der TP-Distanz nach dem SL auf Entry
                           gezogen wird (0.0 = kein Breakeven, Trailing startet
                           sofort wenn trail_distance > 0).
        trail_distance:    Absoluter Abstand des Trailing-Stops vom besten
                           erreichten Preis (0.0 = kein Trailing).
        trail_tp_dist:     Absoluter Abstand des Trailing-TPs vom besten
                           erreichten Preis (0.0 = kein Trailing-TP).

    Returns:
        (result, exit_idx, exit_price, exit_reason)
        result: 1.0=Win, -1.0=Loss, 0.0=kein Exit
    """
    entry_idx = idx + 1
    n = len(closes)

    if entry_idx >= n:
        return 0.0, -1, 0.0, -1

    entry_price = opens[entry_idx]

    if direction == 1:  # Long
        entry = entry_price + spread + slippage
        tp = entry + tp_distance
        sl = entry - sl_distance
        be_trigger_price = entry + tp_distance * breakeven_trigger
    else:  # Short
        entry = entry_price - spread - slippage
        tp = entry - tp_distance
        sl = entry + sl_distance
        be_trigger_price = entry - tp_distance * breakeven_trigger

    end_idx = min(entry_idx + max_bars, n)

    timeout_idx = -1
    if timeout_bars > 0:
        timeout_idx = min(entry_idx + timeout_bars - 1, n - 1)

    best_price = entry
    trailing_active = breakeven_trigger <= 0.0

    for j in range(entry_idx, end_idx):
        if timeout_idx > 0 and j >= timeout_idx:
            exit_price = closes[j]
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            return (1.0 if pnl > 0 else -1.0), j, exit_price, 2

        if direction == 1:
            if highs[j] > best_price:
                best_price = highs[j]
        else:
            if lows[j] < best_price:
                best_price = lows[j]

        if not trailing_active and breakeven_trigger > 0.0:
            if direction == 1 and best_price >= be_trigger_price:
                trailing_active = True
                if entry > sl:
                    sl = entry
            elif direction == -1 and best_price <= be_trigger_price:
                trailing_active = True
                if entry < sl:
                    sl = entry

        if trailing_active and trail_distance > 0.0:
            if direction == 1:
                new_sl = best_price - trail_distance
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = best_price + trail_distance
                if new_sl < sl:
                    sl = new_sl

        if trailing_active and trail_tp_dist > 0.0:
            if direction == 1:
                new_tp = best_price + trail_tp_dist
                if new_tp > tp:
                    tp = new_tp
            else:
                new_tp = best_price - trail_tp_dist
                if new_tp < tp:
                    tp = new_tp

        if direction == 1:
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

        if sl_hit:
            if direction == 1:
                result = 1.0 if sl > entry else -1.0
            else:
                result = 1.0 if sl < entry else -1.0
            return result, j, sl, 1

        if tp_hit:
            return 1.0, j, tp, 0

    return 0.0, -1, 0.0, -1


@njit(cache=True, parallel=False)
def _compute_targets_trailing_numba(
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
    timeout_val: int,
    breakeven_trigger: float,
    trail_atr_mult: float,
    trail_tp_atr_mult: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        if i + 1 >= n:
            continue

        atr = atr_values[i]
        tp_distance = max(atr * tp_mult, min_tp_distance)
        sl_distance = max(atr * sl_mult, min_sl_distance)
        trail_distance = atr * trail_atr_mult if trail_atr_mult > 0.0 else 0.0
        trail_tp_dist = atr * trail_tp_atr_mult if trail_tp_atr_mult > 0.0 else 0.0

        result_long, _, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val, breakeven_trigger, trail_distance, trail_tp_dist,
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        result_short, _, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val, breakeven_trigger, trail_distance, trail_tp_dist,
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short


@njit(cache=True, parallel=False)
def _compute_targets_trailing_with_durations_numba(
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
    timeout_val: int,
    breakeven_trigger: float,
    trail_atr_mult: float,
    trail_tp_atr_mult: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)
    durations_long = np.zeros(n, dtype=np.int64)
    durations_short = np.zeros(n, dtype=np.int64)

    for i in range(n - 1):
        if i + 1 >= n:
            continue

        atr = atr_values[i]
        tp_distance = max(atr * tp_mult, min_tp_distance)
        sl_distance = max(atr * sl_mult, min_sl_distance)
        trail_distance = atr * trail_atr_mult if trail_atr_mult > 0.0 else 0.0
        trail_tp_dist = atr * trail_tp_atr_mult if trail_tp_atr_mult > 0.0 else 0.0

        result_long, exit_long, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val, breakeven_trigger, trail_distance, trail_tp_dist,
        )
        if result_long == 1.0:
            targets_long[i] = 1.0
        durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

        result_short, exit_short, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val, breakeven_trigger, trail_distance, trail_tp_dist,
        )
        if result_short == 1.0:
            targets_short[i] = 1.0
        durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

    return targets_long, targets_short, durations_long, durations_short


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

@register_exit_modifier("trailing_stop")
class TrailingStopModifier(BaseExitModifier):
    """
    ATR-basierter Trailing-Stop Exit-Modifier.

    Ersetzt die Standard-Simulation von atr_based mit Trailing-Stop- und
    Breakeven-Stop-Logik. Wird optional zur Pipeline hinzugefügt.

    Parameter (in exit_modifier_params):
        breakeven_trigger (float, default=0.5):
            Bruchteil der TP-Distanz nach dem SL auf Entry gezogen wird.
            0.0 = kein Breakeven (Trailing startet sofort).

        trail_atr_mult (float, default=0.5):
            ATR-Multiplikator für den Trailing-Stop-Abstand.
            0.0 = nur Breakeven, kein Trailing.
    """

    def compute_targets(
        self,
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
        timeout_val: int,
        return_durations: bool = False,
        breakeven_trigger: float = 0.5,
        trail_atr_mult: float = 0.5,
        trail_tp_atr_mult: float = 0.0,
        **kwargs,
    ):
        if return_durations:
            return _compute_targets_trailing_with_durations_numba(
                opens, closes, highs, lows, atr_values,
                tp_mult, sl_mult, spread, slippage,
                min_tp_distance, min_sl_distance,
                max_bars, timeout_val,
                breakeven_trigger, trail_atr_mult, trail_tp_atr_mult,
            )

        return _compute_targets_trailing_numba(
            opens, closes, highs, lows, atr_values,
            tp_mult, sl_mult, spread, slippage,
            min_tp_distance, min_sl_distance,
            max_bars, timeout_val,
            breakeven_trigger, trail_atr_mult, trail_tp_atr_mult,
        )


__all__ = ["TrailingStopModifier"]
