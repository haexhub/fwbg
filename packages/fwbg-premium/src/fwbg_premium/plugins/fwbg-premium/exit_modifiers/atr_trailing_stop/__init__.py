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
import pathlib
from typing import Tuple

import numpy as np
from numba import njit

from fwbg_sdk import BaseExitModifier, register_exit_modifier
from fwbg.simulation import _simulate_trade_trailing_numba  # shared kernel

_CACHE_DIR = pathlib.Path(__file__).parent / "__pycache__"


def _clear_numba_cache():
    for pattern in ("*.nbi", "*.nbc"):
        for f in _CACHE_DIR.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


def _call_numba(func, *args):
    """Call a Numba-JIT function with automatic stale-cache recovery."""
    try:
        return func(*args)
    except ModuleNotFoundError:
        _clear_numba_cache()
        return func(*args)


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
            return _call_numba(_compute_targets_trailing_with_durations_numba,
                opens, closes, highs, lows, atr_values,
                tp_mult, sl_mult, spread, slippage,
                min_tp_distance, min_sl_distance,
                max_bars, timeout_val,
                breakeven_trigger, trail_atr_mult, trail_tp_atr_mult,
            )

        return _call_numba(_compute_targets_trailing_numba,
            opens, closes, highs, lows, atr_values,
            tp_mult, sl_mult, spread, slippage,
            min_tp_distance, min_sl_distance,
            max_bars, timeout_val,
            breakeven_trigger, trail_atr_mult, trail_tp_atr_mult,
        )


__all__ = ["TrailingStopModifier"]
