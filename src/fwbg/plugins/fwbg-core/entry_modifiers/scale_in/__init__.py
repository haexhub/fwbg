"""
Scale-In Entry Modifier.

Fügt bei konfigurierbaren Retracement-Levels zusätzliche Positionen hinzu.
Nach jedem Nachkauf wird der TP basierend auf dem neuen gewichteten
Durchschnittspreis angepasst.

Konfiguration in der Strategy-JSON:
    {
      "exit_strategy": "atr_based",
      "entry_modifier": "scale_in",
      "entry_modifier_params": {
        "levels": [0.2, 0.4, 0.6],
        "qty_multiplier": 1.0
      }
    }

Levels:
    Retracement als Bruchteil der Entry→SL-Distanz.
    0.2 = Nachkauf bei 20% Retracement Richtung SL.

Qty Multiplier:
    Positionsgröße pro Nachkauf relativ zur Initial-Position.
    1.0 = gleiche Größe, 0.5 = halbe Größe.
"""
import pathlib
from typing import Tuple

import numpy as np
from numba import njit

from fwbg_sdk import BaseEntryModifier, register_entry_modifier
from fwbg.simulation import _simulate_trade_scale_in_numba

_CACHE_DIR = pathlib.Path(__file__).parent / "__pycache__"
_MAX_LEVELS = 10


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


def _pack_levels(levels):
    """Pack Python list into fixed-size Numba-compatible array."""
    arr = np.full(_MAX_LEVELS, -1.0, dtype=np.float64)
    n = min(len(levels), _MAX_LEVELS)
    for i in range(n):
        arr[i] = float(levels[i])
    return arr, n


# ---------------------------------------------------------------------------
# Numba JIT wrappers
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=False)
def _compute_targets_scale_in_numba(
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
    scale_levels: np.ndarray,
    n_levels: int,
    scale_qty_mult: float,
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
        trail_dist = atr * trail_atr_mult if trail_atr_mult > 0.0 else 0.0
        trail_tp_dist = atr * trail_tp_atr_mult if trail_tp_atr_mult > 0.0 else 0.0

        result_long, _, _, _, _, _, _ = _simulate_trade_scale_in_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val,
            scale_levels, n_levels, scale_qty_mult,
            breakeven_trigger, trail_dist, trail_tp_dist,
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        result_short, _, _, _, _, _, _ = _simulate_trade_scale_in_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val,
            scale_levels, n_levels, scale_qty_mult,
            breakeven_trigger, trail_dist, trail_tp_dist,
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short


@njit(cache=True, parallel=False)
def _compute_targets_scale_in_with_durations_numba(
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
    scale_levels: np.ndarray,
    n_levels: int,
    scale_qty_mult: float,
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
        trail_dist = atr * trail_atr_mult if trail_atr_mult > 0.0 else 0.0
        trail_tp_dist = atr * trail_tp_atr_mult if trail_tp_atr_mult > 0.0 else 0.0

        result_long, exit_long, _, _, _, _, _ = _simulate_trade_scale_in_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val,
            scale_levels, n_levels, scale_qty_mult,
            breakeven_trigger, trail_dist, trail_tp_dist,
        )
        if result_long == 1.0:
            targets_long[i] = 1.0
        durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

        result_short, exit_short, _, _, _, _, _ = _simulate_trade_scale_in_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_val,
            scale_levels, n_levels, scale_qty_mult,
            breakeven_trigger, trail_dist, trail_tp_dist,
        )
        if result_short == 1.0:
            targets_short[i] = 1.0
        durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

    return targets_long, targets_short, durations_long, durations_short


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

@register_entry_modifier("scale_in")
class ScaleInModifier(BaseEntryModifier):
    """
    Scale-In Entry Modifier.

    Fügt Positionen bei Retracement-Levels hinzu (Bruchteil der Entry→SL-Distanz).
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
        # Scale-in params
        levels=None,
        qty_multiplier: float = 1.0,
        # Trailing params (pass-through from exit modifier)
        breakeven_trigger: float = 0.0,
        trail_atr_mult: float = 0.0,
        trail_tp_atr_mult: float = 0.0,
        **kwargs,
    ):
        if levels is None:
            levels = [0.2, 0.4, 0.6]

        scale_arr, n_levels = _pack_levels(levels)

        if return_durations:
            return _call_numba(
                _compute_targets_scale_in_with_durations_numba,
                opens, closes, highs, lows, atr_values,
                tp_mult, sl_mult, spread, slippage,
                min_tp_distance, min_sl_distance,
                max_bars, timeout_val,
                scale_arr, n_levels, float(qty_multiplier),
                float(breakeven_trigger), float(trail_atr_mult), float(trail_tp_atr_mult),
            )

        return _call_numba(
            _compute_targets_scale_in_numba,
            opens, closes, highs, lows, atr_values,
            tp_mult, sl_mult, spread, slippage,
            min_tp_distance, min_sl_distance,
            max_bars, timeout_val,
            scale_arr, n_levels, float(qty_multiplier),
            float(breakeven_trigger), float(trail_atr_mult), float(trail_tp_atr_mult),
        )

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "levels": [0.2, 0.4, 0.6],
            "qty_multiplier": 1.0,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "levels": {
                "type": "list[float]",
                "default": [0.2, 0.4, 0.6],
                "description": "Retracement-Levels als Bruchteil der Entry→SL-Distanz. 0.2 = Nachkauf bei 20% Richtung SL.",
            },
            "qty_multiplier": {
                "type": "float",
                "default": 1.0,
                "description": "Positionsgröße pro Nachkauf relativ zur Initial-Position. 1.0 = gleiche Größe.",
                "min": 0.1,
                "max": 3.0,
                "step": 0.1,
            },
        }


__all__ = ["ScaleInModifier"]
