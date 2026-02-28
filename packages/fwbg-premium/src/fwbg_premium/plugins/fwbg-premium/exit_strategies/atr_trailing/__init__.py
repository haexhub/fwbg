"""
ATR-Based Exit Strategy with Trailing Stop + Breakeven Stop.

Extends atr_based with two additional mechanics:

  Breakeven Stop:
    Once price moves `breakeven_trigger` fraction of TP distance in our
    favour, SL is moved to entry price.  A winning trade can no longer
    become a loss — worst case is breakeven (minus spread).

  Trailing Stop:
    After breakeven activation the SL follows the best price seen since
    entry at a fixed distance of `trail_atr_mult * ATR`.  Locks in
    progressively more profit as the trade runs.

Both can be combined or used independently:
  - breakeven_trigger=0.5, trail_atr_mult=0.5  (recommended)
  - breakeven_trigger=0.5, trail_atr_mult=0.0  (breakeven only)
  - breakeven_trigger=0.0, trail_atr_mult=0.5  (trailing from entry)
"""
from typing import Dict, Any, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from numba import njit

from fwbg_sdk import BaseExitStrategy, register_exit_strategy
from fwbg.core import GridParams

if TYPE_CHECKING:
    from fwbg.core.context import SimulationContext


# ---------------------------------------------------------------------------
# Numba helpers (all simulation logic stays inside the plugin)
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
) -> tuple:
    """
    Single-trade simulation mit Breakeven- und Trailing-Stop.

    Args:
        breakeven_trigger: Bruchteil der TP-Distanz nach dem SL auf Entry
                           gezogen wird (0.0 = kein Breakeven, Trailing startet
                           sofort wenn trail_distance > 0).
        trail_distance:    Absoluter Abstand des Trailing-Stops vom besten
                           erreichten Preis in Preiseinheiten
                           (0.0 = kein Trailing, nur Breakeven-Stop).

    Returns:
        (result, exit_idx, exit_price, exit_reason)
        result:      1.0=Win, -1.0=Loss, 0.0=Kein Exit
        exit_reason: 0=TP, 1=SL/Trail/BE, 2=Timeout, -1=Kein Exit
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

    # Trailing startet sofort wenn kein BE-Trigger konfiguriert
    best_price = entry
    trailing_active = breakeven_trigger <= 0.0

    for j in range(entry_idx, end_idx):
        # Timeout (höchste Priorität)
        if timeout_idx > 0 and j >= timeout_idx:
            exit_price = closes[j]
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            return (1.0 if pnl > 0 else -1.0), j, exit_price, 2

        # Besten Preis nachführen
        if direction == 1:
            if highs[j] > best_price:
                best_price = highs[j]
        else:
            if lows[j] < best_price:
                best_price = lows[j]

        # Breakeven-Trigger prüfen
        if not trailing_active and breakeven_trigger > 0.0:
            if direction == 1 and best_price >= be_trigger_price:
                trailing_active = True
                if entry > sl:
                    sl = entry  # SL mindestens auf Entry-Preis
            elif direction == -1 and best_price <= be_trigger_price:
                trailing_active = True
                if entry < sl:
                    sl = entry

        # Trailing-Stop nachziehen
        if trailing_active and trail_distance > 0.0:
            if direction == 1:
                new_sl = best_price - trail_distance
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = best_price + trail_distance
                if new_sl < sl:
                    sl = new_sl

        # TP/SL prüfen
        if direction == 1:
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

        if sl_hit:
            # Win wenn Trailing-SL über den Entry-Preis gezogen wurde
            if direction == 1:
                result = 1.0 if sl > entry else -1.0
            else:
                result = 1.0 if sl < entry else -1.0
            return result, j, sl, 1

        if tp_hit:
            return 1.0, j, tp, 0

    return 0.0, -1, 0.0, -1


@njit(cache=True, parallel=False)
def _compute_targets_atr_trailing_numba(
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
    breakeven_trigger: float,
    trail_atr_mult: float,
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

        result_long, _, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars, breakeven_trigger, trail_distance,
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        result_short, _, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars, breakeven_trigger, trail_distance,
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short


@njit(cache=True, parallel=False)
def _compute_targets_atr_trailing_with_durations_numba(
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
    breakeven_trigger: float,
    trail_atr_mult: float,
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

        result_long, exit_long, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars, breakeven_trigger, trail_distance,
        )
        if result_long == 1.0:
            targets_long[i] = 1.0
        durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

        result_short, exit_short, _, _ = _simulate_trade_trailing_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars, breakeven_trigger, trail_distance,
        )
        if result_short == 1.0:
            targets_short[i] = 1.0
        durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

    return targets_long, targets_short, durations_long, durations_short


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

@register_exit_strategy("atr_trailing")
class AtrTrailingExitStrategy(BaseExitStrategy):
    """
    ATR-basierte Exit-Strategie mit Breakeven- und Trailing-Stop.

    Breakeven Stop:
        Sobald Preis `breakeven_trigger × tp_distance` in die Gewinnzone läuft,
        wird SL auf Entry-Preis gezogen. Einmal profitable Trades können nicht
        mehr als Verlust enden.

    Trailing Stop:
        Nach Breakeven-Aktivierung folgt SL dem besten erreichten Preis mit
        `trail_atr_mult × ATR` Abstand und sichert wachsende Gewinne.

    Beide Parameter können unabhängig konfiguriert werden:
        breakeven_trigger=0.5, trail_atr_mult=0.5  → beide aktiv (empfohlen)
        breakeven_trigger=0.5, trail_atr_mult=0.0  → nur Breakeven
        breakeven_trigger=0.0, trail_atr_mult=0.5  → nur Trailing (ab Entry)
    """

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params: Union[GridParams, None] = None,
        tp_mult: float = 3.0,
        sl_mult: float = 1.5,
        atr_period: int = 14,
        min_tp_pips: int = 8,
        min_sl_pips: int = 12,
        timeout_bars: int = None,
        breakeven_trigger: float = 0.5,
        trail_atr_mult: float = 0.5,
        return_durations: bool = False,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if params is not None:
            tp_mult = params.tp_value
            sl_mult = params.sl_value
            if params.timeout_bars is not None:
                timeout_bars = params.timeout_bars
            if params.extra:
                atr_period = params.extra.get("atr_period", atr_period)
                min_tp_pips = params.extra.get("min_tp_pips", min_tp_pips)
                min_sl_pips = params.extra.get("min_sl_pips", min_sl_pips)
                breakeven_trigger = params.extra.get("breakeven_trigger", breakeven_trigger)
                trail_atr_mult = params.extra.get("trail_atr_mult", trail_atr_mult)

        opn_v = df["O"].values.astype(np.float64)
        cls_v = df["C"].values.astype(np.float64)
        hgh_v = df["H"].values.astype(np.float64)
        low_v = df["L"].values.astype(np.float64)

        if "_atr" in df.columns:
            atr_v = df["_atr"].values.astype(np.float64)
        elif "vol_atr" in df.columns:
            atr_v = df["vol_atr"].values.astype(np.float64)
        else:
            import ta
            atr_v = ta.volatility.average_true_range(
                df["H"], df["L"], df["C"], window=atr_period
            ).values.astype(np.float64)

        atr_v = np.nan_to_num(atr_v, nan=0.0)

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips
        slippage = ctx.spread * 0.5
        max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)
        timeout_val = timeout_bars if timeout_bars else 0

        if return_durations:
            return _compute_targets_atr_trailing_with_durations_numba(
                opn_v, cls_v, hgh_v, low_v,
                atr_v, tp_mult, sl_mult,
                ctx.spread, slippage,
                min_tp_distance, min_sl_distance,
                max_bars, timeout_val,
                breakeven_trigger, trail_atr_mult,
            )

        return _compute_targets_atr_trailing_numba(
            opn_v, cls_v, hgh_v, low_v,
            atr_v, tp_mult, sl_mult,
            ctx.spread, slippage,
            min_tp_distance, min_sl_distance,
            max_bars, timeout_val,
            breakeven_trigger, trail_atr_mult,
        )

    def resolve_distances(
        self,
        df: pd.DataFrame,
        tp: float,
        sl: float,
        ctx: "SimulationContext",
    ):
        """ATR-basierte TP/SL-Distanzen (gleich wie atr_based)."""
        exit_params = ctx.exit_params if ctx.exit_params else {}
        atr_period = exit_params.get("atr_period", 14)
        min_tp_pips = exit_params.get("min_tp_pips", 8)
        min_sl_pips = exit_params.get("min_sl_pips", 12)

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips

        if "_atr" in df.columns:
            atr_v = df["_atr"].values.astype(np.float64)
        elif "vol_atr" in df.columns:
            atr_v = df["vol_atr"].values.astype(np.float64)
        else:
            import ta
            atr_v = ta.volatility.average_true_range(
                df["H"], df["L"], df["C"], window=atr_period
            ).values.astype(np.float64)

        atr_v = np.nan_to_num(atr_v, nan=0.0)
        tp_dists = np.maximum(atr_v * tp, min_tp_distance)
        sl_dists = np.maximum(atr_v * sl, min_sl_distance)
        return tp_dists, sl_dists

    def get_cache_key(self, params: dict) -> str:
        tp = params.get("tp_mult", 0)
        sl = params.get("sl_mult", 0)
        timeout = params.get("timeout_bars")
        be = params.get("breakeven_trigger", 0.5)
        trail = params.get("trail_atr_mult", 0.5)
        timeout_str = str(timeout) if timeout else "none"
        return f"atr_trail_tp{tp:.2f}_sl{sl:.2f}_be{be:.2f}_tr{trail:.2f}_to{timeout_str}"

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "tp_mult": 3.0,
            "sl_mult": 1.5,
            "atr_period": 14,
            "min_tp_pips": 8,
            "min_sl_pips": 12,
            "timeout_bars": None,
            "breakeven_trigger": 0.5,
            "trail_atr_mult": 0.5,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "tp_mult": {
                "type": "float", "default": 3.0,
                "description": "ATR-Multiplikator für Take-Profit",
                "min": 0.1, "max": 20.0, "step": 0.1,
            },
            "sl_mult": {
                "type": "float", "default": 1.5,
                "description": "ATR-Multiplikator für Stop-Loss",
                "min": 0.1, "max": 20.0, "step": 0.1,
            },
            "atr_period": {
                "type": "int", "default": 14,
                "description": "ATR-Periode (Fallback wenn keine _atr Spalte)",
                "min": 1, "max": 1000, "step": 1,
            },
            "min_tp_pips": {
                "type": "int", "default": 8,
                "description": "Mindest-TP in Spread-Vielfachen",
                "min": 1, "max": 500, "step": 1,
            },
            "min_sl_pips": {
                "type": "int", "default": 12,
                "description": "Mindest-SL in Spread-Vielfachen",
                "min": 1, "max": 500, "step": 1,
            },
            "timeout_bars": {
                "type": "int", "default": None,
                "description": "Trade nach N Bars schließen wenn weder TP noch SL",
                "min": 1, "max": 500, "step": 1, "required": False,
            },
            "breakeven_trigger": {
                "type": "float", "default": 0.5,
                "description": (
                    "Bruchteil der TP-Distanz ab dem SL auf Entry gezogen wird. "
                    "0.5 = Breakeven-Aktivierung wenn Preis 50% Richtung TP läuft. "
                    "0.0 = deaktiviert (Trailing startet sofort)."
                ),
                "min": 0.0, "max": 1.0, "step": 0.05,
            },
            "trail_atr_mult": {
                "type": "float", "default": 0.5,
                "description": (
                    "ATR-Multiplikator für den Trailing-Stop-Abstand vom besten Preis. "
                    "0.5 = Trailing-Stop 0.5×ATR hinter dem Hochpunkt (Long). "
                    "0.0 = nur Breakeven-Stop, kein Trailing."
                ),
                "min": 0.0, "max": 5.0, "step": 0.1,
            },
        }


__all__ = ["AtrTrailingExitStrategy"]
