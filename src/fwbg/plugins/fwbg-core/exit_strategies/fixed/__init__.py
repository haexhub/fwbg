"""
Fixed Exit Strategy Plugin.

Verwendet fixe TP/SL-Werte basierend auf Spread-Multiplikatoren.
"""
from typing import Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

from fwbg_sdk import BaseExitStrategy, register_exit_strategy
from fwbg.simulation import compute_targets_numba

if TYPE_CHECKING:
    from fwbg.core.context import SimulationContext


@register_exit_strategy("fixed")
class FixedExitStrategy(BaseExitStrategy):
    """
    Exit-Strategie mit fixen TP/SL-Werten.

    TP und SL werden als Multiplikatoren des Spreads angegeben.
    Beispiel: tp=30 bei spread=0.0001 -> TP = 30 Pips = 0.003
    """

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params=None,
        return_durations: bool = False,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten (Spalten: O, H, L, C)
            ctx: SimulationContext mit spread, max_trade_bars, etc.
            params: GridParams mit tp_value, sl_value, timeout_bars
            return_durations: Wenn True, auch Durations zurückgeben

        Returns:
            (targets_long, targets_short) oder
            (targets_long, targets_short, durations_long, durations_short)
        """
        if params is not None:
            tp = params.tp_value
            sl = params.sl_value
            timeout_bars = params.timeout_bars
        else:
            tp = kwargs.get("tp", 30)
            sl = kwargs.get("sl", 20)
            timeout_bars = kwargs.get("timeout_bars")

        opn_v = df["O"].values.astype(np.float64)
        cls_v = df["C"].values.astype(np.float64)
        hgh_v = df["H"].values.astype(np.float64)
        low_v = df["L"].values.astype(np.float64)

        n = len(df)
        tp_distances = np.full(n, ctx.spread * tp)
        sl_distances = np.full(n, ctx.spread * sl)
        slippage = ctx.spread * 0.5
        max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)
        timeout_val = timeout_bars if timeout_bars else 0

        # === ENTRY MODIFIER DISPATCH ===
        entry_modifier_name = getattr(ctx, "entry_modifier", None)
        if entry_modifier_name and isinstance(entry_modifier_name, str):
            from fwbg.core.registry import get_entry_modifier
            entry_mod_cls = get_entry_modifier(entry_modifier_name)
            entry_mod = entry_mod_cls()
            entry_mod_params = getattr(ctx, "entry_modifier_params", {}) or {}

            # Fixed strategy uses spread-based distances, but entry modifier
            # expects ATR arrays + multipliers. Compute ATR for the modifier.
            if "_atr" in df.columns:
                atr_v = df["_atr"].values.astype(np.float64)
            elif "vol_atr" in df.columns:
                atr_v = df["vol_atr"].values.astype(np.float64)
            else:
                import ta
                atr_series = ta.volatility.average_true_range(
                    df["H"], df["L"], df["C"], window=14
                )
                atr_v = atr_series.values.astype(np.float64)
            atr_v = np.nan_to_num(atr_v, nan=0.0)

            # Pass through trailing params from exit modifier
            modifier_params = getattr(ctx, "exit_modifier_params", {}) or {}
            em_breakeven = modifier_params.get("breakeven_trigger", 0.0)
            em_trail = modifier_params.get("trail_atr_mult", 0.0)
            em_trail_tp = modifier_params.get("trail_tp_atr_mult", 0.0)

            # Fixed strategy: pass tp_mult/sl_mult=0 so min distances act as
            # the effective distances (max(atr*0, fixed_dist) = fixed_dist).
            return entry_mod.compute_targets(
                opn_v, cls_v, hgh_v, low_v, atr_v,
                0.0, 0.0,
                ctx.spread, slippage,
                tp_distances[0], sl_distances[0],
                max_bars, timeout_val,
                return_durations=return_durations,
                breakeven_trigger=em_breakeven,
                trail_atr_mult=em_trail,
                trail_tp_atr_mult=em_trail_tp,
                **entry_mod_params,
            )

        result = compute_targets_numba(
            opn_v, cls_v, hgh_v, low_v,
            tp_distances, sl_distances, ctx.spread, slippage,
            max_bars, timeout_val
        )
        if return_durations:
            return result  # (tgt_l, tgt_s, dur_l, dur_s)
        return result[0], result[1]

    def resolve_distances(
        self,
        df: pd.DataFrame,
        tp: float,
        sl: float,
        ctx: "SimulationContext",
    ):
        """Fixe Distanzen: spread * Multiplikator, konstant für alle Bars."""
        n = len(df)
        return np.full(n, ctx.spread * tp), np.full(n, ctx.spread * sl)

    def get_cache_key(self, params: dict) -> str:
        """
        Gibt eindeutigen Cache-Key für diese Parameter zurück.

        Format: "fixed_tp{tp}_sl{sl}_to{timeout}"
        """
        tp = params.get("tp", 0)
        sl = params.get("sl", 0)
        timeout = params.get("timeout_bars")
        timeout_str = str(timeout) if timeout else "none"
        return f"fixed_tp{int(tp)}_sl{int(sl)}_to{timeout_str}"

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "tp": 30,
            "sl": 20,
            "timeout_bars": None,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "tp": {
                "type": "int",
                "default": 30,
                "description": "Take-profit distance as spread multiplier (e.g. 30 = 30 pips at spread 0.0001)",
                "min": 1,
                "max": 1000,
                "step": 1,
            },
            "sl": {
                "type": "int",
                "default": 20,
                "description": "Stop-loss distance as spread multiplier (e.g. 20 = 20 pips at spread 0.0001)",
                "min": 1,
                "max": 1000,
                "step": 1,
            },
            "timeout_bars": {
                "type": "int",
                "default": None,
                "description": "Close trade after N bars if neither TP nor SL is hit (None = no timeout)",
                "min": 1,
                "max": 500,
                "step": 1,
                "required": False,
            },
        }


__all__ = ["FixedExitStrategy"]
