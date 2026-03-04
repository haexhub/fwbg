"""
Structural R:R Exit Strategy.

Uses a pre-computed structural stop-loss distance from an indicator column
and derives take-profit as a fixed multiple of that distance (R:R).

Designed to work with the pullback_momentum indicator, which outputs:
  tpm_sl_dist_long  — SL distance at long entry bars (entry − pullback_low + ATR buffer)
  tpm_sl_dist_short — SL distance at short entry bars (pullback_high − entry + ATR buffer)

TP = SL × r_multiple  (default: 2.0, i.e. 2R)

The r_multiple is the primary optimization parameter and maps to `tp_value` in
the grid system (set "tp": <value> in the exit strategy config params).
"""
from typing import Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from fwbg_sdk import BaseExitStrategy, register_exit_strategy
from fwbg.simulation import compute_targets_numba

if TYPE_CHECKING:
    from fwbg.core.context import SimulationContext


@register_exit_strategy("structural_rr")
class StructuralRRExitStrategy(BaseExitStrategy):
    """
    Exit strategy with structural SL (from indicator column) and fixed R:R TP.

    Stop-loss distance is read from pre-computed indicator columns.
    Take-profit = SL distance × r_multiple.
    """

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params=None,
        return_durations: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # --- Parameter resolution (grid params take precedence over kwargs) ---
        r_multiple = 2.0
        timeout_bars = None

        if params is not None:
            # r_multiple is mapped to tp_value in the grid system
            if hasattr(params, "tp_value") and params.tp_value is not None:
                r_multiple = float(params.tp_value)
            # Also check params.extra for frameworks that support it
            if hasattr(params, "extra") and params.extra:
                r_multiple = float(params.extra.get("r_multiple", r_multiple))
            timeout_bars = getattr(params, "timeout_bars", None)

        # kwargs override (direct call without grid)
        r_multiple = float(kwargs.get("r_multiple", r_multiple))
        timeout_bars = kwargs.get("timeout_bars", timeout_bars)

        sl_col_long = kwargs.get("sl_dist_column_long", "tpm_sl_dist_long")
        sl_col_short = kwargs.get("sl_dist_column_short", "tpm_sl_dist_short")
        min_tp_pips = int(kwargs.get("min_tp_pips", 10))
        min_sl_pips = int(kwargs.get("min_sl_pips", 5))

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips

        # --- Build per-bar SL distance array ---
        # Long and short entries fire at different bars, so we merge both columns:
        # at long-entry bars sl_col_long is set (sl_col_short is NaN), and vice versa.
        sl_long_raw = (
            df[sl_col_long].values.astype(np.float64)
            if sl_col_long in df.columns
            else np.full(len(df), np.nan)
        )
        sl_short_raw = (
            df[sl_col_short].values.astype(np.float64)
            if sl_col_short in df.columns
            else np.full(len(df), np.nan)
        )

        # Prefer long SL when both are set (shouldn't happen in normal operation)
        sl_dist_arr = np.where(~np.isnan(sl_long_raw), sl_long_raw, sl_short_raw)
        sl_dist_arr = np.nan_to_num(sl_dist_arr, nan=min_sl_distance)
        sl_dist_arr = np.maximum(sl_dist_arr, min_sl_distance)

        tp_dist_arr = np.maximum(sl_dist_arr * r_multiple, min_tp_distance)

        # --- Run Numba simulation ---
        opn_v = df["O"].values.astype(np.float64)
        cls_v = df["C"].values.astype(np.float64)
        hgh_v = df["H"].values.astype(np.float64)
        low_v = df["L"].values.astype(np.float64)
        slippage = ctx.spread * 0.5
        max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)
        timeout_val = timeout_bars if timeout_bars else 0

        result = compute_targets_numba(
            opn_v, cls_v, hgh_v, low_v,
            tp_dist_arr, sl_dist_arr, ctx.spread, slippage,
            max_bars, timeout_val,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve per-bar TP/SL distances from indicator columns.

        tp is used as r_multiple, sl is ignored (SL comes from the indicator).
        """
        r_multiple = tp if tp else 2.0

        ep = ctx.exit_params or {}
        sl_col_long = ep.get("sl_dist_column_long", "tpm_sl_dist_long")
        sl_col_short = ep.get("sl_dist_column_short", "tpm_sl_dist_short")
        min_sl_pips = int(ep.get("min_sl_pips", 5))
        min_tp_pips = int(ep.get("min_tp_pips", 10))

        min_sl_distance = ctx.spread * min_sl_pips
        min_tp_distance = ctx.spread * min_tp_pips

        sl_long_raw = (
            df[sl_col_long].values.astype(np.float64)
            if sl_col_long in df.columns
            else np.full(len(df), np.nan)
        )
        sl_short_raw = (
            df[sl_col_short].values.astype(np.float64)
            if sl_col_short in df.columns
            else np.full(len(df), np.nan)
        )

        sl_dist = np.where(~np.isnan(sl_long_raw), sl_long_raw, sl_short_raw)
        sl_dist = np.nan_to_num(sl_dist, nan=min_sl_distance)
        sl_dist = np.maximum(sl_dist, min_sl_distance)

        tp_dist = np.maximum(sl_dist * r_multiple, min_tp_distance)
        return tp_dist, sl_dist

    def get_cache_key(self, params: dict) -> str:
        r = params.get("r_multiple", params.get("tp", 2.0))
        timeout = params.get("timeout_bars")
        timeout_str = str(timeout) if timeout else "none"
        return f"structural_rr_r{r}_to{timeout_str}"

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "r_multiple": 2.0,
            "sl_dist_column_long": "tpm_sl_dist_long",
            "sl_dist_column_short": "tpm_sl_dist_short",
            "min_tp_pips": 10,
            "min_sl_pips": 5,
            "timeout_bars": None,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "r_multiple": {
                "type": "float",
                "default": 2.0,
                "description": (
                    "Risk-reward ratio. TP = structural SL distance × r_multiple. "
                    "Maps to 'tp' in the grid config for optimization."
                ),
                "min": 0.5,
                "max": 5.0,
                "step": 0.5,
            },
            "sl_dist_column_long": {
                "type": "string",
                "default": "tpm_sl_dist_long",
                "description": "Column containing the structural SL distance for long trades.",
            },
            "sl_dist_column_short": {
                "type": "string",
                "default": "tpm_sl_dist_short",
                "description": "Column containing the structural SL distance for short trades.",
            },
            "min_tp_pips": {
                "type": "int",
                "default": 10,
                "description": "Minimum TP distance in pips (spread multiples). Floor for very tight structures.",
                "min": 1,
                "max": 100,
                "step": 1,
            },
            "min_sl_pips": {
                "type": "int",
                "default": 5,
                "description": "Minimum SL distance in pips (spread multiples). Floor for very tight structures.",
                "min": 1,
                "max": 100,
                "step": 1,
            },
            "timeout_bars": {
                "type": "int",
                "default": None,
                "description": "Close trade after N bars if TP/SL not hit. None = no timeout.",
                "min": 1,
                "max": 500,
                "step": 1,
                "required": False,
            },
        }


__all__ = ["StructuralRRExitStrategy"]
