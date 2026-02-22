"""
ORB-Based Exit Strategy Plugin.

SL = orb_sl_dist column (half the ORB range) when available, fallback to ATR * sl_mult.
TP = ATR * tp_mult (optimizer searches best multiple).

Design rationale:
  - SL is anchored to the ORB range: if price falls below ORB Low after an
    upside breakout, the ORB thesis is invalidated — the SL should be exactly there.
    For midpoint entry: SL distance = (or_high - or_low) / 2 = orb_sl_dist.
  - TP remains ATR-based so the optimizer can find the best risk-reward multiple.
"""
from typing import Dict, Any, Iterator, Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

from fwbg_sdk import BaseExitStrategy, register_exit_strategy
from fwbg.simulation import _simulate_trade_numba
from fwbg.core import GridParams

if TYPE_CHECKING:
    from fwbg.core.context import SimulationContext


@register_exit_strategy("orb_based")
class OrbExitStrategy(BaseExitStrategy):
    """
    Exit strategy for ORB (Opening Range Breakout) setups.

    SL: Uses `orb_sl_dist` (or any `*_sl_dist`) column when present — this equals
        half the ORB range, i.e. the distance from the midpoint entry to the ORB Low.
        Falls back to ATR * sl_mult when no sl_dist column is found.

    TP: ATR * tp_mult (optimizer searches 1.5x–4x).

    The default sl_mult = 1.0 means "use orb_sl_dist as-is", which gives the
    exact ORB-range stop. Values > 1.0 add a buffer above/below the ORB boundary.
    """

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params: Union[GridParams, None] = None,
        tp_mult: float = 2.0,
        sl_mult: float = 1.0,
        atr_period: int = 14,
        min_tp_pips: int = 8,
        min_sl_pips: int = 5,
        timeout_bars: int = None,
        return_durations: bool = False,
        **kwargs,
    ) -> tuple:
        """Compute win/loss targets using ORB-range SL and ATR-based TP."""
        if params is not None:
            tp_mult = params.tp_value
            sl_mult = params.sl_value
            if params.timeout_bars is not None:
                timeout_bars = params.timeout_bars
            if params.extra:
                atr_period = params.extra.get("atr_period", atr_period)
                min_tp_pips = params.extra.get("min_tp_pips", min_tp_pips)
                min_sl_pips = params.extra.get("min_sl_pips", min_sl_pips)

        opn_v = df["O"].values.astype(np.float64)
        cls_v = df["C"].values.astype(np.float64)
        hgh_v = df["H"].values.astype(np.float64)
        low_v = df["L"].values.astype(np.float64)

        atr_v = self._get_atr(df, atr_period)
        sl_dist_v = self._get_sl_dist(df, atr_v, sl_mult)

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips
        slippage = ctx.spread * 0.5
        max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)
        timeout_val = timeout_bars if timeout_bars else 0

        n = len(cls_v)
        targets_long = np.zeros(n, dtype=np.float64)
        targets_short = np.zeros(n, dtype=np.float64)
        if return_durations:
            durations_long = np.zeros(n, dtype=np.int64)
            durations_short = np.zeros(n, dtype=np.int64)

        for i in range(n - 1):
            tp_distance = max(atr_v[i] * tp_mult, min_tp_distance)
            sl_distance = max(sl_dist_v[i], min_sl_distance)

            result_long, exit_long, _, _ = _simulate_trade_numba(
                opn_v, cls_v, hgh_v, low_v, i, 1,
                tp_distance, sl_distance, ctx.spread, slippage,
                max_bars, timeout_val,
            )
            if result_long == 1.0:
                targets_long[i] = 1.0
            if return_durations:
                durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

            result_short, exit_short, _, _ = _simulate_trade_numba(
                opn_v, cls_v, hgh_v, low_v, i, -1,
                tp_distance, sl_distance, ctx.spread, slippage,
                max_bars, timeout_val,
            )
            if result_short == 1.0:
                targets_short[i] = 1.0
            if return_durations:
                durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

        if return_durations:
            return targets_long, targets_short, durations_long, durations_short
        return targets_long, targets_short

    def resolve_distances(
        self,
        df: pd.DataFrame,
        tp: float,
        sl: float,
        ctx: "SimulationContext",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """TP = ATR * tp_mult per bar; SL = orb_sl_dist * sl_mult or ATR * sl_mult fallback."""
        exit_params = ctx.exit_params if ctx.exit_params else {}
        atr_period = exit_params.get("atr_period", 14)
        min_tp_pips = exit_params.get("min_tp_pips", 8)
        min_sl_pips = exit_params.get("min_sl_pips", 5)

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips

        atr_v = self._get_atr(df, atr_period)
        sl_raw = self._get_sl_dist(df, atr_v, sl)

        tp_dists = np.maximum(atr_v * tp, min_tp_distance)
        sl_dists = np.maximum(sl_raw, min_sl_distance)
        return tp_dists, sl_dists

    def iterate_grid(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> Iterator[dict]:
        """Iterate over tp_mult x sl_mult (buffer) combinations.

        TP grid: optimizer finds best ATR multiple (default 1.5-4.0).
        SL grid: defaults to [1.0] (exact ORB range, no buffer).
        """
        tp_values = grid_config.get("tp_mult", grid_config.get("tp", [1.5, 2.0, 2.5, 3.0, 4.0]))
        sl_values = grid_config.get("sl_mult", grid_config.get("sl", [1.0]))
        timeout_values = grid_config.get("timeout_bars", [None])
        min_rrr = grid_config.get("min_rrr", 0)

        exit_params = ctx.exit_params if ctx.exit_params else {}
        atr_period = exit_params.get("atr_period", 14)
        min_tp_pips = exit_params.get("min_tp_pips", 8)
        min_sl_pips = exit_params.get("min_sl_pips", 5)

        for tp_mult in tp_values:
            for sl_mult in sl_values:
                rrr = tp_mult / sl_mult if sl_mult > 0 else 0
                if min_rrr > 0 and rrr < min_rrr:
                    continue
                for timeout in timeout_values:
                    yield {
                        "tp_mult": float(tp_mult),
                        "sl_mult": float(sl_mult),
                        "timeout_bars": timeout,
                        "atr_period": atr_period,
                        "min_tp_pips": min_tp_pips,
                        "min_sl_pips": min_sl_pips,
                    }

    def get_cache_key(self, params: dict) -> str:
        tp_mult = params.get("tp_mult", 0)
        sl_mult = params.get("sl_mult", 0)
        timeout = params.get("timeout_bars")
        timeout_str = str(timeout) if timeout else "none"
        return f"orb_tp{tp_mult:.2f}_sl{sl_mult:.2f}_to{timeout_str}"

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "tp_mult": 2.0,
            "sl_mult": 1.0,
            "atr_period": 14,
            "min_tp_pips": 8,
            "min_sl_pips": 5,
            "timeout_bars": None,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "tp_mult": {
                "type": "float",
                "default": 2.0,
                "description": "ATR multiplier for take-profit (optimizer searches best value)",
                "min": 0.5,
                "max": 10.0,
                "step": 0.5,
            },
            "sl_mult": {
                "type": "float",
                "default": 1.0,
                "description": (
                    "Multiplier on orb_sl_dist (= half ORB range). "
                    "1.0 = exact ORB Low stop, >1.0 adds buffer beyond ORB Low. "
                    "Fallback to ATR * sl_mult if no orb_sl_dist column present."
                ),
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for TP calculation and SL fallback",
                "min": 1,
                "max": 200,
                "step": 1,
            },
            "min_tp_pips": {
                "type": "int",
                "default": 8,
                "description": "Minimum TP in spread multiples (floor for low-vol environments)",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "min_sl_pips": {
                "type": "int",
                "default": 5,
                "description": "Minimum SL in spread multiples (floor for low-vol environments)",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "timeout_bars": {
                "type": "int",
                "default": None,
                "description": "Close trade after N bars if neither TP nor SL is hit",
                "min": 1,
                "max": 500,
                "step": 1,
                "required": False,
            },
        }

    # --- Private helpers ---

    @staticmethod
    def _get_atr(df: pd.DataFrame, atr_period: int) -> np.ndarray:
        """Return ATR array: use precomputed column if available, else compute."""
        if "_atr" in df.columns:
            atr_v = df["_atr"].values.astype(np.float64)
        elif "vol_atr" in df.columns:
            atr_v = df["vol_atr"].values.astype(np.float64)
        else:
            import ta
            atr_series = ta.volatility.average_true_range(
                df["H"], df["L"], df["C"], window=atr_period
            )
            atr_v = atr_series.values.astype(np.float64)
        return np.nan_to_num(atr_v, nan=0.0)

    @staticmethod
    def _get_sl_dist(df: pd.DataFrame, atr_v: np.ndarray, sl_mult: float) -> np.ndarray:
        """Return raw SL distances: use orb_sl_dist * sl_mult if available, else ATR * sl_mult.

        Detects any column ending in '_sl_dist' (covers both orb_sl_dist and
        session-specific orb_s08_sl_dist etc.).
        """
        sl_col = next(
            (c for c in df.columns if c == "orb_sl_dist" or c.endswith("_sl_dist")),
            None,
        )
        if sl_col is not None:
            sl_vals = df[sl_col].fillna(pd.Series(atr_v * sl_mult, index=df.index)).values
            return (sl_vals * sl_mult).astype(np.float64)
        return (atr_v * sl_mult).astype(np.float64)


__all__ = ["OrbExitStrategy"]
