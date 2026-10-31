"""Signal Exit Strategy Plugin.

Trades have no take-profit or stop-loss: they close at the close of the first
bar after entry where the indicator's exit column for their side is 1. The
simulator reads the columns named by ``exit_signal_long_column`` and
``exit_signal_short_column`` from the exit params.
"""
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pandas as pd

from fwbg_sdk import BaseExitStrategy, register_exit_strategy

if TYPE_CHECKING:
    from fwbg.core.context import SimulationContext

# TP/SL levels this far from the entry are never reached, so only the exit
# columns (or the end of the data) close a trade.
UNREACHABLE_PRICE_MULTIPLE = 10.0


def next_flag_index(flags: np.ndarray) -> np.ndarray:
    """For each bar, the index of the first flagged bar strictly after it (len(flags) if none)."""
    n = len(flags)
    flagged_at = np.where(flags, np.arange(n), n)
    next_from_here = np.minimum.accumulate(flagged_at[::-1])[::-1]
    return np.append(next_from_here[1:], n)


@register_exit_strategy("signal_exit")
class SignalExitStrategy(BaseExitStrategy):
    """Exit at the close of the bar where the indicator's exit column fires."""

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params=None,
        return_durations: bool = False,
        **kwargs,
    ) -> Tuple[np.ndarray, ...]:
        """Label bar i a win when entering at its close and exiting at the next flagged close pays.

        Entry pays the spread plus half a spread of slippage, as in the simulator.
        Bars without a later exit flag are no win.
        """
        exit_params = dict(params.extra) if params is not None else dict(kwargs)
        closes = df["C"].values.astype(np.float64)
        cost = ctx.spread * 1.5
        n = len(df)
        results = []
        for direction, param in ((1, "exit_signal_long_column"), (-1, "exit_signal_short_column")):
            flags = self._flags(df, exit_params.get(param))
            nxt = next_flag_index(flags)
            has_exit = nxt < n
            exit_close = closes[np.minimum(nxt, n - 1)]
            pnl = direction * (exit_close - closes) - cost
            results.append((np.where(has_exit & (pnl > 0), 1.0, 0.0), np.where(has_exit, nxt - np.arange(n), 0)))
        (tgt_l, dur_l), (tgt_s, dur_s) = results
        if return_durations:
            return tgt_l, tgt_s, dur_l, dur_s
        return tgt_l, tgt_s

    @staticmethod
    def _flags(df: pd.DataFrame, column: Optional[str]) -> np.ndarray:
        if not column:
            return np.zeros(len(df), dtype=bool)
        if column not in df.columns:
            raise KeyError(f"signal_exit: exit column '{column}' is not in the frame")
        return np.nan_to_num(df[column].values.astype(np.float64), nan=0.0) >= 1.0

    def resolve_distances(self, df: pd.DataFrame, tp: float, sl: float, ctx: "SimulationContext"):
        """TP and SL distances no price move reaches; the exit columns close trades."""
        far = np.abs(df["C"].values.astype(np.float64)) * UNREACHABLE_PRICE_MULTIPLE
        return far, far.copy()

    def get_cache_key(self, params: dict) -> str:
        long_col = params.get("exit_signal_long_column")
        short_col = params.get("exit_signal_short_column")
        return f"signal_exit_{long_col}_{short_col}"

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "exit_signal_long_column": None,
            "exit_signal_short_column": None,
            "entry_delay": 0,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "exit_signal_long_column": {
                "type": "string",
                "default": None,
                "description": "Column whose value 1 closes an open long at that bar's close",
            },
            "exit_signal_short_column": {
                "type": "string",
                "default": None,
                "description": "Column whose value 1 closes an open short at that bar's close",
            },
            "entry_delay": {
                "type": "int",
                "default": 0,
                "description": "0 = enter at the signal bar's close, 1 = at the next bar's open",
                "min": 0,
                "max": 1,
                "step": 1,
            },
        }


__all__ = ["SignalExitStrategy", "next_flag_index"]
