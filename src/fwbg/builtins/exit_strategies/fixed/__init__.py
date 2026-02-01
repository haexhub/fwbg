"""
Fixed Exit Strategy Plugin.

Verwendet fixe TP/SL-Werte basierend auf Spread-Multiplikatoren.
"""
from typing import Dict, Any, Iterator, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

from fwbg.plugins import BaseExitStrategy
from fwbg.core import register_exit_strategy
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
        tp: float = 30,
        sl: float = 20,
        timeout_bars: int = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten (Spalten: O, H, L, C)
            ctx: SimulationContext mit spread, max_trade_bars, etc.
            tp: Take-Profit als Spread-Multiplikator
            sl: Stop-Loss als Spread-Multiplikator
            timeout_bars: Optional - Trade schließen nach X Bars

        Returns:
            (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
        """
        # OHLC-Arrays extrahieren
        opn_v = df["O"].values.astype(np.float64)
        cls_v = df["C"].values.astype(np.float64)
        hgh_v = df["H"].values.astype(np.float64)
        low_v = df["L"].values.astype(np.float64)

        # Distanzen berechnen (Spread * Multiplikator)
        tp_distance = ctx.spread * tp
        sl_distance = ctx.spread * sl
        slippage = ctx.spread * 0.5

        # max_bars: Wie weit maximal simuliert wird
        max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)

        # timeout_bars: Wann Trade geschlossen wird (0 = kein Timeout)
        timeout_val = timeout_bars if timeout_bars else 0

        return compute_targets_numba(
            opn_v, cls_v, hgh_v, low_v,
            tp_distance, sl_distance, ctx.spread, slippage,
            max_bars, timeout_val
        )

    def iterate_grid(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> Iterator[dict]:
        """
        Iteriert über alle TP x SL x Timeout Kombinationen.

        Args:
            grid_config: Grid-Konfiguration mit tp, sl, timeout_bars Listen
            ctx: SimulationContext

        Yields:
            Dict mit Parameter-Kombination für compute_targets
        """
        tp_values = grid_config.get("tp", [15, 20, 25, 30, 40, 50])
        sl_values = grid_config.get("sl", [15, 20, 25, 30, 40, 50])
        timeout_values = grid_config.get("timeout_bars", [None])
        min_rrr = grid_config.get("min_rrr", 0)

        if timeout_values is None:
            timeout_values = [None]

        for tp in tp_values:
            for sl in sl_values:
                # RRR-Filter
                rrr = tp / sl if sl > 0 else 0
                if min_rrr > 0 and rrr < min_rrr:
                    continue

                for timeout in timeout_values:
                    yield {
                        "tp": float(tp),
                        "sl": float(sl),
                        "timeout_bars": timeout,
                    }

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


__all__ = ["FixedExitStrategy"]
