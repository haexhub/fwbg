"""
Fixed Exit Strategy.

Verwendet fixe TP/SL-Werte basierend auf Spread-Multiplikatoren.
Dies ist die Default-Exit-Strategie und entspricht dem bisherigen Verhalten.
"""
from typing import Dict, Any, Iterator, Tuple
import numpy as np
import pandas as pd

from ..base import BaseExitStrategy, GridParams
from .. import register
from .config import FixedExitConfig
from .compute import compute_targets_fixed


@register("fixed")
class FixedExitStrategy(BaseExitStrategy):
    """
    Exit-Strategie mit fixen TP/SL-Werten.

    TP und SL werden als Multiplikatoren des Spreads angegeben.
    Diese Werte bleiben für alle Trades gleich, unabhängig von
    Marktbedingungen.
    """

    def __init__(self, config: FixedExitConfig = None):
        self.config = config or FixedExitConfig()

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params: GridParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten
            ctx: SimulationContext
            params: Grid-Parameter (tp_value, sl_value als Spread-Multiplikatoren)

        Returns:
            (targets_long, targets_short)
        """
        return compute_targets_fixed(
            df=df,
            tp=int(params.tp_value),
            sl=int(params.sl_value),
            ctx=ctx,
            timeout_bars=params.timeout_bars,
        )

    def iterate_grid(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> Iterator[GridParams]:
        """
        Iteriert über alle TP x SL x Timeout Kombinationen.

        Args:
            grid_config: Grid-Konfiguration mit tp, sl, timeout_bars Listen
            ctx: SimulationContext

        Yields:
            GridParams für jede valide Kombination
        """
        # Grid-Werte aus Config extrahieren
        tp_values = grid_config.get("tp", self.config.tp)
        sl_values = grid_config.get("sl", self.config.sl)
        timeout_values = grid_config.get("timeout_bars", self.config.timeout_bars)
        min_rrr = grid_config.get("min_rrr", self.config.min_rrr)

        # Timeout-Werte normalisieren
        if timeout_values is None:
            timeout_values = [None]

        for tp in tp_values:
            for sl in sl_values:
                # RRR-Filter
                rrr = tp / sl if sl > 0 else 0
                if min_rrr > 0 and rrr < min_rrr:
                    continue

                for timeout in timeout_values:
                    yield GridParams(
                        tp_value=float(tp),
                        sl_value=float(sl),
                        timeout_bars=timeout,
                    )

    def get_cache_key(self, params: GridParams) -> str:
        """
        Gibt eindeutigen Cache-Key für diese Parameter zurück.

        Format: "fixed_tp{tp}_sl{sl}_to{timeout}"
        """
        timeout_str = str(params.timeout_bars) if params.timeout_bars else "none"
        return f"fixed_tp{int(params.tp_value)}_sl{int(params.sl_value)}_to{timeout_str}"

    def total_combinations(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> int:
        """
        Berechnet Gesamtzahl der Grid-Kombinationen (optimiert).
        """
        tp_values = grid_config.get("tp", self.config.tp)
        sl_values = grid_config.get("sl", self.config.sl)
        timeout_values = grid_config.get("timeout_bars", self.config.timeout_bars)
        min_rrr = grid_config.get("min_rrr", self.config.min_rrr)

        if timeout_values is None:
            timeout_values = [None]

        # Zähle valide Kombinationen
        valid_pairs = 0
        for tp in tp_values:
            for sl in sl_values:
                rrr = tp / sl if sl > 0 else 0
                if min_rrr <= 0 or rrr >= min_rrr:
                    valid_pairs += 1

        return valid_pairs * len(timeout_values)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FixedExitStrategy":
        """Erstellt Strategie aus Config-Dictionary."""
        exit_config = FixedExitConfig.from_dict(config.get("fixed", {}))
        return cls(config=exit_config)


__all__ = ["FixedExitStrategy", "FixedExitConfig", "compute_targets_fixed"]
