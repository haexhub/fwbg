"""
ATR-Based Exit Strategy.

Verwendet dynamische TP/SL-Werte basierend auf Average True Range.
TP/SL werden pro Trade basierend auf der Volatilität bei Entry berechnet.
"""
from typing import Dict, Any, Iterator, Tuple
import numpy as np
import pandas as pd

from ..base import BaseExitStrategy, GridParams
from .. import register
from .config import AtrExitConfig
from .compute import compute_targets_atr


@register("atr_based")
class AtrExitStrategy(BaseExitStrategy):
    """
    Exit-Strategie mit ATR-basierten TP/SL-Werten.

    TP und SL werden als Multiplikatoren des ATR angegeben.
    Die tatsächlichen Werte variieren pro Trade je nach Volatilität
    zum Zeitpunkt der Trade-Eröffnung.

    Vorteile:
    - Passt sich automatisch an Marktvolatilität an
    - Größere TP/SL bei hoher Volatilität
    - Engere TP/SL bei niedriger Volatilität
    """

    def __init__(self, config: AtrExitConfig = None):
        self.config = config or AtrExitConfig()

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params: GridParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten und _atr Spalte
            ctx: SimulationContext
            params: Grid-Parameter (tp_value, sl_value als ATR-Multiplikatoren)

        Returns:
            (targets_long, targets_short)
        """
        return compute_targets_atr(
            df=df,
            tp_mult=params.tp_value,
            sl_mult=params.sl_value,
            ctx=ctx,
            atr_period=self.config.atr_period,
            min_tp_pips=self.config.min_tp_pips,
            min_sl_pips=self.config.min_sl_pips,
            timeout_bars=params.timeout_bars,
        )

    def iterate_grid(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> Iterator[GridParams]:
        """
        Iteriert über alle TP-Mult x SL-Mult x Timeout Kombinationen.

        Args:
            grid_config: Grid-Konfiguration mit atr_tp_mult, atr_sl_mult Listen
            ctx: SimulationContext

        Yields:
            GridParams für jede valide Kombination
        """
        # Grid-Werte aus Config extrahieren
        # Unterstütze beide Namenskonventionen
        tp_mults = grid_config.get("atr_tp_mult",
                    grid_config.get("tp_mult", self.config.tp_mult))
        sl_mults = grid_config.get("atr_sl_mult",
                    grid_config.get("sl_mult", self.config.sl_mult))
        timeout_values = grid_config.get("timeout_bars", self.config.timeout_bars)
        min_rrr = grid_config.get("min_rrr", self.config.min_rrr)

        # Timeout-Werte normalisieren
        if timeout_values is None:
            timeout_values = [None]

        for tp_mult in tp_mults:
            for sl_mult in sl_mults:
                # RRR-Filter (basierend auf Multiplikatoren)
                rrr = tp_mult / sl_mult if sl_mult > 0 else 0
                if min_rrr > 0 and rrr < min_rrr:
                    continue

                for timeout in timeout_values:
                    yield GridParams(
                        tp_value=float(tp_mult),
                        sl_value=float(sl_mult),
                        timeout_bars=timeout,
                        extra={
                            "atr_period": self.config.atr_period,
                            "min_tp_pips": self.config.min_tp_pips,
                            "min_sl_pips": self.config.min_sl_pips,
                        }
                    )

    def get_cache_key(self, params: GridParams) -> str:
        """
        Gibt eindeutigen Cache-Key für diese Parameter zurück.

        Format: "atr_tp{tp_mult}_sl{sl_mult}_to{timeout}"
        """
        timeout_str = str(params.timeout_bars) if params.timeout_bars else "none"
        return f"atr_tp{params.tp_value:.2f}_sl{params.sl_value:.2f}_to{timeout_str}"

    def total_combinations(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> int:
        """
        Berechnet Gesamtzahl der Grid-Kombinationen (optimiert).
        """
        tp_mults = grid_config.get("atr_tp_mult",
                    grid_config.get("tp_mult", self.config.tp_mult))
        sl_mults = grid_config.get("atr_sl_mult",
                    grid_config.get("sl_mult", self.config.sl_mult))
        timeout_values = grid_config.get("timeout_bars", self.config.timeout_bars)
        min_rrr = grid_config.get("min_rrr", self.config.min_rrr)

        if timeout_values is None:
            timeout_values = [None]

        # Zähle valide Kombinationen
        valid_pairs = 0
        for tp_mult in tp_mults:
            for sl_mult in sl_mults:
                rrr = tp_mult / sl_mult if sl_mult > 0 else 0
                if min_rrr <= 0 or rrr >= min_rrr:
                    valid_pairs += 1

        return valid_pairs * len(timeout_values)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AtrExitStrategy":
        """Erstellt Strategie aus Config-Dictionary."""
        exit_config = AtrExitConfig.from_dict(config.get("atr_based", config))
        return cls(config=exit_config)


__all__ = ["AtrExitStrategy", "AtrExitConfig", "compute_targets_atr"]
