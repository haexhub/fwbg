"""
Konfiguration für ATR-basierte Exit Strategy.

Dynamische TP/SL-Werte basierend auf Average True Range.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from ..base import ExitConfig


@dataclass
class AtrExitConfig(ExitConfig):
    """
    Konfiguration für ATR-basierte TP/SL-Werte.

    TP/SL werden als Multiplikatoren des ATR angegeben.
    Beispiel: tp_mult=2.0 bei ATR=0.005 -> TP = 100 Pips

    Die tatsächlichen TP/SL-Werte variieren pro Trade basierend auf
    der Volatilität zum Zeitpunkt der Trade-Eröffnung.
    """
    # ATR-Periode für Berechnung
    atr_period: int = 14

    # Grid-Werte für TP (ATR-Multiplikatoren)
    tp_mult: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5, 3.0])

    # Grid-Werte für SL (ATR-Multiplikatoren)
    sl_mult: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 2.5])

    # Mindest-TP in Pips (Spread-Schutz)
    min_tp_pips: int = 10

    # Mindest-SL in Pips
    min_sl_pips: int = 15

    # Timeout in Bars (None = kein Timeout)
    timeout_bars: List[Optional[int]] = field(default_factory=lambda: [None])

    # Mindest-RRR Filter (0 = kein Filter)
    min_rrr: float = 0.0

    # Separate Grids für Long/Short
    long_tp_mult: List[float] = None
    long_sl_mult: List[float] = None
    short_tp_mult: List[float] = None
    short_sl_mult: List[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "AtrExitConfig":
        """Erstellt Config aus Dictionary."""
        return cls(
            atr_period=data.get("atr_period", 14),
            tp_mult=data.get("atr_tp_mult", data.get("tp_mult", [1.0, 1.5, 2.0, 2.5, 3.0])),
            sl_mult=data.get("atr_sl_mult", data.get("sl_mult", [1.0, 1.5, 2.0, 2.5])),
            min_tp_pips=data.get("min_tp_pips", 10),
            min_sl_pips=data.get("min_sl_pips", 15),
            timeout_bars=data.get("timeout_bars", [None]),
            min_rrr=data.get("min_rrr", 0.0),
            long_tp_mult=data.get("long_tp_mult"),
            long_sl_mult=data.get("long_sl_mult"),
            short_tp_mult=data.get("short_tp_mult"),
            short_sl_mult=data.get("short_sl_mult"),
        )

    def get_long_grid(self) -> tuple:
        """Gibt (tp_mult, sl_mult) Grid für Long Trades zurück."""
        return (
            self.long_tp_mult if self.long_tp_mult is not None else self.tp_mult,
            self.long_sl_mult if self.long_sl_mult is not None else self.sl_mult,
        )

    def get_short_grid(self) -> tuple:
        """Gibt (tp_mult, sl_mult) Grid für Short Trades zurück."""
        return (
            self.short_tp_mult if self.short_tp_mult is not None else self.tp_mult,
            self.short_sl_mult if self.short_sl_mult is not None else self.sl_mult,
        )
