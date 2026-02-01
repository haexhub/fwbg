"""
Konfiguration für Fixed Exit Strategy.

Fixe TP/SL-Werte basierend auf Spread-Multiplikatoren.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from ..base import ExitConfig


@dataclass
class FixedExitConfig(ExitConfig):
    """
    Konfiguration für fixe TP/SL-Werte.

    TP/SL werden als Multiplikatoren des Spreads angegeben.
    Beispiel: tp=30, sl=50 bei spread=0.0001 -> TP=30 Pips, SL=50 Pips
    """
    # Grid-Werte für TP (Spread-Multiplikatoren)
    tp: List[int] = field(default_factory=lambda: [15, 20, 25, 30, 40, 50, 60, 80])

    # Grid-Werte für SL (Spread-Multiplikatoren)
    sl: List[int] = field(default_factory=lambda: [15, 20, 25, 30, 40, 50, 60, 80])

    # Timeout in Bars (None = kein Timeout)
    timeout_bars: List[Optional[int]] = field(default_factory=lambda: [None])

    # Mindest-RRR Filter (0 = kein Filter)
    min_rrr: float = 0.0

    # Separate Grids für Long/Short
    long_tp: List[int] = None
    long_sl: List[int] = None
    short_tp: List[int] = None
    short_sl: List[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "FixedExitConfig":
        """Erstellt Config aus Dictionary."""
        # Default-Werte müssen hier definiert werden, da dataclass field defaults
        # nicht über cls.attribut zugänglich sind
        default_tp = [15, 20, 25, 30, 40, 50, 60, 80]
        default_sl = [15, 20, 25, 30, 40, 50, 60, 80]

        return cls(
            tp=data.get("tp", default_tp),
            sl=data.get("sl", default_sl),
            timeout_bars=data.get("timeout_bars", [None]),
            min_rrr=data.get("min_rrr", 0.0),
            long_tp=data.get("long_tp"),
            long_sl=data.get("long_sl"),
            short_tp=data.get("short_tp"),
            short_sl=data.get("short_sl"),
        )

    def get_long_grid(self) -> tuple:
        """Gibt (tp, sl) Grid für Long Trades zurück."""
        return (
            self.long_tp if self.long_tp is not None else self.tp,
            self.long_sl if self.long_sl is not None else self.sl,
        )

    def get_short_grid(self) -> tuple:
        """Gibt (tp, sl) Grid für Short Trades zurück."""
        return (
            self.short_tp if self.short_tp is not None else self.tp,
            self.short_sl if self.short_sl is not None else self.sl,
        )
