"""
BaseExitModifier - Abstrakte Basisklasse für Exit-Modifier-Plugins.

Exit-Modifier sind optionale Add-ons, die die Trade-Simulation einer
Exit-Strategie (z.B. atr_based) mit zusätzlicher Logik erweitern:
- Trailing Stops
- Breakeven-Stops
- Tighten-after-profit etc.

Sie werden in der Pipeline-Konfiguration optional angegeben und können
so unabhängig von der Basis-Exit-Strategie getestet werden.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseExitModifier(ABC):
    """
    Basisklasse für Exit-Modifier-Plugins.

    Ein Exit-Modifier ersetzt die Trade-Simulation einer Basis-Exit-Strategie
    (z.B. atr_based) und implementiert zusätzliche Logik wie Trailing Stops
    oder Breakeven-Schutz.

    Der Modifier erhält dieselben vorberechneten Arrays (OHLC, ATR, Distanzen)
    und gibt die gleichen Target-Arrays (targets_long, targets_short) zurück.

    Strategy-Konfiguration (pipeline JSON):
        {
          "exit_strategy": "atr_based",
          "exit_params": "atr_intraday",
          "exit_modifier": "trailing_stop",
          "exit_modifier_params": {
            "breakeven_trigger": 0.5,
            "trail_atr_mult": 0.5
          }
        }
    """

    name: str = "base"

    @abstractmethod
    def compute_targets(
        self,
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
        timeout_val: int,
        return_durations: bool = False,
        **params,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Führt die Trade-Simulation mit modifier-spezifischer Logik durch.

        Args:
            opens:            Open-Preise
            closes:           Close-Preise
            highs:            High-Preise
            lows:             Low-Preise
            atr_values:       ATR-Werte pro Bar
            tp_mult:          TP ATR-Multiplikator
            sl_mult:          SL ATR-Multiplikator
            spread:           Spread in Preiseinheiten
            slippage:         Slippage in Preiseinheiten
            min_tp_distance:  Mindest-TP-Distanz
            min_sl_distance:  Mindest-SL-Distanz
            max_bars:         Maximale Trade-Laufzeit
            timeout_val:      Timeout in Bars (0 = kein Timeout)
            return_durations: Wenn True, auch Durations zurückgeben
            **params:         Modifier-spezifische Parameter

        Returns:
            (targets_long, targets_short) — Arrays mit 1.0 für Win, 0.0 für Loss
            Wenn return_durations=True: (targets_long, targets_short, durations_long, durations_short)
        """
        pass
