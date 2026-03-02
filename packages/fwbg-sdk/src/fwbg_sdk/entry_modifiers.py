"""
BaseEntryModifier - Abstrakte Basisklasse für Entry-Modifier-Plugins.

Entry-Modifier sind optionale Add-ons, die das Entry-Verhalten einer
Strategie mit zusätzlicher Logik erweitern:
- Scale-In bei Retracement-Levels
- Nachkauf-Logik bei bestimmten Preislevels
- Pyramidisierung etc.

Sie werden in der Pipeline-Konfiguration optional angegeben und können
so unabhängig von der Basis-Entry-Logik getestet werden.
"""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from fwbg_sdk.base import _infer_param_type


class BaseEntryModifier(ABC):
    """
    Basisklasse für Entry-Modifier-Plugins.

    Ein Entry-Modifier erweitert das Entry-Verhalten einer Strategie
    und implementiert zusätzliche Logik wie Scale-In bei Retracement-Levels
    oder Nachkauf-Strategien.

    Der Modifier erhält dieselben vorberechneten Arrays (OHLC, ATR, Distanzen)
    und gibt die gleichen Target-Arrays (targets_long, targets_short) zurück.

    Strategy-Konfiguration (pipeline JSON):
        {
          "entry_modifier": "scale_in",
          "entry_modifier_params": {
            "retracement_pct": 0.5,
            "max_additions": 2
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
        Führt die Trade-Simulation mit modifier-spezifischer Entry-Logik durch.

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

    @classmethod
    def get_default_params(cls) -> dict:
        """Default parameters for this entry modifier."""
        return {}

    @classmethod
    def get_param_schema(cls) -> dict:
        """Parameter schema for UI rendering. Auto-inferred from defaults if not overridden."""
        defaults = cls.get_default_params()
        schema = {}
        for key, value in defaults.items():
            schema[key] = {
                "type": _infer_param_type(value),
                "default": value,
                "description": "",
            }
        return schema
