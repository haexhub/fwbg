"""
ATR-Based Exit Strategy Plugin.

Verwendet dynamische TP/SL-Werte basierend auf Average True Range.
TP/SL werden pro Trade basierend auf der Volatilität bei Entry berechnet.
"""
from typing import Dict, Any, Iterator, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
from numba import njit

from fwbg.plugins import BaseExitStrategy
from fwbg.core import register_exit_strategy
from fwbg.simulation import _simulate_trade_numba

if TYPE_CHECKING:
    from fwbg.core.context import SimulationContext


@njit(cache=True, parallel=False)
def _compute_targets_atr_numba(
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
    timeout_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Win/Loss Targets mit ATR-basierten TP/SL.

    TP/SL werden pro Trade basierend auf ATR bei Entry berechnet:
    - tp_distance = max(atr[entry_idx] * tp_mult, min_tp_distance)
    - sl_distance = max(atr[entry_idx] * sl_mult, min_sl_distance)
    """
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        entry_idx = i + 1
        if entry_idx >= n:
            continue

        # ATR bei Signal-Bar verwenden
        atr_at_signal = atr_values[i]

        # Dynamische TP/SL berechnen mit Mindest-Werten
        tp_distance = max(atr_at_signal * tp_mult, min_tp_distance)
        sl_distance = max(atr_at_signal * sl_mult, min_sl_distance)

        # Long Trade simulieren
        result_long, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        # Short Trade simulieren
        result_short, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short


@register_exit_strategy("atr_based")
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

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        tp_mult: float = 2.0,
        sl_mult: float = 1.5,
        atr_period: int = 14,
        min_tp_pips: int = 10,
        min_sl_pips: int = 15,
        timeout_bars: int = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten und optional _atr Spalte
            ctx: SimulationContext
            tp_mult: ATR-Multiplikator für Take-Profit
            sl_mult: ATR-Multiplikator für Stop-Loss
            atr_period: ATR-Periode (für Fallback-Berechnung)
            min_tp_pips: Mindest-TP in Pips (Spread-Schutz)
            min_sl_pips: Mindest-SL in Pips
            timeout_bars: Optional - Trade schließen nach X Bars

        Returns:
            (targets_long, targets_short)
        """
        # OHLC-Arrays extrahieren
        opn_v = df["O"].values.astype(np.float64)
        cls_v = df["C"].values.astype(np.float64)
        hgh_v = df["H"].values.astype(np.float64)
        low_v = df["L"].values.astype(np.float64)

        # ATR-Array - verwende vorberechnete Spalte oder berechne
        if "_atr" in df.columns:
            atr_v = df["_atr"].values.astype(np.float64)
        elif "vol_atr" in df.columns:
            atr_v = df["vol_atr"].values.astype(np.float64)
        else:
            # Fallback: ATR berechnen
            import ta
            atr_series = ta.volatility.average_true_range(
                df["H"], df["L"], df["C"], window=atr_period
            )
            atr_v = atr_series.values.astype(np.float64)

        # NaN durch 0 ersetzen (am Anfang der Serie)
        atr_v = np.nan_to_num(atr_v, nan=0.0)

        # Mindest-Distanzen in Preiseinheiten
        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips

        slippage = ctx.spread * 0.5

        # max_bars: Wie weit maximal simuliert wird
        max_bars = ctx.max_trade_bars if ctx.max_trade_bars else len(df)

        # timeout_bars: Wann Trade geschlossen wird (0 = kein Timeout)
        timeout_val = timeout_bars if timeout_bars else 0

        return _compute_targets_atr_numba(
            opn_v, cls_v, hgh_v, low_v,
            atr_v, tp_mult, sl_mult,
            ctx.spread, slippage,
            min_tp_distance, min_sl_distance,
            max_bars, timeout_val
        )

    def iterate_grid(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> Iterator[dict]:
        """
        Iteriert über alle TP-Mult x SL-Mult x Timeout Kombinationen.

        Unterstützt:
        - tp/sl: Interpretiert als ATR-Multiplikatoren
        - atr_tp_mult/atr_sl_mult: Explizite Benennung
        """
        # Grid-Werte extrahieren (beide Namenskonventionen)
        tp_mults = grid_config.get("atr_tp_mult",
                    grid_config.get("tp_mult",
                    grid_config.get("tp", [1.0, 1.5, 2.0, 2.5])))
        sl_mults = grid_config.get("atr_sl_mult",
                    grid_config.get("sl_mult",
                    grid_config.get("sl", [1.0, 1.5, 2.0])))
        timeout_values = grid_config.get("timeout_bars", [None])
        min_rrr = grid_config.get("min_rrr", 0)

        if timeout_values is None:
            timeout_values = [None]

        # Exit-Parameter aus Context oder Defaults
        exit_params = ctx.exit_params if ctx.exit_params else {}
        atr_period = exit_params.get("atr_period", 14)
        min_tp_pips = exit_params.get("min_tp_pips", 10)
        min_sl_pips = exit_params.get("min_sl_pips", 15)

        for tp_mult in tp_mults:
            for sl_mult in sl_mults:
                # RRR-Filter
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
        """
        Gibt eindeutigen Cache-Key für diese Parameter zurück.

        Format: "atr_tp{tp_mult}_sl{sl_mult}_to{timeout}"
        """
        tp_mult = params.get("tp_mult", 0)
        sl_mult = params.get("sl_mult", 0)
        timeout = params.get("timeout_bars")
        timeout_str = str(timeout) if timeout else "none"
        return f"atr_tp{tp_mult:.2f}_sl{sl_mult:.2f}_to{timeout_str}"

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "tp_mult": 2.0,
            "sl_mult": 1.5,
            "atr_period": 14,
            "min_tp_pips": 10,
            "min_sl_pips": 15,
            "timeout_bars": None,
        }


__all__ = ["AtrExitStrategy"]
