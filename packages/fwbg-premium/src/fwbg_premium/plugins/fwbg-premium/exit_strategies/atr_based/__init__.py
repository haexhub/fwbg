"""
ATR-Based Exit Strategy Plugin.

Verwendet dynamische TP/SL-Werte basierend auf Average True Range.
TP/SL werden pro Trade basierend auf der Volatilität bei Entry berechnet.

NEU: Adaptiver Timeout - Der Timeout wird pro Trade dynamisch berechnet
basierend auf der aktuellen Volatilität relativ zur mittleren Volatilität.
Bei hoher Volatilität wird der Timeout verkürzt (schnellere Preisbewegungen),
bei niedriger Volatilität verlängert.
"""
import pathlib
from typing import Tuple, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from numba import njit

_CACHE_DIR = pathlib.Path(__file__).parent / "__pycache__"


def _clear_numba_cache():
    """Delete stale Numba .nbi/.nbc cache files for this module.

    Called automatically when a ModuleNotFoundError is raised during Numba cache
    loading (which happens after package restructuring / module renames).
    """
    for pattern in ("*.nbi", "*.nbc"):
        for f in _CACHE_DIR.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


def _call_numba(func, *args):
    """Call a Numba-JIT function with automatic stale-cache recovery.

    If Numba's pickle cache references a module that no longer exists
    (e.g. after package restructuring), it raises ModuleNotFoundError.
    We clear the cache and retry once — Numba will recompile from source.
    """
    try:
        return func(*args)
    except ModuleNotFoundError:
        _clear_numba_cache()
        return func(*args)

from fwbg_sdk import BaseExitStrategy, register_exit_strategy  # noqa: E402
from fwbg.simulation import _simulate_trade_numba  # noqa: E402
from fwbg.core import GridParams  # noqa: E402

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


@njit(cache=True, parallel=False)
def _compute_targets_atr_with_durations_numba(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wie _compute_targets_atr_numba, gibt zusätzlich Trade-Durations zurück."""
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)
    durations_long = np.zeros(n, dtype=np.int64)
    durations_short = np.zeros(n, dtype=np.int64)

    for i in range(n - 1):
        entry_idx = i + 1
        if entry_idx >= n:
            continue

        atr_at_signal = atr_values[i]
        tp_distance = max(atr_at_signal * tp_mult, min_tp_distance)
        sl_distance = max(atr_at_signal * sl_mult, min_sl_distance)

        result_long, exit_long, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars
        )
        if result_long == 1.0:
            targets_long[i] = 1.0
        durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

        result_short, exit_short, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, timeout_bars
        )
        if result_short == 1.0:
            targets_short[i] = 1.0
        durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

    return targets_long, targets_short, durations_long, durations_short


@njit(cache=True, parallel=False)
def _compute_targets_atr_adaptive_timeout_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_values: np.ndarray,
    atr_ma_values: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    spread: float,
    slippage: float,
    min_tp_distance: float,
    min_sl_distance: float,
    max_bars: int,
    base_timeout: int,
    min_timeout: int,
    max_timeout: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Berechnet Win/Loss Targets mit ATR-basierten TP/SL und adaptivem Timeout.

    Der Timeout wird pro Trade dynamisch berechnet:
    - timeout = base_timeout * (atr_ma / atr_current)
    - Bei hoher Volatilität (atr > atr_ma): kürzerer Timeout
    - Bei niedriger Volatilität (atr < atr_ma): längerer Timeout
    - Begrenzt durch min_timeout und max_timeout
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
        atr_ma_at_signal = atr_ma_values[i]

        # Dynamische TP/SL berechnen mit Mindest-Werten
        tp_distance = max(atr_at_signal * tp_mult, min_tp_distance)
        sl_distance = max(atr_at_signal * sl_mult, min_sl_distance)

        # Adaptiver Timeout berechnen
        # Ratio: atr_ma / atr_current
        # Bei hoher Vol (atr > ma): ratio < 1 → kürzerer Timeout
        # Bei niedriger Vol (atr < ma): ratio > 1 → längerer Timeout
        if atr_at_signal > 0 and atr_ma_at_signal > 0:
            vol_ratio = atr_ma_at_signal / atr_at_signal
            # Begrenzen um extreme Werte zu vermeiden
            vol_ratio = max(0.25, min(4.0, vol_ratio))
            adaptive_timeout = int(base_timeout * vol_ratio)
            adaptive_timeout = max(min_timeout, min(max_timeout, adaptive_timeout))
        else:
            adaptive_timeout = base_timeout

        # Long Trade simulieren
        result_long, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, adaptive_timeout
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        # Short Trade simulieren
        result_short, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, adaptive_timeout
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short


@njit(cache=True, parallel=False)
def _compute_targets_atr_adaptive_timeout_with_durations_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_values: np.ndarray,
    atr_ma_values: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    spread: float,
    slippage: float,
    min_tp_distance: float,
    min_sl_distance: float,
    max_bars: int,
    base_timeout: int,
    min_timeout: int,
    max_timeout: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wie adaptive timeout variant, gibt zusätzlich Trade-Durations zurück."""
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)
    durations_long = np.zeros(n, dtype=np.int64)
    durations_short = np.zeros(n, dtype=np.int64)

    for i in range(n - 1):
        entry_idx = i + 1
        if entry_idx >= n:
            continue

        atr_at_signal = atr_values[i]
        atr_ma_at_signal = atr_ma_values[i]
        tp_distance = max(atr_at_signal * tp_mult, min_tp_distance)
        sl_distance = max(atr_at_signal * sl_mult, min_sl_distance)

        if atr_at_signal > 0 and atr_ma_at_signal > 0:
            vol_ratio = atr_ma_at_signal / atr_at_signal
            vol_ratio = max(0.25, min(4.0, vol_ratio))
            adaptive_timeout = int(base_timeout * vol_ratio)
            adaptive_timeout = max(min_timeout, min(max_timeout, adaptive_timeout))
        else:
            adaptive_timeout = base_timeout

        result_long, exit_long, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, adaptive_timeout
        )
        if result_long == 1.0:
            targets_long[i] = 1.0
        durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

        result_short, exit_short, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage,
            max_bars, adaptive_timeout
        )
        if result_short == 1.0:
            targets_short[i] = 1.0
        durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

    return targets_long, targets_short, durations_long, durations_short


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

    Adaptiver Timeout (NEU):
    - Wenn adaptive_timeout=True, wird der Timeout pro Trade berechnet
    - Basierend auf aktueller ATR relativ zur mittleren ATR
    - Hohe Volatilität → kürzerer Timeout (schnellere Bewegungen)
    - Niedrige Volatilität → längerer Timeout (langsame Bewegungen)
    """

    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params: Union[GridParams, None] = None,
        tp_mult: float = 2.0,
        sl_mult: float = 1.5,
        atr_period: int = 14,
        min_tp_pips: int = 10,
        min_sl_pips: int = 15,
        timeout_bars: int = None,
        adaptive_timeout: bool = False,
        base_timeout: int = 48,
        min_timeout: int = 12,
        max_timeout: int = 96,
        atr_ma_period: int = 200,
        return_durations: bool = False,
        **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten und optional _atr Spalte
            ctx: SimulationContext
            params: Optional GridParams-Objekt (überschreibt tp_mult/sl_mult/timeout_bars)
            tp_mult: ATR-Multiplikator für Take-Profit
            sl_mult: ATR-Multiplikator für Stop-Loss
            atr_period: ATR-Periode (für Fallback-Berechnung)
            min_tp_pips: Mindest-TP in Pips (Spread-Schutz)
            min_sl_pips: Mindest-SL in Pips
            timeout_bars: Fixer Timeout - Trade schließen nach X Bars (ignoriert bei adaptive_timeout=True)
            adaptive_timeout: Wenn True, wird Timeout dynamisch basierend auf Volatilität berechnet
            base_timeout: Basis-Timeout bei durchschnittlicher Volatilität (Default: 48 Bars = 2 Tage)
            min_timeout: Minimaler Timeout bei hoher Volatilität (Default: 12 Bars = 12 Stunden)
            max_timeout: Maximaler Timeout bei niedriger Volatilität (Default: 96 Bars = 4 Tage)
            atr_ma_period: Periode für den gleitenden ATR-Durchschnitt (Default: 200)

        Returns:
            (targets_long, targets_short)
        """
        # GridParams überschreibt einzelne Parameter
        if params is not None:
            tp_mult = params.tp_value
            sl_mult = params.sl_value
            if params.timeout_bars is not None:
                timeout_bars = params.timeout_bars
            # Extra-Parameter aus GridParams
            if params.extra:
                atr_period = params.extra.get("atr_period", atr_period)
                min_tp_pips = params.extra.get("min_tp_pips", min_tp_pips)
                min_sl_pips = params.extra.get("min_sl_pips", min_sl_pips)
                # Adaptive Timeout Parameter
                adaptive_timeout = params.extra.get("adaptive_timeout", adaptive_timeout)
                base_timeout = params.extra.get("base_timeout", base_timeout)
                min_timeout = params.extra.get("min_timeout", min_timeout)
                max_timeout = params.extra.get("max_timeout", max_timeout)
                atr_ma_period = params.extra.get("atr_ma_period", atr_ma_period)

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

        # === ENTRY MODIFIER DISPATCH ===
        # When an entry modifier (e.g. scale_in) is configured, it takes priority
        # over exit modifiers because its kernel handles both scale-in AND trailing
        # stop internally.
        entry_modifier_name = getattr(ctx, "entry_modifier", None)
        if entry_modifier_name and isinstance(entry_modifier_name, str):
            from fwbg.core.registry import get_entry_modifier
            entry_mod_cls = get_entry_modifier(entry_modifier_name)
            entry_mod = entry_mod_cls()
            entry_mod_params = getattr(ctx, "entry_modifier_params", {}) or {}

            # Pre-compute TP/SL distances using ATR
            tp_dist_arr = np.maximum(atr_v * tp_mult, min_tp_distance)
            sl_dist_arr = np.maximum(atr_v * sl_mult, min_sl_distance)

            # Trailing distances from exit modifier
            modifier_params = getattr(ctx, "exit_modifier_params", {}) or {}
            em_breakeven = modifier_params.get("breakeven_trigger", 0.0)
            em_trail_mult = modifier_params.get("trail_atr_mult", 0.0)
            em_trail_tp_mult = modifier_params.get("trail_tp_atr_mult", 0.0)
            trail_dist_arr = atr_v * em_trail_mult if em_trail_mult > 0.0 else np.zeros_like(atr_v)
            trail_tp_dist_arr = atr_v * em_trail_tp_mult if em_trail_tp_mult > 0.0 else np.zeros_like(atr_v)

            timeout_val = timeout_bars if timeout_bars else 0
            return entry_mod.compute_targets(
                opn_v, cls_v, hgh_v, low_v,
                tp_dist_arr, sl_dist_arr, trail_dist_arr,
                ctx.spread, slippage,
                max_bars, timeout_val,
                return_durations=return_durations,
                breakeven_trigger=em_breakeven,
                trail_tp_dist_arr=trail_tp_dist_arr,
                **entry_mod_params,
            )

        # === EXIT MODIFIER DISPATCH ===
        # Wenn ein Exit-Modifier konfiguriert ist, delegiert die Simulation an diesen.
        # Der Modifier erhält dieselben Arrays und gibt (targets_long, targets_short) zurück.
        exit_modifier_name = getattr(ctx, "exit_modifier", None)
        if exit_modifier_name:
            from fwbg.core.registry import get_exit_modifier
            modifier_cls = get_exit_modifier(exit_modifier_name)
            modifier = modifier_cls()
            modifier_params = getattr(ctx, "exit_modifier_params", {}) or {}
            timeout_val = timeout_bars if timeout_bars else 0
            return modifier.compute_targets(
                opn_v, cls_v, hgh_v, low_v, atr_v,
                tp_mult, sl_mult,
                ctx.spread, slippage,
                min_tp_distance, min_sl_distance,
                max_bars, timeout_val,
                return_durations=return_durations,
                **modifier_params,
            )

        # === ADAPTIVER TIMEOUT ===
        if adaptive_timeout:
            atr_ma_v = pd.Series(atr_v).rolling(window=atr_ma_period, min_periods=1).mean().values
            atr_ma_v = np.nan_to_num(atr_ma_v, nan=0.0).astype(np.float64)

            if return_durations:
                return _call_numba(_compute_targets_atr_adaptive_timeout_with_durations_numba,
                    opn_v, cls_v, hgh_v, low_v,
                    atr_v, atr_ma_v,
                    tp_mult, sl_mult,
                    ctx.spread, slippage,
                    min_tp_distance, min_sl_distance,
                    max_bars,
                    base_timeout, min_timeout, max_timeout
                )

            return _call_numba(_compute_targets_atr_adaptive_timeout_numba,
                opn_v, cls_v, hgh_v, low_v,
                atr_v, atr_ma_v,
                tp_mult, sl_mult,
                ctx.spread, slippage,
                min_tp_distance, min_sl_distance,
                max_bars,
                base_timeout, min_timeout, max_timeout
            )

        # === FIXER TIMEOUT ===
        timeout_val = timeout_bars if timeout_bars else 0

        if return_durations:
            return _call_numba(_compute_targets_atr_with_durations_numba,
                opn_v, cls_v, hgh_v, low_v,
                atr_v, tp_mult, sl_mult,
                ctx.spread, slippage,
                min_tp_distance, min_sl_distance,
                max_bars, timeout_val
            )

        return _call_numba(_compute_targets_atr_numba,
            opn_v, cls_v, hgh_v, low_v,
            atr_v, tp_mult, sl_mult,
            ctx.spread, slippage,
            min_tp_distance, min_sl_distance,
            max_bars, timeout_val
        )

    def resolve_distances(
        self,
        df: pd.DataFrame,
        tp: float,
        sl: float,
        ctx: "SimulationContext",
    ):
        """ATR-basierte Distanzen: atr[i] * Multiplikator pro Bar."""
        exit_params = ctx.exit_params if ctx.exit_params else {}
        atr_period = exit_params.get("atr_period", 14)
        min_tp_pips = exit_params.get("min_tp_pips", 10)
        min_sl_pips = exit_params.get("min_sl_pips", 15)

        min_tp_distance = ctx.spread * min_tp_pips
        min_sl_distance = ctx.spread * min_sl_pips

        # ATR-Array — vorberechnete Spalte oder Fallback
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

        atr_v = np.nan_to_num(atr_v, nan=0.0)

        tp_dists = np.maximum(atr_v * tp, min_tp_distance)
        sl_dists = np.maximum(atr_v * sl, min_sl_distance)
        return tp_dists, sl_dists

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
            # Adaptive Timeout
            "adaptive_timeout": False,
            "base_timeout": 48,
            "min_timeout": 12,
            "max_timeout": 96,
            "atr_ma_period": 200,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "tp_mult": {
                "type": "float",
                "default": 2.0,
                "description": "ATR multiplier for take-profit distance (tp_distance = ATR * tp_mult)",
                "min": 0.1,
                "max": 20.0,
                "step": 0.1,
            },
            "sl_mult": {
                "type": "float",
                "default": 1.5,
                "description": "ATR multiplier for stop-loss distance (sl_distance = ATR * sl_mult)",
                "min": 0.1,
                "max": 20.0,
                "step": 0.1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR lookback period in bars (used only as fallback if no precomputed ATR column exists)",
                "min": 1,
                "max": 1000,
                "step": 1,
            },
            "min_tp_pips": {
                "type": "int",
                "default": 10,
                "description": "Minimum TP distance in pips (spread multiples) to prevent too-tight targets in low-vol environments",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "min_sl_pips": {
                "type": "int",
                "default": 15,
                "description": "Minimum SL distance in pips (spread multiples) to prevent too-tight stops in low-vol environments",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "timeout_bars": {
                "type": "int",
                "default": None,
                "description": "Fixed timeout: close trade after N bars if neither TP nor SL is hit (ignored when adaptive_timeout is enabled)",
                "min": 1,
                "max": 500,
                "step": 1,
                "required": False,
            },
            "adaptive_timeout": {
                "type": "bool",
                "default": False,
                "description": "Enable per-trade adaptive timeout based on current vs average volatility",
            },
            "base_timeout": {
                "type": "int",
                "default": 48,
                "description": "Base timeout bars at average volatility (adaptive mode); scales up/down with vol ratio",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "min_timeout": {
                "type": "int",
                "default": 12,
                "description": "Minimum adaptive timeout bars (floor for high-volatility periods)",
                "min": 1,
                "max": 200,
                "step": 1,
            },
            "max_timeout": {
                "type": "int",
                "default": 96,
                "description": "Maximum adaptive timeout bars (cap for low-volatility periods)",
                "min": 10,
                "max": 1000,
                "step": 1,
            },
            "atr_ma_period": {
                "type": "int",
                "default": 200,
                "description": "Moving average period over ATR for computing the vol ratio in adaptive timeout mode",
                "min": 10,
                "max": 1000,
                "step": 10,
            },
        }


__all__ = ["AtrExitStrategy"]
