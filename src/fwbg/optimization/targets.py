"""
Target computation, trade simulation, and validation evaluation.

Extracted from nested_cv.py for modularity (keeping files under 600 lines).
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from xgboost import XGBClassifier

from fwbg.core.context import SimulationContext
from fwbg.core import get_exit_strategy as get_strategy, GridParams
from fwbg.simulation.trade import simulate_pro_trade


def _validate_targets(
    targets_long: np.ndarray,
    targets_short: np.ndarray,
    ctx: SimulationContext
) -> Tuple[bool, bool]:
    """
    Prüft ob genug Targets für Long/Short vorhanden sind.

    Args:
        targets_long: Long-Targets Array
        targets_short: Short-Targets Array
        ctx: SimulationContext

    Returns:
        (has_long, has_short) - Boolean Tuple
    """
    min_per_direction = ctx.min_trades // 2
    n_long = np.count_nonzero(targets_long)
    n_short = np.count_nonzero(targets_short)
    has_long = ctx.long_enabled and n_long >= min_per_direction
    has_short = ctx.short_enabled and n_short >= min_per_direction
    return has_long, has_short


def _simulate_trades_core(
    df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    ct_long: float,
    ct_short: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    return_detailed: bool = False,
    timeout_bars: int = None,
    direction_filter: int = None,
) -> Dict[str, Any]:
    """
    Kern-Funktion für Trade-Simulation (konsolidiert aus 3 ähnlichen Funktionen).

    Args:
        df: DataFrame mit OHLC-Daten und _regime_ok
        probs_long: Wahrscheinlichkeiten für Long-Trades (oder None)
        probs_short: Wahrscheinlichkeiten für Short-Trades (oder None)
        long_win_idx: Index der Win-Klasse im Long-Modell
        short_win_idx: Index der Win-Klasse im Short-Modell
        ct_long: Confidence Threshold für Long
        ct_short: Confidence Threshold für Short
        tp: Take-Profit Multiplikator
        sl: Stop-Loss Multiplikator
        ctx: SimulationContext mit allen Parametern
        return_detailed: Wenn True, auch volle Trade-Details zurückgeben
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen
        direction_filter: None=beide, 1=nur Long, -1=nur Short

    Returns:
        dict mit trades und optional trades_detailed
    """
    opn = df["O"].values
    cls = df["C"].values
    hgh = df["H"].values
    low = df["L"].values
    # ATR ist optional - wenn nicht vorhanden (volatility indicator nicht konfiguriert), Dummy-Array
    atr = df["_atr"].values if "_atr" in df.columns else np.zeros(len(df))
    regime = df["_regime_ok"].values
    timestamps = df.index.values

    trades = []
    trades_detailed = [] if return_detailed else None
    next_allowed_entry = 0

    # Simuliere bis zum vorletzten Bar (letzter Bar kann kein Entry sein)
    for i in range(len(df) - 1):
        if i < next_allowed_entry:
            continue

        if not regime[i]:
            continue

        direction = None
        # Long-Check (wenn nicht gefiltert)
        if direction_filter in (None, 1):
            if ctx.long_enabled and probs_long is not None and probs_long[i, long_win_idx] >= ct_long:
                direction = 1
        # Short-Check (wenn nicht gefiltert und Long nicht getroffen)
        if direction is None and direction_filter in (None, -1):
            if ctx.short_enabled and probs_short is not None and probs_short[i, short_win_idx] >= ct_short:
                direction = -1

        if direction:
            trade = simulate_pro_trade(
                cls, hgh, low, atr, i, direction, tp, sl, ctx.spread,
                timestamps=timestamps, symbol=ctx.symbol, opens=opn,
                max_bars=ctx.max_trade_bars,
                timeout_bars=timeout_bars
            )
            if trade:
                trades.append({"result": trade["result"], "pnl_raw": trade["pnl_raw"]})
                next_allowed_entry = trade["exit_idx"] + 1

                if return_detailed:
                    trade["ct"] = ct_long if direction == 1 else ct_short
                    trade["hour"] = df.index[i].hour
                    trades_detailed.append(trade)

    result = {"trades": trades}
    if return_detailed:
        result["trades_detailed"] = trades_detailed
    return result


def simulate_trades_sequential(
    df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    ct: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    return_detailed: bool = False,
    timeout_bars: int = None,
) -> Dict[str, Any]:
    """Simuliert Trades sequentiell mit gleichem CT für Long/Short."""
    return _simulate_trades_core(
        df, probs_long, probs_short, long_win_idx, short_win_idx,
        ct, ct, tp, sl, ctx, return_detailed, timeout_bars
    )


def _simulate_single_direction(
    df: pd.DataFrame,
    probs: np.ndarray,
    win_idx: int,
    ct: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    direction: int,
    timeout_bars: int = None,
) -> Dict[str, Any]:
    """Simuliert Trades für eine einzelne Richtung (Long oder Short)."""
    if direction == 1:
        return _simulate_trades_core(
            df, probs, None, win_idx, None,
            ct, 0.0, tp, sl, ctx, False, timeout_bars, direction_filter=1
        )
    else:
        return _simulate_trades_core(
            df, None, probs, None, win_idx,
            0.0, ct, tp, sl, ctx, False, timeout_bars, direction_filter=-1
        )


def simulate_trades_sequential_separate_ct(
    df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    ct_long: float,
    ct_short: float,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    return_detailed: bool = False,
    timeout_bars: int = None,
) -> Dict[str, Any]:
    """Simuliert Trades mit separaten CT-Thresholds für Long und Short."""
    return _simulate_trades_core(
        df, probs_long, probs_short, long_win_idx, short_win_idx,
        ct_long, ct_short, tp, sl, ctx, return_detailed, timeout_bars
    )


def compute_targets(
    df: pd.DataFrame,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """
    Berechnet Long/Short Targets für einen DataFrame.

    Args:
        df: DataFrame mit OHLC-Daten
        tp: Take-Profit Multiplikator
        sl: Stop-Loss Multiplikator
        ctx: SimulationContext
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen

    Returns:
        (targets_long, targets_short, has_long, has_short)
    """
    targets_long = np.zeros(len(df))
    targets_short = np.zeros(len(df))

    opn_v = df["O"].values
    cls_v = df["C"].values
    hgh_v = df["H"].values
    low_v = df["L"].values
    # ATR ist optional - wenn nicht vorhanden (volatility indicator nicht konfiguriert), Dummy-Array
    atr_v = df["_atr"].values if "_atr" in df.columns else np.zeros(len(df))
    timestamps = df.index.values

    # Simuliere bis zum vorletzten Bar (letzter Bar kann kein Entry sein)
    for i in range(len(df) - 1):
        trade_long = simulate_pro_trade(
            cls_v, hgh_v, low_v, atr_v, i, 1, tp, sl, ctx.spread,
            timestamps=timestamps, symbol=ctx.symbol, opens=opn_v,
            max_bars=ctx.max_trade_bars,  # None = kein Limit
            timeout_bars=timeout_bars  # Time-based Exit
        )
        trade_short = simulate_pro_trade(
            cls_v, hgh_v, low_v, atr_v, i, -1, tp, sl, ctx.spread,
            timestamps=timestamps, symbol=ctx.symbol, opens=opn_v,
            max_bars=ctx.max_trade_bars,  # None = kein Limit
            timeout_bars=timeout_bars  # Time-based Exit
        )
        if trade_long and trade_long["result"] == 1.0:
            targets_long[i] = 1
        if trade_short and trade_short["result"] == 1.0:
            targets_short[i] = 1

    has_long, has_short = _validate_targets(targets_long, targets_short, ctx)

    return targets_long, targets_short, has_long, has_short


def compute_targets_cached(
    full_df: pd.DataFrame,
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
    exit_strategy_mode: str = "fixed",
    grid_params: GridParams = None,
    return_durations: bool = False,
) -> tuple:
    """
    Berechnet Targets einmal auf dem gesamten DataFrame (für Caching).

    Dispatcht an die Exit-Strategie via Plugin-Registry.

    Args:
        full_df: Gesamter Inner-DataFrame (nicht nur ein Fold!)
        tp: Take-Profit Wert
        sl: Stop-Loss Wert
        ctx: SimulationContext
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen
        exit_strategy_mode: Name der Exit-Strategie (Plugin-Registry Key)
        grid_params: GridParams-Objekt (wenn vorhanden, werden tp/sl ignoriert)
        return_durations: Wenn True, auch Trade-Durations zurückgeben (für Sample Weights)

    Returns:
        (targets_long, targets_short) oder
        (targets_long, targets_short, durations_long, durations_short) wenn return_durations=True
    """
    # Dispatch to exit strategy plugin
    strategy_cls = get_strategy(exit_strategy_mode)
    strategy = strategy_cls()

    extra = {}
    if hasattr(ctx, 'exit_params') and ctx.exit_params:
        extra = ctx.exit_params.copy()

    if grid_params is None:
        grid_params = GridParams(
            tp_value=float(tp),
            sl_value=float(sl),
            timeout_bars=timeout_bars,
            extra=extra,
        )

    return strategy.compute_targets(
        full_df, ctx, params=grid_params, return_durations=return_durations
    )


def slice_targets_for_fold(
    full_targets_long: np.ndarray,
    full_targets_short: np.ndarray,
    full_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    ctx: SimulationContext
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """
    Extrahiert Targets für einen bestimmten Fold aus gecachten Gesamt-Targets.

    Args:
        full_targets_long: Gecachte Long-Targets für gesamten DataFrame
        full_targets_short: Gecachte Short-Targets für gesamten DataFrame
        full_df: Der gesamte DataFrame (für Index-Mapping)
        fold_df: Der Fold-DataFrame (Train oder Val)
        ctx: SimulationContext

    Returns:
        (targets_long, targets_short, has_long, has_short)
    """
    # Finde die Positionen des Fold im Gesamt-DataFrame
    # Verwende searchsorted für robuste Index-Suche (auch bei nicht-eindeutigen Indices)
    try:
        # Versuche get_loc (schnell bei eindeutigen Indices)
        start_loc = full_df.index.get_loc(fold_df.index[0])
        end_loc = full_df.index.get_loc(fold_df.index[-1])

        # get_loc kann slice, int oder array zurückgeben
        if isinstance(start_loc, slice):
            start_idx = start_loc.start if start_loc.start is not None else 0
        elif isinstance(start_loc, np.ndarray):
            start_idx = np.where(start_loc)[0][0]
        else:
            start_idx = start_loc

        if isinstance(end_loc, slice):
            end_idx = end_loc.stop if end_loc.stop is not None else len(full_df)
        elif isinstance(end_loc, np.ndarray):
            end_idx = np.where(end_loc)[0][-1] + 1
        else:
            end_idx = end_loc + 1

    except (KeyError, IndexError):
        # Fallback: Nutze get_indexer (vectorisiert, O(log n) bei sortiertem Index)
        fold_start = fold_df.index[0]
        fold_end = fold_df.index[-1]

        indices = full_df.index.get_indexer([fold_start, fold_end])
        if indices[0] >= 0 and indices[1] >= 0:
            start_idx = indices[0]
            end_idx = indices[1] + 1
        else:
            # Index nicht gefunden - verwende Länge des fold_df
            # Dies sollte nur passieren wenn fold_df nicht aus full_df stammt
            start_idx = 0
            end_idx = len(fold_df)

    # Slice die gecachten Targets
    fold_targets_long = full_targets_long[start_idx:end_idx]
    fold_targets_short = full_targets_short[start_idx:end_idx]

    # Prüfe ob genug Targets vorhanden (nutzt konsolidierte Funktion)
    has_long, has_short = _validate_targets(fold_targets_long, fold_targets_short, ctx)

    return fold_targets_long, fold_targets_short, has_long, has_short


def _get_probs(
    model: Optional[XGBClassifier],
    df: pd.DataFrame,
    features: Optional[List[str]]
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Berechnet Wahrscheinlichkeiten für ein Modell."""
    if not features or model is None:
        return None, None
    probs = model.predict_proba(df[features])
    if 1 in model.classes_:
        win_idx = np.where(model.classes_ == 1)[0][0]
        return probs, win_idx
    return None, None


def evaluate_on_validation(
    val_df: pd.DataFrame,
    mod_long: Optional[XGBClassifier],
    mod_short: Optional[XGBClassifier],
    features_long: Optional[List[str]],
    features_short: Optional[List[str]],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
) -> Tuple[Optional[float], float, Dict[float, List[float]]]:
    """
    Evaluiert Modelle auf Validation-Set und findet besten CT.

    Bei separate_long_short=True werden separate CTs für Long und Short optimiert.

    Args:
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen

    Returns:
        (best_ct, best_pnl, trades_by_ct)
        Bei separate_long_short: best_ct ist ein Tuple (ct_long, ct_short)
    """
    probs_long, long_win_idx = _get_probs(mod_long, val_df, features_long)
    probs_short, short_win_idx = _get_probs(mod_short, val_df, features_short)

    # Separate CT-Optimierung wenn aktiviert
    if ctx.separate_long_short:
        return _evaluate_separate_ct(
            val_df, probs_long, probs_short, long_win_idx, short_win_idx,
            tp, sl, ctx, timeout_bars
        )

    # Standard: Gemeinsamer CT für Long und Short
    # SEQUENTIELLE Evaluierung - kein nested Threading
    # (Feature-Gruppen sind bereits parallelisiert)
    trades_by_ct = {}
    for ct in ctx.grid_ct:
        result = simulate_trades_sequential(
            val_df, probs_long, probs_short, long_win_idx, short_win_idx,
            ct, tp, sl, ctx, return_detailed=False, timeout_bars=timeout_bars
        )
        trades_by_ct[ct] = result["trades"]

    # Besten CT finden
    best_ct = None
    best_pnl = float("-inf")
    for ct, ct_trades in trades_by_ct.items():
        if len(ct_trades) >= 10:
            ct_pnl = sum(t["pnl_raw"] for t in ct_trades)
            if ct_pnl > best_pnl:
                best_pnl = ct_pnl
                best_ct = ct

    return best_ct, best_pnl, trades_by_ct


def _optimize_ct_for_direction(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    win_idx: int,
    ct_values: List[float],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    direction: int,
    timeout_bars: int = None,
    min_trades: int = 5,
) -> Tuple[Optional[float], float, Dict[float, List[float]]]:
    """
    Optimiert CT für eine einzelne Richtung (Long oder Short).

    Args:
        val_df: Validation DataFrame
        probs: Wahrscheinlichkeiten
        win_idx: Win-Index
        ct_values: Liste der CT-Werte zum Testen
        tp/sl: Take-Profit/Stop-Loss
        ctx: SimulationContext
        direction: 1=Long, -1=Short
        timeout_bars: Optional Timeout
        min_trades: Minimum Trades für gültigen CT

    Returns:
        (best_ct, best_pnl, trades_by_ct)
    """
    # SEQUENTIELLE Evaluierung - kein nested Threading
    trades_by_ct = {}
    for ct in ct_values:
        result = _simulate_single_direction(
            val_df, probs, win_idx, ct, tp, sl, ctx,
            direction=direction, timeout_bars=timeout_bars
        )
        trades_by_ct[ct] = result["trades"]

    # Besten CT finden
    best_ct = None
    best_pnl = float("-inf")
    for ct, trades in trades_by_ct.items():
        if len(trades) >= min_trades:
            pnl = sum(t["pnl_raw"] for t in trades)
            if pnl > best_pnl:
                best_pnl = pnl
                best_ct = ct

    return best_ct, best_pnl, trades_by_ct


def _evaluate_separate_ct(
    val_df: pd.DataFrame,
    probs_long: Optional[np.ndarray],
    probs_short: Optional[np.ndarray],
    long_win_idx: Optional[int],
    short_win_idx: Optional[int],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
) -> Tuple[Optional[tuple], float, Dict]:
    """
    Optimiert CT separat für Long und Short Trades.

    Long und Short werden UNABHÄNGIG voneinander optimiert:
    - Finde besten CT für Long (nur Long-Trades simulieren)
    - Finde besten CT für Short (nur Short-Trades simulieren)
    - Kombiniere die besten CTs

    Das reduziert die Komplexität von O(n²) auf O(2n).
    Bei 6 CT-Werten: 12 Simulationen statt 36.

    Args:
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen

    Returns:
        ((ct_long, ct_short), best_combined_pnl, trades_info)
    """
    trades_info = {"long": {}, "short": {}, "combined": {}}

    # Grid für Long und Short CTs
    long_cts = ctx.long_grid_ct if ctx.long_grid_ct else ctx.grid_ct
    short_cts = ctx.short_grid_ct if ctx.short_grid_ct else ctx.grid_ct

    # === LONG CT OPTIMIERUNG (unabhängig) ===
    best_ct_long, best_pnl_long, long_trades_by_ct = None, float("-inf"), {}
    if ctx.long_enabled and probs_long is not None:
        best_ct_long, best_pnl_long, long_trades_by_ct = _optimize_ct_for_direction(
            val_df, probs_long, long_win_idx, long_cts, tp, sl, ctx,
            direction=1, timeout_bars=timeout_bars
        )
        trades_info["long"] = long_trades_by_ct

    # === SHORT CT OPTIMIERUNG (unabhängig) ===
    best_ct_short, best_pnl_short, short_trades_by_ct = None, float("-inf"), {}
    if ctx.short_enabled and probs_short is not None:
        best_ct_short, best_pnl_short, short_trades_by_ct = _optimize_ct_for_direction(
            val_df, probs_short, short_win_idx, short_cts, tp, sl, ctx,
            direction=-1, timeout_bars=timeout_bars
        )
        trades_info["short"] = short_trades_by_ct

    # === KOMBINATION ===
    # Wenn nur eine Richtung aktiviert/erfolgreich ist, verwende Default-CT für die andere
    if best_ct_long is None and best_ct_short is None:
        return None, float("-inf"), trades_info

    # Fallback auf mittleren CT-Wert wenn eine Richtung keine Trades hat
    if best_ct_long is None:
        best_ct_long = long_cts[len(long_cts) // 2] if long_cts else 0.5

    if best_ct_short is None:
        best_ct_short = short_cts[len(short_cts) // 2] if short_cts else 0.5

    # Kombinierter PnL (für Vergleich mit anderen Grid-Kombinationen)
    # Simuliere einmal mit den optimalen CTs um echten kombinierten PnL zu bekommen
    combined_result = simulate_trades_sequential_separate_ct(
        val_df, probs_long, probs_short, long_win_idx, short_win_idx,
        best_ct_long, best_ct_short, tp, sl, ctx, return_detailed=False,
        timeout_bars=timeout_bars
    )
    combined_trades = combined_result["trades"]
    combined_pnl = sum(t["pnl_raw"] for t in combined_trades) if len(combined_trades) >= 10 else float("-inf")

    trades_info["combined"] = {
        "ct_long": best_ct_long,
        "ct_short": best_ct_short,
        "trades": combined_trades,
    }

    return (best_ct_long, best_ct_short), combined_pnl, trades_info
