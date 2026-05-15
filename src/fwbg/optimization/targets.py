"""
Target computation, trade simulation, and validation evaluation.

Extracted from nested_cv.py for modularity (keeping files under 600 lines).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from fwbg.core.context import SimulationContext
from fwbg.core import get_exit_strategy, GridParams

if TYPE_CHECKING:
    from fwbg_sdk.models import BaseModel
from fwbg.pipeline.features import REGIME_LONG, REGIME_SHORT
from fwbg.simulation.trade import simulate_pro_trade, compute_session_mask


def _resolve_distances(df: pd.DataFrame, tp: float, sl: float, ctx: SimulationContext):
    """Delegiert Distance-Berechnung an das Exit-Strategy-Plugin."""
    exit_strategy_class = get_exit_strategy(ctx.exit_strategy)
    exit_strategy = exit_strategy_class()
    return exit_strategy.resolve_distances(df, tp, sl, ctx)


def _compute_signal_events(
    probs: Optional[np.ndarray], win_idx: Optional[int], ct: float
) -> Optional[np.ndarray]:
    """Assign event IDs to contiguous runs of signal >= ct.

    Each contiguous block of bars where the model probability exceeds the
    confidence threshold is one "signal event".  When the probability drops
    below ct and rises again, a new event begins.

    Returns int32 array where 0 = no signal, >0 = event ID.
    """
    if probs is None or win_idx is None:
        return None
    n = len(probs)
    events = np.zeros(n, dtype=np.int32)
    event_id = 0
    in_event = False
    for i in range(n):
        if probs[i, win_idx] >= ct:
            if not in_event:
                event_id += 1
                in_event = True
            events[i] = event_id
        else:
            in_event = False
    return events


def _validate_targets(
    targets_long: np.ndarray, targets_short: np.ndarray, ctx: SimulationContext
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


def simulate_trades(
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
    per_trade_params: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Canonical trade-simulation entry point.

    Combines previously-separate wrappers (joint CT, separate CT, single
    direction) into one public function. The three legacy wrappers
    (`simulate_trades_sequential`, `simulate_trades_sequential_separate_ct`,
    `_simulate_single_direction`) remain as thin shims for back-compat.

    Args:
        df: DataFrame mit OHLC-Daten und _regime bitmask column
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
    regime = (
        df["_regime"].values
        if "_regime" in df.columns
        else np.full(len(df), 7, dtype=np.int8)
    )
    timestamps = df.index.values
    has_rv = "vol_rv_20" in df.columns
    rv_values = df["vol_rv_20"].values if has_rv else None

    # TP/SL-Distanzen vom Exit-Strategy-Plugin berechnen lassen
    tp_dists, sl_dists = _resolve_distances(df, tp, sl, ctx)

    # Per-trade TP/SL overrides from model (xgboost_rrr, xgboost_mfe)
    if per_trade_params is not None:
        tp_dists = per_trade_params[:, 0].copy()
        sl_dists = per_trade_params[:, 1].copy()

    # Absolute SL levels: when sl_level is set in exit_params, SL is
    # anchored to the structural price level from that column (e.g. OR midpoint)
    # instead of computed as entry ± sl_dist.
    # sl_level is a suffix like "or_midpoint" — auto-detect the full column name.
    sl_levels = None
    exit_params = ctx.exit_params if ctx.exit_params else {}
    sl_level_suffix = exit_params.get("sl_level")
    if sl_level_suffix and sl_level_suffix != "none":
        sl_col = next(
            (c for c in df.columns if c.endswith(f"_{sl_level_suffix}")),
            None,
        )
        if sl_col is not None:
            sl_levels = df[sl_col].values.astype(np.float64)

    # Entry delay: 0 = entry at signal bar close (breakout stop-orders),
    # 1 = entry at next bar open (default, no look-ahead).
    entry_delay = exit_params.get("entry_delay", 1)

    # Trailing stop from exit_modifier_params (separate composable plugin).
    modifier_params = getattr(ctx, "exit_modifier_params", None) or {}
    breakeven_trigger = modifier_params.get("breakeven_trigger", 0.0)
    trail_atr_mult = modifier_params.get("trail_atr_mult", 0.0)

    # Resolve per-bar trail distances: range mode uses the OR range as trail
    # distance (structural), ATR mode uses ATR * trail_atr_mult.
    trail_dists = np.zeros(len(df), dtype=np.float64)
    if trail_atr_mult > 0.0:
        tp_mode = exit_params.get("tp_mode", "atr")
        if tp_mode == "range":
            exit_strategy_cls = get_exit_strategy(ctx.exit_strategy)
            es = exit_strategy_cls()
            if hasattr(es, "_get_range"):
                range_v = es._get_range(df, exit_params)
                trail_dists = np.where(range_v > 0.0, range_v, 0.0).astype(np.float64)
        else:
            atr_col = "_atr" if "_atr" in df.columns else ("vol_atr" if "vol_atr" in df.columns else None)
            if atr_col:
                atr_v = np.nan_to_num(df[atr_col].values.astype(np.float64), nan=0.0)
            else:
                import ta
                atr_period = exit_params.get("atr_period", 14)
                atr_v = np.nan_to_num(
                    ta.volatility.average_true_range(
                        df["H"], df["L"], df["C"], window=atr_period,
                    ).values.astype(np.float64),
                    nan=0.0,
                )
            trail_dists = atr_v * trail_atr_mult

    # Entry modifier params for scale-in
    entry_mod_params = getattr(ctx, "entry_modifier_params", None) or {}
    scale_levels = entry_mod_params.get("levels", None)
    scale_qty_mult = entry_mod_params.get("qty_multiplier", 1.0)

    # Session-aware exits: only exit during session hours.
    # Trades may run through off-session periods (overnight holds).
    # Prefer exit_session hours (wider CFD window), fall back to session hours.
    in_session = None
    s_start = getattr(ctx, "exit_session_start_hour", None)
    if s_start is None:
        s_start = getattr(ctx, "session_start_hour", None)
    s_end = getattr(ctx, "exit_session_end_hour", None)
    if s_end is None:
        s_end = getattr(ctx, "session_end_hour", None)
    if isinstance(s_start, int) and isinstance(s_end, int):
        in_session = compute_session_mask(
            df.index, s_start, s_end,
            ohlc=(opn, hgh, low, cls),
        )

    # Signal rules pre-filter: when _composed_signal_long/short columns exist,
    # only allow entries on bars where the signal is active (== 1.0).
    # This lets signal_rules act as entry gates for ML models.
    # Skipped when return_detailed=False during inner-CV evaluation (too few
    # signal bars in small validation windows would starve the grid search).
    signal_long = None
    signal_short = None
    if return_detailed:
        signal_long = (
            df["_composed_signal_long"].values
            if "_composed_signal_long" in df.columns
            else None
        )
        signal_short = (
            df["_composed_signal_short"].values
            if "_composed_signal_short" in df.columns
            else None
        )

    # Signal event limiting: prevent re-entry into the same persistent signal.
    # A "signal event" is a contiguous run of bars where P(win) >= ct.
    # max_trades_per_signal=1 means one trade per breakout event (ORB default).
    max_per_signal = exit_params.get("max_trades_per_signal", 0)
    long_event_ids = None
    short_event_ids = None
    long_event_trades: Dict[int, int] = {}
    short_event_trades: Dict[int, int] = {}
    if max_per_signal > 0:
        long_event_ids = _compute_signal_events(probs_long, long_win_idx, ct_long)
        short_event_ids = _compute_signal_events(probs_short, short_win_idx, ct_short)

    trades = []
    trades_detailed = [] if return_detailed else None
    next_allowed_entry = 0

    # Simuliere bis zum vorletzten Bar (letzter Bar kann kein Entry sein)
    for i in range(len(df) - 1):
        if i < next_allowed_entry:
            continue

        direction = None
        # Long-Check: regime bitmask must have REGIME_LONG bit set
        if direction_filter in (None, 1):
            if (
                regime[i] & REGIME_LONG
                and ctx.long_enabled
                and probs_long is not None
                and probs_long[i, long_win_idx] >= ct_long
                and (signal_long is None or signal_long[i] >= 1.0)
            ):
                # Check signal event limit
                if max_per_signal > 0 and long_event_ids is not None:
                    eid = long_event_ids[i]
                    if eid > 0 and long_event_trades.get(eid, 0) < max_per_signal:
                        direction = 1
                else:
                    direction = 1
        # Short-Check: regime bitmask must have REGIME_SHORT bit set
        if direction is None and direction_filter in (None, -1):
            if (
                regime[i] & REGIME_SHORT
                and ctx.short_enabled
                and probs_short is not None
                and probs_short[i, short_win_idx] >= ct_short
                and (signal_short is None or signal_short[i] >= 1.0)
            ):
                if max_per_signal > 0 and short_event_ids is not None:
                    eid = short_event_ids[i]
                    if eid > 0 and short_event_trades.get(eid, 0) < max_per_signal:
                        direction = -1
                else:
                    direction = -1

        if direction:
            sl_abs = None
            if sl_levels is not None:
                v = sl_levels[i]
                if not np.isnan(v):
                    sl_abs = v
            td = trail_dists[i]
            trade = simulate_pro_trade(
                cls,
                hgh,
                low,
                i,
                direction,
                tp_dists[i],
                sl_dists[i],
                ctx.spread,
                timestamps=timestamps,
                symbol=ctx.symbol,
                opens=opn,
                max_bars=ctx.max_trade_bars,
                timeout_bars=timeout_bars,
                in_session=in_session,
                sl_level_abs=sl_abs,
                entry_delay=entry_delay,
                breakeven_trigger=breakeven_trigger,
                trail_distance=td if breakeven_trigger > 0.0 else 0.0,
                scale_levels=scale_levels,
                scale_qty_mult=scale_qty_mult,
            )
            if trade:
                t = {"result": trade["result"], "pnl_raw": trade["pnl_raw"],
                     "mae": trade["mae"], "mfe": trade["mfe"]}
                if has_rv:
                    rv_val = float(rv_values[i])
                    if not np.isnan(rv_val):
                        t["rv_at_entry"] = rv_val
                trades.append(t)
                next_allowed_entry = trade["exit_idx"] + 1

                # Record trade against its signal event
                if max_per_signal > 0:
                    if direction == 1 and long_event_ids is not None:
                        eid = long_event_ids[i]
                        long_event_trades[eid] = long_event_trades.get(eid, 0) + 1
                    elif direction == -1 and short_event_ids is not None:
                        eid = short_event_ids[i]
                        short_event_trades[eid] = short_event_trades.get(eid, 0) + 1

                if return_detailed:
                    trade["ct"] = ct_long if direction == 1 else ct_short
                    trade["hour"] = df.index[i].hour
                    trades_detailed.append(trade)

    result = {"trades": trades}
    if return_detailed:
        result["trades_detailed"] = trades_detailed
    return result


# Back-compat alias — older callers (api/runs.py, tests) import this name.
_simulate_trades_core = simulate_trades


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
    per_trade_params: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Simuliert Trades sequentiell mit gleichem CT für Long/Short.

    Back-compat wrapper around :func:`simulate_trades`.
    """
    return simulate_trades(
        df,
        probs_long,
        probs_short,
        long_win_idx,
        short_win_idx,
        ct,
        ct,
        tp,
        sl,
        ctx,
        return_detailed,
        timeout_bars,
        per_trade_params=per_trade_params,
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
    per_trade_params: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Simuliert Trades für eine einzelne Richtung (Long oder Short).

    Back-compat wrapper around :func:`simulate_trades` with a direction filter.
    """
    if direction == 1:
        return simulate_trades(
            df,
            probs,
            None,
            win_idx,
            None,
            ct,
            0.0,
            tp,
            sl,
            ctx,
            False,
            timeout_bars,
            direction_filter=1,
            per_trade_params=per_trade_params,
        )
    else:
        return simulate_trades(
            df,
            None,
            probs,
            None,
            win_idx,
            0.0,
            ct,
            tp,
            sl,
            ctx,
            False,
            timeout_bars,
            direction_filter=-1,
            per_trade_params=per_trade_params,
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
    per_trade_params: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Simuliert Trades mit separaten CT-Thresholds für Long und Short.

    Back-compat wrapper around :func:`simulate_trades`.
    """
    return simulate_trades(
        df,
        probs_long,
        probs_short,
        long_win_idx,
        short_win_idx,
        ct_long,
        ct_short,
        tp,
        sl,
        ctx,
        return_detailed,
        timeout_bars,
        per_trade_params=per_trade_params,
    )


def compute_targets(
    df: pd.DataFrame, tp: int, sl: int, ctx: SimulationContext, timeout_bars: int = None
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """
    Berechnet Long/Short Targets für einen DataFrame.

    Delegiert an compute_targets_cached (Plugin-Dispatch).

    Args:
        df: DataFrame mit OHLC-Daten
        tp: Take-Profit Wert
        sl: Stop-Loss Wert
        ctx: SimulationContext
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen

    Returns:
        (targets_long, targets_short, has_long, has_short)
    """
    result = compute_targets_cached(
        df,
        tp,
        sl,
        ctx,
        timeout_bars,
        exit_strategy_mode=ctx.exit_strategy,
    )
    targets_long, targets_short = result[0], result[1]
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
    exit_strategy_class = get_exit_strategy(exit_strategy_mode)
    exit_strategy = exit_strategy_class()

    extra = {}
    if hasattr(ctx, "exit_params") and ctx.exit_params:
        extra = ctx.exit_params.copy()

    if grid_params is None:
        grid_params = GridParams(
            tp_value=float(tp),
            sl_value=float(sl),
            timeout_bars=timeout_bars,
            extra=extra,
        )

    return exit_strategy.compute_targets(
        full_df, ctx, params=grid_params, return_durations=return_durations
    )


def compute_mfe_targets(
    df: pd.DataFrame,
    sl_atr: float,
    max_bars: int = 50,
    spread: float = 0.0,
    atr_col: str = "_atr",
    timeout_bars: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Maximum Favorable Excursion in ATR multiples per bar.

    For each bar, simulates a hypothetical long and short trade with the
    given SL (in ATR multiples). Tracks the maximum favorable price movement
    before the trade is stopped out or times out. Returns MFE normalized
    by ATR.

    Args:
        df: DataFrame with OHLC and ATR column.
        sl_atr: Stop-loss in ATR multiples.
        max_bars: Maximum trade duration to consider.
        spread: Bid-ask spread.
        atr_col: Name of ATR column (default "_atr").
        timeout_bars: Optional trade timeout.

    Returns:
        (mfe_long, mfe_short): Arrays of shape (n,) with MFE in ATR multiples.
    """
    if atr_col not in df.columns:
        fallback = "vol_atr" if "vol_atr" in df.columns else None
        if fallback:
            atr_col = fallback
        else:
            raise ValueError(f"ATR column '{atr_col}' not found in DataFrame")

    closes = df["C"].values
    highs = df["H"].values
    lows = df["L"].values
    opens = df["O"].values
    atr = df[atr_col].values.astype(np.float64)
    n = len(df)
    effective_timeout = timeout_bars or max_bars

    mfe_long = np.zeros(n, dtype=np.float64)
    mfe_short = np.zeros(n, dtype=np.float64)

    # Mirror the trade simulator's spread/slippage model so MFE-derived targets
    # use the same effective fill price the live simulator would. Slippage is
    # half the spread (see simulation/trade.py::simulate_pro_trade).
    slippage = spread * 0.5

    for i in range(n - 1):
        atr_i = atr[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        sl_dist = atr_i * sl_atr
        # entry_delay=1: enter at next bar open
        mid_entry = opens[i + 1] if i + 1 < n else closes[i]

        # Long: pay the ask + slippage on entry; exit at the mark (TP/SL acts
        # as a trigger level on the mid). Spread/slippage are already baked
        # into the effective entry so no extra subtraction at exit.
        long_entry = mid_entry + spread + slippage
        max_favorable = 0.0
        for j in range(i + 1, min(i + 1 + effective_timeout, n)):
            favorable = highs[j] - long_entry
            if favorable > max_favorable:
                max_favorable = favorable
            adverse = long_entry - lows[j]
            if adverse >= sl_dist:
                break
        mfe_long[i] = max_favorable / atr_i

        # Short: sell the bid - slippage on entry.
        short_entry = mid_entry - spread - slippage
        max_favorable = 0.0
        for j in range(i + 1, min(i + 1 + effective_timeout, n)):
            favorable = short_entry - lows[j]
            if favorable > max_favorable:
                max_favorable = favorable
            adverse = highs[j] - short_entry
            if adverse >= sl_dist:
                break
        mfe_short[i] = max_favorable / atr_i

    return mfe_long, mfe_short


def slice_targets_for_fold(
    fold_df: pd.DataFrame,
    ctx: SimulationContext,
    tp: float,
    sl: float,
    timeout_bars: Optional[int] = None,
    return_durations: bool = False,
) -> Tuple:
    """Compute targets for one inner-CV fold using ONLY the fold's own bars.

    Embargo fix: the previous implementation precomputed targets on the entire
    inner_df and sliced per fold by index — that leaked val information into
    train targets near the fold boundary, because the forward TP/SL window of
    the last train bars overlapped the val region. Recomputing per fold
    eliminates that leak: target at bar t now uses only bars in [t+1, t+max_bars]
    inside fold_df.

    Args:
        fold_df: The fold's DataFrame (train slice in production).
        ctx: SimulationContext (exit_strategy, max_trade_bars, …).
        tp: Take-profit multiplier.
        sl: Stop-loss multiplier.
        timeout_bars: Optional trade-timeout override.
        return_durations: When True, also return per-trade durations.

    Returns:
        return_durations=False -> (targets_long, targets_short, has_long, has_short)
        return_durations=True  -> (targets_long, targets_short, dur_long, dur_short,
                                   has_long, has_short)
    """
    exit_strategy_mode = getattr(ctx, "exit_strategy", "fixed")
    result = compute_targets_cached(
        fold_df, tp, sl, ctx, timeout_bars,
        exit_strategy_mode=exit_strategy_mode,
        return_durations=return_durations,
    )
    if return_durations:
        targets_long, targets_short, dur_long, dur_short = result
    else:
        targets_long, targets_short = result

    has_long, has_short = _validate_targets(targets_long, targets_short, ctx)
    if return_durations:
        return targets_long, targets_short, dur_long, dur_short, has_long, has_short
    return targets_long, targets_short, has_long, has_short


def _get_probs(
    model: Optional["BaseModel"], df: pd.DataFrame, features: Optional[List[str]]
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Berechnet Wahrscheinlichkeiten für ein Modell."""
    if not features or model is None:
        return None, None
    X = df[features].copy()
    X_vals = X.values.copy()
    inf_mask = np.isinf(X_vals)
    if inf_mask.any():
        X_vals[inf_mask] = np.nan
        X = pd.DataFrame(X_vals, columns=X.columns, index=X.index)
    probs = model.predict_probability(X)
    if 1 in model.trained_classes:
        win_idx = np.where(model.trained_classes == 1)[0][0]
        return probs, win_idx
    return None, None


def _apply_meta_filter(
    df: Optional[pd.DataFrame],
    probs: np.ndarray,
    win_idx: int,
    features: List[str],
    meta_model: Optional[Any],
) -> np.ndarray:
    """
    Apply meta-model filter to zero out low-confidence predictions (AFML Ch. 3).

    The meta-model predicts whether the primary signal will be profitable.
    Bars where meta-model says 'skip' (P(trade) < 0.5) get zeroed probs.

    Args:
        df: DataFrame with feature columns (None when no meta_model)
        probs: Primary model's probability array (n_samples, n_classes)
        win_idx: Index of the win class in probs
        features: Feature column names used by primary model
        meta_model: Trained meta-model (or None to pass through)

    Returns:
        Filtered probability array (same shape as probs)
    """
    if meta_model is None:
        return probs

    primary_probs = probs[:, win_idx]
    X_meta = np.column_stack([df[features].values, primary_probs])

    X_meta_df = pd.DataFrame(X_meta, columns=features + ["oof_prob"])
    meta_probs = meta_model.predict_probability(X_meta_df)
    if 1 in meta_model.trained_classes:
        meta_win_idx = np.where(meta_model.trained_classes == 1)[0][0]
    else:
        return probs

    # Zero out bars where meta-model predicts "skip trade"
    mask = meta_probs[:, meta_win_idx] < 0.5
    filtered = probs.copy()
    filtered[mask] = 0.0
    return filtered


def evaluate_on_validation(
    val_df: pd.DataFrame,
    mod_long: Optional["BaseModel"],
    mod_short: Optional["BaseModel"],
    features_long: Optional[List[str]],
    features_short: Optional[List[str]],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    timeout_bars: int = None,
    meta_mod_long: Optional[Any] = None,
    meta_mod_short: Optional[Any] = None,
) -> Tuple[Optional[float], float, Dict[float, List[float]]]:
    """
    Evaluiert Modelle auf Validation-Set und findet besten CT.

    Bei separate_long_short=True werden separate CTs für Long und Short optimiert.
    Bei meta_mod_long/short: Meta-Labeling Filter wird angewandt (AFML Ch. 3).

    Args:
        timeout_bars: Optional - nach X Bars ohne TP/SL zum Close schließen
        meta_mod_long: Optional meta-model for filtering long predictions
        meta_mod_short: Optional meta-model for filtering short predictions

    Returns:
        (best_ct, best_pnl, trades_by_ct)
        Bei separate_long_short: best_ct ist ein Tuple (ct_long, ct_short)
    """
    probs_long, long_win_idx = _get_probs(mod_long, val_df, features_long)
    probs_short, short_win_idx = _get_probs(mod_short, val_df, features_short)

    # Per-trade TP/SL overrides from model (xgboost_rrr, xgboost_mfe)
    per_trade_params = None
    atr_col = "_atr" if "_atr" in val_df.columns else ("vol_atr" if "vol_atr" in val_df.columns else None)
    atr_vals = val_df[atr_col].values if atr_col else None
    if mod_long is not None:
        ptp = mod_long.get_per_trade_params(val_df[features_long], atr=atr_vals)
        if ptp is not None:
            per_trade_params = ptp
    if per_trade_params is None and mod_short is not None:
        ptp = mod_short.get_per_trade_params(val_df[features_short], atr=atr_vals)
        if ptp is not None:
            per_trade_params = ptp

    # Meta-Labeling: filter predictions via meta-model
    if meta_mod_long is not None and probs_long is not None:
        probs_long = _apply_meta_filter(
            val_df, probs_long, long_win_idx, features_long, meta_mod_long
        )
    if meta_mod_short is not None and probs_short is not None:
        probs_short = _apply_meta_filter(
            val_df, probs_short, short_win_idx, features_short, meta_mod_short
        )

    # Probability Calibration: EV-optimal threshold replaces CT grid
    min_eval = ctx.min_eval_trades
    if ctx.probability_calibration:
        ct_ev = sl / (tp + sl)
        if ctx.separate_long_short:
            result = simulate_trades_sequential_separate_ct(
                val_df,
                probs_long,
                probs_short,
                long_win_idx,
                short_win_idx,
                ct_ev,
                ct_ev,
                tp,
                sl,
                ctx,
                return_detailed=False,
                timeout_bars=timeout_bars,
                per_trade_params=per_trade_params,
            )
            trades = result["trades"]
            pnl = (
                sum(t["pnl_raw"] for t in trades)
                if len(trades) >= min_eval
                else float("-inf")
            )
            best_ct = (ct_ev, ct_ev)
        else:
            result = simulate_trades_sequential(
                val_df,
                probs_long,
                probs_short,
                long_win_idx,
                short_win_idx,
                ct_ev,
                tp,
                sl,
                ctx,
                return_detailed=False,
                timeout_bars=timeout_bars,
                per_trade_params=per_trade_params,
            )
            trades = result["trades"]
            pnl = (
                sum(t["pnl_raw"] for t in trades)
                if len(trades) >= min_eval
                else float("-inf")
            )
            best_ct = ct_ev
        return best_ct, pnl, {ct_ev: trades}

    # Separate CT-Optimierung wenn aktiviert
    if ctx.separate_long_short:
        return _evaluate_separate_ct(
            val_df,
            probs_long,
            probs_short,
            long_win_idx,
            short_win_idx,
            tp,
            sl,
            ctx,
            timeout_bars,
            per_trade_params=per_trade_params,
        )

    # Standard: Gemeinsamer CT für Long und Short
    # SEQUENTIELLE Evaluierung - kein nested Threading
    # (Feature-Gruppen sind bereits parallelisiert)
    trades_by_ct = {}
    for ct in ctx.grid_ct:
        result = simulate_trades_sequential(
            val_df,
            probs_long,
            probs_short,
            long_win_idx,
            short_win_idx,
            ct,
            tp,
            sl,
            ctx,
            return_detailed=False,
            timeout_bars=timeout_bars,
            per_trade_params=per_trade_params,
        )
        trades_by_ct[ct] = result["trades"]

    # Besten CT finden
    best_ct = None
    best_pnl = float("-inf")
    for ct, ct_trades in trades_by_ct.items():
        if len(ct_trades) >= min_eval:
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
    min_trades: int = 1,
    per_trade_params: Optional[np.ndarray] = None,
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
        per_trade_params: Optional per-trade TP/SL overrides

    Returns:
        (best_ct, best_pnl, trades_by_ct)
    """
    # SEQUENTIELLE Evaluierung - kein nested Threading
    trades_by_ct = {}
    for ct in ct_values:
        result = _simulate_single_direction(
            val_df,
            probs,
            win_idx,
            ct,
            tp,
            sl,
            ctx,
            direction=direction,
            timeout_bars=timeout_bars,
            per_trade_params=per_trade_params,
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
    per_trade_params: Optional[np.ndarray] = None,
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
            val_df,
            probs_long,
            long_win_idx,
            long_cts,
            tp,
            sl,
            ctx,
            direction=1,
            timeout_bars=timeout_bars,
            per_trade_params=per_trade_params,
        )
        trades_info["long"] = long_trades_by_ct

    # === SHORT CT OPTIMIERUNG (unabhängig) ===
    best_ct_short, best_pnl_short, short_trades_by_ct = None, float("-inf"), {}
    if ctx.short_enabled and probs_short is not None:
        best_ct_short, best_pnl_short, short_trades_by_ct = _optimize_ct_for_direction(
            val_df,
            probs_short,
            short_win_idx,
            short_cts,
            tp,
            sl,
            ctx,
            direction=-1,
            timeout_bars=timeout_bars,
            per_trade_params=per_trade_params,
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
        val_df,
        probs_long,
        probs_short,
        long_win_idx,
        short_win_idx,
        best_ct_long,
        best_ct_short,
        tp,
        sl,
        ctx,
        return_detailed=False,
        timeout_bars=timeout_bars,
        per_trade_params=per_trade_params,
    )
    combined_trades = combined_result["trades"]
    min_eval = ctx.min_eval_trades
    combined_pnl = (
        sum(t["pnl_raw"] for t in combined_trades)
        if len(combined_trades) >= min_eval
        else float("-inf")
    )

    trades_info["combined"] = {
        "ct_long": best_ct_long,
        "ct_short": best_ct_short,
        "trades": combined_trades,
    }

    return (best_ct_long, best_ct_short), combined_pnl, trades_info
