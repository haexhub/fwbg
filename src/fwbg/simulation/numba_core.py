"""
Numba-optimierte Kernfunktionen für Trade-Simulation.

Cache versioning: bump _CACHE_VERSION whenever the return signature of any
@njit function changes, or a kernel's behaviour changes: cached callers in
plugins keep their compiled copy of a changed callee otherwise.  At import time we compare against a stamp file;
on mismatch we wipe every .nbi/.nbc we can find and rewrite the stamp.
"""
import pathlib
import numpy as np
from numba import njit, prange

# Bump this whenever a @njit function signature/return type or kernel behaviour changes.
_CACHE_VERSION = "7"

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_STAMP_FILE = _THIS_DIR / "__pycache__" / ".numba_cache_version"


def _clear_numba_cache():
    """Delete ALL Numba .nbi/.nbc cache files reachable from this package."""
    removed = 0
    # Walk upward to find project root (directory containing src/ or packages/)
    search_roots = []
    for parent in _THIS_DIR.parents:
        src = parent / "src"
        pkg = parent / "packages"
        if src.is_dir() or pkg.is_dir():
            if src.is_dir():
                search_roots.append(src)
            if pkg.is_dir():
                search_roots.append(pkg)
            break
    # Fallback: at least clear our own directory tree
    if not search_roots:
        search_roots.append(_THIS_DIR)
    for root in search_roots:
        for ext in ("*.nbi", "*.nbc"):
            for f in root.rglob(ext):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    if removed:
        import logging
        logging.getLogger(__name__).info(f"Cleared {removed} stale Numba cache files")


def _check_cache_version():
    """Clear all Numba caches if the version stamp doesn't match."""
    _STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        current = _STAMP_FILE.read_text().strip()
    except (FileNotFoundError, OSError):
        current = ""
    if current != _CACHE_VERSION:
        _clear_numba_cache()
        try:
            _STAMP_FILE.write_text(_CACHE_VERSION)
        except OSError:
            pass


# IMPORTANT: Must run BEFORE @njit definitions so stale caches are cleared
# before Numba loads them into memory at decoration time.
_check_cache_version()


# --- Shared exit rules -------------------------------------------------------
# One definition of how a bar exits a trade, used by every kernel below and,
# via ``.py_func``, by the Python evaluator ``simulate_pro_trade``.  Order per
# bar: TP/SL against the levels active at the bar's start, then the timeout at
# the bar's close, then trailing updates that take effect from the next bar
# (OHLC has no intrabar order).


@njit(cache=True)
def _entry_levels(direction, entry_price, spread, slippage, tp_distance, sl_distance):
    """Cost-adjusted entry and TP/SL trigger levels.

    Spread and slippage always work against the trader and are folded into the
    entry, so exits at the trigger levels are already net of costs.
    """
    if direction == 1:
        entry = entry_price + spread + slippage
        return entry, entry + tp_distance, entry - sl_distance
    entry = entry_price - spread - slippage
    return entry, entry - tp_distance, entry + sl_distance


@njit(cache=True)
def _timeout_index(entry_idx, timeout_bars, n):
    """Bar index at which a bar-count timeout fires (-1 = no timeout)."""
    if timeout_bars > 0:
        return min(entry_idx + timeout_bars - 1, n - 1)
    return -1


@njit(cache=True)
def _bar_hits(direction, high, low, tp, sl):
    """(tp_hit, sl_hit) for one bar."""
    if direction == 1:
        return high >= tp, low <= sl
    return low <= tp, high >= sl


@njit(cache=True)
def _stop_fill(direction, sl, bar_open):
    """Stop exit price: a gap through the stop fills at the open."""
    if direction == 1:
        return min(sl, bar_open)
    return max(sl, bar_open)


@njit(cache=True)
def _pnl(direction, exit_price, ref_price, qty):
    if direction == 1:
        return (exit_price - ref_price) * qty
    return (ref_price - exit_price) * qty


@njit(cache=True)
def _outcome(direction, exit_price, ref_price, qty):
    """1.0 if exiting at *exit_price* is a net win against *ref_price*, else -1.0."""
    return 1.0 if _pnl(direction, exit_price, ref_price, qty) > 0 else -1.0


@njit(cache=True)
def _scale_in_triggered(direction, high, low, trigger, sl):
    """A scale-in level fills when touched, unless it lies beyond the stop."""
    if direction == 1:
        return low <= trigger and trigger > sl
    return high >= trigger and trigger < sl


@njit(cache=True)
def _trail_after_bar(direction, high, low, best_price, sl, tp, trailing_active,
                     use_breakeven, be_trigger_price, be_sl,
                     trail_distance, trail_tp_dist):
    """Advance breakeven/trailing levels after a bar has been checked.

    Returns (best_price, sl, tp, trailing_active).  Levels only ever tighten.
    """
    if direction == 1:
        best_price = max(best_price, high)
        if not trailing_active and use_breakeven and best_price >= be_trigger_price:
            trailing_active = True
            sl = max(sl, be_sl)
        if trailing_active and trail_distance > 0.0:
            sl = max(sl, best_price - trail_distance)
        if trailing_active and trail_tp_dist > 0.0:
            tp = max(tp, best_price + trail_tp_dist)
    else:
        best_price = min(best_price, low)
        if not trailing_active and use_breakeven and best_price <= be_trigger_price:
            trailing_active = True
            sl = min(sl, be_sl)
        if trailing_active and trail_distance > 0.0:
            sl = min(sl, best_price + trail_distance)
        if trailing_active and trail_tp_dist > 0.0:
            tp = min(tp, best_price - trail_tp_dist)
    return best_price, sl, tp, trailing_active


@njit(cache=True)
def _simulate_trade_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    direction: int,
    tp_distance: float,
    sl_distance: float,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
) -> tuple:
    """
    Numba-optimierte Trade-Simulation (Kern-Loop).

    Args:
        opens, closes, highs, lows: OHLC-Arrays
        idx: Signal-Index (Entry bei idx+1)
        direction: 1=Long, -1=Short
        tp_distance: Take-Profit Distanz in Preiseinheiten
        sl_distance: Stop-Loss Distanz in Preiseinheiten
        spread: Bid-Ask Spread
        slippage: Slippage-Kosten
        max_bars: Maximale Simulation-Länge
        timeout_bars: Trade schließen nach X Bars (0 = kein Timeout)

    Returns:
        (result, exit_idx, exit_price, exit_reason)
        result: 1.0=Win, -1.0=Loss, 0.0=Kein Ergebnis
        exit_reason: 0=TP, 1=SL, 2=Timeout, -1=Kein Exit
    """
    entry_idx = idx + 1
    n = len(closes)

    if entry_idx >= n:
        return 0.0, -1, 0.0, -1

    entry, tp, sl = _entry_levels(
        direction, opens[entry_idx], spread, slippage, tp_distance, sl_distance
    )
    end_idx = min(entry_idx + max_bars, n)
    timeout_idx = _timeout_index(entry_idx, timeout_bars, n)

    for j in range(entry_idx, end_idx):
        tp_hit, sl_hit = _bar_hits(direction, highs[j], lows[j], tp, sl)
        # Beide im selben Bar - konservativ: SL
        if sl_hit:
            fill = _stop_fill(direction, sl, opens[j])
            return _outcome(direction, fill, entry, 1.0), j, fill, 1
        if tp_hit:
            return 1.0, j, tp, 0
        if timeout_idx > 0 and j >= timeout_idx:
            return _outcome(direction, closes[j], entry, 1.0), j, closes[j], 2

    # Kein Exit (weder TP/SL noch Timeout innerhalb max_bars)
    return 0.0, -1, 0.0, -1


@njit(cache=True)
def _simulate_trade_session_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    direction: int,
    tp_distance: float,
    sl_distance: float,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
    in_session: np.ndarray,
) -> tuple:
    """
    Session-aware trade simulation: exits only during session hours.

    Trades may run through off-session periods (overnight holds allowed).
    TP/SL checks and timeout counting only happen on in-session bars.

    Args:
        in_session: boolean array (len n). True = bar is within trading session.
        Other args identical to _simulate_trade_numba.

    Returns:
        (result, exit_idx, exit_price, exit_reason)
    """
    entry_idx = idx + 1
    n = len(closes)

    if entry_idx >= n:
        return 0.0, -1, 0.0, -1

    entry, tp, sl = _entry_levels(
        direction, opens[entry_idx], spread, slippage, tp_distance, sl_distance
    )
    end_idx = min(entry_idx + max_bars, n)
    session_bars_elapsed = 0

    for j in range(entry_idx, end_idx):
        if not in_session[j]:
            continue

        session_bars_elapsed += 1

        tp_hit, sl_hit = _bar_hits(direction, highs[j], lows[j], tp, sl)
        if sl_hit:
            fill = _stop_fill(direction, sl, opens[j])
            return _outcome(direction, fill, entry, 1.0), j, fill, 1
        if tp_hit:
            return 1.0, j, tp, 0
        if timeout_bars > 0 and session_bars_elapsed >= timeout_bars:
            return _outcome(direction, closes[j], entry, 1.0), j, closes[j], 2

    return 0.0, -1, 0.0, -1


@njit(cache=True)
def _simulate_trade_trailing_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    direction: int,
    tp_distance: float,
    sl_distance: float,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
    breakeven_trigger: float,
    trail_distance: float,
    trail_tp_dist: float,
    breakeven_offset: float = 0.0,
) -> tuple:
    """
    Single-trade simulation with breakeven and trailing stop.

    Args:
        breakeven_trigger: Fraction of tp_distance at which SL moves to breakeven.
                           0.0 = no breakeven (trailing starts immediately if trail_distance > 0).
        trail_distance:    Absolute trailing stop distance from best price.
                           0.0 = no trailing.
        trail_tp_dist:     Absolute trailing TP distance from best price.
                           0.0 = no trailing TP.
        breakeven_offset:  Fraction of tp_distance to add above entry when breakeven
                           triggers. E.g. 0.1 = SL moves to entry + 10% of TP distance
                           (guarantees profit). 0.0 = SL moves exactly to entry.

    Returns:
        (result, exit_idx, exit_price, exit_reason)
        result: 1.0=Win, -1.0=Loss, 0.0=no exit
        exit_reason: 0=TP, 1=SL, 2=Timeout, -1=no exit
    """
    entry_idx = idx + 1
    n = len(closes)

    if entry_idx >= n:
        return 0.0, -1, 0.0, -1

    entry, tp, sl = _entry_levels(
        direction, opens[entry_idx], spread, slippage, tp_distance, sl_distance
    )
    be_trigger_price = entry + direction * tp_distance * breakeven_trigger
    be_sl = entry + direction * breakeven_offset * tp_distance
    end_idx = min(entry_idx + max_bars, n)
    timeout_idx = _timeout_index(entry_idx, timeout_bars, n)

    best_price = entry
    trailing_active = breakeven_trigger <= 0.0

    for j in range(entry_idx, end_idx):
        tp_hit, sl_hit = _bar_hits(direction, highs[j], lows[j], tp, sl)
        if sl_hit:
            fill = _stop_fill(direction, sl, opens[j])
            return _outcome(direction, fill, entry, 1.0), j, fill, 1
        if tp_hit:
            return 1.0, j, tp, 0
        if timeout_idx > 0 and j >= timeout_idx:
            return _outcome(direction, closes[j], entry, 1.0), j, closes[j], 2

        best_price, sl, tp, trailing_active = _trail_after_bar(
            direction, highs[j], lows[j], best_price, sl, tp, trailing_active,
            breakeven_trigger > 0.0, be_trigger_price, be_sl,
            trail_distance, trail_tp_dist,
        )

    return 0.0, -1, 0.0, -1


@njit(cache=True)
def _simulate_trade_scale_in_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    idx: int,
    direction: int,
    tp_distance: float,
    sl_distance: float,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
    scale_levels: np.ndarray,
    n_levels: int,
    scale_qty_mult: float,
    breakeven_trigger: float,
    trail_distance: float,
    trail_tp_dist: float,
) -> tuple:
    """
    Trade simulation with scale-in (multiple entries at retracement levels).

    Supports additional entries at configurable retracement levels (fractions
    of the entry-to-SL distance).  After each fill the average price is
    recalculated and the TP is adjusted from the new average.  The SL stays
    at the original level unless trailing tightens it.

    Args:
        scale_levels: shape (max_levels,), fractions 0-1, -1.0=unused
        n_levels:     actual number of active levels
        scale_qty_mult: quantity per scale-in (1.0 = same as initial)
        breakeven_trigger: fraction of tp_distance; 0.0 = off
        trail_distance: absolute trailing stop distance; 0.0 = off
        trail_tp_dist:  absolute trailing TP distance; 0.0 = off

    Returns:
        (result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills)
        result: 1.0=Win, -1.0=Loss, 0.0=no exit
        exit_reason: 0=TP, 1=SL, 2=Timeout, -1=no exit
    """
    entry_idx = idx + 1
    n = len(closes)

    if entry_idx >= n:
        return 0.0, -1, 0.0, -1, 0.0, 0.0, 0

    entry, tp, sl = _entry_levels(
        direction, opens[entry_idx], spread, slippage, tp_distance, sl_distance
    )

    # --- Scale-in preparation ---
    total_qty = 1.0
    weighted_sum = entry * 1.0
    avg_price = entry
    n_fills = 1

    scale_price = np.empty(n_levels, dtype=np.float64)
    levels_filled = np.zeros(n_levels, dtype=np.int8)
    for k in range(n_levels):
        scale_price[k] = entry - direction * scale_levels[k] * sl_distance

    end_idx = min(entry_idx + max_bars, n)
    timeout_idx = _timeout_index(entry_idx, timeout_bars, n)

    best_price = entry
    trailing_active = breakeven_trigger <= 0.0

    for j in range(entry_idx, end_idx):
        # --- Scale-in fills (TP and breakeven follow the new average) ---
        for k in range(n_levels):
            if levels_filled[k] == 1:
                continue
            if _scale_in_triggered(direction, highs[j], lows[j], scale_price[k], sl):
                weighted_sum += scale_price[k] * scale_qty_mult
                total_qty += scale_qty_mult
                avg_price = weighted_sum / total_qty
                tp = avg_price + direction * tp_distance
                levels_filled[k] = 1
                n_fills += 1

        tp_hit, sl_hit = _bar_hits(direction, highs[j], lows[j], tp, sl)
        if sl_hit:
            fill = _stop_fill(direction, sl, opens[j])
            result = _outcome(direction, fill, avg_price, total_qty)
            return result, j, fill, 1, avg_price, total_qty, n_fills
        if tp_hit:
            return 1.0, j, tp, 0, avg_price, total_qty, n_fills
        if timeout_idx > 0 and j >= timeout_idx:
            result = _outcome(direction, closes[j], avg_price, total_qty)
            return result, j, closes[j], 2, avg_price, total_qty, n_fills

        best_price, sl, tp, trailing_active = _trail_after_bar(
            direction, highs[j], lows[j], best_price, sl, tp, trailing_active,
            breakeven_trigger > 0.0,
            avg_price + direction * tp_distance * breakeven_trigger, avg_price,
            trail_distance, trail_tp_dist,
        )

    return 0.0, -1, 0.0, -1, avg_price, total_qty, n_fills


@njit(cache=True, parallel=True)
def compute_targets_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_distances: np.ndarray,
    sl_distances: np.ndarray,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
) -> tuple:
    """
    Berechnet Long/Short Targets und Durations für alle Bars.

    PARALLELISIERT: Nutzt alle verfügbaren CPU-Kerne für 2-4x Speedup.

    Args:
        opens, closes, highs, lows: OHLC-Arrays
        tp_distances: Per-bar TP-Distanzen (Array, len == n)
        sl_distances: Per-bar SL-Distanzen (Array, len == n)

    Returns:
        (targets_long, targets_short, durations_long, durations_short)
    """
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)
    durations_long = np.zeros(n, dtype=np.int64)
    durations_short = np.zeros(n, dtype=np.int64)

    for i in prange(n - 1):
        result_long, exit_long, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distances[i], sl_distances[i], spread, slippage, max_bars, timeout_bars
        )
        if result_long == 1.0:
            targets_long[i] = 1.0
        durations_long[i] = (exit_long - i) if exit_long >= 0 else max_bars

        result_short, exit_short, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distances[i], sl_distances[i], spread, slippage, max_bars, timeout_bars
        )
        if result_short == 1.0:
            targets_short[i] = 1.0
        durations_short[i] = (exit_short - i) if exit_short >= 0 else max_bars

    return targets_long, targets_short, durations_long, durations_short
