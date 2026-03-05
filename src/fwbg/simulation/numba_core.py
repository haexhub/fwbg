"""
Numba-optimierte Kernfunktionen für Trade-Simulation.

Cache versioning: bump _CACHE_VERSION whenever the return signature of any
@njit function changes.  At import time we compare against a stamp file;
on mismatch we wipe every .nbi/.nbc we can find and rewrite the stamp.
"""
import pathlib
import numpy as np
from numba import njit, prange

# Bump this whenever a @njit function signature/return type changes.
_CACHE_VERSION = "6"

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

    # Entry-Preis
    entry_price = opens[entry_idx]

    # TP/SL-Levels berechnen
    # WICHTIG: Slippage wirkt IMMER gegen den Trader:
    # - Entry-Slippage: schlechterer Einstieg
    # - Exit-Slippage: wird bei PnL-Berechnung berücksichtigt, nicht bei Level
    # Die TP/SL-Levels sind die TRIGGER-Levels, der tatsächliche Exit-Preis
    # wäre schlechter durch Slippage (bereits in entry eingerechnet für Netto-PnL)
    if direction == 1:  # Long
        entry = entry_price + spread + slippage  # Kaufe teurer
        tp = entry + tp_distance  # TP-Level (Trigger)
        sl = entry - sl_distance  # SL-Level (Trigger)
    else:  # Short
        entry = entry_price - spread - slippage  # Verkaufe billiger
        tp = entry - tp_distance  # TP-Level (Trigger)
        sl = entry + sl_distance  # SL-Level (Trigger)

    # Maximale Simulation-Länge
    end_idx = min(entry_idx + max_bars, n)

    # Timeout-Index berechnen (falls aktiviert)
    # WICHTIG: Timeout wird INNERHALB des Loops geprüft, nicht erst danach!
    timeout_idx = -1
    if timeout_bars > 0:
        timeout_idx = min(entry_idx + timeout_bars - 1, n - 1)

    # Simulation-Loop
    for j in range(entry_idx, end_idx):
        # Timeout-Check ZUERST (Timeout hat Priorität über TP/SL nach timeout_bars)
        if timeout_idx > 0 and j >= timeout_idx:
            exit_price = closes[j]
            # PnL berechnen (Exit-Slippage bereits in entry eingerechnet)
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            result = 1.0 if pnl > 0 else -1.0
            return result, j, exit_price, 2  # exit_reason=2 (Timeout)

        if direction == 1:  # Long
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:  # Short
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

        if tp_hit and sl_hit:
            # Beide im selben Bar - konservativ: Loss
            return -1.0, j, sl, 1

        if tp_hit:
            return 1.0, j, tp, 0

        if sl_hit:
            return -1.0, j, sl, 1

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

    entry_price = opens[entry_idx]

    if direction == 1:
        entry = entry_price + spread + slippage
        tp = entry + tp_distance
        sl = entry - sl_distance
    else:
        entry = entry_price - spread - slippage
        tp = entry - tp_distance
        sl = entry + sl_distance

    end_idx = min(entry_idx + max_bars, n)
    session_bars_elapsed = 0

    for j in range(entry_idx, end_idx):
        if not in_session[j]:
            continue

        session_bars_elapsed += 1

        # Timeout check (counts only session bars)
        if timeout_bars > 0 and session_bars_elapsed >= timeout_bars:
            exit_price = closes[j]
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            result = 1.0 if pnl > 0 else -1.0
            return result, j, exit_price, 2

        if direction == 1:
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

        if tp_hit and sl_hit:
            return -1.0, j, sl, 1
        if tp_hit:
            return 1.0, j, tp, 0
        if sl_hit:
            return -1.0, j, sl, 1

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

    entry_price = opens[entry_idx]

    if direction == 1:  # Long
        entry = entry_price + spread + slippage
        tp = entry + tp_distance
        sl = entry - sl_distance
        be_trigger_price = entry + tp_distance * breakeven_trigger
    else:  # Short
        entry = entry_price - spread - slippage
        tp = entry - tp_distance
        sl = entry + sl_distance
        be_trigger_price = entry - tp_distance * breakeven_trigger

    end_idx = min(entry_idx + max_bars, n)

    timeout_idx = -1
    if timeout_bars > 0:
        timeout_idx = min(entry_idx + timeout_bars - 1, n - 1)

    best_price = entry
    trailing_active = breakeven_trigger <= 0.0

    for j in range(entry_idx, end_idx):
        if timeout_idx > 0 and j >= timeout_idx:
            exit_price = closes[j]
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            return (1.0 if pnl > 0 else -1.0), j, exit_price, 2

        if direction == 1:
            if highs[j] > best_price:
                best_price = highs[j]
        else:
            if lows[j] < best_price:
                best_price = lows[j]

        if not trailing_active and breakeven_trigger > 0.0:
            if direction == 1 and best_price >= be_trigger_price:
                trailing_active = True
                be_sl = entry + breakeven_offset * tp_distance
                if be_sl > sl:
                    sl = be_sl
            elif direction == -1 and best_price <= be_trigger_price:
                trailing_active = True
                be_sl = entry - breakeven_offset * tp_distance
                if be_sl < sl:
                    sl = be_sl

        if trailing_active and trail_distance > 0.0:
            if direction == 1:
                new_sl = best_price - trail_distance
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = best_price + trail_distance
                if new_sl < sl:
                    sl = new_sl

        if trailing_active and trail_tp_dist > 0.0:
            if direction == 1:
                new_tp = best_price + trail_tp_dist
                if new_tp > tp:
                    tp = new_tp
            else:
                new_tp = best_price - trail_tp_dist
                if new_tp < tp:
                    tp = new_tp

        if direction == 1:
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

        if sl_hit:
            if direction == 1:
                result = 1.0 if sl > entry else -1.0
            else:
                result = 1.0 if sl < entry else -1.0
            return result, j, sl, 1

        if tp_hit:
            return 1.0, j, tp, 0

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

    entry_price = opens[entry_idx]

    if direction == 1:  # Long
        entry = entry_price + spread + slippage
        tp = entry + tp_distance
        sl = entry - sl_distance
    else:  # Short
        entry = entry_price - spread - slippage
        tp = entry - tp_distance
        sl = entry + sl_distance

    # --- Scale-in preparation ---
    total_qty = 1.0
    weighted_sum = entry * 1.0
    avg_price = entry
    n_fills = 1

    # Precompute scale-in trigger prices
    scale_price = np.empty(n_levels, dtype=np.float64)
    levels_filled = np.zeros(n_levels, dtype=np.int8)
    for k in range(n_levels):
        if direction == 1:
            scale_price[k] = entry - scale_levels[k] * sl_distance
        else:
            scale_price[k] = entry + scale_levels[k] * sl_distance

    # --- Trailing / breakeven state ---
    end_idx = min(entry_idx + max_bars, n)

    timeout_idx = -1
    if timeout_bars > 0:
        timeout_idx = min(entry_idx + timeout_bars - 1, n - 1)

    best_price = entry
    trailing_active = breakeven_trigger <= 0.0

    # Breakeven trigger computed from avg_price
    if direction == 1:
        be_trigger_price = avg_price + tp_distance * breakeven_trigger
    else:
        be_trigger_price = avg_price - tp_distance * breakeven_trigger

    for j in range(entry_idx, end_idx):
        # --- Timeout check ---
        if timeout_idx > 0 and j >= timeout_idx:
            exit_price = closes[j]
            if direction == 1:
                pnl = (exit_price - avg_price) * total_qty
            else:
                pnl = (avg_price - exit_price) * total_qty
            result = 1.0 if pnl > 0 else -1.0
            return result, j, exit_price, 2, avg_price, total_qty, n_fills

        # --- Scale-in trigger check ---
        for k in range(n_levels):
            if levels_filled[k] == 1:
                continue
            if direction == 1:
                if lows[j] <= scale_price[k] and scale_price[k] > sl:
                    fill_price = scale_price[k]
                    weighted_sum += fill_price * scale_qty_mult
                    total_qty += scale_qty_mult
                    avg_price = weighted_sum / total_qty
                    tp = avg_price + tp_distance
                    levels_filled[k] = 1
                    n_fills += 1
                    # Recalculate breakeven trigger from new avg_price
                    if breakeven_trigger > 0.0:
                        be_trigger_price = avg_price + tp_distance * breakeven_trigger
            else:
                if highs[j] >= scale_price[k] and scale_price[k] < sl:
                    fill_price = scale_price[k]
                    weighted_sum += fill_price * scale_qty_mult
                    total_qty += scale_qty_mult
                    avg_price = weighted_sum / total_qty
                    tp = avg_price - tp_distance
                    levels_filled[k] = 1
                    n_fills += 1
                    # Recalculate breakeven trigger from new avg_price
                    if breakeven_trigger > 0.0:
                        be_trigger_price = avg_price - tp_distance * breakeven_trigger

        # --- Update best price ---
        if direction == 1:
            if highs[j] > best_price:
                best_price = highs[j]
        else:
            if lows[j] < best_price:
                best_price = lows[j]

        # --- Breakeven activation ---
        if not trailing_active and breakeven_trigger > 0.0:
            if direction == 1 and best_price >= be_trigger_price:
                trailing_active = True
                if avg_price > sl:
                    sl = avg_price
            elif direction == -1 and best_price <= be_trigger_price:
                trailing_active = True
                if avg_price < sl:
                    sl = avg_price

        # --- Trailing stop ---
        if trailing_active and trail_distance > 0.0:
            if direction == 1:
                new_sl = best_price - trail_distance
                if new_sl > sl:
                    sl = new_sl
            else:
                new_sl = best_price + trail_distance
                if new_sl < sl:
                    sl = new_sl

        # --- Trailing TP ---
        if trailing_active and trail_tp_dist > 0.0:
            if direction == 1:
                new_tp = best_price + trail_tp_dist
                if new_tp > tp:
                    tp = new_tp
            else:
                new_tp = best_price - trail_tp_dist
                if new_tp < tp:
                    tp = new_tp

        # --- TP / SL hit check ---
        if direction == 1:
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl
        else:
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

        if sl_hit:
            if direction == 1:
                pnl = (sl - avg_price) * total_qty
            else:
                pnl = (avg_price - sl) * total_qty
            result = 1.0 if pnl > 0 else -1.0
            return result, j, sl, 1, avg_price, total_qty, n_fills

        if tp_hit:
            return 1.0, j, tp, 0, avg_price, total_qty, n_fills

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
