"""MFE/MAE (Maximum Favorable/Adverse Excursion) computation via Numba.

Computes for every bar how far price moves in favor (MFE) and against (MAE)
within a forward-looking window. Used to determine optimal TP/SL ranges.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def compute_mfe_mae(open_, high, low, max_bars):
    """Compute MFE/MAE for every bar looking forward max_bars.

    Entry = next bar's open (no look-ahead bias).
    Returns 4 arrays of length n: mfe_long, mae_long, mfe_short, mae_short.
    Bars where forward window is insufficient are NaN.
    """
    n = len(open_)
    mfe_long = np.full(n, np.nan)
    mae_long = np.full(n, np.nan)
    mfe_short = np.full(n, np.nan)
    mae_short = np.full(n, np.nan)

    for i in prange(n - 1):
        entry = open_[i + 1]
        end = min(i + 1 + max_bars, n)
        if end <= i + 1:
            continue

        best_high = high[i + 1]
        worst_low = low[i + 1]
        for j in range(i + 2, end):
            if high[j] > best_high:
                best_high = high[j]
            if low[j] < worst_low:
                worst_low = low[j]

        mfe_long[i] = best_high - entry
        mae_long[i] = entry - worst_low
        mfe_short[i] = entry - worst_low
        mae_short[i] = best_high - entry

    return mfe_long, mae_long, mfe_short, mae_short


@njit(cache=True, parallel=True)
def compute_capture_rates(open_, high, low, atr, tp_values, sl_values, max_bars):
    """Compute win rates for each TP/SL combination (in ATR multiples).

    For each bar, iterates forward to determine if TP or SL is hit first.
    Simultaneous hit in same bar = loss (conservative, matches production sim).

    Returns:
        wr_long: (n_tp, n_sl) win rate for long trades
        wr_short: (n_tp, n_sl) win rate for short trades
        trade_counts: (n_tp, n_sl) number of resolved trades (not timed out)
    """
    n = len(open_)
    n_tp = len(tp_values)
    n_sl = len(sl_values)

    wr_long = np.zeros((n_tp, n_sl))
    wr_short = np.zeros((n_tp, n_sl))
    trade_counts = np.zeros((n_tp, n_sl))

    for ti in prange(n_tp):
        for si in range(n_sl):
            wins_l = 0
            wins_s = 0
            resolved_l = 0
            resolved_s = 0

            for i in range(n - 1):
                entry = open_[i + 1]
                if entry <= 0.0 or atr[i] <= 0.0:
                    continue

                tp_dist = atr[i] * tp_values[ti]
                sl_dist = atr[i] * sl_values[si]
                end = min(i + 1 + max_bars, n)

                # Track long and short independently
                hit_l = 0  # 0=unresolved, 1=tp, -1=sl
                hit_s = 0

                for j in range(i + 1, end):
                    # --- Long ---
                    if hit_l == 0:
                        tp_hit = high[j] >= entry + tp_dist
                        sl_hit = low[j] <= entry - sl_dist
                        if tp_hit and sl_hit:
                            hit_l = -1  # Conservative: loss
                        elif tp_hit:
                            hit_l = 1
                        elif sl_hit:
                            hit_l = -1

                    # --- Short ---
                    if hit_s == 0:
                        tp_hit = low[j] <= entry - tp_dist
                        sl_hit = high[j] >= entry + sl_dist
                        if tp_hit and sl_hit:
                            hit_s = -1
                        elif tp_hit:
                            hit_s = 1
                        elif sl_hit:
                            hit_s = -1

                    if hit_l != 0 and hit_s != 0:
                        break

                if hit_l != 0:
                    resolved_l += 1
                    if hit_l == 1:
                        wins_l += 1
                if hit_s != 0:
                    resolved_s += 1
                    if hit_s == 1:
                        wins_s += 1

            total = max(resolved_l, resolved_s)
            if resolved_l > 0:
                wr_long[ti, si] = wins_l / resolved_l
            if resolved_s > 0:
                wr_short[ti, si] = wins_s / resolved_s
            trade_counts[ti, si] = total

    return wr_long, wr_short, trade_counts
