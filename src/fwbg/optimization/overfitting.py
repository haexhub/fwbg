"""Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO).

References:
- DSR: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
- PBO: Bailey, Borwein, López de Prado, Zhu (2017),
       "The Probability of Backtest Overfitting"
"""

from itertools import combinations
from math import comb, log

import numpy as np
from scipy.stats import norm


# === Deflated Sharpe Ratio ===


def expected_max_sr(n_strategies: int) -> float:
    """Expected maximum Sharpe ratio under null hypothesis (all SR=0).

    Uses the approximation from Bailey & López de Prado (2014):
        E[max(SR)] ≈ √(2 * ln(N)) - (ln(π) + ln(ln(N))) / (2 * √(2 * ln(N)))

    where N is the number of independent strategies tested.
    """
    if n_strategies <= 1:
        return 0.0

    ln_n = log(max(n_strategies, 2))
    sqrt_term = np.sqrt(2.0 * ln_n)

    if sqrt_term == 0:
        return 0.0

    euler_mascheroni = 0.5772156649
    return float(
        sqrt_term
        - (log(np.pi) + log(ln_n)) / (2.0 * sqrt_term)
        + euler_mascheroni / sqrt_term
    )


def sr_std(
    observed_sr: float,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Standard deviation of the Sharpe ratio estimator.

    From Lo (2002) and Bailey & López de Prado (2014):
        Var(SR) ≈ (1 + 0.5*SR² - γ₃*SR + (γ₄/4)*SR²) / (T-1)

    where γ₃ = skewness, γ₄ = excess kurtosis.
    """
    if n_trades <= 1:
        return np.inf

    excess_kurtosis = kurtosis - 3.0
    sr2 = observed_sr**2
    variance = (1.0 + 0.5 * sr2 - skewness * observed_sr + (excess_kurtosis / 4.0) * sr2) / (
        n_trades - 1
    )
    return float(np.sqrt(max(variance, 0.0)))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trades: int,
    n_strategies: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """Compute the Deflated Sharpe Ratio.

    DSR = Φ((SR_obs - E[max(SR)]) / σ(SR))

    Returns dict with: dsr, observed_sr, expected_max_sr, n_strategies, is_significant
    """
    e_max = expected_max_sr(n_strategies)
    sigma = sr_std(observed_sr, n_trades, skewness, kurtosis)

    if sigma <= 0 or not np.isfinite(sigma):
        dsr_value = 0.0
    else:
        z = (observed_sr - e_max) / sigma
        dsr_value = float(norm.cdf(z))

    return {
        "dsr": dsr_value,
        "observed_sr": observed_sr,
        "expected_max_sr": e_max,
        "n_strategies": n_strategies,
        "is_significant": dsr_value > 0.95,
    }


# === Probability of Backtest Overfitting ===


def build_performance_matrix(
    grid_results_by_fold: dict,
) -> tuple[np.ndarray, list[tuple]]:
    """Build performance matrix M(n_combos, n_folds) from grid results.

    Each row is a strategy combo (tp_mult, sl_mult, timeout_bars),
    each column is a fold. Only combos present in ALL folds are included.

    Returns (matrix, combo_keys) where combo_keys is list of (tp, sl, timeout) tuples.
    """
    if not grid_results_by_fold:
        return np.empty((0, 0)), []

    fold_ids = sorted(grid_results_by_fold.keys())
    n_folds = len(fold_ids)

    # Index results by combo key per fold
    fold_combo_maps: list[dict[tuple, float]] = []
    for fid in fold_ids:
        combo_map = {}
        for gr in grid_results_by_fold[fid]:
            key = (gr["tp_mult"], gr["sl_mult"], gr.get("timeout_bars", 0))
            combo_map[key] = gr["inner_val_pnl"]
        fold_combo_maps.append(combo_map)

    # Find combos present in ALL folds
    common_keys = set(fold_combo_maps[0].keys())
    for cm in fold_combo_maps[1:]:
        common_keys &= set(cm.keys())

    if not common_keys:
        return np.empty((0, n_folds)), []

    combo_keys = sorted(common_keys)
    matrix = np.zeros((len(combo_keys), n_folds))
    for i, key in enumerate(combo_keys):
        for j, cm in enumerate(fold_combo_maps):
            matrix[i, j] = cm[key]

    return matrix, combo_keys


def probability_of_backtest_overfitting(
    performance_matrix: np.ndarray,
) -> dict:
    """Compute PBO using Combinatorial Symmetric Cross-Validation (CSCV).

    Splits S folds into C(S, S/2) pairs of IS/OOS halves.
    For each split, finds best IS combo and checks its OOS rank.
    PBO = fraction of splits where best IS combo underperforms OOS median.

    Returns dict with: pbo, n_cscv_splits, is_overfit, degradation, logit_mean
    """
    n_combos, n_folds = performance_matrix.shape

    if n_combos == 0 or n_folds < 4:
        return {
            "pbo": None,
            "n_cscv_splits": 0,
            "is_overfit": None,
            "degradation": None,
            "logit_mean": None,
        }

    half = n_folds // 2
    n_splits = comb(n_folds, half)
    fold_indices = list(range(n_folds))

    logits = []
    n_underperform = 0

    for is_folds in combinations(fold_indices, half):
        oos_folds = [f for f in fold_indices if f not in is_folds]

        # IS and OOS performance per combo (mean across respective folds)
        is_perf = performance_matrix[:, list(is_folds)].mean(axis=1)
        oos_perf = performance_matrix[:, oos_folds].mean(axis=1)

        # Best IS combo
        best_is_idx = np.argmax(is_perf)
        best_oos_val = oos_perf[best_is_idx]

        # Rank of best IS combo in OOS (relative rank 0..1)
        oos_rank = float(np.mean(oos_perf <= best_oos_val))

        if oos_rank <= 0.5:
            n_underperform += 1

        # Logit of the relative rank (clamp to avoid log(0))
        clamped = np.clip(oos_rank, 1e-6, 1.0 - 1e-6)
        logits.append(float(np.log(clamped / (1.0 - clamped))))

    pbo = n_underperform / n_splits
    logit_mean = float(np.mean(logits)) if logits else 0.0

    # Degradation: mean OOS rank of best IS combo (1.0 = always best, 0 = always worst)
    # Values < 0.5 indicate systematic overfitting
    mean_rank = 1.0 - pbo  # Simplified: fraction of times best IS is above OOS median

    return {
        "pbo": float(pbo),
        "n_cscv_splits": n_splits,
        "is_overfit": pbo > 0.50,
        "degradation": float(mean_rank),
        "logit_mean": logit_mean,
    }


# === Entry Point ===


def compute_overfitting_metrics(
    trade_returns: list[float],
    observed_sr: float,
    n_strategies: int,
    grid_results_by_fold: dict,
    n_trades: int,
) -> dict:
    """Compute both DSR and PBO metrics.

    Args:
        trade_returns: List of per-trade returns (for skewness/kurtosis).
        observed_sr: Non-annualized Sharpe ratio of the strategy.
        n_strategies: Total number of grid combinations tested.
        grid_results_by_fold: {fold_id: [grid_result_dicts]} for PBO.
        n_trades: Number of trades.

    Returns dict with 'dsr' and 'pbo' sub-dicts.
    """
    # Compute skewness and kurtosis from trade returns
    returns_arr = np.array(trade_returns, dtype=float)
    if len(returns_arr) > 2:
        skewness = float(
            np.mean(((returns_arr - returns_arr.mean()) / max(returns_arr.std(), 1e-10)) ** 3)
        )
        kurtosis = float(
            np.mean(((returns_arr - returns_arr.mean()) / max(returns_arr.std(), 1e-10)) ** 4)
        )
    else:
        skewness = 0.0
        kurtosis = 3.0

    dsr_result = deflated_sharpe_ratio(
        observed_sr=observed_sr,
        n_trades=n_trades,
        n_strategies=n_strategies,
        skewness=skewness,
        kurtosis=kurtosis,
    )

    # PBO
    matrix, _ = build_performance_matrix(grid_results_by_fold)
    pbo_result = probability_of_backtest_overfitting(matrix)

    return {"dsr": dsr_result, "pbo": pbo_result}
