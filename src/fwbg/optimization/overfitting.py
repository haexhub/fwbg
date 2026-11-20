"""Deflated Sharpe Ratio (DSR) and Probability of Backtest Overfitting (PBO).

References:
- DSR: Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"
- PBO: Bailey, Borwein, López de Prado, Zhu (2017),
       "The Probability of Backtest Overfitting"
"""

import logging
import warnings
from itertools import combinations
from math import comb

import numpy as np

from fwbg.research import dsr as _dsr

logger = logging.getLogger(__name__)


# === Deflated Sharpe Ratio ===


def expected_max_sr(n_strategies: int, sr_variance: float = 1.0) -> float:
    """Expected maximum Sharpe ratio of *n_strategies* independent zero-edge trials.

    Bailey & Lopez de Prado (2014), the paper's SR_0:
        E[max SR] = sqrt(V[SR]) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N*e)))

    where g is the Euler-Mascheroni constant and V[SR] the variance of the
    Sharpe-ratio estimates ACROSS the trials. The result scales with the
    standard deviation of the trials' SRs, so it is in the unit of the SR it is
    compared with (per trade here). The default variance of 1.0 returns the
    value in units of one trial standard deviation.
    """
    return _dsr.expected_max_sharpe(n_strategies, sr_variance)


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
    variance = (
        1.0 + 0.5 * sr2 - skewness * observed_sr + (excess_kurtosis / 4.0) * sr2
    ) / (n_trades - 1)
    return float(np.sqrt(max(variance, 0.0)))


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trades: int,
    n_strategies: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance: float | None = None,
) -> dict:
    """Compute the Deflated Sharpe Ratio (delegates to ``fwbg.research.dsr``).

    DSR = PSR(SR_obs; SR_0) with SR_0 = E[max SR] of *n_strategies* zero-edge
    trials whose SR estimates have variance *sr_variance*.

    *observed_sr* is the per-trade (non-annualised) Sharpe ratio, *kurtosis* the
    plain fourth standardised moment (3 for normal returns). When *sr_variance*
    (the variance of the trials' SRs) is not known it defaults to 1/(T-1), the
    sampling variance of a per-trade SR estimate with no edge, i.e. the trials
    are assumed to be independent zero-edge draws. The real dispersion across
    trials is usually larger, so this default penalises the search no more
    than the data justify; pass the measured variance when it is available.

    Returns dict with: dsr (None when the SR's variance is undefined), observed_sr,
    expected_max_sr, sr_variance, n_strategies, is_significant.
    """
    if sr_variance is None:
        sr_variance = 1.0 / (n_trades - 1) if n_trades > 1 else 0.0
    e_max = _dsr.expected_max_sharpe(n_strategies, sr_variance)
    dsr_value = _dsr.probabilistic_sharpe_ratio(
        observed_sr, e_max, n_trades, skewness, kurtosis
    )

    return {
        "dsr": dsr_value,
        "observed_sr": observed_sr,
        "expected_max_sr": e_max,
        "sr_variance": sr_variance,
        "n_strategies": n_strategies,
        "is_significant": dsr_value is not None and dsr_value > 0.95,
    }


# === Probability of Backtest Overfitting ===


# The grid's choices that identify a combo. "ct" is a grid dimension of signal models;
# an ML entry's "conf_thresh" is chosen inside its inner CV and would split one combo
# into a row per fold, so it is not part of the key.
_COMBO_FIELDS = (
    "exit_strategy",
    "timeout_bars",
    "ct",
    "exit_params",
    "exit_modifier_params",
    "entry_modifier",
    "entry_modifier_params",
    "model_hyperparameters",
    "regime_filter",
)


def _frozen(value):
    """A hashable stand-in for a grid value (dicts and lists become sorted tuples)."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _frozen(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    return value


def _combo_key(grid_result: dict) -> tuple:
    """The full combo of a grid entry (KeyError without tp_mult and sl_mult)."""
    tp_sl = (grid_result["tp_mult"], grid_result["sl_mult"])
    return tp_sl + tuple(_frozen(grid_result.get(field)) for field in _COMBO_FIELDS)


def count_trials(grid_results: list[dict], n_variants: int = 1) -> int:
    """Number of distinct configurations that were evaluated (the DSR's N).

    Counts distinct combos (TP, SL and every _COMBO_FIELDS value) across all
    folds, not one per (fold, combo) grid row: the same combo re-evaluated in
    every fold is one trial. *n_variants* multiplies the count: each indicator
    grid variant evaluated its own combos and the best one was picked. An entry
    that cannot be keyed counts as its own trial. Never below 1.
    """
    keys: set[tuple] = set()
    unkeyed = 0
    for grid_result in grid_results:
        try:
            key = _combo_key(grid_result)
            hash(key)
        except (KeyError, TypeError):
            unkeyed += 1
            continue
        keys.add(key)
    return max(1, (len(keys) + unkeyed) * max(1, n_variants))


def build_performance_matrix(
    grid_results_by_fold: dict,
) -> tuple[np.ndarray, list[tuple]]:
    """Build performance matrix M(n_combos, n_folds) from grid results.

    Each row is a strategy combo (TP, SL and every _COMBO_FIELDS value, so combos that
    differ only by CT, exit, HP variant or regime filter keep their own rows),
    each column is a fold. Only combos present in ALL folds are included.

    Returns (matrix, combo_keys) where combo_keys is the list of combo tuples.
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
            # A signal combo without inner trades is ineligible (never chosen): no candidate.
            # ML entries carry no inner_trades.
            if gr.get("inner_trades") == 0:
                continue
            try:
                key = _combo_key(gr)
                hash(key)  # an unhashable grid value (e.g. a set) cannot key a row
                val = gr["inner_val_pnl"]
            except (KeyError, TypeError):
                continue
            if val is None or not np.isfinite(val):
                continue
            combo_map[key] = val
        fold_combo_maps.append(combo_map)

    # Find combos present in ALL folds
    common_keys = set(fold_combo_maps[0].keys())
    for cm in fold_combo_maps[1:]:
        common_keys &= set(cm.keys())

    if not common_keys:
        return np.empty((0, n_folds)), []

    # repr: a total order even where a field mixes None and numbers (timeout_bars).
    combo_keys = sorted(common_keys, key=repr)
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

    Returns dict with: pbo, n_cscv_splits, is_overfit, degradation, logit_mean.
    PBO is undefined (pbo None, with a "reason") without at least 2 combos: with a
    single combo there is nothing to select from, so no selection can overfit
    (0.0 would read as "not overfit"). It is also undefined below 4 folds.
    """
    n_combos, n_folds = performance_matrix.shape

    undefined = {
        "pbo": None,
        "n_cscv_splits": 0,
        "is_overfit": None,
        "degradation": None,
        "logit_mean": None,
    }
    if n_combos == 1:
        message = "PBO undefined: only one combo in the performance matrix"
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return {**undefined, "reason": message}
    if n_combos == 0 or n_folds < 4:
        return undefined

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
    sr_variance: float | None = None,
) -> dict:
    """Compute both DSR and PBO metrics.

    Args:
        trade_returns: List of per-trade returns (for skewness/kurtosis).
        observed_sr: Non-annualized Sharpe ratio of the strategy.
        n_strategies: Number of distinct configurations evaluated (see
            ``count_trials``).
        sr_variance: Variance of the trials' SRs, when known (see
            ``deflated_sharpe_ratio`` for the default).
        grid_results_by_fold: {fold_id: [grid_result_dicts]} for PBO.
        n_trades: Number of trades.

    Returns dict with 'dsr' and 'pbo' sub-dicts.
    """
    # Compute skewness and kurtosis from trade returns
    returns_arr = np.array(trade_returns, dtype=float)
    if len(returns_arr) > 2:
        skewness = float(
            np.mean(
                ((returns_arr - returns_arr.mean()) / max(returns_arr.std(), 1e-10))
                ** 3
            )
        )
        kurtosis = float(
            np.mean(
                ((returns_arr - returns_arr.mean()) / max(returns_arr.std(), 1e-10))
                ** 4
            )
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
        sr_variance=sr_variance,
    )

    # PBO on its own: a failure here must not drop the DSR.
    try:
        matrix, combo_keys = build_performance_matrix(grid_results_by_fold)
        pbo_result = probability_of_backtest_overfitting(matrix)
        # Survivor coverage: only combos with a finite inner score in EVERY fold enter
        # the matrix (successive halving prunes weak combos in later folds).
        all_results = [gr for rows in grid_results_by_fold.values() for gr in rows]
        pbo_result["n_combos_in_matrix"] = len(combo_keys)
        pbo_result["n_combos_evaluated"] = (
            count_trials(all_results) if all_results else 0
        )
    except Exception as exc:
        pbo_result = {
            **probability_of_backtest_overfitting(np.empty((0, 0))),
            "error": str(exc),
        }

    return {"dsr": dsr_result, "pbo": pbo_result}
