"""Fixed, long-only monthly allocation rules, using completed months only."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .engine import PortfolioSettings

STRATEGIES = {
    "sma",
    "absolute_momentum",
    "relative_momentum",
    "dual_momentum",
    "equity_dual_momentum",
    "reversal",
    "inverse_volatility",
}


def allocation(
    month_ends: np.ndarray, base: np.ndarray, assets: list[str], settings: "PortfolioSettings"
) -> tuple[np.ndarray, np.ndarray]:
    """Return risky target weights and the per-asset decision statistic.

    Momentum uses L+1 closes for L returns. SMA uses exactly L closes.
    A rejected dual-momentum slot remains cash; it is never redistributed.
    Ties are resolved by asset name, independently of input column order.
    """
    strategy = settings.strategy
    if strategy == "sma":
        score = month_ends.mean(axis=0)
        return base * (month_ends[-1] > score), score
    score = month_ends[-1] / month_ends[0] - 1
    if strategy == "absolute_momentum":
        return base * (score > 0), score
    target = np.zeros(len(assets))
    if strategy == "inverse_volatility":
        score = (month_ends[1:] / month_ends[:-1] - 1).std(axis=0, ddof=1)
        usable = np.isfinite(score) & (score > 1e-12)
        target[usable] = 1 / score[usable]
        if target.sum() > 0:
            target /= target.sum()
    elif strategy == "equity_dual_momentum":
        equities = [assets.index(asset) for asset in settings.equity_assets]
        best = sorted(equities, key=lambda j: (-score[j], assets[j]))[0]
        assert settings.defensive_asset is not None  # Validated by PortfolioSettings.
        defensive = assets.index(settings.defensive_asset)
        if score[best] > 0:
            target[best] = 1.0
        elif score[defensive] > 0:
            target[defensive] = 1.0
    else:
        sign = 1 if strategy == "reversal" else -1
        ranked = sorted(range(len(assets)), key=lambda j: (sign * score[j], assets[j]))
        for j in ranked[: settings.selection_count]:
            if strategy != "dual_momentum" or score[j] > 0:
                target[j] = 1 / settings.selection_count
    return target, score
