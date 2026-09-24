import pandas as pd

from fwbg.core.context import SimulationContext
from fwbg.core.grid_params import GridParams
from fwbg.core.registry import get_exit_strategy
from fwbg.data.assets import get_asset


def compute_labels(
    df: pd.DataFrame, symbol: str, tp: float, sl: float, timeout_bars: int
) -> pd.DataFrame:
    """Triple-barrier long/short labels via FixedExitStrategy.

    tp/sl are spread multipliers (e.g. tp=20 means 20x the asset's spread),
    not raw price distances -- matching FixedExitStrategy's convention.
    """
    asset = get_asset(symbol)
    strategy = get_exit_strategy("fixed")()
    ctx = SimulationContext(
        symbol=asset.symbol,
        asset_class=asset.asset_class,
        spread=asset.spread,
        point=asset.point,
        max_trade_bars=timeout_bars,
    )
    params = GridParams(tp_value=tp, sl_value=sl, timeout_bars=timeout_bars)

    targets_long, targets_short = strategy.compute_targets(df, ctx, params=params)
    return pd.DataFrame(
        {"targets_long": targets_long, "targets_short": targets_short}, index=df.index
    )
