import pandas as pd

from fwbg.core.registry import get_exit_strategy
from fwbg.data.assets import get_asset


class _MinimalContext:
    """Just enough of SimulationContext for FixedExitStrategy.compute_targets()."""

    def __init__(self, spread: float, max_trade_bars: int):
        self.spread = spread
        self.max_trade_bars = max_trade_bars
        self.entry_modifier = None


def compute_labels(
    df: pd.DataFrame, symbol: str, tp: float, sl: float, timeout_bars: int
) -> pd.DataFrame:
    asset = get_asset(symbol)
    strategy = get_exit_strategy("fixed")()
    ctx = _MinimalContext(spread=asset.spread, max_trade_bars=timeout_bars)
    _timeout_bars = timeout_bars

    class _Params:
        tp_value = tp
        sl_value = sl
        timeout_bars = _timeout_bars

    targets_long, targets_short = strategy.compute_targets(df, ctx, params=_Params())
    return pd.DataFrame(
        {"targets_long": targets_long, "targets_short": targets_short}, index=df.index
    )
