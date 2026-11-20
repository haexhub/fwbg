"""Long and short signalling on the same bar is no trade, not a long."""

import numpy as np
import pandas as pd

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import simulate_trades


def _frame(long_sig, short_sig):
    n = len(long_sig)
    idx = pd.date_range("2024-01-02 06:00", periods=n, freq="h")
    return pd.DataFrame(
        {
            "O": 100.0,
            "H": 100.5,
            "L": 99.5,
            "C": 100.0,
            "_composed_signal_long": long_sig,
            "_composed_signal_short": short_sig,
        },
        index=idx,
    )


def _run(df, **kw):
    ctx = SimulationContext(
        symbol="T",
        asset_class="FOREX",
        spread=0.0,
        point=0.0001,
        long_enabled=True,
        short_enabled=True,
        min_trades=1,
    )
    probs = np.zeros((len(df), 2))
    probs[:, 1] = 1.0
    res = simulate_trades(
        df, probs, probs, 1, 1, 0.5, 0.5, 1, 1, ctx, return_detailed=True,
        timeout_bars=1, **kw,
    )  # fmt: skip
    return [(t["signal_idx"], t["direction"]) for t in res["trades_detailed"]]


def test_tie_bar_takes_no_trade():
    df = _frame([1, 1, 0, 1, 0], [0, 1, 0, 0, 0])
    # bar 1 signals both ways: skipped; bar 0 long and bar 3 long still trade
    assert _run(df) == [(0, "LONG"), (3, "LONG")]


def test_single_sided_bars_keep_their_direction():
    df = _frame([1, 0, 0, 0, 0], [0, 0, 1, 0, 0])
    assert _run(df) == [(0, "LONG"), (2, "SHORT")]


def test_direction_filter_run_still_trades_the_tie_bar():
    df = _frame([0, 1, 0, 0], [0, 1, 0, 0])
    assert _run(df, direction_filter=1) == [(1, "LONG")]
    assert _run(df, direction_filter=-1) == [(1, "SHORT")]
