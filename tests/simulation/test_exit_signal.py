"""Rule exits: a strategy column closes the open trade at the bar's close."""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.columns import ColumnRoleError
from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import simulate_trades
from fwbg.simulation.trade import simulate_pro_trade

FAR = 1_000.0


def _trade(closes, exit_signal, direction=1, entry_delay=0, sl_distance=FAR):
    closes = np.asarray(closes, dtype=float)
    return simulate_pro_trade(
        closes=closes, highs=closes + 0.5, lows=closes - 0.5, opens=closes,
        idx=0, direction=direction, tp_distance=FAR, sl_distance=sl_distance,
        spread=0.0, entry_delay=entry_delay, exit_signal=np.asarray(exit_signal, dtype=bool),
    )


def test_flag_closes_at_that_bars_close():
    trade = _trade([100, 101, 103, 102, 99], [0, 0, 1, 0, 0])
    assert trade["exit_idx"] == 2
    assert trade["exit_reason"] == "exit_signal"
    assert trade["exit_price"] == pytest.approx(103.0)
    assert trade["result"] == 1.0


def test_flag_on_the_entry_bar_of_a_close_entry_is_ignored():
    trade = _trade([100, 99, 98, 97], [1, 0, 1, 0])
    assert trade["exit_idx"] == 2
    assert trade["result"] < 0


def test_stop_hit_on_the_flag_bar_wins_over_the_flag():
    trade = _trade([100, 100, 90, 95], [0, 0, 1, 0], sl_distance=5.0)
    assert trade["exit_idx"] == 2
    assert trade.get("exit_reason") != "exit_signal"
    assert trade["exit_price"] == pytest.approx(90.0)  # the bar gapped through the stop at 95


def test_without_flags_the_trade_runs_to_the_end_of_data():
    trade = _trade([100, 101, 102], [0, 0, 0])
    assert trade["exit_reason"] == "end_of_data"


def _frame(closes, long_sig, short_sig, exit_long, exit_short):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "O": closes, "H": closes + 0.5, "L": closes - 0.5, "C": closes,
            "_composed_signal_long": long_sig, "_composed_signal_short": short_sig,
            "x_exit_long": exit_long, "x_exit_short": exit_short,
        },
        index=pd.date_range("2024-01-02 15:00", periods=len(closes), freq="15min"),
    )


def _ctx(entry_delay):
    return SimulationContext(
        symbol="TEST", asset_class="index", spread=0.0, point=0.1,
        exit_strategy="signal_exit",
        exit_params={
            "exit_signal_long_column": "x_exit_long",
            "exit_signal_short_column": "x_exit_short",
            "entry_delay": entry_delay,
        },
    )


def _run(df, ctx):
    probs = np.zeros((len(df), 2))
    probs[:, 1] = 1.0
    return simulate_trades(df, probs, probs, 1, 1, 0.5, 0.5, 1, 1, ctx, return_detailed=True)


def test_close_entry_flips_on_the_exit_bar():
    df = _frame(
        closes=[100, 102, 104, 101, 98, 97],
        long_sig=[1, 0, 0, 0, 0, 0],
        short_sig=[0, 0, 0, 1, 0, 0],
        exit_long=[0, 0, 0, 1, 0, 0],
        exit_short=[0, 0, 0, 0, 0, 1],
    )
    trades = _run(df, _ctx(entry_delay=0))["trades_detailed"]
    assert [(t["direction"], t["signal_idx"], t["exit_idx"]) for t in trades] == [
        ("LONG", 0, 3),
        ("SHORT", 3, 5),
    ]
    assert trades[1]["entry_price"] == pytest.approx(101.0)


def test_next_open_entry_waits_for_the_bar_after_the_exit():
    df = _frame(
        closes=[100, 102, 104, 101, 98, 97],
        long_sig=[1, 0, 0, 0, 0, 0],
        short_sig=[0, 0, 0, 1, 0, 0],
        exit_long=[0, 0, 0, 1, 0, 0],
        exit_short=[0, 0, 0, 0, 0, 1],
    )
    trades = _run(df, _ctx(entry_delay=1))["trades_detailed"]
    assert [t["direction"] for t in trades] == ["LONG"]


def test_a_configured_exit_column_that_is_missing_raises():
    df = _frame([100, 101, 102], [1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]).drop(columns="x_exit_long")
    with pytest.raises(ColumnRoleError, match="exit_signal_long_column"):
        _run(df, _ctx(entry_delay=0))
