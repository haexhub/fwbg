"""hour_filter / day_filter gate on the bar an entry fills on, not the signal bar."""

import numpy as np
import pandas as pd

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import simulate_trades
from fwbg.signals.evaluator import evaluate_condition


def _hourly(start="2024-01-02 06:00", periods=8):
    idx = pd.date_range(start, periods=periods, freq="h")
    return pd.DataFrame({"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0}, index=idx)


def test_hour_filter_uses_the_following_bars_hour():
    df = _hourly()  # 06:00 .. 13:00
    mask = evaluate_condition({"type": "hour_filter", "hours": [9]}, df)
    # the order fills at the open of the 09:00 bar, decided on the 08:00 bar
    assert [t.hour for t in df.index[mask.values]] == [8]


def test_hour_filter_exclude_and_last_bar_never_enters():
    df = _hourly()
    mask = evaluate_condition(
        {"type": "hour_filter", "hours": [9], "exclude": True}, df
    )
    assert [t.hour for t in df.index[~mask.values]] == [8, 13]  # 13:00 is the last bar


def test_day_filter_uses_the_entry_bars_weekday_across_a_weekend():
    idx = pd.DatetimeIndex(
        ["2024-01-05 22:00", "2024-01-05 23:00", "2024-01-08 00:00", "2024-01-08 01:00"]
    )  # Fri, Fri, Mon, Mon
    df = pd.DataFrame({"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0}, index=idx)
    mon = evaluate_condition({"type": "day_filter", "days": [0]}, df)
    # Friday 23:00 signals the Monday 00:00 entry; Monday 00:00 signals Monday 01:00
    assert list(mon.values) == [False, True, True, False]


def test_filtered_signal_enters_on_an_allowed_hour():
    df = _hourly(periods=10)
    sig = evaluate_condition({"type": "hour_filter", "hours": [9, 10]}, df)
    df["_composed_signal_long"] = sig.astype(float)
    ctx = SimulationContext(
        symbol="T",
        asset_class="FOREX",
        spread=0.0,
        point=0.0001,
        long_enabled=True,
        short_enabled=False,
        min_trades=1,
    )
    probs = np.zeros((len(df), 2))
    probs[:, 1] = 1.0
    res = simulate_trades(
        df,
        probs,
        None,
        1,
        None,
        0.5,
        0.5,
        1,
        1,
        ctx,
        return_detailed=True,
        timeout_bars=1,
    )
    entry_hours = [df.index[t["entry_idx"]].hour for t in res["trades_detailed"]]
    assert entry_hours and set(entry_hours) <= {9, 10}
