"""filters.entry_hours: entries only when the entry bar closes inside a session."""

import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.data import trading_hours
from fwbg.data.assets import get_asset
from fwbg.data.trading_hours import TradingHoursError, parse_trading_hours
from fwbg.signals.evaluator import evaluate_condition

LONDON = {"tz": "Europe/London", "windows": [{"days": "Mon-Fri", "start": "08:00", "end": "17:00"}]}


def hourly(start: str, periods: int) -> pd.DataFrame:
    return pd.DataFrame({"C": 1.0}, index=pd.date_range(start, periods=periods, freq="h"))


def allowed(df: pd.DataFrame, hours, bar_minutes: int = 60) -> list[str]:
    cond = {
        "type": "session_filter",
        "hours": parse_trading_hours(hours),
        "bar_minutes": bar_minutes,
    }
    mask = evaluate_condition(cond, df)
    return list(df.index[mask.to_numpy()].strftime("%H:%M"))


def test_bar_closes_inside_the_session_in_local_time_across_dst():
    # winter: London = UTC, bars starting 07:00 .. 15:00 UTC close 08:00 .. 16:00 -> (08:00, 17:00]
    winter = allowed(hourly("2026-01-05 00:00", 24), LONDON)
    assert winter[0] == "08:00" and winter[-1] == "16:00" and len(winter) == 9
    # summer: London = UTC+1, the same local hours are one hour earlier in UTC
    summer = allowed(hourly("2026-07-06 00:00", 24), LONDON)
    assert summer[0] == "07:00" and summer[-1] == "15:00"
    assert allowed(hourly("2026-01-10 00:00", 24), LONDON) == []  # Saturday


def test_the_bar_ending_at_the_close_may_enter():
    cash = {
        "tz": "America/New_York",
        "windows": [{"days": "Mon-Fri", "start": "09:30", "end": "16:00"}],
    }
    m15 = pd.DataFrame(
        {"C": 1.0}, index=pd.date_range("2026-03-02 14:00", periods=30, freq="15min")
    )
    times = allowed(m15, cash, bar_minutes=15)  # UTC starts; EST = UTC-5
    assert times[0] == "14:30" and times[-1] == "20:45"  # closes 09:45 .. 16:00 New York


def test_daily_bars_are_rejected():
    with pytest.raises(ValueError, match="intraday"):
        allowed(hourly("2026-01-05", 3), LONDON, bar_minutes=1440)


def test_session_name_resolves_per_asset(monkeypatch):
    monkeypatch.setattr(trading_hours, "load_asset_overrides", lambda: {})
    strategy = StrategyConfig.from_dict({"name": "x", "filters": {"entry_hours": "cash"}})
    ctx = SimulationContext.create(get_asset("NAS100"), strategy)
    assert ctx.entry_hours.windows[0].start == 9 * 60 + 30
    with pytest.raises(TradingHoursError, match="cash"):
        SimulationContext.create(get_asset("EURUSD"), strategy)
    explicit = StrategyConfig.from_dict({"name": "x", "filters": {"entry_hours": LONDON}})
    assert SimulationContext.create(get_asset("EURUSD"), explicit).entry_hours.tz == "Europe/London"
    assert SimulationContext.create(get_asset("EURUSD"), StrategyConfig()).entry_hours is None


@pytest.mark.parametrize("bad", [{"tz": "UTC"}, "", 9, ["08:00"]])
def test_invalid_entry_hours_are_rejected_when_the_config_loads(bad):
    with pytest.raises(ValueError, match="entry_hours"):
        StrategyConfig.from_dict({"name": "x", "filters": {"entry_hours": bad}})


def test_entry_hours_round_trip_in_the_saved_config():
    strategy = StrategyConfig.from_dict({"name": "x", "filters": {"entry_hours": "cash"}})
    assert strategy.to_dict()["filters"]["entry_hours"] == "cash"


def test_signal_fold_entries_combine_rules_and_entry_hours(monkeypatch):
    """Integration: a signal fold's composed entry column honours rules AND entry hours."""
    import numpy as np

    from fwbg.data import config as data_config
    from fwbg.optimization.robust_validation import WalkForwardFold
    from fwbg.optimization.signal_fold import prepare_signal_fold_data

    monkeypatch.setattr(data_config, "TIMEFRAME", "HOUR_1")
    rows, split = 24 * 14, 24 * 7
    idx = pd.date_range("2026-01-05", periods=rows, freq="h")  # winter: London = UTC
    price = 1.1 + np.cumsum(np.random.default_rng(0).normal(0, 0.0005, rows))
    df = pd.DataFrame(
        {
            "O": price,
            "H": price + 0.001,
            "L": price - 0.001,
            "C": price,
            "flag": (np.arange(rows) % 2).astype(float),
        },
        index=idx,
    )
    fold = WalkForwardFold(
        fold_id=0,
        train_start=0,
        train_end=split,
        test_start=split,
        test_end=rows,
        train_df=df.iloc[:split].copy(),
        test_df=df.iloc[split:].copy(),
    )
    strategy = StrategyConfig.from_dict(
        {
            "name": "x",
            "model": {"type": "signal", "trade_directions": ["long"]},
            "filters": {"entry_hours": LONDON},
            "signal_rules": {
                "long": {
                    "operator": "AND",
                    "conditions": [
                        {"type": "value_check", "column": "flag", "op": ">", "value": 0.5}
                    ],
                }
            },
            "exit_strategies": [{"name": "fixed", "params": {"tp_mult": 1.0, "sl_mult": 1.0}}],
        }
    )
    ctx = SimulationContext.create(get_asset("EURUSD"), strategy)
    _, test_df, _ = prepare_signal_fold_data(
        fold,
        fold_indicators=[],
        precomputed_raw_df=None,
        preprocessing_configs=None,
        ctx=ctx,
        sym="EURUSD",
    )
    entries = test_df.index[test_df["_composed_signal_long"] > 0]
    closes = entries + pd.Timedelta(hours=1)
    assert len(entries) > 0
    assert (test_df.loc[entries, "flag"] == 1.0).all()
    assert ((closes.hour >= 9) & (closes.hour <= 17) & (closes.dayofweek < 5)).all()
