"""Overnight financing: CFD positions pay a rate p.a. of the entry price per night held."""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.data.assets import AssetConfig
from fwbg.simulation.trade import simulate_pro_trade

FAR = 1_000.0
DAILY = pd.date_range("2024-01-01", periods=5, freq="D").values  # Mon..Fri


def _trade(closes, exit_at, rate, timestamps=DAILY):
    closes = np.asarray(closes, dtype=float)
    flags = np.zeros(len(closes), dtype=bool)
    flags[exit_at] = True
    return simulate_pro_trade(
        closes=closes, highs=closes, lows=closes, opens=closes, idx=0, direction=1,
        tp_distance=FAR, sl_distance=FAR, spread=0.0, entry_delay=0,
        exit_signal=flags, timestamps=timestamps, financing_rate=rate,
    )


def test_each_night_held_costs_the_daily_rate_of_the_entry_price():
    trade = _trade([100, 104, 107, 110, 111], exit_at=3, rate=0.365)
    assert trade["financing"] == pytest.approx(100 * 0.001 * 3)
    assert trade["pnl_raw"] == pytest.approx(10 - 0.3)
    assert trade["result"] == 1.0


def test_financing_turns_a_small_gain_into_a_loss():
    trade = _trade([100, 100, 100, 100.1, 100], exit_at=3, rate=0.365)
    assert trade["pnl_raw"] == pytest.approx(0.1 - 0.3)
    assert trade["result"] == -1.0


def test_nights_are_calendar_nights_across_a_weekend():
    fri_to_mon = np.array(["2024-01-05", "2024-01-08"], dtype="datetime64[ns]")
    trade = _trade([100, 101], exit_at=1, rate=0.365, timestamps=fri_to_mon)
    assert trade["financing"] == pytest.approx(100 * 0.001 * 3)


def test_without_timestamps_or_rate_nothing_is_charged():
    assert _trade([100, 101, 102], exit_at=2, rate=0.365, timestamps=None)["financing"] == 0.0
    assert _trade([100, 101, 102], exit_at=2, rate=0.0)["financing"] == 0.0


def _strategy(**extra):
    return StrategyConfig.from_dict({"name": "t", **extra})


def test_strategy_financing_pct_reaches_the_context_as_a_fraction():
    asset = AssetConfig(symbol="NAS100", asset_class="INDEX", spread=1.0, point=0.1, currencies=["USD"])
    ctx = SimulationContext.create(asset, _strategy(financing_pct=6.5))
    assert ctx.financing_rate == pytest.approx(0.065)
    assert SimulationContext.create(asset, _strategy()).financing_rate == 0.0


@pytest.mark.parametrize("bad", [-1, "abc", float("nan")])
def test_invalid_financing_pct_is_rejected(bad):
    with pytest.raises(ValueError, match="financing_pct"):
        _strategy(financing_pct=bad)
