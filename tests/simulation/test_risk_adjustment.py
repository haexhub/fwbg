"""adjust_risk_for_target_dd must land near the drawdown target, not at the floor."""

import logging
import random

import pytest

from fwbg.simulation.trade import (
    MIN_ADJUSTED_RISK,
    adjust_risk_for_target_dd,
    calculate_max_drawdown,
)

RRR = 1.5
TARGET = 0.30


def _random_trades():
    """The 87-trade series that used to collapse to the 0.001 floor (1.4% DD)."""
    rng = random.Random(1)
    out = []
    for _ in range(101):
        n = rng.randint(30, 120)
        out.append([1.0 if rng.random() < 0.5 else -1.0 for _ in range(n)])
    return out


def test_result_stays_within_target_and_close_to_it_for_random_series():
    for trades in _random_trades():
        res = adjust_risk_for_target_dd(trades, 0.05, RRR, TARGET)
        if res["original_dd"] <= TARGET:
            assert res["adjusted_risk"] == 0.05
            continue
        assert res["adjusted_dd"] <= TARGET + 1e-9
        # the old search returned 0.001 (DD 1.4%) for one of these series
        assert res["adjusted_dd"] > TARGET - 0.03
        assert res["adjusted_risk"] > MIN_ADJUSTED_RISK
        assert res["adjusted_dd"] == pytest.approx(
            calculate_max_drawdown(trades, res["adjusted_risk"], RRR)
        )
        assert res["scale_factor"] == pytest.approx(res["adjusted_risk"] / 0.05)


def test_floor_is_explicit_and_warned_when_the_target_is_unreachable(caplog):
    trades = [-1.0] * 50
    with caplog.at_level(logging.WARNING, logger="fwbg.simulation.trade"):
        res = adjust_risk_for_target_dd(trades, 0.05, 2.0, target_max_dd=0.01)
    assert res["adjusted_risk"] == MIN_ADJUSTED_RISK
    assert res["adjusted_dd"] > 0.01
    assert "risk floor" in caplog.text


def test_base_risk_below_floor_is_never_raised():
    trades = [-1.0] * 400
    res = adjust_risk_for_target_dd(trades, 0.0005, 2.0, target_max_dd=0.01)
    assert res["adjusted_risk"] <= 0.0005
