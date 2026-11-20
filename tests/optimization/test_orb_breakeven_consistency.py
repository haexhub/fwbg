"""orb_based breakeven: the trade evaluator models what the training labels model.

The labels move the stop to ``entry + breakeven_offset * tp_distance`` when the
breakeven trigger is reached, and in range mode trail at the opening range from
then on, with or without ``trail_atr_mult``.  The evaluator used to ignore the
offset and to leave the range-mode trade without trailing.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core import GridParams
from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import (
    resolve_breakeven_offset,
    simulate_trades,
)
from fwbg.plugins import import_plugin_module
from fwbg.simulation import numba_core
from fwbg.simulation.trade import simulate_pro_trade

_orb = import_plugin_module("fwbg-premium", "exit_strategies", "orb_based")
needs_orb = pytest.mark.skipif(_orb is None, reason="orb_based ships with fwbg-premium")

BASE = {"min_tp_pips": 0, "min_sl_pips": 0}

# Entry 100 (ATR 1): SL 2 below, breakeven trigger 0.5 of the TP distance.
# Bar 1 reaches 102.5, so the stop moves up.  With offset 0.25 of a TP of 4 it
# sits at 101 and bar 2 (low 100.8) takes it out for a win; at the entry it
# does not, and bar 3 (low 98.5) stops the trade at 100 for no gain.
OFFSET_ROWS = [
    (100, 100, 100, 100),
    (100, 102.5, 99.9, 102),
    (102, 102.2, 100.8, 101.5),
    (101.5, 101.6, 98.5, 99),
    (99, 99, 99, 99),
]

# Range mode (range 2, TP = 3 x range = 6): bar 1 triggers breakeven at 102 and
# the stop trails the high (103) by the range, to 101.  Bar 2 (low 100.5) takes
# it out for a win.  Without trailing the stop stays at the entry until bar 3.
RANGE_ROWS = [
    (100, 100, 100, 100),
    (100, 103, 99.9, 102.5),
    (102.5, 102.6, 100.5, 101),
    (101, 101.2, 99, 99.5),
    (99.5, 99.5, 99.5, 99.5),
]


def _df(rows):
    a = np.array(rows, dtype=float)
    df = pd.DataFrame(
        {"O": a[:, 0], "H": a[:, 1], "L": a[:, 2], "C": a[:, 3]},
        index=pd.date_range("2024-01-01", periods=len(a), freq="h"),
    )
    df["_atr"] = 1.0
    return df


def _ctx(exit_params, strategy="orb_based"):
    return SimulationContext(
        symbol="T",
        asset_class="FOREX",
        spread=0.0,
        point=0.0001,
        long_enabled=True,
        short_enabled=False,
        min_trades=1,
        exit_strategy=strategy,
        exit_params={**BASE, **exit_params},
    )


def _label_and_trade(rows, exit_params, tp_mult, sl_mult):
    df = _df(rows)
    ctx = _ctx(exit_params)
    label = _orb.OrbExitStrategy().compute_targets(
        df, ctx, params=GridParams(tp_value=tp_mult, sl_value=sl_mult, extra={})
    )[0][0]
    probs = np.zeros((len(df), 2))
    probs[0, 1] = 1.0
    trade = simulate_trades(
        df, probs, None, 1, None, 0.5, 0.5, tp_mult, sl_mult, ctx,
        return_detailed=True,
    )["trades_detailed"][0]  # fmt: skip
    return label, trade


@needs_orb
@pytest.mark.parametrize("offset, win", [(0.25, True), (0.0, False)])
def test_breakeven_offset_label_and_trade_agree(offset, win):
    label, trade = _label_and_trade(
        OFFSET_ROWS,
        {"breakeven_trigger": 0.5, "breakeven_offset": offset},
        tp_mult=4.0,
        sl_mult=2.0,
    )
    assert bool(label == 1.0) is win
    assert bool(trade["result"] == 1.0) is win
    if win:
        assert trade["exit_idx"] == 2 and trade["exit_price"] == pytest.approx(101.0)
        assert trade["pnl_raw"] == pytest.approx(1.0)


@needs_orb
def test_range_mode_breakeven_trails_at_the_range():
    exit_params = {"tp_mode": "range", "breakeven_trigger": 0.5}
    rows = _df(RANGE_ROWS)
    rows["orb_range"] = 2.0
    ctx = _ctx(exit_params)
    label = _orb.OrbExitStrategy().compute_targets(
        rows, ctx, params=GridParams(tp_value=3.0, sl_value=2.0, extra={})
    )[0][0]
    probs = np.zeros((len(rows), 2))
    probs[0, 1] = 1.0
    trade = simulate_trades(
        rows, probs, None, 1, None, 0.5, 0.5, 3.0, 2.0, ctx, return_detailed=True
    )["trades_detailed"][0]
    assert label == 1.0
    assert trade["result"] == 1.0
    assert trade["exit_idx"] == 2 and trade["exit_price"] == pytest.approx(101.0)


class TestResolveBreakevenOffset:
    def test_orb_reads_the_exit_param(self):
        assert resolve_breakeven_offset(_ctx({"breakeven_offset": 0.2})) == 0.2
        assert resolve_breakeven_offset(_ctx({})) == 0.0

    def test_a_modifier_may_override_it(self):
        ctx = _ctx({"breakeven_offset": 0.2})
        ctx.exit_modifier = "trailing_stop"
        ctx.exit_modifier_params = {"breakeven_offset": 0.1}
        assert resolve_breakeven_offset(ctx) == 0.1

    def test_other_strategies_have_no_offset(self):
        assert resolve_breakeven_offset(_ctx({"breakeven_offset": 0.2}, "fixed")) == 0.0


@pytest.mark.parametrize("offset", [0.0, 0.25])
@pytest.mark.parametrize("direction", [1, -1])
def test_evaluator_offset_matches_the_label_kernel(offset, direction):
    """Premium-free: the kernel the labels use and simulate_pro_trade agree."""
    a = np.array(OFFSET_ROWS, dtype=float)
    if direction == -1:
        a = 200.0 - a[:, [0, 2, 1, 3]]  # mirror the series: highs <-> lows
    opens, highs, lows, closes = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    kernel = numba_core._simulate_trade_trailing_numba(
        opens, closes, highs, lows, 0, direction, 4.0, 2.0, 0.0, 0.0,
        10, 0, 0.5, 0.0, 0.0, offset,
    )  # fmt: skip
    trade = simulate_pro_trade(
        closes, highs, lows, 0, direction, 4.0, 2.0, 0.0,
        opens=opens, breakeven_trigger=0.5, breakeven_offset=offset,
    )  # fmt: skip
    assert trade["result"] == kernel[0]
    assert trade["exit_idx"] == kernel[1]
    assert trade["exit_price"] == pytest.approx(kernel[2])
