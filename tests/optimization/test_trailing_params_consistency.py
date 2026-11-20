"""The trade evaluator simulates the breakeven/trailing exit of the strategy.

``atr_trailing`` carries ``breakeven_trigger``/``trail_atr_mult`` in
``exit_params``; the trade evaluator used to read only
``exit_modifier_params`` and so simulated a plain SL/TP exit. Training labels
resolve the same parameters through ``resolve_trailing_params``; the label
side is tested with the plugin.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import resolve_trailing_params, simulate_trades
from fwbg.plugins import import_plugin_module

needs_atr_trailing = pytest.mark.skipif(
    import_plugin_module("fwbg-premium", "exit_strategies", "atr_trailing") is None,
    reason="the evaluator looks up atr_trailing, which ships with fwbg-premium",
)

# Entry 100 (ATR 1, SL 98, TP out of reach).  Bar 1 runs to 104, so a stop
# trailing 1.0 behind the high sits at 103 from bar 2 on; bar 2 touches it.
# Without trailing the trade stays open and ends at the last close (100).
ROWS = [
    (100, 100, 100, 100),
    (100, 104, 100.5, 103),
    (103, 103.5, 102.5, 100),
    (100, 100.5, 99.5, 100),
]


def _df():
    a = np.array(ROWS, dtype=float)
    df = pd.DataFrame(
        {"O": a[:, 0], "H": a[:, 1], "L": a[:, 2], "C": a[:, 3]},
        index=pd.date_range("2024-01-01", periods=len(a), freq="h"),
    )
    df["_atr"] = 1.0
    return df


def _ctx(exit_params, exit_modifier_params=None, strategy="atr_trailing"):
    return SimulationContext(
        symbol="T",
        asset_class="FOREX",
        spread=0.0,
        point=0.0001,
        long_enabled=True,
        short_enabled=False,
        min_trades=1,
        exit_strategy=strategy,
        exit_params=exit_params,
        exit_modifier_params=exit_modifier_params or {},
    )


def _evaluated(ctx):
    probs = np.zeros((len(ROWS), 2))
    probs[0, 1] = 1.0
    res = simulate_trades(_df(), probs, None, 1, None, 0.5, 0.5, 50, 2, ctx)
    return res["trades"][0]


BASE = {"min_tp_pips": 0, "min_sl_pips": 0}


@needs_atr_trailing
def test_exit_params_trailing_reaches_the_evaluator():
    ctx = _ctx({**BASE, "breakeven_trigger": 0.0, "trail_atr_mult": 1.0})
    trade = _evaluated(ctx)
    assert trade["result"] == 1.0
    assert trade["pnl_raw"] == pytest.approx(3.0)


@needs_atr_trailing
def test_evaluator_without_trailing_keeps_the_trade_open():
    ctx = _ctx({**BASE, "breakeven_trigger": 0.0, "trail_atr_mult": 0.0})
    assert _evaluated(ctx)["result"] == -1.0


@needs_atr_trailing
def test_modifier_params_override_the_strategy_params():
    ctx = _ctx(
        {**BASE, "breakeven_trigger": 0.0, "trail_atr_mult": 0.0},
        {"breakeven_trigger": 0.0, "trail_atr_mult": 1.0},
    )
    assert _evaluated(ctx)["result"] == 1.0


class TestResolveTrailingParams:
    def test_atr_trailing_defaults(self):
        assert resolve_trailing_params(_ctx({})) == (0.5, 0.5, 0)

    def test_other_strategies_use_modifier_params_only(self):
        ctx = _ctx({"breakeven_trigger": 0.9}, {"trail_atr_mult": 2.0}, "fixed")
        assert resolve_trailing_params(ctx) == (0.0, 2.0, 0)

    def test_orb_direct_params(self):
        ctx = _ctx({"breakeven_trigger": 0.4, "trail_pips": 7}, None, "orb_based")
        assert resolve_trailing_params(ctx) == (0.4, 0.0, 7)
