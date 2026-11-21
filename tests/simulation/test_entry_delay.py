"""Labels and evaluated trades enter at the same price for either entry_delay.

``entry_delay=1`` fills at the next open, ``entry_delay=0`` at the signal bar's
close; exits are checked from the next bar on in both modes.  The label kernels
used to enter at the next open always, and the exit strategies read different
defaults than the trade evaluator when the config left the setting out.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core import GridParams
from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import compute_targets_cached, simulate_trades
from fwbg.plugins import import_plugin_module
from fwbg.simulation import numba_core
from fwbg.simulation.trade import (
    DEFAULT_ENTRY_DELAY,
    resolve_entry_delay,
    simulate_pro_trade,
)

# Signal on bar 0.  Entering at its close (98) with +-1.5 puts TP at 99.5, which
# bar 1 (high 99.8, low 97.6) reaches before the stop at 96.5: a win.  Entering
# at bar 1's open (100) puts TP at 101.5 and the stop at 98.5, which bar 1's
# low breaks: a loss.
OPN = np.array([98.0, 100.0, 100.0, 100.0])
HGH = np.array([98.5, 99.8, 100.0, 100.0])
LOW = np.array([97.5, 97.6, 99.5, 99.5])
CLS = np.array([98.0, 99.0, 100.0, 100.0])
DIST = 1.5


def _kernel_labels(entry_delay, max_bars=3):
    n = len(CLS)
    return numba_core.compute_targets_numba(
        OPN, CLS, HGH, LOW, np.full(n, DIST), np.full(n, DIST), 0.0, 0.0, max_bars, 0,
        False, entry_delay,
    )  # fmt: skip


def _trade(direction, entry_delay):
    return simulate_pro_trade(
        CLS, HGH, LOW, 0, direction, DIST, DIST, 0.0, opens=OPN, entry_delay=entry_delay
    )


@pytest.mark.parametrize(
    "entry_delay, long_label, short_label", [(1, 0.0, 1.0), (0, 1.0, 0.0)]
)
def test_label_kernel_enters_where_the_delay_says(entry_delay, long_label, short_label):
    long_, short, _, _ = _kernel_labels(entry_delay)
    assert long_[0] == long_label
    assert short[0] == short_label


@pytest.mark.parametrize("entry_delay", [0, 1])
@pytest.mark.parametrize("direction", [1, -1])
def test_label_agrees_with_the_evaluated_trade(entry_delay, direction):
    long_, short, _, _ = _kernel_labels(entry_delay)
    label = long_[0] if direction == 1 else short[0]
    trade = _trade(direction, entry_delay)
    assert (trade["result"] == 1.0) == (label == 1.0)


def test_close_entry_price_is_the_signal_close():
    assert _trade(1, 0)["entry_price_raw"] == CLS[0]
    assert _trade(1, 1)["entry_price_raw"] == OPN[1]


@pytest.mark.parametrize("entry_delay", [0, 1])
def test_other_kernels_use_the_same_entry(entry_delay):
    args = (OPN, CLS, HGH, LOW, 0, 1, DIST, DIST, 0.0, 0.0, 3, 0)
    plain = numba_core._simulate_trade_numba(*args, entry_delay)
    trailing = numba_core._simulate_trade_trailing_numba(
        *args, 0.0, 0.0, 0.0, 0.0, entry_delay
    )
    session = numba_core._simulate_trade_session_numba(
        *args, np.ones(len(CLS), dtype=np.bool_), entry_delay
    )
    scale = numba_core._simulate_trade_scale_in_numba(
        *args, np.full(1, -1.0), 0, 1.0, 0.0, 0.0, 0.0, entry_delay
    )
    expected = 1.0 if entry_delay == 0 else -1.0
    assert plain[0] == trailing[0] == session[0] == scale[0] == expected


# --- the default is one value -------------------------------------------------


def _ctx(exit_params=None, spread=0.1):
    return SimulationContext(
        symbol="T",
        asset_class="FOREX",
        spread=spread,
        point=0.0001,
        long_enabled=True,
        short_enabled=False,
        min_trades=1,
        exit_strategy="fixed",
        exit_params=exit_params or {},
    )


def _frame():
    df = pd.DataFrame(
        {"O": OPN, "H": HGH, "L": LOW, "C": CLS},
        index=pd.date_range("2024-01-01", periods=len(CLS), freq="h"),
    )
    return df


def _evaluated_entry(ctx):
    probs = np.zeros((len(CLS), 2))
    probs[0, 1] = 1.0
    res = simulate_trades(
        _frame(), probs, None, 1, None, 0.5, 0.5, 15, 15, ctx, return_detailed=True
    )
    return res["trades_detailed"][0]


def test_default_is_the_next_open():
    assert DEFAULT_ENTRY_DELAY == 1
    assert resolve_entry_delay(None) == 1
    assert resolve_entry_delay({}) == 1
    assert resolve_entry_delay({"entry_delay": 0}) == 0


@pytest.mark.parametrize("bad", [2, -1, 0.5, True, None])
def test_other_delays_are_refused(bad):
    with pytest.raises(ValueError, match="entry_delay"):
        resolve_entry_delay({"entry_delay": bad})


def test_evaluator_without_the_setting_enters_at_the_next_open():
    assert _evaluated_entry(_ctx())["entry_idx"] == 1
    assert _evaluated_entry(_ctx({"entry_delay": 0}))["entry_idx"] == 0


@pytest.mark.parametrize("entry_delay", [None, 0, 1])
def test_signal_exit_labels_and_evaluator_share_the_default(entry_delay):
    signal_exit = import_plugin_module("fwbg-core", "exit_strategies", "signal_exit")
    df = _frame()
    df["xl"] = [0, 0, 1, 0]
    extra = {"exit_signal_long_column": "xl"}
    if entry_delay is not None:
        extra["entry_delay"] = entry_delay
    labels = signal_exit.SignalExitStrategy().compute_targets(
        df,
        _ctx(extra),
        params=GridParams(tp_value=1.0, sl_value=1.0, extra=extra),
    )[0]
    # exit at bar 2's close (100); entry 98 (delay 0) or 100 (delay 1, no gain)
    assert labels[0] == (1.0 if entry_delay == 0 else 0.0)
    probs = np.zeros((len(CLS), 2))
    probs[0, 1] = 1.0
    ctx = SimulationContext(
        symbol="T", asset_class="FOREX", spread=0.0, point=0.0001,
        long_enabled=True, short_enabled=False, min_trades=1,
        exit_strategy="signal_exit", exit_params=extra,
    )  # fmt: skip
    trade = simulate_trades(
        df, probs, None, 1, None, 0.5, 0.5, 1, 1, ctx, return_detailed=True
    )["trades_detailed"][0]
    assert trade["entry_idx"] == (0 if entry_delay == 0 else 1)
    assert (trade["result"] == 1.0) == (labels[0] == 1.0)


@pytest.mark.parametrize("entry_delay", [0, 1])
def test_fixed_strategy_labels_honour_entry_delay(entry_delay):
    ctx = _ctx({"entry_delay": entry_delay})
    ctx.max_trade_bars = 3
    long_, _ = compute_targets_cached(_frame(), 15, 15, ctx, exit_strategy_mode="fixed")
    trade = _evaluated_entry(ctx)
    assert long_[0] == (1.0 if entry_delay == 0 else 0.0)
    assert (trade["result"] == 1.0) == (long_[0] == 1.0)
