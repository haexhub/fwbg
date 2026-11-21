"""Scale-in labels follow the open-window rule of the plain kernel.

A window cut by the end of the data has no outcome yet: NaN, not a loss.  A TP or
SL reached before the end settles the label, and complete windows stay 0.0.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import compute_targets_cached
from fwbg.plugins import import_plugin_module

FLAT = (100.0, 100.5, 99.5, 100.0)
_scale_in = import_plugin_module("fwbg-core", "entry_modifiers", "scale_in")


def _frame(rows):
    a = np.array(rows, dtype=float)
    df = pd.DataFrame(
        {"O": a[:, 0], "H": a[:, 1], "L": a[:, 2], "C": a[:, 3]},
        index=pd.date_range("2024-01-01", periods=len(a), freq="h"),
    )
    df["_atr"] = 1.0  # the modifier path reads it; the fixed distances ignore it
    return df


def _ctx(**kw):
    # spread 0.1: tp = sl = 30 -> 3.0 points, never touched by FLAT bars
    params = {
        "symbol": "T",
        "asset_class": "FOREX",
        "spread": 0.1,
        "point": 0.0001,
        "min_trades": 1,
        "exit_strategy": "fixed",
        "max_trade_bars": 3,
        "entry_modifier": "scale_in",
        "entry_modifier_params": {"levels": [0.2]},
    }
    params.update(kw)
    return SimulationContext(**params)


def _labels(rows, return_durations=False, **kw):
    return compute_targets_cached(
        _frame(rows),
        30,
        30,
        _ctx(**kw),
        exit_strategy_mode="fixed",
        return_durations=return_durations,
    )


@pytest.mark.parametrize("return_durations", [False, True])
def test_open_windows_are_unknown_complete_ones_are_zero(return_durations):
    result = _labels([FLAT] * 6, return_durations)
    long_, short = result[0], result[1]
    assert long_[:3].tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(long_[3:]).all() and np.isnan(short[3:]).all()


def test_tp_near_the_end_settles_the_label():
    rows = [FLAT] * 5 + [(100.0, 104.0, 99.9, 103.0)]  # TP on the last bar
    long_, _ = _labels(rows)
    assert long_[4] == 1.0


def test_sl_inside_a_cut_window_settles_a_loss():
    rows = [FLAT, FLAT, FLAT, (100.0, 100.5, 85.0, 90.0)]
    long_, _ = _labels(rows)
    assert long_[1] == 0.0 and long_[2] == 0.0


def test_truncated_timeout_is_unknown():
    long_, _ = compute_targets_cached(
        _frame([FLAT] * 6),
        30,
        30,
        _ctx(max_trade_bars=10),
        timeout_bars=3,
        exit_strategy_mode="fixed",
    )
    # the reached timeouts are settled (a win or a loss, whatever the scale-in fill gives)
    assert not np.isnan(long_[:3]).any() and np.isnan(long_[3:]).all()


def test_the_modifier_keeps_zero_labels_unless_asked():
    n = 6
    a = np.array([FLAT] * n)
    out = _scale_in.ScaleInModifier().compute_targets(
        a[:, 0], a[:, 3], a[:, 1], a[:, 2],
        np.full(n, 3.0), np.full(n, 3.0), np.zeros(n), 0.1, 0.05, 3, 0,
    )  # fmt: skip
    assert not np.isnan(out[0]).any() and not np.isnan(out[1]).any()
