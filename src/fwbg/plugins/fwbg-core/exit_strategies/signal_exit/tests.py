"""Tests for the signal_exit exit strategy plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.core import GridParams
from fwbg.core.context import SimulationContext
from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "exit_strategies", "signal_exit")
SignalExitStrategy = _mod.SignalExitStrategy
next_flag_index = _mod.next_flag_index


def _ctx(spread=0.0):
    return SimulationContext(symbol="TEST", asset_class="index", spread=spread, point=0.1)


def _params(**extra):
    return GridParams(tp_value=1.0, sl_value=1.0, extra=extra)


def _df(closes, exit_long, exit_short):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"O": closes, "H": closes, "L": closes, "C": closes,
                         "xl": exit_long, "xs": exit_short})


def test_next_flag_index_is_strictly_after_each_bar():
    flags = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    assert next_flag_index(flags).tolist() == [1, 4, 4, 4, 6, 6]


def test_labels_compare_the_close_with_the_next_flagged_close():
    df = _df([100, 101, 103, 99, 98], exit_long=[0, 0, 1, 0, 1], exit_short=[0, 0, 0, 1, 0])
    params = _params(exit_signal_long_column="xl", exit_signal_short_column="xs")
    tgt_l, tgt_s, dur_l, dur_s = SignalExitStrategy().compute_targets(
        df, _ctx(), params=params, return_durations=True
    )
    assert tgt_l.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]  # 100->103, 101->103, 103->98, 99->98, no exit
    assert tgt_s.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]  # all fall to 99; bars 3-4 have no later exit
    assert dur_l.tolist() == [2, 1, 2, 1, 0]
    assert dur_s.tolist() == [3, 2, 1, 0, 0]


def test_labels_pay_the_entry_cost():
    df = _df([100, 101], exit_long=[0, 1], exit_short=[0, 0])
    tgt_l, _ = SignalExitStrategy().compute_targets(
        df, _ctx(spread=1.0), params=_params(exit_signal_long_column="xl")
    )
    assert tgt_l.tolist() == [0.0, 0.0]  # +1 move does not cover 1.5 spread


def test_next_open_entry_labels_use_the_next_bar_open():
    df = _df([100, 105, 103], exit_long=[0, 0, 1], exit_short=[0, 0, 0])
    df["O"] = [99, 102, 103]
    tgt_l, _, dur_l, _ = SignalExitStrategy().compute_targets(
        df, _ctx(), params=_params(
            exit_signal_long_column="xl", entry_delay=1
        ), return_durations=True
    )
    assert tgt_l.tolist() == [1.0, 0.0, 0.0]  # enter at 102, exit at 103
    assert dur_l.tolist() == [2, 1, 0]  # duration includes the signal-to-entry bar


def test_next_open_entry_cannot_be_labeled_on_the_final_bar():
    df = _df([100, 101], exit_long=[0, 1], exit_short=[0, 0])
    _, _, dur_l, _ = SignalExitStrategy().compute_targets(
        df, _ctx(), params=_params(
            exit_signal_long_column="xl", entry_delay=1
        ), return_durations=True
    )
    assert dur_l.tolist() == [1, 0]


def test_missing_exit_column_raises():
    df = _df([100, 101], exit_long=[0, 1], exit_short=[0, 0])
    with pytest.raises(KeyError, match="nope"):
        SignalExitStrategy().compute_targets(df, _ctx(), params=_params(exit_signal_long_column="nope"))


def test_distances_are_never_reached():
    df = _df([100, 50], exit_long=[0, 0], exit_short=[0, 0])
    tp, sl = SignalExitStrategy().resolve_distances(df, 1.0, 1.0, _ctx())
    assert np.all(tp >= 10 * df["C"].values) and np.all(sl >= 10 * df["C"].values)
