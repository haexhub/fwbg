"""Labels whose outcome window runs past the end of the frame are unknown, not losses."""

import numpy as np

from fwbg.simulation.numba_core import compute_targets_numba

FLAT = (100.0, 100.5, 99.5, 100.0)


def _run(rows, tp, sl, max_bars, timeout_bars=0, nan_incomplete=True):
    a = np.array(rows, dtype=float)
    n = len(a)
    return compute_targets_numba(
        a[:, 0], a[:, 3], a[:, 1], a[:, 2],
        np.full(n, float(tp)), np.full(n, float(sl)), 0.0, 0.0,
        max_bars, timeout_bars, nan_incomplete,
    )  # fmt: skip


def test_open_window_at_the_frame_end_is_nan():
    # 6 bars, nothing ever touches TP (10) or SL (10), max_bars = 3
    rows = [FLAT] * 6
    long_, short, _, _ = _run(rows, 10, 10, max_bars=3)
    # signal i enters at i+1 and needs bars i+1..i+3: complete only for i <= 2
    assert not np.isnan(long_[:3]).any() and (long_[:3] == 0.0).all()
    assert np.isnan(long_[3:]).all()
    assert np.isnan(short[3:]).all()


def test_a_hit_inside_a_short_window_stays_a_label():
    # bar 5 (the last) reaches TP for a long entered on bar 4's signal
    rows = [FLAT, FLAT, FLAT, FLAT, FLAT, (100.0, 112.0, 99.5, 111.0)]
    long_, _, _, _ = _run(rows, 10, 10, max_bars=4)
    assert long_[4] == 1.0  # window cut after one bar, but TP already hit


def test_stop_inside_a_cut_window_is_a_settled_loss_not_nan():
    rows = [FLAT, FLAT, FLAT, (100.0, 100.5, 85.0, 90.0)]
    long_, _, _, _ = _run(rows, 10, 10, max_bars=5)
    assert long_[1] == 0.0 and long_[2] == 0.0  # SL on bar 3 settles both


def test_truncated_timeout_is_nan_but_a_reached_timeout_is_not():
    rows = [FLAT] * 6
    long_, _, _, _ = _run(rows, 10, 10, max_bars=10, timeout_bars=3)
    # timeout at the close of bar i+3: reached for i <= 2
    assert (long_[:3] == 0.0).all()
    assert np.isnan(long_[3:]).all()


def test_last_bar_has_no_entry_and_is_nan():
    long_, short, _, _ = _run([FLAT] * 4, 10, 10, max_bars=1)
    assert np.isnan(long_[-1]) and np.isnan(short[-1])


def test_default_keeps_the_zero_labels():
    long_, short, _, _ = _run([FLAT] * 6, 10, 10, max_bars=3, nan_incomplete=False)
    assert not np.isnan(long_).any() and not np.isnan(short).any()


def test_a_cut_timeout_is_unknown_even_when_the_last_close_is_in_profit():
    # The timeout (4 bars) of the signal on bar 3 would fire after the data ends;
    # closing at the last bar's profitable 102 is not its outcome.
    rows = [FLAT] * 4 + [(100.0, 101.5, 99.9, 101.0), (101.0, 102.5, 100.9, 102.0)]
    long_, _, _, _ = _run(rows, 10, 10, max_bars=10, timeout_bars=4)
    assert long_[0] == 1.0  # the timeout at bar 4 is reached: a settled win
    assert np.isnan(long_[3])
