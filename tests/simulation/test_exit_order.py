"""Per-bar exit order shared by the Python evaluator and the numba kernels.

TP/SL are checked against the levels active at the bar's start, a gap through
the stop fills at the open, the timeout exits at the close only when neither
level was touched, and trailing updates take effect from the next bar.
"""
import numpy as np
import pytest

from fwbg.simulation.numba_core import (
    _simulate_trade_numba,
    _simulate_trade_scale_in_numba,
    _simulate_trade_session_numba,
    _simulate_trade_trailing_numba,
)
from fwbg.simulation.trade import simulate_pro_trade

NO_LEVELS = np.full(1, -1.0)


def _bars(*rows):
    """rows of (open, high, low, close) → opens, closes, highs, lows."""
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 3], a[:, 1], a[:, 2]


def _mirror(opens, closes, highs, lows):
    """Reflect prices around 100 so a long scenario becomes the short one."""
    return 200 - opens, 200 - closes, 200 - lows, 200 - highs


def _python(o, c, h, lo, direction, tp, sl, timeout=None, trail=0.0):
    t = simulate_pro_trade(c, h, lo, 0, direction, tp, sl, 0.0, opens=o,
                           timeout_bars=timeout, trail_distance=trail)
    return t["result"], t["exit_idx"], t["exit_price"]


def _kernels(o, c, h, lo, direction, tp, sl, timeout=0, trail=0.0):
    """(result, exit_idx, exit_price) from every kernel that models the case."""
    args = (o, c, h, lo, 0, direction, tp, sl, 0.0, 0.0, 100, timeout)
    out = [_simulate_trade_trailing_numba(*args, 0.0, trail, 0.0)[:3],
           _simulate_trade_scale_in_numba(*args, NO_LEVELS, 0, 1.0, 0.0, trail, 0.0)[:3]]
    if trail == 0.0:
        out.append(_simulate_trade_numba(*args)[:3])
        out.append(_simulate_trade_session_numba(*args, np.ones(len(o), dtype=np.bool_))[:3])
    return out


FLAT = (100, 100.5, 99.5, 100)


@pytest.fixture(params=[1, -1], ids=["long", "short"])
def direction(request):
    return request.param


def _scenario(direction, *rows):
    bars = _bars(*rows)
    return bars if direction == 1 else _mirror(*bars)


def _price(direction, p):
    return p if direction == 1 else 200 - p


def test_stop_on_timeout_bar_beats_timeout_close(direction):
    o, c, h, lo = _scenario(direction, FLAT, FLAT, (99.5, 100, 94, 99.5), FLAT)

    expected = (-1.0, 2, _price(direction, 95))
    assert _python(o, c, h, lo, direction, 50, 5, timeout=2) == pytest.approx(expected)
    for got in _kernels(o, c, h, lo, direction, 50, 5, timeout=2):
        assert got == pytest.approx(expected)


def test_gap_through_stop_fills_at_open(direction):
    o, c, h, lo = _scenario(direction, FLAT, FLAT, (80, 82, 79, 81), FLAT)

    expected = (-1.0, 2, _price(direction, 80))
    assert _python(o, c, h, lo, direction, 50, 5) == pytest.approx(expected)
    for got in _kernels(o, c, h, lo, direction, 50, 5):
        assert got == pytest.approx(expected)


def test_trailing_stop_moves_from_next_bar(direction):
    # Bar 1 runs to 110 and back to 99: the trail (3) may only use that high
    # from bar 2 on, where the 107 stop is then hit.
    o, c, h, lo = _scenario(
        direction, FLAT, (100, 110, 99, 109), (109, 109.5, 106, 107), FLAT,
    )

    expected = (1.0, 2, _price(direction, 107))
    assert _python(o, c, h, lo, direction, 50, 3, trail=3.0) == pytest.approx(expected)
    for got in _kernels(o, c, h, lo, direction, 50, 3, trail=3.0):
        assert got == pytest.approx(expected)


def test_trailed_stop_gapped_below_entry_is_a_loss(direction):
    o, c, h, lo = _scenario(
        direction, FLAT, (100, 110, 99, 109), (95, 96, 94, 95), FLAT,
    )

    expected = (-1.0, 2, _price(direction, 95))
    assert _python(o, c, h, lo, direction, 50, 3, trail=3.0) == pytest.approx(expected)
    for got in _kernels(o, c, h, lo, direction, 50, 3, trail=3.0):
        assert got == pytest.approx(expected)
