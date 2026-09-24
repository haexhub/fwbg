"""Regression tests for the shared target and trade-simulation contracts."""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.core import GridParams
from fwbg.optimization.targets import simulate_trades
from fwbg.plugins import import_plugin_module


_fixed = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
if _fixed is None:
    pytest.skip("fwbg-core exit_strategies plugin not available", allow_module_level=True)

FixedExitStrategy = _fixed.FixedExitStrategy


def _scale_context():
    return SimulationContext(
        symbol="TEST",
        asset_class="forex",
        spread=1.0,
        point=0.01,
        max_trade_bars=10,
        entry_modifier="scale_in",
        entry_modifier_params={"levels": [0.5], "qty_multiplier": 1.0},
    )


@pytest.mark.parametrize(
    ("highs", "lows", "direction_index"),
    [
        (
            [100.0, 100.0, 105.0, 100.0],
            [100.0, 94.0, 99.0, 100.0],
            0,
        ),
        (
            [100.0, 106.0, 101.0, 100.0],
            [100.0, 99.0, 96.0, 100.0],
            1,
        ),
    ],
    ids=["long", "short"],
)
def test_fixed_scale_in_supports_array_contract_and_durations(
    highs, lows, direction_index
):
    """Fixed + scale-in works for both directions with either output shape."""
    values = np.asarray([100.0, 100.0, 100.0, 100.0])
    df = pd.DataFrame(
        {"O": values, "H": highs, "L": lows, "C": values, "_atr": np.ones(4)},
        index=pd.date_range("2024-01-01", periods=4, freq="h"),
    )
    strategy = FixedExitStrategy()
    params = GridParams(tp_value=5.0, sl_value=10.0, timeout_bars=None)

    plain = strategy.compute_targets(df, _scale_context(), params=params)
    detailed = strategy.compute_targets(
        df, _scale_context(), params=params, return_durations=True
    )

    assert len(plain) == 2
    assert len(detailed) == 4
    targets = plain[direction_index]
    durations = detailed[2 + direction_index]
    assert targets[0] == 1.0
    assert durations[0] == 2


def test_trailing_only_modifier_is_active_without_breakeven():
    """A trail distance is honored when breakeven_trigger is zero."""
    n = 5
    df = pd.DataFrame(
        {
            "O": [100.0, 100.0, 104.0, 104.0, 104.0],
            "H": [100.0, 105.0, 104.5, 104.0, 104.0],
            "L": [100.0, 104.5, 103.5, 104.0, 104.0],
            "C": [100.0, 104.5, 104.0, 104.0, 104.0],
            "_atr": np.full(n, 2.0),
            "_regime": np.array([4, 0, 0, 0, 0], dtype=np.int8),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    ctx = SimulationContext(
        symbol="TEST",
        asset_class="forex",
        spread=1.0,
        point=0.01,
        max_trade_bars=4,
        long_enabled=True,
        short_enabled=False,
        exit_modifier_params={"breakeven_trigger": 0.0, "trail_atr_mult": 0.5},
    )
    probs = np.zeros((n, 2))
    probs[:, 1] = 0.9

    result = simulate_trades(
        df,
        probs,
        None,
        1,
        None,
        0.5,
        0.5,
        10,
        10,
        ctx,
        return_detailed=True,
    )

    assert len(result["trades_detailed"]) == 1
    trade = result["trades_detailed"][0]
    assert trade["exit_idx"] == 2
    assert trade["exit_price"] == pytest.approx(104.0)
    assert trade["result"] == 1.0


def test_return_detailed_only_changes_output_shape():
    """Composed signal filters apply identically in both output modes."""
    n = 6
    values = np.full(n, 100.0)
    df = pd.DataFrame(
        {
            "O": values,
            "H": values + 0.1,
            "L": values - 0.1,
            "C": values,
            "_regime": np.full(n, 4, dtype=np.int8),
            "_composed_signal_long": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "_composed_signal_short": np.zeros(n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    ctx = SimulationContext(
        symbol="TEST",
        asset_class="forex",
        spread=0.1,
        point=0.01,
        max_trade_bars=4,
        long_enabled=True,
        short_enabled=False,
    )
    probs = np.zeros((n, 2))
    probs[:, 1] = 0.9

    compact = simulate_trades(
        df, probs, None, 1, None, 0.5, 0.5, 10, 10, ctx, timeout_bars=1
    )
    detailed = simulate_trades(
        df,
        probs,
        None,
        1,
        None,
        0.5,
        0.5,
        10,
        10,
        ctx,
        return_detailed=True,
        timeout_bars=1,
    )

    assert len(compact["trades"]) == len(detailed["trades"]) == 1
    assert compact["trades"][0]["pnl_raw"] == pytest.approx(
        detailed["trades"][0]["pnl_raw"]
    )
    assert detailed["trades_detailed"][0]["signal_idx"] == 1
