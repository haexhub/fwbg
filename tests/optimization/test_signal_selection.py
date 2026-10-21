"""Signal-model combo enumeration, inner-window scoring and selection."""

import numpy as np
import pandas as pd

from fwbg.core.config import ExitStrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization.robust_validation import WalkForwardFold
from fwbg.optimization.signal_fold import process_signal_fold
from fwbg.optimization.signal_selection import signal_combos

SIGNALS = ["_composed_signal_long", "_composed_signal_short"]


def _ctx(**overrides) -> SimulationContext:
    defaults = dict(
        symbol="TEST",
        asset_class="forex",
        spread=1.0,
        point=0.01,
        min_trades=2,
        model_type="signal",
        n_inner_folds=2,
        embargo_bars=0,
        exit_strategies=[
            ExitStrategyConfig(name="fixed", params={"tp_mult": 2, "sl_mult": 4}),
            ExitStrategyConfig(name="fixed", params={"tp_mult": 6, "sl_mult": 4}, ct=[0.5, 0.6]),
        ],
    )
    defaults.update(overrides)
    return SimulationContext(**defaults)


def _frame(n_bars, start="2024-01-01"):
    """Flat hourly bars with a long signal every 24th bar."""
    idx = pd.date_range(start, periods=n_bars, freq="h")
    signal = np.zeros(n_bars)
    signal[5::24] = 1.0
    return pd.DataFrame(
        {"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0, SIGNALS[0]: signal, SIGNALS[1]: 0.0},
        index=idx,
    )


def _fold(train_df, test_df):
    return WalkForwardFold(
        fold_id=0,
        train_start=0,
        train_end=len(train_df),
        test_start=len(train_df),
        test_end=len(train_df) + len(test_df),
        train_df=train_df,
        test_df=test_df,
    )


def test_combos_cover_exits_x_hp_x_ct_in_config_order():
    exits = [
        ExitStrategyConfig(
            name="fixed",
            params={"tp_mult": 2, "sl_mult": 4, "timeout_bars": 12},
            exit_modifier="trail",
            exit_modifier_params={"step": 1},
            entry_modifier="delay",
            entry_modifier_params={"bars": 2},
        ),
        ExitStrategyConfig(
            name="fixed",
            params={"tp_mult": 6, "sl_mult": 3},
            ct=[0.5, 0.6],
            long_ct=[0.6],
        ),
    ]
    ctx = _ctx(
        exit_strategies=exits,
        model_hyperparameters={"base": 1},
        grid_model_hyperparameters=[None, {"signal_column_long": "x"}],
    )

    combos = signal_combos(ctx, SIGNALS)

    assert [c.candidate["params"] for c in combos] == [
        (2, 4, 0.5),
        (2, 4, 0.5),
        (6, 3, 0.5),
        (6, 3, 0.6),
        (6, 3, 0.5),
        (6, 3, 0.6),
    ]
    assert [c.candidate["model_hyperparameters"] for c in combos] == [
        {"base": 1},
        {"base": 1, "signal_column_long": "x"},
        {"base": 1},
        {"base": 1},
        {"base": 1, "signal_column_long": "x"},
        {"base": 1, "signal_column_long": "x"},
    ]
    first, last = combos[0], combos[-1]
    assert first.candidate["timeout_bars"] == 12
    assert first.candidate["selected_features_long"] == SIGNALS
    assert first.candidate["exit_modifier_params"] == {"step": 1}
    assert first.ctx.exit_strategy == "fixed"
    assert first.ctx.exit_params == exits[0].params
    assert (first.ctx.exit_modifier, first.ctx.exit_modifier_params) == ("trail", {"step": 1})
    assert (first.ctx.entry_modifier, first.ctx.entry_modifier_params) == ("delay", {"bars": 2})
    assert first.ctx.separate_long_short is False
    assert last.ctx.separate_long_short is True
    assert last.ctx.model_hyperparameters == {"base": 1, "signal_column_long": "x"}
    assert first.config is exits[0] and last.config is exits[1]


def test_signal_fold_grid_entries_keep_the_combo_order():
    train_df, test_df = _frame(960), _frame(240, start="2024-02-10")

    _, grid = process_signal_fold(
        _fold(train_df, test_df), train_df, test_df, SIGNALS, _ctx(), "TEST"
    )

    assert [(e["tp_mult"], e["sl_mult"], e["ct"]) for e in grid] == [
        (2, 4, 0.5),
        (6, 4, 0.5),
        (6, 4, 0.6),
    ]
