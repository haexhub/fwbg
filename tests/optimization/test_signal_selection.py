"""Signal-model combo enumeration, inner-window scoring and selection."""

import numpy as np
import pandas as pd

from fwbg.core.config import ExitStrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization import signal_selection
from fwbg.optimization.nested_cv import nested_cv_split
from fwbg.optimization.robust_validation import WalkForwardFold
from fwbg.optimization.signal_fold import process_signal_fold
from fwbg.optimization.signal_selection import (
    inner_windows,
    select_combo,
    signal_combos,
)

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


def _three_exits_ctx(**overrides):
    exits = [
        ExitStrategyConfig(name="fixed", params={"tp_mult": tp, "sl_mult": 4}) for tp in (2, 4, 6)
    ]
    return _ctx(exit_strategies=exits, **overrides)


def _canned(monkeypatch, results):
    """Two inner windows; evaluate_on_holdout answers (pnl, n_trades) per TP and window."""
    frame = _frame(300)
    windows = [(frame.iloc[:100], frame.iloc[100:200]), (frame.iloc[:200], frame.iloc[200:300])]
    starts = [val.index[0] for _, val in windows]

    def fake(holdout_df, inner_df, candidate, ctx):
        pnl, n_trades = results[candidate["params"][0]][starts.index(holdout_df.index[0])]
        return {
            "trades": [],
            "trades_detailed": [],
            "pnl": pnl,
            "win_rate": 0,
            "n_trades": n_trades,
        }

    monkeypatch.setattr(signal_selection, "evaluate_on_holdout", fake)
    return windows


def _chosen_tp(choice):
    combo, _ = choice
    return combo.candidate["params"][0]


def test_windows_match_the_ml_split():
    train_df = _frame(960)
    ctx = _ctx(n_inner_folds=3, embargo_bars=5)

    windows = inner_windows(train_df, ctx)

    expected = nested_cv_split(train_df, 0.0, 3, embargo_bars=5)["inner_folds"]
    assert len(windows) == len(expected) >= 2
    for (inner_train, val), (exp_train, exp_val) in zip(windows, expected):
        pd.testing.assert_frame_equal(inner_train, exp_train)
        pd.testing.assert_frame_equal(val, exp_val)
        assert inner_train.index.isin(train_df.index).all()
        assert val.index.isin(train_df.index).all()


def test_picks_highest_mean_inner_pnl(monkeypatch):
    windows = _canned(monkeypatch, {2: [(1, 3), (1, 3)], 4: [(4, 3), (-1, 3)], 6: [(0, 3), (2, 3)]})

    choice, _ = select_combo(signal_combos(_three_exits_ctx(), SIGNALS), windows)

    assert _chosen_tp(choice) == 4
    assert choice[1]["inner_val_pnl"] == 1.5


def test_window_without_trades_counts_as_zero(monkeypatch):
    windows = _canned(monkeypatch, {2: [(3, 2), (0, 0)], 4: [(2, 2), (2, 2)], 6: [(0, 0), (0, 0)]})

    choice, entries = select_combo(signal_combos(_three_exits_ctx(), SIGNALS), windows)

    assert _chosen_tp(choice) == 4
    assert entries[0]["inner_val_pnl"] == 1.5
    assert entries[0]["fold_pnls"] == [3, 0]


def test_combo_without_inner_trades_is_ineligible(monkeypatch):
    windows = _canned(
        monkeypatch, {2: [(0, 0), (0, 0)], 4: [(-1, 2), (-1, 2)], 6: [(-3, 2), (0, 0)]}
    )

    choice, _ = select_combo(signal_combos(_three_exits_ctx(), SIGNALS), windows)

    # No sign gate: the best eligible combo is chosen even though it loses.
    assert _chosen_tp(choice) == 4
    assert choice[1]["inner_val_pnl"] == -1


def test_no_eligible_combo_returns_none(monkeypatch):
    windows = _canned(monkeypatch, {tp: [(0, 0), (0, 0)] for tp in (2, 4, 6)})

    choice, entries = select_combo(signal_combos(_three_exits_ctx(), SIGNALS), windows)

    assert choice is None
    assert len(entries) == 3


def test_no_windows_returns_none():
    choice, entries = select_combo(signal_combos(_three_exits_ctx(), SIGNALS), [])

    assert choice is None
    assert [e["inner_trades"] for e in entries] == [0, 0, 0]


def test_tie_keeps_config_order(monkeypatch):
    windows = _canned(monkeypatch, {2: [(1, 1), (1, 1)], 4: [(2, 1), (0, 1)], 6: [(0, 1), (2, 1)]})

    choice, _ = select_combo(signal_combos(_three_exits_ctx(), SIGNALS), windows)

    assert _chosen_tp(choice) == 2


def test_grid_entries_carry_inner_metrics(monkeypatch):
    windows = _canned(monkeypatch, {2: [(3, 2), (-1, 1)], 4: [(1, 1), (1, 1)], 6: [(0, 0), (0, 0)]})
    ctx = _three_exits_ctx(model_hyperparameters={"signal_column_long": "x"})

    _, entries = select_combo(signal_combos(ctx, SIGNALS), windows)

    assert entries[0] == {
        "exit_strategy": "fixed",
        "tp_mult": 2,
        "sl_mult": 4,
        "ct": 0.5,
        "timeout_bars": None,
        "model_hyperparameters": {"signal_column_long": "x"},
        "inner_val_pnl": 1.0,
        "inner_trades": 3,
        "fold_pnls": [3, -1],
        "fold_stability": 0.5,
    }
    assert all("pnl" not in e for e in entries)


def test_scoring_only_sees_train_rows(monkeypatch):
    frame = _frame(1200)
    train_df, test_df = frame.iloc[:960], frame.iloc[960:]
    seen = []
    real = signal_selection.evaluate_on_holdout

    def spy(holdout_df, inner_df, candidate, ctx):
        seen.extend([holdout_df.index, inner_df.index])
        return real(holdout_df, inner_df, candidate, ctx)

    monkeypatch.setattr(signal_selection, "evaluate_on_holdout", spy)
    ctx = _ctx()

    select_combo(signal_combos(ctx, SIGNALS), inner_windows(train_df, ctx))

    assert len(seen) == 3 * 2 * 2  # combos x windows x (val, inner train)
    assert all(idx.isin(train_df.index).all() for idx in seen)
    assert not any(idx.isin(test_df.index).any() for idx in seen)
