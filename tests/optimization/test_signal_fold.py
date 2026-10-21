"""Signal folds choose their combo on the training window, never on the test fold.

Synthetic hourly bars with a long signal every 24th bar. In the training data
each signal is followed by +3 then -5 (TP 2 / SL 4 wins, TP 6 / SL 4 loses);
in the test data by +8 (both win, TP 6 by more). Selecting on the test fold
would pick TP 6, selecting on the training window picks TP 2.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import ExitStrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.optimization import signal_fold, signal_selection
from fwbg.optimization.nested_cv import evaluate_on_holdout
from fwbg.optimization.robust_validation import WalkForwardFold
from fwbg.optimization.signal_fold import process_signal_fold
from fwbg.optimization.signal_selection import inner_windows, signal_combos

SIGNALS = ["_composed_signal_long", "_composed_signal_short"]
SPREAD = 1.0
# The simulator fills a long at the next bar's open + spread + slippage (0.5 x spread).
ENTRY = 100.0 + 1.5 * SPREAD
TRAIN_DAYS, TEST_DAYS = 40, 15


def _ctx() -> SimulationContext:
    return SimulationContext(
        symbol="TEST",
        asset_class="forex",
        spread=SPREAD,
        point=0.01,
        min_trades=2,
        model_type="signal",
        n_inner_folds=2,
        embargo_bars=0,
        exit_strategies=[
            ExitStrategyConfig(name="fixed", params={"tp_mult": 2, "sl_mult": 4}),
            ExitStrategyConfig(name="fixed", params={"tp_mult": 6, "sl_mult": 4}),
        ],
    )


def _bars(n_days, start, rise, drop=0.0, signals=True):
    """Flat bars at 100; a signal at hour 5 is followed by ENTRY+rise, then ENTRY-drop.

    A spike at hour 15 reaches every TP, so each combo has training targets and the
    model is trained regardless of how its signals played out.
    """
    idx = pd.date_range(start, periods=n_days * 24, freq="h")
    high = np.full(len(idx), 100.0)
    low = np.full(len(idx), 100.0)
    signal = np.zeros(len(idx))
    for day_start in range(0, len(idx), 24):
        signal_bar = day_start + 5
        signal[signal_bar] = 1.0 if signals else 0.0
        high[signal_bar + 2] = ENTRY + rise
        if drop:
            low[signal_bar + 3] = ENTRY - drop
        high[day_start + 15] = ENTRY + 8.5
    return pd.DataFrame(
        {"O": 100.0, "H": high, "L": low, "C": 100.0, SIGNALS[0]: signal, SIGNALS[1]: 0.0},
        index=idx,
    )


def _train(signals=True):
    return _bars(TRAIN_DAYS, "2024-01-01", rise=3, drop=5, signals=signals)


def _test(rise=8, drop=0.0, signals=True):
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=TRAIN_DAYS)
    return _bars(TEST_DAYS, start, rise=rise, drop=drop, signals=signals)


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


def _run(train_df, test_df):
    return process_signal_fold(_fold(train_df, test_df), train_df, test_df, SIGNALS, _ctx(), "TEST")


def _tp2_tp6():
    tp2, tp6 = signal_combos(_ctx(), SIGNALS)
    assert tp2.candidate["params"][0] == 2 and tp6.candidate["params"][0] == 6
    return tp2, tp6


def _pnl(holdout_df, inner_df, combo):
    result = evaluate_on_holdout(holdout_df, inner_df, combo.candidate, combo.ctx)
    assert result["n_trades"] > 0
    return result["pnl"]


def _assert_train_prefers_tp2(train_df):
    tp2, tp6 = _tp2_tp6()
    for inner_train, val_df in inner_windows(train_df, _ctx()):
        assert _pnl(val_df, inner_train, tp2) > 0 > _pnl(val_df, inner_train, tp6)


def _assert_test_prefers_tp6(train_df, test_df):
    tp2, tp6 = _tp2_tp6()
    assert _pnl(test_df, train_df, tp6) > _pnl(test_df, train_df, tp2) > 0


def _spy(monkeypatch):
    calls = []

    def wrap(real):
        def spy(holdout_df, inner_df, candidate, ctx):
            calls.append((holdout_df.index, inner_df.index, candidate["params"]))
            return real(holdout_df, inner_df, candidate, ctx)

        return spy

    for module in (signal_fold, signal_selection):
        monkeypatch.setattr(module, "evaluate_on_holdout", wrap(module.evaluate_on_holdout))
    return calls


def _touching(calls, test_df):
    return [
        params
        for holdout_idx, inner_idx, params in calls
        if holdout_idx.isin(test_df.index).any() or inner_idx.isin(test_df.index).any()
    ]


def test_choice_ignores_the_test_fold():
    train_df = _train()
    test_a, test_b = _test(rise=8), _test(rise=3, drop=5)
    _assert_train_prefers_tp2(train_df)
    _assert_test_prefers_tp6(train_df, test_a)
    tp2, tp6 = _tp2_tp6()
    assert _pnl(test_b, train_df, tp2) > _pnl(test_b, train_df, tp6)

    result_a, _ = _run(train_df, test_a)
    result_b, _ = _run(train_df, test_b)

    assert result_a["best_config"] == result_b["best_config"]
    assert result_a["inner_val_pnl"] == result_b["inner_val_pnl"]


def test_train_best_combo_is_chosen_even_if_another_wins_on_test():
    train_df, test_df = _train(), _test()
    _assert_train_prefers_tp2(train_df)
    _assert_test_prefers_tp6(train_df, test_df)
    tp2, _ = _tp2_tp6()

    result, _ = _run(train_df, test_df)

    assert (result["best_config"]["tp"], result["best_config"]["sl"]) == (2, 4)
    assert result["test_pnl"] == _pnl(test_df, train_df, tp2)


def test_test_fold_is_evaluated_once(monkeypatch):
    train_df, test_df = _train(), _test()
    _assert_train_prefers_tp2(train_df)
    _assert_test_prefers_tp6(train_df, test_df)
    calls = _spy(monkeypatch)

    _run(train_df, test_df)

    assert len(calls) == 2 * 2 + 1  # combos x inner windows + one test evaluation
    assert _touching(calls, test_df) == [(2, 4, 0.5)]


def test_no_inner_trades_returns_no_fold_and_never_touches_test(monkeypatch):
    train_df, test_df = _train(signals=False), _test()
    tp2, _ = _tp2_tp6()
    assert _pnl(test_df, train_df, tp2) > 0  # the test fold would trade
    calls = _spy(monkeypatch)

    result, grid = _run(train_df, test_df)

    assert result is None
    assert [e["inner_trades"] for e in grid] == [0, 0]
    assert calls and _touching(calls, test_df) == []


def test_chosen_combo_without_test_trades_keeps_the_fold():
    train_df, test_df = _train(), _test(signals=False)
    _assert_train_prefers_tp2(train_df)

    result, _ = _run(train_df, test_df)

    assert result is not None
    assert result["best_config"]["tp"] == 2
    assert (result["test_trades"], result["test_pnl"]) == (0, 0)


def test_fold_result_reports_inner_metrics():
    train_df, test_df = _train(), _test()
    _assert_train_prefers_tp2(train_df)
    _assert_test_prefers_tp6(train_df, test_df)
    tp2, _ = _tp2_tp6()
    windows = inner_windows(train_df, _ctx())
    inner_mean = np.mean([_pnl(val_df, inner_train, tp2) for inner_train, val_df in windows])

    result, grid = _run(train_df, test_df)

    assert result["inner_val_pnl"] == pytest.approx(inner_mean)
    assert result["inner_val_pnl"] != result["test_pnl"]
    assert result["inner_fold_stability"] == 1.0
    assert result["n_candidates"] == 2
    assert [e["fold_id"] for e in grid] == [0, 0]
    assert grid[0]["inner_val_pnl"] == pytest.approx(inner_mean)
