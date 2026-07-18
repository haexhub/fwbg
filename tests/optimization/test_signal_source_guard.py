"""Regression tests for the signal-source guard and the run-level backstop.

Bug: a `type: "signal"` strategy with no signal source (no signal_rules, no
model.required_features, no allowed_hours/allowed_days) got dispatched, silently
skipped every walk-forward fold with an empty feature pool, and "completed" in a
few seconds as no_successful_folds. Index/seasonality strategies hit this
constantly. Two guards now prevent it:

1. process_symbol returns `no_signal_source` early (before folds).
2. run_optimizer fails the whole run (returns None → exit 1) when no symbol
   actually executed a backtest.
"""
import numpy as np
import pandas as pd
from unittest.mock import patch

from fwbg.core.config import (
    ExitStrategyConfig,
    FilterConfig,
    ModelConfig,
    StrategyConfig,
)
from fwbg.cli.main import _ran_real_backtest
from fwbg.optimization.process import process_symbol


def _ohlc_df(n_rows: int = 1000) -> pd.DataFrame:
    """Minimal OHLC frame large enough to pass the insufficient-data check."""
    idx = pd.date_range("2022-01-01", periods=n_rows, freq="h")
    close = 100 + np.cumsum(np.random.RandomState(1).randn(n_rows) * 0.1)
    return pd.DataFrame(
        {"O": close, "H": close + 0.1, "L": close - 0.1, "C": close},
        index=idx,
    )


def _signal_strategy(**filter_kwargs) -> StrategyConfig:
    return StrategyConfig(
        model=ModelConfig(type="signal", required_features=[]),
        filters=FilterConfig(**filter_kwargs),
        exit_strategies=[
            ExitStrategyConfig(
                name="atr_based",
                params={"tp_mult": 2.0, "sl_mult": 1.0, "atr_period": 14},
                ct=[0.5],
            )
        ],
        signal_rules=None,
    )


class TestSignalSourceGuard:
    def test_no_source_returns_no_signal_source(self):
        """A signal model with no signal source fails fast, before folds."""
        strat = _signal_strategy()  # no allowed_hours, no required_features
        with patch(
            "fwbg.optimization.process.load_data_aligned", return_value=_ohlc_df()
        ):
            result = process_symbol("EURUSD_HOUR_1.csv", strat)
        assert result["status"] == "no_signal_source"

    def test_time_filter_passes_guard(self):
        """allowed_hours is a valid signal source — the guard must not fire."""
        strat = _signal_strategy(allowed_hours=[8, 9, 10])
        # Stop right after the guard so we don't run the full pipeline: force the
        # fold step to bail. Reaching it proves the guard was passed.
        with patch(
            "fwbg.optimization.process.load_data_aligned", return_value=_ohlc_df()
        ), patch(
            "fwbg.optimization.process.create_walk_forward_folds",
            side_effect=ValueError("stop after guard"),
        ):
            result = process_symbol("EURUSD_HOUR_1.csv", strat)
        assert result["status"] != "no_signal_source"
        assert result["status"] == "insufficient_data_for_folds"

    def test_required_features_passes_guard(self):
        """Non-empty model.required_features is a valid signal source too."""
        strat = _signal_strategy()
        strat.model.required_features = ["my_signal_col"]
        with patch(
            "fwbg.optimization.process.load_data_aligned", return_value=_ohlc_df()
        ), patch(
            "fwbg.optimization.process.create_walk_forward_folds",
            side_effect=ValueError("stop after guard"),
        ):
            result = process_symbol("EURUSD_HOUR_1.csv", strat)
        assert result["status"] != "no_signal_source"

    def test_non_signal_model_is_not_gated(self):
        """A non-signal (e.g. ML) model without a signal source is not gated."""
        strat = StrategyConfig(
            model=ModelConfig(type="xgboost", required_features=[]),
            filters=FilterConfig(),
            exit_strategies=[
                ExitStrategyConfig(name="atr_based", params={"tp_mult": 2.0, "sl_mult": 1.0})
            ],
        )
        with patch(
            "fwbg.optimization.process.load_data_aligned", return_value=_ohlc_df()
        ), patch(
            "fwbg.optimization.process.create_walk_forward_folds",
            side_effect=ValueError("stop after guard"),
        ):
            result = process_symbol("EURUSD_HOUR_1.csv", strat)
        assert result["status"] != "no_signal_source"


class TestRanRealBacktest:
    def test_all_no_work_is_false(self):
        results = [
            {"symbol": "A", "status": "no_signal_source"},
            {"symbol": "B", "status": "no_data"},
            {"symbol": "C", "status": "no_successful_folds"},
            None,
        ]
        assert _ran_real_backtest(results) is False

    def test_one_executed_is_true(self):
        assert _ran_real_backtest([
            {"status": "no_data"},
            {"status": "no_edge"},
        ]) is True

    def test_ok_is_true(self):
        assert _ran_real_backtest([{"status": "ok"}]) is True

    def test_no_unified_trades_counts_as_executed(self):
        # Folds ran but produced no trades — a real (if empty) result, not broken.
        assert _ran_real_backtest([{"status": "no_unified_trades"}]) is True

    def test_empty_is_false(self):
        assert _ran_real_backtest([]) is False
