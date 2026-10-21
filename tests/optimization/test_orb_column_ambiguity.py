"""An ORB run whose exit cannot tell which stop column to use ends as that symbol's error.

With retracement enabled the opening_range plugin emits both the session's
``*_sl_dist`` and its ``*_rl50_sl_dist``; orb_based refuses to guess between
them, and the run reports the refusal instead of trading on either one.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.data import config as data_config
from fwbg.optimization import process

pytest.importorskip("fwbg_premium", reason="orb_based exit strategy is a premium plugin")

BREAKOUT = "rb4_prb2_orb_s00_breakout_"
ORB = {
    "range_bars": [4],
    "sessions": [0],
    "enable_retracement": True,
    "pre_range_bars": [2],
    "candle_span": "body",
    "retracement_levels": 0.5,
}


def _m15_frame(rows: int = 12000) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    close = 7000 * np.exp(np.cumsum(rng.normal(0, 0.0015, rows)))
    opn = close * (1 + rng.normal(0, 0.0005, rows))
    return pd.DataFrame(
        {
            "O": opn,
            "H": np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.001, rows))),
            "L": np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.001, rows))),
            "C": close,
        },
        index=pd.date_range("2023-01-02", periods=rows, freq="15min"),
    )


def _orb_exit(**exit_params) -> dict:
    return {
        "name": "orb_based",
        "params": {"tp_mode": "range", "tp_mult": 0.8, "sl_mult": 2, **exit_params},
        "ct": [0.5],
    }


ATR_EXIT = {"name": "atr_based", "params": {"tp_mult": 2.0, "sl_mult": 1.0}, "ct": [0.5]}


def _strategy(model: dict, exits: list[dict]) -> StrategyConfig:
    return StrategyConfig.from_dict(
        {
            "name": "orb-ambiguity",
            "timeframe": "MINUTE_15",
            "pipeline": {"indicators": [{"name": "opening_range", "params": ORB}]},
            "exit_strategies": exits,
            "model": model,
            "signal_rules": {
                "long": {"conditions": [{"type": "signal_active", "column": BREAKOUT + "up"}]},
                "short": {"conditions": [{"type": "signal_active", "column": BREAKOUT + "down"}]},
            },
            "validation": {"method": "walk_forward", "folds": 4},
        }
    )


@pytest.fixture
def m15_run(monkeypatch):
    tf_cfg = data_config.resolve_tf_config("MINUTE_15")
    monkeypatch.setenv("TIMEFRAME", "MINUTE_15")
    monkeypatch.setattr(data_config, "TIMEFRAME", "MINUTE_15")
    monkeypatch.setattr(data_config, "tf_cfg", tf_cfg)
    monkeypatch.setattr(data_config, "OOS_SIZE", tf_cfg["oos_size"])
    monkeypatch.setattr(data_config, "WINDOW_SIZE", tf_cfg["window_size"])
    frame = _m15_frame()
    monkeypatch.setattr(process, "load_data_aligned", lambda path: frame)
    return lambda strategy: process.process_symbol("ASX200_MINUTE_15.csv", strategy)


SIGNAL = {"type": "signal", "trade_directions": ["long", "short"]}


def test_ambiguous_stop_column_is_the_symbols_error(m15_run):
    result = m15_run(_strategy(SIGNAL, [_orb_exit()]))
    assert result["status"] == "error"
    assert "sl_dist_column" in result["error"]
    assert "rb4_prb2_orb_s00_rl50_sl_dist" in result["error"]


def test_explicit_stop_column_runs(m15_run):
    result = m15_run(_strategy(SIGNAL, [_orb_exit(sl_dist_column="rb4_prb2_orb_s00_sl_dist")]))
    assert result["status"] != "error"
    assert result["walk_forward"]["total_trades"] > 0


@pytest.mark.parametrize(
    "exits",
    [[_orb_exit()], [ATR_EXIT, _orb_exit()]],
    ids=["feature-selection-targets", "grid-search-combo"],
)
def test_ambiguous_stop_column_is_the_symbols_error_for_a_trained_model(m15_run, exits):
    result = m15_run(_strategy({"type": "xgboost"}, exits))
    assert result["status"] == "error"
    assert "sl_dist_column" in result["error"]
