"""Hash-seed independence: the same configuration gives the same backtest in every process.

Python randomises str hashing per process (PYTHONHASHSEED), so iterating a set of
column names yields a different order in each process. orb_based used to take
the first ``*_sl_dist`` column, so a hash-ordered column layout changed trades
between runs; it now refuses several candidates, and the backtest below names
its stop column. Each check runs in fresh interpreters with different hash seeds.
"""

import json
import os
import subprocess
import sys

import pytest

SEEDS = ("0", "5", "12345")

ORB_PARAMS = {
    "range_bars": [4],
    "sessions": [0, 8],
    "enable_retracement": True,
    "pre_range_bars": [2],
    "candle_span": "body",
    "retracement_levels": 0.5,
}

OPENING_RANGE_COLUMNS = """
import json, sys
import numpy as np, pandas as pd
from fwbg.plugins import import_plugin_module

n = 2000
close = 100 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.001, n)))
frame = pd.DataFrame(
    {"O": close, "H": close * 1.002, "L": close * 0.998, "C": close},
    index=pd.date_range("2024-01-01", periods=n, freq="15min"),
)
orb = import_plugin_module("fwbg-core", "indicators", "opening_range")
out = orb.OpeningRangeIndicator().compute(frame, **json.loads(sys.argv[2]))
json.dump(list(out.columns), open(sys.argv[1], "w"))
"""

ORB_BACKTEST = """
import json, os, sys
os.environ["TIMEFRAME"] = "MINUTE_15"
import numpy as np, pandas as pd
from fwbg.core.config import StrategyConfig
from fwbg.data import config as data_config
from fwbg.optimization import process

n = 12000
rng = np.random.default_rng(3)
close = 7000 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
opn = close * (1 + rng.normal(0, 0.0005, n))
frame = pd.DataFrame(
    {
        "O": opn,
        "H": np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.001, n))),
        "L": np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.001, n))),
        "C": close,
    },
    index=pd.date_range("2023-01-02", periods=n, freq="15min"),
)
process.load_data_aligned = lambda path: frame
tf_cfg = data_config.resolve_tf_config("MINUTE_15")
data_config.TIMEFRAME = "MINUTE_15"
data_config.tf_cfg = tf_cfg
data_config.OOS_SIZE = tf_cfg["oos_size"]
data_config.WINDOW_SIZE = tf_cfg["window_size"]

breakout = "rb4_prb2_orb_s00_breakout_"
strategy = StrategyConfig.from_dict({
    "name": "hash-seed",
    "timeframe": "MINUTE_15",
    "pipeline": {"indicators": [{"name": "opening_range", "params": json.loads(sys.argv[2])}]},
    "exit_strategies": [{
        "name": "orb_based",
        "params": {"tp_mode": "range", "tp_mult": 0.8, "sl_mult": 2,
                   "min_tp_pips": 1, "min_sl_pips": 1,
                   "sl_dist_column": "rb4_prb2_orb_s00_sl_dist"},
        "ct": [0.5],
    }],
    "model": {"type": "signal", "trade_directions": ["long", "short"]},
    "signal_rules": {
        "long": {"conditions": [{"type": "signal_active", "column": breakout + "up"}]},
        "short": {"conditions": [{"type": "signal_active", "column": breakout + "down"}]},
    },
    "validation": {"method": "walk_forward", "folds": 4},
})
result = process.process_symbol("ASX200_MINUTE_15.csv", strategy)
folds = result.get("walk_forward") or {}
json.dump(
    [result.get("status"), folds.get("total_trades"), folds.get("mean_win_rate")],
    open(sys.argv[1], "w"),
)
"""


def _run_with_hash_seed(script: str, seed: str, params: dict, out_path) -> object:
    env = {
        **os.environ,
        "PYTHONHASHSEED": seed,
        "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
    }
    subprocess.run(
        [sys.executable, "-c", script, str(out_path), json.dumps(params)],
        env=env,
        check=True,
        capture_output=True,
        timeout=300,
    )
    return json.loads(out_path.read_text())


def test_opening_range_column_order_is_independent_of_the_hash_seed(tmp_path):
    orders = {
        seed: _run_with_hash_seed(OPENING_RANGE_COLUMNS, seed, ORB_PARAMS, tmp_path / seed)
        for seed in SEEDS
    }
    assert orders["5"] == orders["0"]
    assert orders["12345"] == orders["0"]
    sl_dist = [c for c in orders["0"] if c.endswith("_sl_dist")]
    # The plugin's feature order: the session's own stop distance comes before
    # its retracement variant.
    assert sl_dist[:2] == ["rb4_prb2_orb_s00_sl_dist", "rb4_prb2_orb_s00_rl50_sl_dist"]


def test_orb_backtest_is_independent_of_the_hash_seed(tmp_path):
    pytest.importorskip("fwbg_premium", reason="orb_based exit strategy is a premium plugin")
    params = {**ORB_PARAMS, "sessions": [0]}
    results = {
        seed: _run_with_hash_seed(ORB_BACKTEST, seed, params, tmp_path / seed) for seed in SEEDS
    }
    assert results["0"][1] > 0
    assert results["5"] == results["0"]
    assert results["12345"] == results["0"]
