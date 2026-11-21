"""Hash-seed independence: the same configuration gives the same backtest in every process.

Python randomises str hashing per process (PYTHONHASHSEED), so iterating a set of
column names yields a different order in each process. A hash-ordered column layout
could change which ``*_sl_dist`` column an exit strategy picks, so the indicator's
output order must not depend on the hash seed. Each check runs in fresh interpreters
with different hash seeds.
"""

import json
import os
import subprocess
import sys


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
        seed: _run_with_hash_seed(
            OPENING_RANGE_COLUMNS, seed, ORB_PARAMS, tmp_path / seed
        )
        for seed in SEEDS
    }
    assert orders["5"] == orders["0"]
    assert orders["12345"] == orders["0"]
    sl_dist = [c for c in orders["0"] if c.endswith("_sl_dist")]
    # The plugin's feature order: the session's own stop distance comes before
    # its retracement variant.
    assert sl_dist[:2] == ["rb4_prb2_orb_s00_sl_dist", "rb4_prb2_orb_s00_rl50_sl_dist"]
