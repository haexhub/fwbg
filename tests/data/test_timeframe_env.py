"""Regression: optimizer workers must derive timeframe config from the
TIMEFRAME env var.

The optimizer offloads each symbol to a ProcessPoolExecutor. On Python 3.14 the
default multiprocessing start method is "forkserver", so workers re-import
`fwbg.data.config` fresh and do NOT inherit the module-global mutations the CLI
makes in the parent. If the timeframe isn't propagated via the environment, an
M15 backtest silently uses the HOUR defaults (bars_per_hour=1), inflating
test_period_years ~4x and halving the annualized Sharpe. `cli.main` sets
os.environ["TIMEFRAME"] to prevent exactly this — verify a fresh interpreter
honors it.
"""

import os
import subprocess
import sys
import textwrap


def _reimport_with_timeframe(tf: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        """
        import fwbg.data.config as c
        print(c.TIMEFRAME, c.tf_cfg["bars_per_hour"], c.WINDOW_SIZE, c.OOS_SIZE)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TIMEFRAME": tf},
        capture_output=True,
        text=True,
    )


def test_fresh_interpreter_derives_m15_from_env():
    r = _reimport_with_timeframe("MINUTE_15")
    assert r.returncode == 0, r.stderr
    tf, bph, window, oos = r.stdout.split()
    assert tf == "MINUTE_15"
    assert bph == "4"  # NOT the HOUR default of 1 → bars_per_year is correct
    assert window == "50000"
    assert oos == "8000"


def test_fresh_interpreter_defaults_to_hour_without_env():
    # No TIMEFRAME in env → HOUR fallback (unchanged legacy behavior).
    env = {k: v for k, v in os.environ.items() if k != "TIMEFRAME"}
    r = subprocess.run(
        [sys.executable, "-c", "import fwbg.data.config as c; print(c.TIMEFRAME)"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "HOUR"
