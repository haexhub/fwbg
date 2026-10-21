"""Prefix stability: features computed on [start, end) equal those computed on [start, cutoff).

A holdout look (fwbg.optimization.holdout) precomputes raw indicators and runs
the data-loading pipeline on the whole window, so the first holdout bars have
their warm-up. Its development rows then match the development run only if
every row before the cutoff comes out the same when the later rows are cut
off. tests/test_lookahead_bias.py (future scramble) cannot see statistics that
depend on the frame's length or global moments; this cuts the frame instead.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.data_sources import _DATA_SOURCES, register_csv_source
from fwbg.data.loader import run_data_loading
from fwbg.pipeline import get_registry
from fwbg.pipeline.features import compute_indicator_pool, normalize_plugin_name

# Trigger plugin discovery before the parametrize lists are built.
get_registry().auto_discover()

from fwbg_sdk import DATA_LOADER_REGISTRY, INDICATOR_REGISTRY  # noqa: E402

ROWS = 1600
CUT = 1100
# Meta-indicators read other indicators' (or macro) columns; they run inside a pool.
POOL = ["volatility", "regime", "momentum"]
META = ["market_regime", "adversarial_validation"]
MACRO = ["macro_vix", "macro_hyg", "macro_lqd", "macro_spx", "macro_tlt"]


def _ohlcv(rows: int = ROWS, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.normal(0, 0.001, rows))
    opens = close + rng.normal(0, 0.0003, rows)
    return pd.DataFrame(
        {
            "O": opens,
            "H": np.maximum(close, opens) + np.abs(rng.normal(0, 0.0005, rows)),
            "L": np.minimum(close, opens) - np.abs(rng.normal(0, 0.0005, rows)),
            "C": close,
            "V": rng.integers(100, 10000, rows).astype(float),
        },
        index=pd.date_range("2021-01-04", periods=rows, freq="h"),
    )


def _assert_prefix_stable(full: pd.DataFrame, prefix: pd.DataFrame, columns) -> None:
    """Every column of *full* cut at len(prefix) equals *prefix* (NaN equals NaN)."""
    n = len(prefix)
    assert full.index[:n].equals(prefix.index)
    differing = []
    for col in columns:
        if col not in prefix.columns:
            differing.append(f"{col} (missing)")
            continue
        a = pd.to_numeric(full[col].iloc[:n], errors="coerce").to_numpy(float)
        b = pd.to_numeric(prefix[col], errors="coerce").to_numpy(float)
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same.all():
            differing.append(f"{col} ({int((~same).sum())} rows, first {int(np.argmax(~same))})")
    assert not differing, f"not prefix-stable: {differing}"


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return _ohlcv()


@pytest.mark.parametrize("name", sorted(INDICATOR_REGISTRY))
def test_indicator_is_prefix_stable(name, ohlcv):
    if get_registry().get(normalize_plugin_name(name)).stateful:
        pytest.skip("stateful: fit on each fold's training frame, never on the full window")
    cls = INDICATOR_REGISTRY[name]
    full = cls().compute(ohlcv.copy())
    columns = [c for c in full.columns if c not in ohlcv.columns]
    if not columns:
        pytest.skip("meta-indicator without input columns (see the pool test)")
    _assert_prefix_stable(full, cls().compute(ohlcv.iloc[:CUT].copy()), columns)


def test_meta_indicators_in_a_pool_are_prefix_stable(ohlcv):
    rng = np.random.default_rng(5)
    frame = ohlcv.copy()
    for col in MACRO:
        frame[col] = 100 + np.cumsum(rng.normal(0, 0.5, len(frame)))
    base = compute_indicator_pool(frame, indicators=POOL)
    full = compute_indicator_pool(frame, indicators=POOL + META)
    assert set(full.columns) - set(base.columns), "the meta-indicators added no column"
    prefix = compute_indicator_pool(frame.iloc[:CUT], indicators=POOL + META)
    _assert_prefix_stable(full, prefix, full.columns)


# --- data loading: DataSource alignment plus the DATA_LOADING plugins ----------

LOADING_ROWS = 24 * 200  # COT windows: z-score from 1560 bars, 26-week change 3120 bars
LOADING_CUT = 24 * 150
SOURCE = "_prefix_stability"
ITEMS = {
    "VIX_DAY": "vix",
    "TNX_DAY": "tnx",
    "IRX_DAY": "irx",
    "COT_EURUSD_DAY": "cot_eurusd",
}


@pytest.fixture
def csv_source(tmp_path):
    """Synthetic daily series behind a CSV DataSource; COT carries a release calendar."""
    rng = np.random.default_rng(11)
    days = pd.date_range("2020-12-01", periods=260, freq="D")
    for stem in ITEMS:
        close = 20 + np.cumsum(rng.normal(0, 0.3, len(days)))
        frame = pd.DataFrame({"Date": days.strftime("%Y-%m-%d"), "Close": close})
        if stem.startswith("COT_"):
            frame = frame.rename(columns={"Date": "Datetime"})
            frame["release_date"] = (days + pd.Timedelta(days=3, hours=21)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        frame.to_csv(tmp_path / f"{stem}.csv", index=False)
    register_csv_source(name=SOURCE, path=tmp_path)
    yield SOURCE
    _DATA_SOURCES.pop(SOURCE, None)


@pytest.mark.parametrize("plugin", sorted(DATA_LOADER_REGISTRY))
def test_data_loading_is_prefix_stable(plugin, csv_source):
    frame = _ohlcv(LOADING_ROWS, seed=7)
    rng = np.random.default_rng(13)
    # Base columns a provider cache would deliver (jev_signal reads them).
    for col in ("jev_official_is_long_win", "jev_official_is_short_win"):
        frame[col] = rng.random(len(frame))

    def load(df):
        configs = [{"source": csv_source, "name": plugin, "params": {"indicators": dict(ITEMS)}}]
        return run_data_loading(df.copy(), configs)

    full = load(frame)
    assert set(full.columns) - set(frame.columns), f"{plugin} added no column"
    _assert_prefix_stable(full, load(frame.iloc[:LOADING_CUT]), full.columns)
