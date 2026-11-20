"""Chart overlays are drawn at the bar they were computed on.

Indicators shift every output column by one bar (``shift_features``) so that a
model never sees bar i's own values at bar i. A chart is a view, not a model
input: an overlay line such as the Supertrend level has to sit at the bar whose
close produced it, so the endpoint undoes that shift for overlay columns.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def ohlc(tmp_path, monkeypatch):
    import fwbg.core.data_sources as ds_mod
    from fwbg.core.data_sources import CSVSourceConfig

    rng = np.random.default_rng(7)
    n = 24 * 12
    close = 100 + np.cumsum(rng.normal(size=n))
    frame = pd.DataFrame(
        {
            "O": close,
            "H": close + rng.random(n),
            "L": close - rng.random(n),
            "C": close,
            "V": 0,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    out = frame.copy()
    out.index.name = "T"
    out.to_csv(tmp_path / "EURUSD_HOUR_1.csv", date_format="%Y-%m-%d %H:%M:%S")
    source = CSVSourceConfig(name="test_src", path=tmp_path)
    monkeypatch.setattr(ds_mod, "_DATA_SOURCES", {"test_src": source})
    return frame


def test_supertrend_overlay_sits_on_the_bar_that_produced_it(ohlc, client):
    from fwbg.plugins import import_plugin_module

    mod = import_plugin_module("fwbg-core", "indicators", "supertrend")
    _, line = mod._supertrend(ohlc["H"], ohlc["L"], ohlc["C"], 14, 3.0)

    resp = client.post(
        "/api/chart/indicator",
        json={
            "fqn": "fwbg-core:supertrend",
            "source": "test_src",
            "symbol": "EURUSD",
            "timeframe": "HOUR_1",
            "params": {"period": 14, "multiplier": 3.0},
            "limit": 1000,
        },
    )
    assert resp.status_code == 200, resp.text
    drawn = pd.Series(resp.json()["overlay_data"]["_st_line"], dtype=float)

    # Row i shows the level computed from the close of bar i. The last bar is
    # blank: the model-facing output never carried its value.
    expected = line.to_numpy()[:-1]
    np.testing.assert_allclose(drawn.to_numpy()[:-1], expected, equal_nan=True)
    assert np.isnan(drawn.iloc[-1])


def test_mtf_overlay_shows_the_previous_finished_candle(ohlc, tmp_path, client):
    """With an indicator timeframe the level of a finished day is held across
    the NEXT day's chart bars, never across the day that produced it."""
    from fwbg.plugins import import_plugin_module

    daily = ohlc.resample("D").agg(
        {"O": "first", "H": "max", "L": "min", "C": "last", "V": "sum"}
    )
    daily.index.name = "T"
    daily.to_csv(tmp_path / "EURUSD_DAY_1.csv", date_format="%Y-%m-%d %H:%M:%S")

    mod = import_plugin_module("fwbg-core", "indicators", "supertrend")
    _, d_line = mod._supertrend(daily["H"], daily["L"], daily["C"], 2, 0.5)

    resp = client.post(
        "/api/chart/indicator",
        json={
            "fqn": "fwbg-core:supertrend",
            "source": "test_src",
            "symbol": "EURUSD",
            "timeframe": "HOUR_1",
            "indicator_timeframe": "DAY_1",
            "params": {"period": 2, "multiplier": 0.5},
            "limit": 1000,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    drawn = pd.Series(
        body["overlay_data"]["_st_line"],
        index=pd.to_datetime(body["timestamps"], unit="ms"),
        dtype=float,
    )
    # days on which the level moved: drawing the same-day level would show
    # d_line[day] instead of d_line[day - 1]
    moved = [d for d in range(3, len(daily)) if d_line.iloc[d] != d_line.iloc[d - 1]]
    assert len(moved) >= 3
    for day in moved:
        rows = drawn[drawn.index.normalize() == daily.index[day]]
        np.testing.assert_allclose(rows.to_numpy(), d_line.iloc[day - 1])
