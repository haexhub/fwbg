"""Tests for the ibs indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

IbsIndicator = import_plugin_module("fwbg-core", "indicators", "ibs").IbsIndicator


def bars(rows):
    o, h, lo, c = zip(*rows)
    return pd.DataFrame({"O": o, "H": h, "L": lo, "C": c},
                        index=pd.date_range("2024-01-02", periods=len(rows), freq="D"))


@pytest.fixture
def out():
    return IbsIndicator().compute(bars([
        (100, 110, 90, 100),   # first bar: no previous bar
        (100, 110, 100, 101),  # IBS 0.1, up close
        (101, 104, 96, 98),    # IBS 0.25, down close
        (98, 106, 97, 105),    # close above the previous high (104)
        (105, 105, 105, 105),  # no range
    ]))


def test_ibs_is_the_close_location_in_the_range(out):
    assert out["ibs"].iloc[1:4].tolist() == pytest.approx([0.1, 0.25, 8 / 9])
    assert np.isnan(out["ibs"].iloc[4])


def test_flags(out):
    assert out["ibs_down"].iloc[1:].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert out["ibs_above_prev_high"].iloc[1:].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_first_bar_is_nan(out):
    assert out[["ibs", "ibs_down", "ibs_above_prev_high"]].iloc[0].isna().all()


def test_prefix_stable(out):
    df = bars([(100, 110, 90, 100), (100, 110, 100, 101), (101, 104, 96, 98), (98, 106, 97, 105)])
    full = IbsIndicator().compute(df)
    prefix = IbsIndicator().compute(df.iloc[:3])
    pd.testing.assert_frame_equal(full.iloc[:3], prefix)
