"""Indicator timing must survive missing bars and extensions of the input series."""

import numpy as np
import pandas as pd
import pytest

from fwbg.data.loader import load_data_aligned
from fwbg.pipeline.features import compute_indicator_pool


@pytest.mark.parametrize(
    ("name", "params", "column", "frequency"),
    [
        (
            "opening_range",
            {"sessions": [8], "range_bars": 2},
            "rb2_orb_s08_range",
            "15min",
        ),
        ("weekly_opening_range", {"range_bars": 2}, "wor_rb2_range", "15min"),
        ("supertrend_mtf", {"period": 5, "d1_bars": 4}, "_st_d1_line", "15min"),
        ("volatility", {}, "vol_rv_20", "h"),
    ],
)
def test_appending_sparse_bars_keeps_earlier_features(name, params, column, frequency):
    index = pd.date_range("2025-01-06 08:00", periods=600, freq=frequency)
    close = 100.0 + np.sin(np.arange(len(index)))
    history = pd.DataFrame(
        {"O": close, "H": close + 1, "L": close - 1, "C": close}, index=index
    )
    later_index = pd.date_range("2025-03-01", periods=72 * 24, freq="h")
    later_index = later_index[later_index.hour < 6]
    later = pd.DataFrame(
        {"O": 100.0, "H": 101.0, "L": 99.0, "C": 100.0}, index=later_index
    )
    if frequency == "15min":
        later = pd.DataFrame(
            {"O": 100.0, "H": 101.0, "L": 99.0, "C": 100.0},
            index=pd.date_range("2025-03-01", periods=len(history) * 2, freq="30min"),
        )
    configs = [{"name": name, "params": params}]
    past = compute_indicator_pool(history, configs)[column]
    extended = compute_indicator_pool(pd.concat([history, later]), configs)[column]
    assert past.notna().any()
    pd.testing.assert_series_equal(past, extended.loc[past.index], check_freq=False)


def test_resampled_breakout_uses_nominal_bar_duration_despite_a_leading_gap(tmp_path):
    index = pd.date_range("2025-01-06", periods=192, freq="15min").delete(
        [1, 2, 3, 5, 6, 7]
    )
    bars = pd.DataFrame({"O": 100.0, "H": 101.0, "L": 99.0, "C": 100.0}, index=index)
    bars.loc["2025-01-07 09:45", ["H", "C"]] = [111.0, 110.0]
    csv = tmp_path / "EURUSD_MINUTE_15.csv"
    bars.rename_axis("T").to_csv(csv)
    bars = load_data_aligned(str(csv))
    columns = [
        {"name": "ema", "params": {"lines": [{"period": 2}]}},
        {"name": "previous_day_levels", "params": {"resample_tf": "1h"}},
    ]
    out = compute_indicator_pool(bars, columns)
    breaks = out["hl_ses_pdl_high_break"]
    assert breaks.index[breaks.eq(1)].tolist() == [pd.Timestamp("2025-01-07 10:00")]
    cutoff = pd.Timestamp("2025-01-07 09:30")
    partial = compute_indicator_pool(bars.loc[:cutoff], columns)
    pd.testing.assert_series_equal(partial[breaks.name], breaks.loc[:cutoff])
