"""Timing and publication-lag regressions for the monthly macro overlays."""

import numpy as np
import pandas as pd
import pytest

from fwbg.portfolio.engine import PortfolioSettings, run_portfolio
from fwbg.portfolio.overlay import OverlaySettings, exposure_mask, read_series


def test_month_end_value_applies_from_the_next_session():
    """A month-end decision must not reach the month-end session itself."""
    condition = pd.Series(
        [True, False],
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"]),
    )
    index = pd.bdate_range("2020-01-30", "2020-03-05")
    mask = exposure_mask(condition, index)
    assert not mask.loc["2020-01-31"]
    assert mask.loc["2020-02-03"]
    assert mask.loc["2020-02-28"]
    assert not mask.loc["2020-03-02"]


def test_read_series_shifts_values_to_publication_date(tmp_path):
    """Values are only visible after the configured publication lag."""
    path = tmp_path / "WALCL.csv"
    path.write_text("observation_date,WALCL\n2020-01-01,100\n2020-01-08,110\n")
    series = read_series(path, lag_days=9)
    assert series.index[0] == pd.Timestamp("2020-01-10")
    assert series.iloc[0] == 100


def test_overlay_sets_the_book_to_cash_when_the_condition_is_false():
    """A false overlay month leaves every sleeve in cash for that month."""
    dates = pd.bdate_range("2020-01-01", "2020-06-30")
    prices = pd.DataFrame({"A": 100.0 + np.arange(len(dates)), "B": 100.0}, index=dates)
    overlay = pd.Series(False, index=dates)
    overlay.loc["2020-04-01":] = True
    settings = PortfolioSettings(
        weights={"A": 0.5, "B": 0.5},
        equity_benchmark="A",
        lookback_months=2,
        one_way_cost_bps=0,
        overlay=overlay,
    )
    result = run_portfolio(prices, settings)
    weights = result.weights.set_index(["date", "portfolio"])
    assert weights.loc[(pd.Timestamp("2020-03-31"), "trend"), "cash"] == pytest.approx(1.0)
    assert weights.loc[(pd.Timestamp("2020-06-30"), "trend"), "cash"] < 1.0
    assert result.equity["trend"].iloc[-1] > result.equity["trend"].iloc[0]


def test_overlay_settings_reject_unknown_names():
    """Only declared overlays are accepted."""
    with pytest.raises(ValueError):
        OverlaySettings(name="unknown").validate()
