"""Timing and publication-lag regressions for the monthly macro overlays."""

import numpy as np
import pandas as pd
import pytest

from fwbg.portfolio.engine import PortfolioSettings, run_portfolio
from fwbg.portfolio.overlay import (
    OverlaySettings,
    _component,
    change_over_weeks,
    exposure_mask,
    overlay_condition,
    read_series,
)


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
    assert weights.loc[(pd.Timestamp("2020-03-31"), "trend"), "cash"] == pytest.approx(
        1.0
    )
    assert weights.loc[(pd.Timestamp("2020-06-30"), "trend"), "cash"] < 1.0
    assert result.equity["trend"].iloc[-1] > result.equity["trend"].iloc[0]


def test_overlay_settings_reject_unknown_names():
    """Only declared overlays are accepted."""
    with pytest.raises(ValueError):
        OverlaySettings(name="unknown").validate()


def _rising_prices(start, end):
    dates = pd.bdate_range(start, end)
    prices = pd.DataFrame({"A": 100.0 + np.arange(len(dates)), "B": 100.0}, index=dates)
    return prices


def _settings(overlay):
    return PortfolioSettings(
        weights={"A": 0.5, "B": 0.5},
        equity_benchmark="A",
        lookback_months=2,
        one_way_cost_bps=0,
        overlay=overlay,
    )


def _cash(result, date, portfolio):
    weights = result.weights.set_index(["date", "portfolio"])
    return weights.loc[(pd.Timestamp(date), portfolio), "cash"]


def test_exposure_mask_output_reaches_the_engine_without_a_second_delay():
    """Jan=True and Feb=False: the March rebalance session already holds cash."""
    prices = _rising_prices("2019-10-01", "2020-04-30")
    condition = pd.Series(
        [True, True, True, True, False, True],
        index=pd.DatetimeIndex(
            [
                "2019-10-31",
                "2019-11-30",
                "2019-12-31",
                "2020-01-31",
                "2020-02-29",
                "2020-03-31",
            ]
        ),
    )
    mask = exposure_mask(condition, prices.index)
    assert mask.loc["2020-02-28"]
    assert not mask.loc["2020-03-02"]
    result = run_portfolio(prices, _settings(mask))
    for portfolio in ("trend", "static", "equity_buy_hold"):
        assert _cash(result, "2020-02-28", portfolio) < 1.0
        assert _cash(result, "2020-03-02", portfolio) == pytest.approx(1.0)
        assert _cash(result, "2020-03-31", portfolio) == pytest.approx(1.0)
        assert _cash(result, "2020-04-01", portfolio) < 1.0


def test_equity_buy_hold_follows_the_overlay_out_and_back_in():
    """A false first month starts in cash, the book re-enters later and exits again."""
    prices = _rising_prices("2019-10-01", "2020-06-30")
    dates = prices.index
    overlay = pd.Series(True, index=dates)
    overlay.loc["2019-12-01":"2020-01-31"] = False
    overlay.loc["2020-03-01":"2020-03-31"] = False
    result = run_portfolio(prices, _settings(overlay))
    first = result.equity.index[1]
    assert _cash(result, first, "equity_buy_hold") == pytest.approx(1.0)
    assert _cash(result, "2020-02-03", "equity_buy_hold") < 1.0
    assert _cash(result, "2020-03-02", "equity_buy_hold") == pytest.approx(1.0)
    assert _cash(result, "2020-04-01", "equity_buy_hold") < 1.0


def test_equity_buy_hold_is_not_rebalanced_without_an_overlay():
    """Without an overlay buy-and-hold trades only once."""
    prices = _rising_prices("2019-10-01", "2020-06-30")
    result = run_portfolio(prices, _settings(None))
    trades = result.trades[result.trades["portfolio"] == "equity_buy_hold"]
    assert len(trades) == 1


def _write(path, name, rows):
    frame = pd.DataFrame(rows, columns=["observation_date", name])
    frame.to_csv(path / f"{name}.csv", index=False)


def test_net_liquidity_converts_rrp_from_billions(tmp_path):
    """WALCL +50,000 million with RRP +100 billion is a net drain, not a gain."""
    _write(tmp_path, "WALCL", [("2020-01-01", 1_000_000), ("2020-04-01", 1_050_000)])
    _write(tmp_path, "WTREGEN", [("2020-01-01", 100_000), ("2020-04-01", 100_000)])
    _write(tmp_path, "RRPONTSYD", [("2020-01-01", 100.0), ("2020-04-01", 200.0)])
    settings = OverlaySettings(name="fed_net_liquidity", weeks=13, lag_days=0)
    condition = overlay_condition(settings, tmp_path)
    assert not condition.loc["2020-04-01"]


def test_change_is_measured_over_elapsed_weeks_not_rows():
    """Thirteen monthly rows are not thirteen weeks."""
    index = pd.date_range("2019-01-31", periods=14, freq="ME")
    level = pd.Series(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7], index=index, dtype=float
    )
    change = change_over_weeks(level, 13)
    assert change.iloc[-1] < 0
    daily = pd.Series(np.arange(200.0), index=pd.bdate_range("2020-01-01", periods=200))
    assert change_over_weeks(daily, 13).iloc[-1] == pytest.approx(65.0)


def test_configured_lag_is_a_floor_for_every_component(tmp_path):
    """lag_days delays every series; the Japan yield never drops below its 60-day floor."""
    _write(tmp_path, "DGS10", [("2020-01-01", 3.0)])
    _write(tmp_path, "IRLTLT01JPM156N", [("2020-01-01", 1.0)])
    published = {}
    for lag in (0, 5, 120):
        settings = OverlaySettings("us_japan_10y_spread", lag_days=lag)
        published[lag] = {
            name: _component(settings, tmp_path, name).index[0]
            - pd.Timestamp("2020-01-01")
            for name in ("DGS10.csv", "IRLTLT01JPM156N.csv")
        }
    assert published[0] == {
        "DGS10.csv": pd.Timedelta(days=0),
        "IRLTLT01JPM156N.csv": pd.Timedelta(days=60),
    }
    assert published[5]["DGS10.csv"] == pd.Timedelta(days=5)
    assert published[5]["IRLTLT01JPM156N.csv"] == pd.Timedelta(days=60)
    assert published[120] == {
        "DGS10.csv": pd.Timedelta(days=120),
        "IRLTLT01JPM156N.csv": pd.Timedelta(days=120),
    }


def test_threshold_must_be_a_finite_number():
    """NaN, infinity and booleans are rejected."""
    for bad in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            OverlaySettings(name="fed_balance_sheet", threshold=bad).validate()
