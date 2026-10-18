"""Accounting, stop ordering, corporate actions and causal daily signals."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fwbg.portfolio.swing import (
    RULES,
    SwingSettings,
    indicators,
    rebalance,
    run_swing,
    validate_bars,
)
from fwbg.portfolio.swing_data import adjust_etf
from fwbg.portfolio.swing_study import worst_p


def bars_from_close(close):
    return pd.DataFrame(
        {
            f"{side}_{field}": close
            for side in ("bid", "ask")
            for field in ("open", "high", "low", "close")
        }
    )


@pytest.fixture
def flat_bars():
    return bars_from_close(pd.Series(100.0, index=pd.bdate_range("2018-01-01", "2019-02-28")))


def test_leverage_target_is_self_financed_after_bidask_and_fees():
    cash, q, delta, fill, fee = rebalance(10000.0, 0.0, 100.0, 102.0, 1.5, 0.001)
    assert fill == 102
    assert cash == pytest.approx(10000 - q * 102 - fee)
    assert q * 100 / (cash + q * 100) == pytest.approx(1.5)
    assert fee == pytest.approx(delta * 102 * 0.001)
    next_cash, next_q, delta, fill, fee = rebalance(cash, q, 80.0, 82.0, 1.5, 0.001)
    assert delta < 0
    assert fill == 80
    assert next_q * 80 / (next_cash + next_q * 80) == pytest.approx(1.5)


def test_buy_hold_uses_ask_and_cash_has_no_yield(flat_bars):
    for field in ("open", "high", "low", "close"):
        flat_bars[f"ask_{field}"] += 2
    settings = SwingSettings("buy_hold", end="2019-01-31", cost_bps=10)
    result = run_swing(flat_bars, settings)
    expected = 10000 / (102 * 1.001) * 100
    np.testing.assert_allclose(result.equity.iloc[1:], expected)
    assert len(result.ledger) == 1
    idle = run_swing(flat_bars, replace(settings, rule="rsi_sma"))
    assert idle.equity.eq(10000).all()
    assert idle.positions.funding.eq(0).all()


def test_financing_charges_actual_borrow_for_weekend_calendar_days(flat_bars):
    settings = SwingSettings(
        "buy_hold", start="2019-01-04", end="2019-01-08", leverage=1.5, cost_bps=0, borrow_rate=0.08
    )
    result = run_swing(flat_bars, settings)
    monday = result.positions.set_index("date").loc["2019-01-07"]
    assert monday.funding == pytest.approx(5000 * 0.08 * 3 / 365)
    assert result.equity.loc["2019-01-07"] == pytest.approx(10000 - monday.funding)
    assert monday.exposure == pytest.approx(1.5)


def test_new_high_cannot_raise_stop_retroactively_within_same_bar(flat_bars):
    flat_bars.loc["2019-01-04", ["bid_high", "ask_high"]] = 120
    flat_bars.loc["2019-01-04", ["bid_low", "ask_low"]] = 99
    flat_bars.loc["2019-01-04", ["bid_close", "ask_close"]] = 110
    flat_bars.loc["2019-01-07"] = 110
    features = indicators(flat_bars)
    features["breakout_entry"] = False
    features.loc["2019-01-03", "breakout_entry"] = True
    features["atr"] = 2.0
    settings = SwingSettings("breakout_trail", start="2019-01-04", end="2019-01-07", cost_bps=0)
    with patch("fwbg.portfolio.swing.indicators", return_value=features):
        result = run_swing(flat_bars, settings)
    assert result.ledger.reason.tolist() == ["entry", "gap_stop"]
    assert result.ledger.iloc[-1].fill == 110  # Worse than yesterday's 114 stop.
    assert result.equity.iloc[-1] == pytest.approx(11000)


def test_entry_bar_can_hit_initial_stop_and_cannot_capture_opening_jump(flat_bars):
    flat_bars.loc["2019-01-04"] = 200
    flat_bars.loc["2019-01-04", ["bid_low", "ask_low"]] = 190
    features = indicators(flat_bars)
    features["breakout_entry"] = False
    features.loc["2019-01-03", "breakout_entry"] = True
    features["atr"] = 2.0
    with patch("fwbg.portfolio.swing.indicators", return_value=features):
        result = run_swing(
            flat_bars,
            SwingSettings("breakout_trail", start="2019-01-04", end="2019-01-04", cost_bps=0),
        )
    assert result.ledger.reason.tolist() == ["entry", "intraday_stop"]
    assert result.ledger.iloc[-1].fill == 194
    assert result.equity.iloc[-1] == pytest.approx(9700)


def test_rsi_time_exit_is_at_next_open_after_ten_sessions(flat_bars):
    features = indicators(flat_bars)
    features["rsi_entry"] = False
    features.loc["2019-01-03", "rsi_entry"] = True
    features["rsi_level_exit"] = False
    with patch("fwbg.portfolio.swing.indicators", return_value=features):
        result = run_swing(
            flat_bars, SwingSettings("rsi_level", start="2019-01-04", end="2019-01-22", cost_bps=0)
        )
    assert result.ledger.reason.tolist() == ["entry", "time_exit"]
    assert result.ledger.iloc[-1].date == pd.Timestamp("2019-01-18")


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("leverage", [1.0, 1.5])
def test_future_bars_cannot_change_prior_results(rule, leverage):
    dates = pd.bdate_range("2018-01-01", "2020-06-30")
    rng = np.random.default_rng(39)
    bars = bars_from_close(
        pd.Series(100 * np.exp(rng.normal(0.0002, 0.02, len(dates)).cumsum()), index=dates)
    )
    settings = SwingSettings(rule, end="2020-06-30", leverage=leverage)
    first = run_swing(bars, settings)
    bars.loc["2020-02-17":] *= 0.2
    second = run_swing(bars, settings)
    pd.testing.assert_series_equal(
        first.equity.loc[:"2020-02-14"], second.equity.loc[:"2020-02-14"]
    )
    pd.testing.assert_frame_equal(
        first.ledger.loc[first.ledger.date < "2020-02-17"],
        second.ledger.loc[second.ledger.date < "2020-02-17"],
    )


def test_bankruptcy_cannot_disappear_from_returns(flat_bars):
    flat_bars.loc["2019-01-07":] = 1.0
    result = run_swing(
        flat_bars,
        SwingSettings("buy_hold", start="2019-01-04", end="2019-01-09", leverage=1.5, cost_bps=0),
    )
    assert result.bankrupt
    assert result.equity.loc["2019-01-07":].eq(0).all()


def test_corporate_split_is_not_mistaken_for_a_crash():
    dates = pd.bdate_range("2020-01-01", periods=3)
    bars = bars_from_close(pd.Series([200.0, 202.0, 102.0], index=dates))
    reference = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "Adj Close": [100.0, 101.0, 102.0],
            "Stock Splits": [0.0, 0.0, 2.0],
        },
        index=dates,
    )
    adjusted, info = adjust_etf(bars, reference)
    np.testing.assert_allclose(adjusted.bid_close, [100, 101, 102])
    assert info["split_adjusted_rows"] == 2
    reference.loc[dates[1], "Close"] = 1
    with pytest.raises(ValueError, match="disagreement"):
        adjust_etf(bars, reference)


def test_bad_quotes_fail_closed(flat_bars):
    flat_bars.loc[flat_bars.index[-1], ["ask_open", "ask_high", "ask_low", "ask_close"]] = 99
    with pytest.raises(ValueError, match="negative"):
        validate_bars(flat_bars)


def test_monte_carlo_noise_cannot_beat_exact_sign_probability_floor():
    check = {
        "status": "evaluated",
        "monte_carlo": {"p_value": 0.0001},
        "block_sign_permutations": {"6": {"p_value": 0.0001, "observations": 10}},
    }
    assert worst_p([check]) == 1 / 1024
