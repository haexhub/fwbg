"""A monthly decision acts from the first session after the month end, never a month late."""

import pandas as pd
import pytest

from fwbg.research.timing import change_over_weeks, monthly_position

SESSIONS = pd.bdate_range("2020-01-01", "2020-05-29")
MONTH_ENDS = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])


def condition(*values):
    return pd.Series(values, index=MONTH_ENDS[: len(values)])


def test_decision_acts_from_the_first_session_after_the_month_end():
    """Jan=True, Feb=False, Mar=True: in for February, out for March, in again for April."""
    position = monthly_position(condition(True, False, True, True), SESSIONS)
    # Nothing is decided before the first month end.
    assert position.loc["2020-01-31"] == 0
    # The first session after Jan 31 executes the trade; the return is earned after it.
    assert position.loc["2020-02-03"] == 0
    assert position.loc["2020-02-04"] == 1
    # Feb 29 said cash, but the executing session of that decision still earns the old position.
    assert position.loc["2020-03-02"] == 1
    assert position.loc["2020-03-03"] == 0
    assert position.loc["2020-04-01"] == 0
    assert position.loc["2020-04-02"] == 1


def test_naive_monthly_shift_is_a_whole_month_late():
    """The pattern this helper replaces: the January decision only acts from March."""
    monthly = condition(True, False, True, True)
    naive = (
        monthly.shift(1).reindex(SESSIONS, method="ffill").fillna(False).astype(float)
    )
    correct = monthly_position(monthly, SESSIONS)
    assert naive.loc["2020-02-04"] == 0 and correct.loc["2020-02-04"] == 1
    assert naive.loc["2020-03-02"] == 1 and correct.loc["2020-03-03"] == 0


def test_month_end_on_a_weekend_acts_from_the_next_monday_session():
    """Jan 31, 2021 is a Sunday: the executing session is Monday Feb 1."""
    sessions = pd.bdate_range("2021-01-25", "2021-02-10")
    position = monthly_position(
        pd.Series([True], index=pd.DatetimeIndex(["2021-01-31"])), sessions
    )
    assert position.loc["2021-02-01"] == 0
    assert position.loc["2021-02-02"] == 1


def test_later_decisions_never_change_earlier_positions():
    """No look-ahead: flipping the April decision leaves every position through May 1 as it was."""
    base = monthly_position(condition(True, False, True, True), SESSIONS)
    flipped = monthly_position(condition(True, False, True, False), SESSIONS)
    assert base.loc[:"2020-05-01"].equals(flipped.loc[:"2020-05-01"])
    assert base.loc["2020-05-04"] == 1 and flipped.loc["2020-05-04"] == 0


def test_position_is_a_zero_one_float_series_on_the_given_index():
    position = monthly_position(condition(True, False), SESSIONS)
    assert position.index.equals(SESSIONS)
    assert set(position.unique()) <= {0.0, 1.0}
    assert position.dtype == float


def test_missing_conditions_are_ignored():
    """NaN values (warm-up) do not count as a decision."""
    monthly = pd.Series([None, True], index=MONTH_ENDS[:2], dtype="object")
    position = monthly_position(monthly, SESSIONS)
    assert position.loc["2020-02-28"] == 0
    assert position.loc["2020-03-03"] == 1


def test_non_datetime_inputs_are_rejected():
    with pytest.raises(ValueError):
        monthly_position(pd.Series([True]), SESSIONS)


def test_change_over_weeks_is_exported_for_study_scripts():
    """Windows are calendar weeks, not rows of a merged weekly/daily index."""
    daily = pd.Series(
        range(200), index=pd.bdate_range("2020-01-01", periods=200), dtype=float
    )
    assert change_over_weeks(daily, 13).iloc[-1] == pytest.approx(65.0)


def test_window_starting_after_a_decision_keeps_the_invested_position():
    """Jan 31 was True and executed on Feb 3: a window that starts on Feb 4 is already in."""
    full = monthly_position(condition(True, False, True, True), SESSIONS)
    window = monthly_position(
        condition(True, False, True, True), SESSIONS[SESSIONS >= "2020-02-04"]
    )
    assert window.iloc[0] == 1
    assert window.equals(full.loc["2020-02-04":])


def test_window_starting_on_the_executing_session_is_not_yet_invested():
    """A window that starts on Feb 3 earns nothing on that session, as in the full index."""
    window = monthly_position(
        condition(True, False, True, True), SESSIONS[SESSIONS >= "2020-02-03"]
    )
    assert window.iloc[0] == 0 and window.loc["2020-02-04"] == 1


def test_empty_index_gives_an_empty_position():
    empty = SESSIONS[:0]
    assert monthly_position(condition(True), empty).empty


@pytest.mark.parametrize("zone", ["UTC", "America/New_York", "Asia/Tokyo"])
@pytest.mark.parametrize("condition_aware", [False, True])
def test_tz_aware_index_gives_the_same_positions_as_a_naive_one(zone, condition_aware):
    """A tz-aware daily index (and/or condition) must not change the rule's semantics."""
    monthly = condition(True, False, True, True)
    expected = monthly_position(monthly, SESSIONS)
    aware_sessions = SESSIONS.tz_localize(zone)
    aware_monthly = monthly.copy()
    if condition_aware:
        aware_monthly.index = aware_monthly.index.tz_localize(zone)

    got = monthly_position(aware_monthly, aware_sessions)

    assert got.index.equals(aware_sessions)
    assert got.to_numpy().tolist() == expected.to_numpy().tolist()
    assert got.loc["2020-02-03"] == 0 and got.loc["2020-02-04"] == 1
