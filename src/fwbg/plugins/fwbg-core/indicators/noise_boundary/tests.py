"""Tests for the noise_boundary indicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "indicators", "noise_boundary")
NoiseBoundaryIndicator = _mod.NoiseBoundaryIndicator

BARS_PER_SESSION = 26  # 09:30-16:00 New York in 15-minute bars
SIGNALS = ["nb_long", "nb_short", "nb_exit_long", "nb_exit_short"]


def sessions(days, step_of_day, volume=True):
    """M15 bars of New York cash sessions, naive UTC index, each opening at the last close.

    Within a session the close of bar k (1-based) is open * (1 + step * k), with
    *step_of_day* giving the step for each day.
    """
    frames, price = [], 100.0
    for day, step in zip(days, step_of_day):
        local = pd.date_range(
            f"{day} 09:30",
            periods=BARS_PER_SESSION,
            freq="15min",
            tz="America/New_York",
        )
        closes = price * (1 + step * np.arange(1, BARS_PER_SESSION + 1))
        opens = np.concatenate([[price], closes[:-1]])
        frame = pd.DataFrame(
            {
                "O": opens,
                "H": np.maximum(opens, closes),
                "L": np.minimum(opens, closes),
                "C": closes,
            },
            index=local.tz_convert("UTC").tz_localize(None),
        )
        if volume:
            frame["V"] = 1.0
        frames.append(frame)
        price = closes[-1]
    return pd.concat(frames)


def business_days(start, n):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


@pytest.fixture
def breakout_day():
    """15 quiet sessions across the 2025 DST switch, then one moving twice as fast."""
    days = business_days("2025-02-24", 16)
    return sessions(days, [0.001] * 15 + [0.002]), days[-1]


def ny(index):
    return index.tz_localize("UTC").tz_convert("America/New_York")


def test_long_fires_at_every_half_hour_check_of_the_breakout_session(breakout_day):
    df, last_day = breakout_day
    out = NoiseBoundaryIndicator().compute(df)
    fired = out.index[out["nb_long"] == 1]
    ends = ny(fired + pd.Timedelta("15min"))
    assert all(str(e.date()) == last_day for e in ends)
    assert sorted(e.strftime("%H:%M") for e in ends) == [
        f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)
    ]
    assert out["nb_short"].sum() == 0


def test_bands_need_the_lookback_sessions_before_them(breakout_day):
    df, _ = breakout_day
    out = NoiseBoundaryIndicator().compute(df, lookback_sessions=14)
    first_banded_day = ny(out.index[out["nb_upper"].notna()][0]).date()
    assert str(first_banded_day) == business_days("2025-02-24", 15)[-1]


def test_upper_band_is_the_average_move_from_the_anchor(breakout_day):
    df, last_day = breakout_day
    out = NoiseBoundaryIndicator().compute(df)
    bar = out[
        ny(out.index + pd.Timedelta("15min")).strftime("%Y-%m-%d %H:%M")
        == f"{last_day} 10:00"
    ]
    session_open = df.loc[
        ny(df.index).strftime("%Y-%m-%d %H:%M") == f"{last_day} 09:30", "O"
    ].iloc[0]
    # 10:00 is the second bar: the quiet sessions moved 0.002 from their open by then
    assert bar["nb_upper"].iloc[0] == pytest.approx(session_open * 1.002)


def test_exit_flags_fire_at_the_session_close_and_outside_the_session(breakout_day):
    df, _ = breakout_day
    after_close = df.index[-1] + pd.Timedelta("15min")
    extra = pd.DataFrame(
        {"O": [100.0], "H": [100.0], "L": [100.0], "C": [100.0], "V": [1.0]},
        index=[after_close],
    )
    out = NoiseBoundaryIndicator().compute(pd.concat([df, extra]))
    banded = out["nb_upper"].notna()
    closing = (
        ny(out.index + pd.Timedelta("15min")).strftime("%H:%M") == "16:00"
    ) & banded.cummax()
    assert (out.loc[closing, ["nb_exit_long", "nb_exit_short"]] == 1).all().all()
    assert (
        out.loc[after_close, "nb_exit_long"] == 1
        and out.loc[after_close, "nb_long"] == 0
    )


def test_warm_up_before_the_first_band_is_nan(breakout_day):
    df, _ = breakout_day
    out = NoiseBoundaryIndicator().compute(df)
    first_band = out.index[out["nb_upper"].notna()][0]
    assert out.loc[out.index < first_band, SIGNALS].isna().all().all()


def test_works_without_volume():
    days = business_days("2025-06-02", 16)
    out = NoiseBoundaryIndicator().compute(
        sessions(days, [0.001] * 15 + [0.002], volume=False)
    )
    assert out["nb_long"].sum() == 12
    assert out["nb_vwap"].notna().any()


def test_prefix_stable(breakout_day):
    df, _ = breakout_day
    full = NoiseBoundaryIndicator().compute(df)
    prefix = NoiseBoundaryIndicator().compute(df.iloc[: len(df) - 30])
    cols = ["nb_upper", "nb_lower", "nb_vwap"] + SIGNALS
    pd.testing.assert_frame_equal(full.loc[prefix.index, cols], prefix[cols])


def _hourly_us_bars(start_minute):
    """Hourly bars (labelled by start) of five New York days, naive UTC index."""
    frames = []
    for day in business_days("2025-02-24", 5):
        local = pd.date_range(
            f"{day} 08:{start_minute:02d}",
            periods=12,
            freq="60min",
            tz="America/New_York",
        )
        frame = pd.DataFrame(
            {"O": 100.0, "H": 101.0, "L": 99.0, "C": 100.0, "V": 1.0},
            index=local.tz_convert("UTC").tz_localize(None),
        )
        frames.append(frame)
    return pd.concat(frames)


def test_bar_grid_that_does_not_match_session_open_is_an_error():
    # hourly bars start on the hour but the session opens at 09:30
    with pytest.raises(ValueError, match="bar grid does not match the session"):
        NoiseBoundaryIndicator().compute(_hourly_us_bars(0))


def test_bars_aligned_to_session_open_and_close_are_accepted():
    # hourly bars from 09:30 end on :30 -> session 09:30-16:30
    out = NoiseBoundaryIndicator().compute(
        _hourly_us_bars(30),
        session_close="16:30",
        first_check="10:30",
        last_check="16:00",
        check_every_minutes=30,
        lookback_sessions=2,
    )
    assert out["nb_upper"].notna().any()


def test_daily_bars_are_an_error():
    df = pd.DataFrame(
        {"O": 1.0, "H": 1.1, "L": 0.9, "C": 1.0, "V": 1.0},
        index=pd.bdate_range("2025-01-01", periods=30),
    )
    with pytest.raises(ValueError, match="not shorter than the session"):
        NoiseBoundaryIndicator().compute(df)


def test_grid_check_does_not_reject_a_slice_that_misses_the_open(breakout_day):
    df, _ = breakout_day
    # a trailing window starting mid-session still fits the grid: NaN, not an error
    out = NoiseBoundaryIndicator().compute(df.iloc[10:20])
    assert out["nb_upper"].isna().all()
