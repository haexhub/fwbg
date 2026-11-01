"""Tests for the OECD monthly release-date convention in the macro fetcher.

Convention: ``release_date = reference_month_end + 8 weeks at 21:00 UTC``
(conservative upper bound of the 4-8 week OECD lag, see
docs/data/macro-release-dates.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# scripts/ is not on the import path; add it for this test module.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_macro_data import _is_oecd_monthly, compute_oecd_release_date  # noqa: E402


@pytest.mark.parametrize(
    "observation, expected_release",
    [
        # Jan 31 + 56 days = Mar 27
        ("2024-01-01", "2024-03-27 21:00:00+00:00"),
        # Feb 29 (leap year) + 56 days = Apr 25
        ("2024-02-01", "2024-04-25 21:00:00+00:00"),
        # Jun 30 + 56 days = Aug 25
        ("2024-06-01", "2024-08-25 21:00:00+00:00"),
        # Dec 31 + 56 days = Feb 25 of the next year
        ("2024-12-01", "2025-02-25 21:00:00+00:00"),
    ],
)
def test_oecd_release_date_matches_convention(observation, expected_release):
    assert compute_oecd_release_date(pd.Timestamp(observation)) == pd.Timestamp(
        expected_release
    )


@pytest.mark.parametrize("obs", ["2024-01-15", "2024-07-31", "2025-02-28"])
def test_release_lands_after_reference_month(obs):
    """Whatever day inside the month is passed, release is after month end."""
    ts = pd.Timestamp(obs)
    month_end = (ts.normalize() + pd.offsets.MonthEnd(0)).tz_localize("UTC")
    assert compute_oecd_release_date(ts) > month_end + pd.Timedelta(weeks=7)


def test_same_month_same_release_regardless_of_day_or_tz():
    naive = compute_oecd_release_date(pd.Timestamp("2024-03-01"))
    mid = compute_oecd_release_date(pd.Timestamp("2024-03-17 13:45"))
    aware = compute_oecd_release_date(pd.Timestamp("2024-03-01", tz="UTC"))
    assert naive == mid == aware


def test_only_oecd_series_are_treated_as_monthly():
    for sid in ("IRLTLT01DEM156N", "IRLTLT01JPM156N", "IRLTLT01GBM156N", "IRLTLT01AUM156N"):
        assert _is_oecd_monthly(sid)
    for sid in ("DGS2", "DGS5", "DGS30"):
        assert not _is_oecd_monthly(sid)
