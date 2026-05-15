"""Tests for the COT fetcher's release-date computation.

Sample report→release pairs are pinned against the CFTC TFF publication
calendar (Tuesday report, Friday ~15:30 ET release; we round up to 21:00
UTC for safety).  Source:
https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# scripts/ is not on the import path; add it for this test module.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_cot_data import compute_release_date  # noqa: E402


@pytest.mark.parametrize(
    "report_tuesday, expected_release",
    [
        ("2024-01-02", "2024-01-05 21:00:00+00:00"),
        ("2024-02-06", "2024-02-09 21:00:00+00:00"),
        ("2024-06-25", "2024-06-28 21:00:00+00:00"),
        ("2024-11-26", "2024-11-29 21:00:00+00:00"),
        ("2025-03-04", "2025-03-07 21:00:00+00:00"),
    ],
)
def test_release_date_matches_cftc_calendar(report_tuesday, expected_release):
    assert compute_release_date(pd.Timestamp(report_tuesday)) == pd.Timestamp(
        expected_release
    )


def test_release_date_handles_tz_aware_input():
    naive = compute_release_date(pd.Timestamp("2024-01-02"))
    aware = compute_release_date(pd.Timestamp("2024-01-02", tz="UTC"))
    assert naive == aware


def test_release_date_ignores_input_time_of_day():
    morning = compute_release_date(pd.Timestamp("2024-01-02 06:00"))
    evening = compute_release_date(pd.Timestamp("2024-01-02 23:30"))
    assert morning == evening == pd.Timestamp("2024-01-05 21:00:00+00:00")
