"""Fixtures shared by the backfill and mapping tests."""

import pytest

from fwbg.research.legacy import scan_runs
from tests.research.factories import make_run


@pytest.fixture
def runs(tmp_path):
    """Scan three fixture runs spanning two strategy names and distinct run dates."""
    results = tmp_path / "results"
    make_run(results, "r1", name="orb-test", timestamp="2026-03-05T10:00:00")
    make_run(results, "r2", name="orb-test", timestamp="2026-03-06T09:00:00")
    make_run(results, "r3", name="pdh retest", timestamp="2026-02-22T08:00:00")
    return scan_runs(results, tmp_path / "presets")
