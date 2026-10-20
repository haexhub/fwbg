"""Fixtures shared by the research tests."""

import pytest

from fwbg.research.card import write_card
from fwbg.research.legacy import scan_runs
from fwbg.research.repo import REGISTER_PATHS, commit_register, init_research_repo
from tests.research.factories import make_card, make_run


@pytest.fixture
def runs(tmp_path):
    """Scan three fixture runs spanning two strategy names and distinct run dates."""
    results = tmp_path / "results"
    make_run(results, "r1", name="orb-test", timestamp="2026-03-05T10:00:00")
    make_run(results, "r2", name="orb-test", timestamp="2026-03-06T09:00:00")
    make_run(results, "r3", name="pdh retest", timestamp="2026-02-22T08:00:00")
    return scan_runs(results, tmp_path / "presets")


@pytest.fixture
def strategies_dir(tmp_path):
    """A research directory with an empty ``strategies/`` folder and no git."""
    research = tmp_path / "research"
    (research / "strategies").mkdir(parents=True)
    return research


@pytest.fixture
def gate_register(tmp_path, git_env):
    """An initialised register with the committed active card S-0007 (cutoff 2025-04-30)."""
    research = tmp_path / "research"
    init_research_repo(research)
    write_card(research, make_card(), "cusum-event-continuation")
    commit_register(research, REGISTER_PATHS, "test card")
    return research
