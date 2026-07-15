"""Tests für den zeitbasierten Equity-Replay (simulate_equity_timeline)."""
import pytest

from fwbg.simulation.equity import simulate_equity_timeline


class TestSimulateEquityTimeline:
    def test_empty_input(self):
        out = simulate_equity_timeline([], [])
        assert out["equity_points"] == []
        assert out["final_equity"] == 100.0
        assert out["max_drawdown"] == 0.0
        assert out["years"] == 0.0

    def test_chronological_order_regardless_of_input_order(self):
        # Loss (t1) passiert VOR dem Win (t2), wird aber als zweites übergeben.
        times = ["2021-01-01T00:00:00", "2020-01-01T00:00:00"]
        returns = [0.10, -0.05]
        out = simulate_equity_timeline(times, returns)

        # Endkapital ist ordnungs-invariant …
        assert out["final_equity"] == pytest.approx(100 * 0.95 * 1.10)
        # … aber der Drawdown entsteht chronologisch beim frühen Loss.
        assert out["max_drawdown"] == pytest.approx(0.05)
        assert out["start_time"].startswith("2020-01-01")
        assert out["end_time"].startswith("2021-01-01")
        # Erster Punkt = Startkapital, danach chronologisch steigende Zeiten
        ts = [p[0] for p in out["equity_points"]]
        assert ts == sorted(ts)
        assert out["equity_points"][0][1] == 100.0

    def test_annual_return_from_calendar_span(self):
        times = ["2020-01-01T00:00:00", "2022-01-01T00:00:00"]
        returns = [0.10, 0.10]
        out = simulate_equity_timeline(times, returns)
        assert out["years"] == pytest.approx(2.0, abs=0.01)
        # (1.21)^(1/2) - 1 = 10 %
        assert out["annual_return"] == pytest.approx(10.0, abs=0.1)

    def test_bankruptcy_caps_drawdown(self):
        times = ["2020-01-01", "2020-06-01", "2020-12-01"]
        returns = [-0.5, -1.5, 0.2]
        out = simulate_equity_timeline(times, returns)
        assert out["final_equity"] == 0.0
        assert out["max_drawdown"] == 1.0

    def test_mismatched_lengths_returns_empty(self):
        out = simulate_equity_timeline(["2020-01-01"], [0.1, 0.2])
        assert out["equity_points"] == []
