"""Tests for MFE/MAE computation and capture rate analysis."""

import json

import numpy as np
import pandas as pd
import pytest

from fwbg.exploration.mfe_mae import compute_capture_rates, compute_mfe_mae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_test_csv(path, n=500):
    """Write a realistic OHLCV CSV for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    open_ = close + rng.standard_normal(n) * 0.1
    high = np.maximum(open_, close) + abs(rng.standard_normal(n)) * 0.2
    low = np.minimum(open_, close) - abs(rng.standard_normal(n)) * 0.2
    vol = rng.integers(100, 10000, size=n)
    df = pd.DataFrame({"T": dates, "O": open_, "H": high, "L": low, "C": close, "V": vol})
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# compute_mfe_mae
# ---------------------------------------------------------------------------

class TestComputeMfeMae:
    def test_basic_uptrend(self):
        """In a steady uptrend, MFE long should be large, MAE long small."""
        n = 100
        open_ = np.arange(100.0, 100.0 + n, dtype=np.float64)
        high = open_ + 0.5
        low = open_ - 0.1
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        valid = slice(None, n - 11)
        assert np.nanmean(mfe_l[valid]) > 1.0
        assert np.nanmean(mae_l[valid]) < np.nanmean(mfe_l[valid])

    def test_basic_downtrend(self):
        """In a steady downtrend, MFE short should be large, MAE short small."""
        n = 100
        open_ = np.arange(200.0, 200.0 - n, -1.0, dtype=np.float64)
        high = open_ + 0.1
        low = open_ - 0.5
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        valid = slice(None, n - 11)
        assert np.nanmean(mfe_s[valid]) > 1.0
        assert np.nanmean(mae_s[valid]) < np.nanmean(mfe_s[valid])

    def test_output_shapes(self):
        n = 50
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        for arr in (mfe_l, mae_l, mfe_s, mae_s):
            assert arr.shape == (n,)

    def test_last_bar_nan(self):
        """Last bar can't look forward — should be NaN."""
        n = 50
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        assert np.isnan(mfe_l[-1])

    def test_entry_is_next_bar_open(self):
        """Entry should be next bar's open, not current bar."""
        open_ = np.array([100.0, 105.0, 110.0, 115.0, 120.0], dtype=np.float64)
        high = open_ + 1.0
        low = open_ - 1.0
        mfe_l, mae_l, _, _ = compute_mfe_mae(open_, high, low, max_bars=2)
        # Bar 0: entry=open_[1]=105. Forward bars [1,2].
        # MFE long = max(high[1], high[2]) - entry = max(106, 111) - 105 = 6.0
        assert abs(mfe_l[0] - 6.0) < 1e-10

    def test_mfe_mae_non_negative(self):
        """MFE and MAE should always be >= 0."""
        rng = np.random.default_rng(42)
        n = 500
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        open_ = close + rng.standard_normal(n) * 0.1
        high = np.maximum(open_, close) + abs(rng.standard_normal(n)) * 0.3
        low = np.minimum(open_, close) - abs(rng.standard_normal(n)) * 0.3
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=20)
        valid = ~np.isnan(mfe_l)
        assert np.all(mfe_l[valid] >= -1e-10)
        assert np.all(mae_l[valid] >= -1e-10)
        assert np.all(mfe_s[valid] >= -1e-10)
        assert np.all(mae_s[valid] >= -1e-10)

    def test_symmetric_for_flat_market(self):
        """In a flat market (constant OHLC), MFE long == MAE short, etc."""
        n = 100
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        mfe_l, mae_l, mfe_s, mae_s = compute_mfe_mae(open_, high, low, max_bars=10)
        valid = ~np.isnan(mfe_l)
        np.testing.assert_allclose(mfe_l[valid], mae_s[valid])
        np.testing.assert_allclose(mae_l[valid], mfe_s[valid])


# ---------------------------------------------------------------------------
# compute_capture_rates
# ---------------------------------------------------------------------------

class TestComputeCaptureRates:
    def test_basic_shape(self):
        n = 200
        rng = np.random.default_rng(42)
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        high = open_ + abs(rng.standard_normal(n)) * 0.5
        low = open_ - abs(rng.standard_normal(n)) * 0.5
        atr = np.full(n, 1.0)
        tp_vals = np.array([0.5, 1.0, 1.5])
        sl_vals = np.array([0.5, 1.0, 1.5])
        wr_long, wr_short, trade_counts = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=20
        )
        assert wr_long.shape == (3, 3)
        assert wr_short.shape == (3, 3)
        assert trade_counts.shape == (3, 3)

    def test_wider_sl_more_wins(self):
        """Wider SL should yield same or higher win rate."""
        n = 1000
        rng = np.random.default_rng(42)
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
        high = open_ + abs(rng.standard_normal(n)) * 0.4
        low = open_ - abs(rng.standard_normal(n)) * 0.4
        atr = np.full(n, 0.5)
        tp_vals = np.array([1.0])
        sl_vals = np.array([0.5, 1.0, 2.0])
        wr_long, _, _ = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=30
        )
        assert wr_long[0, 1] >= wr_long[0, 0] - 0.02
        assert wr_long[0, 2] >= wr_long[0, 1] - 0.02

    def test_tighter_tp_more_wins(self):
        """Tighter TP should yield same or higher win rate."""
        n = 1000
        rng = np.random.default_rng(42)
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
        high = open_ + abs(rng.standard_normal(n)) * 0.4
        low = open_ - abs(rng.standard_normal(n)) * 0.4
        atr = np.full(n, 0.5)
        tp_vals = np.array([0.5, 1.0, 2.0])
        sl_vals = np.array([1.0])
        wr_long, _, _ = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=30
        )
        assert wr_long[0, 0] >= wr_long[1, 0] - 0.02
        assert wr_long[1, 0] >= wr_long[2, 0] - 0.02

    def test_simultaneous_hit_counts_as_loss(self):
        """When TP and SL both hit on same bar, should count as loss."""
        open_ = np.array([100.0, 100.0, 102.0], dtype=np.float64)
        high = np.array([100.5, 102.0, 103.0], dtype=np.float64)
        low = np.array([99.5, 98.0, 101.0], dtype=np.float64)
        atr = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        tp_vals = np.array([1.0])
        sl_vals = np.array([1.0])
        wr_long, _, _ = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=2
        )
        # Bar 0 → entry=100, bar 1 H=102 (TP), L=98 (SL) → both hit → loss
        assert wr_long[0, 0] < 0.5

    def test_win_rates_bounded(self):
        """Win rates should be between 0 and 1."""
        rng = np.random.default_rng(42)
        n = 500
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
        high = open_ + abs(rng.standard_normal(n)) * 0.3
        low = open_ - abs(rng.standard_normal(n)) * 0.3
        atr = np.full(n, 0.5)
        tp_vals = np.array([0.5, 1.0, 2.0])
        sl_vals = np.array([0.5, 1.0, 2.0])
        wr_long, wr_short, _ = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=20
        )
        assert np.all(wr_long >= 0.0)
        assert np.all(wr_long <= 1.0)
        assert np.all(wr_short >= 0.0)
        assert np.all(wr_short <= 1.0)

    def test_trade_counts_positive(self):
        """With enough data and reasonable ATR, we should get resolved trades."""
        rng = np.random.default_rng(42)
        n = 500
        open_ = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        high = open_ + abs(rng.standard_normal(n)) * 0.5
        low = open_ - abs(rng.standard_normal(n)) * 0.5
        atr = np.full(n, 0.5)
        tp_vals = np.array([0.5, 1.0])
        sl_vals = np.array([0.5, 1.0])
        _, _, trade_counts = compute_capture_rates(
            open_, high, low, atr, tp_vals, sl_vals, max_bars=30
        )
        assert np.all(trade_counts > 0)


# ---------------------------------------------------------------------------
# exit_analyzer integration
# ---------------------------------------------------------------------------

class TestAnalyzeAsset:
    def test_returns_valid_structure(self, tmp_path):
        from fwbg.exploration.exit_analyzer import analyze_asset

        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=500)
        result = analyze_asset(
            str(csv_path),
            exit_strategy="atr_based",
            exit_params={"atr_period": 14},
            max_bars=20,
        )
        assert "symbol" in result
        assert "mfe_mae" in result
        assert "capture_matrix" in result
        assert "suggested_grid" in result
        assert "long" in result["mfe_mae"]
        assert "short" in result["mfe_mae"]

    def test_suggested_grid_has_tp_sl(self, tmp_path):
        from fwbg.exploration.exit_analyzer import analyze_asset

        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=1000)
        result = analyze_asset(
            str(csv_path),
            exit_strategy="atr_based",
            exit_params={"atr_period": 14},
            max_bars=20,
        )
        grid = result["suggested_grid"]
        assert "tp" in grid
        assert "sl" in grid
        assert len(grid["tp"]) >= 2
        assert len(grid["sl"]) >= 2

    def test_capture_matrix_sorted_by_edge(self, tmp_path):
        from fwbg.exploration.exit_analyzer import analyze_asset

        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=1000)
        result = analyze_asset(
            str(csv_path),
            exit_strategy="atr_based",
            exit_params={"atr_period": 14},
            max_bars=20,
        )
        if len(result["capture_matrix"]) > 1:
            edges = [r["edge_long"] for r in result["capture_matrix"]]
            assert edges == sorted(edges, reverse=True)

    def test_json_roundtrip(self, tmp_path):
        """Result should be JSON-serializable."""
        from fwbg.exploration.exit_analyzer import analyze_asset

        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=500)
        result = analyze_asset(
            str(csv_path),
            exit_strategy="atr_based",
            exit_params={"atr_period": 14},
            max_bars=20,
        )
        serialized = json.dumps(result)
        loaded = json.loads(serialized)
        assert loaded["symbol"] == result["symbol"]


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    def test_analyze_single_asset(self, tmp_path):
        csv_path = tmp_path / "TEST_HOUR.csv"
        _write_test_csv(csv_path, n=1000)
        out_dir = tmp_path / "output"

        from fwbg.cli._analyze import run_analyze

        run_analyze([str(csv_path), "--output-dir", str(out_dir), "--max-bars", "20"])
        json_file = out_dir / "TEST_HOUR.json"
        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        assert data["symbol"] == "TEST"
        assert len(data["capture_matrix"]) > 0
        assert len(data["suggested_grid"]["tp"]) >= 2
