"""
Regression tests for Numba stale-cache recovery and grid-search error visibility.

Root cause: After package restructuring, Numba's pickle cache references old module
paths (e.g. 'fwbg.plugins.fwbg_premium'). Loading the cache raises ModuleNotFoundError.
Previously this was silently swallowed, producing 0 candidates with no visible error.

Fixes:
- _call_numba() in atr_based and atr_trailing_stop wraps Numba calls with
  auto-clear-and-retry on ModuleNotFoundError.
- run_grid_search() re-raises ImportError/ModuleNotFoundError instead of logging only.
"""
import pathlib
import types
import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import ExitStrategyConfig
from fwbg.plugins import import_plugin_module

_atr = import_plugin_module("fwbg-premium", "exit_strategies", "atr_based")
if _atr is None:
    pytest.skip("fwbg-premium atr_based not available", allow_module_level=True)


# ---------------------------------------------------------------------------
# 1. _call_numba: stale cache recovery
# ---------------------------------------------------------------------------

class TestCallNumba:
    """_call_numba must clear cache and retry when Numba raises ModuleNotFoundError."""

    def test_passes_through_normal_call(self):
        """_call_numba returns the function result unchanged when no error occurs."""
        call_numba = _atr._call_numba

        def _fake_numba(*args):
            return sum(args)

        assert call_numba(_fake_numba, 1, 2, 3) == 6

    def test_retries_on_module_not_found_error(self, tmp_path):
        """_call_numba catches ModuleNotFoundError, clears cache dir, and retries."""
        call_numba = _atr._call_numba
        clear_cache = _atr._clear_numba_cache

        # Create fake .nbi/.nbc cache files in the real cache dir
        cache_dir = _atr._CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        fake_nbi = cache_dir / "_fake_stale.py311.nbi"
        fake_nbc = cache_dir / "_fake_stale.py311.nbc"
        fake_nbi.write_bytes(b"stale")
        fake_nbc.write_bytes(b"stale")

        call_count = {"n": 0}

        def _first_call_raises(*args):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ModuleNotFoundError("No module named 'fwbg.plugins.fwbg_premium'")
            return 42

        result = call_numba(_first_call_raises)

        assert result == 42, "Should succeed on second attempt after cache clear"
        assert call_count["n"] == 2, "Should have called the function exactly twice"
        # Cache files should have been deleted
        assert not fake_nbi.exists(), "Stale .nbi should be deleted after recovery"
        assert not fake_nbc.exists(), "Stale .nbc should be deleted after recovery"

    def test_does_not_catch_other_exceptions(self):
        """_call_numba must NOT swallow non-ModuleNotFoundError exceptions."""
        call_numba = _atr._call_numba

        def _always_fails(*args):
            raise ValueError("computation error")

        with pytest.raises(ValueError, match="computation error"):
            call_numba(_always_fails)

    def test_propagates_module_not_found_on_second_attempt(self):
        """If both attempts raise ModuleNotFoundError, the error propagates."""
        call_numba = _atr._call_numba

        def _always_module_error(*args):
            raise ModuleNotFoundError("persistent error")

        with pytest.raises(ModuleNotFoundError):
            call_numba(_always_module_error)


# ---------------------------------------------------------------------------
# 2. _clear_numba_cache: file deletion
# ---------------------------------------------------------------------------

class TestClearNumbaCache:
    """_clear_numba_cache deletes .nbi/.nbc files, leaves other files untouched."""

    def test_deletes_nbi_and_nbc_files(self, monkeypatch):
        """_clear_numba_cache removes .nbi and .nbc files from the cache dir."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            monkeypatch.setattr(_atr, "_CACHE_DIR", tmp_path)

            (tmp_path / "func.py311.nbi").write_bytes(b"x")
            (tmp_path / "func.py311.nbc").write_bytes(b"x")
            (tmp_path / "func.cpython-311.pyc").write_bytes(b"keep")

            _atr._clear_numba_cache()

            remaining = list(tmp_path.iterdir())
            assert len(remaining) == 1, "Only .pyc should remain"
            assert remaining[0].suffix == ".pyc"

    def test_tolerates_missing_cache_dir(self):
        """_clear_numba_cache does not raise if cache dir does not exist."""
        import tempfile, pathlib
        nonexistent = pathlib.Path(tempfile.mkdtemp()) / "no_such_dir"
        # Temporarily replace _CACHE_DIR
        orig = _atr._CACHE_DIR
        _atr._CACHE_DIR = nonexistent
        try:
            _atr._clear_numba_cache()  # Must not raise
        finally:
            _atr._CACHE_DIR = orig


# ---------------------------------------------------------------------------
# 3. atr_based compute_targets: smoke test (catches Numba build issues)
# ---------------------------------------------------------------------------

class TestAtrComputeTargetsEndToEnd:
    """compute_targets must work end-to-end without raising.

    This test catches Numba compilation failures, stale caches, and
    interface mismatches between compute_targets and the Numba kernels.
    """

    @pytest.fixture
    def df(self):
        n = 500
        np.random.seed(1)
        closes = 1.1 + np.cumsum(np.random.normal(0, 0.001, n))
        df = pd.DataFrame({
            "O": closes + np.random.uniform(-0.0005, 0.0005, n),
            "H": closes + np.abs(np.random.normal(0, 0.001, n)),
            "L": closes - np.abs(np.random.normal(0, 0.001, n)),
            "C": closes,
        })
        df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
        df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
        return df

    @pytest.fixture
    def ctx(self):
        from fwbg.core.context import SimulationContext
        return SimulationContext(
            symbol="TEST", asset_class="FOREX",
            spread=0.0001, point=0.00001,
            long_enabled=True, short_enabled=True,
            min_trades=10, exit_strategy="atr_based",
            exit_params={"atr_period": 14, "min_tp_pips": 5, "min_sl_pips": 5},
        )

    def test_returns_two_arrays(self, df, ctx):
        from fwbg.optimization.targets import compute_targets_cached
        result = compute_targets_cached(df, 2.0, 1.0, ctx, timeout_bars=None,
                                        exit_strategy_mode="atr_based")
        assert len(result) == 2
        tl, ts = result
        assert len(tl) == len(df)
        assert len(ts) == len(df)

    def test_returns_four_values_with_durations(self, df, ctx):
        from fwbg.optimization.targets import compute_targets_cached
        result = compute_targets_cached(df, 2.0, 1.0, ctx, timeout_bars=None,
                                        exit_strategy_mode="atr_based",
                                        return_durations=True)
        assert len(result) == 4, "Should return (targets_long, targets_short, dur_long, dur_short)"

    def test_targets_are_binary(self, df, ctx):
        from fwbg.optimization.targets import compute_targets_cached
        tl, ts = compute_targets_cached(df, 2.0, 1.0, ctx,
                                         exit_strategy_mode="atr_based")
        unique_l = set(np.unique(tl[~np.isnan(tl)]))
        unique_s = set(np.unique(ts[~np.isnan(ts)]))
        assert unique_l <= {0.0, 1.0}, f"Unexpected long target values: {unique_l}"
        assert unique_s <= {0.0, 1.0}, f"Unexpected short target values: {unique_s}"

    def test_has_some_wins(self, df, ctx):
        from fwbg.optimization.targets import compute_targets_cached
        tl, ts = compute_targets_cached(df, 2.0, 1.0, ctx,
                                         exit_strategy_mode="atr_based")
        assert tl.sum() > 0, "Long targets should have some wins"
        assert ts.sum() > 0, "Short targets should have some wins"


# ---------------------------------------------------------------------------
# 4. run_grid_search: ImportError must surface, not be swallowed silently
# ---------------------------------------------------------------------------

class TestGridSearchErrorPropagation:
    """ImportError/ModuleNotFoundError from target computation must not be swallowed."""

    def _make_minimal_folds(self):
        n = 300
        np.random.seed(0)
        closes = 1.1 + np.cumsum(np.random.normal(0, 0.001, n))
        df = pd.DataFrame({
            "O": closes, "H": closes + 0.001, "L": closes - 0.001, "C": closes,
            "_regime": np.ones(n, dtype=np.int8) * 7,
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n),
            "feat_c": np.random.randn(n),
        }, index=pd.date_range("2020-01-01", periods=n, freq="15min"))
        train = df.iloc[:200]
        val = df.iloc[200:]
        return [(train, val)], df

    def _make_ctx(self, **overrides):
        from fwbg.core.context import SimulationContext
        defaults = dict(
            symbol="TEST", asset_class="FOREX",
            spread=0.0001, point=0.00001,
            long_enabled=True, short_enabled=True,
            min_trades=10,
            exit_strategy="atr_based",
            exit_params={"atr_period": 14, "min_tp_pips": 5, "min_sl_pips": 5},
            exit_strategies=[
                ExitStrategyConfig(
                    name="atr_based",
                    params={"tp_mult": 2.0, "sl_mult": 1.0, "atr_period": 14,
                            "min_tp_pips": 5, "min_sl_pips": 5},
                    ct=[0.55],
                ),
            ],
            early_pruning_enabled=False,
        )
        defaults.update(overrides)
        return SimulationContext(**defaults)

    def test_import_error_propagates_from_grid_search(self, monkeypatch):
        """An ImportError in target computation must raise, not return 0 candidates."""
        from fwbg.optimization import grid_search as gs

        inner_folds, inner_df = self._make_minimal_folds()
        ctx = self._make_ctx()

        # Patch _compute_cached_targets to raise ImportError
        def _broken_compute(*args, **kwargs):
            raise ImportError("No module named 'fwbg.plugins.fwbg_premium'")

        monkeypatch.setattr(gs, "_compute_cached_targets", _broken_compute)

        with pytest.raises(ImportError):
            gs.run_grid_search(
                full_pool=["feat_a", "feat_b", "feat_c"],
                inner_folds=inner_folds,
                ctx=ctx,
                regime_config={},
                sym="TEST",
                inner_df=inner_df,
            )

    def test_other_exceptions_are_logged_not_raised(self, monkeypatch, capsys):
        """Non-ImportError exceptions are caught but printed to stderr."""
        from fwbg.optimization import grid_search as gs

        inner_folds, inner_df = self._make_minimal_folds()
        ctx = self._make_ctx()

        def _broken_compute(*args, **kwargs):
            raise RuntimeError("unexpected computation error")

        monkeypatch.setattr(gs, "_compute_cached_targets", _broken_compute)

        # Should NOT raise — but should print error to stderr
        candidates, grid_results = gs.run_grid_search(
            full_pool=["feat_a", "feat_b", "feat_c"],
            inner_folds=inner_folds,
            ctx=ctx,
            regime_config={},
            sym="TEST",
            inner_df=inner_df,
        )

        captured = capsys.readouterr()
        assert "[ERROR]" in captured.err, "RuntimeError should be printed to stderr"
        assert len(candidates) == 0
