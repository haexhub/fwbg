"""
Grid-search error visibility.

run_grid_search() must re-raise ImportError/ModuleNotFoundError raised while computing
targets (e.g. a stale Numba cache pointing at module paths that no longer exist) instead
of logging only, which used to produce 0 candidates with no visible error.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import ExitStrategyConfig


# ---------------------------------------------------------------------------
# run_grid_search: ImportError must surface, not be swallowed silently
# ---------------------------------------------------------------------------


class TestGridSearchErrorPropagation:
    """ImportError/ModuleNotFoundError from target computation must not be swallowed."""

    def _make_minimal_folds(self):
        n = 300
        np.random.seed(0)
        closes = 1.1 + np.cumsum(np.random.normal(0, 0.001, n))
        df = pd.DataFrame(
            {
                "O": closes,
                "H": closes + 0.001,
                "L": closes - 0.001,
                "C": closes,
                "_regime": np.ones(n, dtype=np.int8) * 7,
                "feat_a": np.random.randn(n),
                "feat_b": np.random.randn(n),
                "feat_c": np.random.randn(n),
            },
            index=pd.date_range("2020-01-01", periods=n, freq="15min"),
        )
        train = df.iloc[:200]
        val = df.iloc[200:]
        return [(train, val)], df

    def _make_ctx(self, **overrides):
        from fwbg.core.context import SimulationContext

        defaults = dict(
            symbol="TEST",
            asset_class="FOREX",
            spread=0.0001,
            point=0.00001,
            long_enabled=True,
            short_enabled=True,
            min_trades=10,
            exit_strategy="fixed",
            exit_params={},
            exit_strategies=[
                ExitStrategyConfig(
                    name="fixed",
                    params={"tp_mult": 30, "sl_mult": 20},
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
            raise ImportError("No module named 'fwbg.plugins.stale_package'")

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
