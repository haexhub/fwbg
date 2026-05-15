"""Regression test: regime label for bar t must not depend on bars > t.

The regime bitmask is attached to fold DataFrames before training. If the
underlying regime computation uses any non-causal operation (rolling without
shift, ewm with center=True, etc.), mutating bars after t will change the
regime at t. This test pins that property so future refactors of the regime
helper can't introduce a lookahead leak.
"""
import numpy as np
import pandas as pd

from fwbg.core.config import RegimeFilterConfig
from fwbg.optimization.process_fold import _attach_regime_to_fold


def _make_synthetic_df(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    base = np.linspace(1.0, 1.1, n)
    return pd.DataFrame({
        "O": base,
        "H": base + 0.001,
        "L": base - 0.001,
        "C": base,
    }, index=idx)


def test_regime_at_t_independent_of_future_bars():
    df = _make_synthetic_df(500)
    # Add a manually-controlled indicator column the regime can reference.
    # Strictly increasing → the > 0.4 condition crosses around bar 200.
    df["indicator"] = np.linspace(0.0, 1.0, 500)

    regime_cfg = RegimeFilterConfig.from_dict({
        "conditions": [{
            "column": "indicator",
            "operator": ">",
            "value": 0.4,
            "directions": 6,        # Long + Short
            "else_directions": 0,   # Blocked
        }],
    })

    baseline = _attach_regime_to_fold(df.copy(), regime_cfg)

    # Mutate the indicator column from bar 250 onwards.
    mutated = df.copy()
    mutated.iloc[250:, mutated.columns.get_loc("indicator")] = 999.0
    perturbed = _attach_regime_to_fold(mutated, regime_cfg)

    # Regime labels on the first 250 bars MUST be unchanged. If they differ,
    # _attach_regime_to_fold is reading future bars at some position t < 250.
    np.testing.assert_array_equal(
        baseline["_regime"].values[:250],
        perturbed["_regime"].values[:250],
        err_msg="Regime on train bars changed when test bars were mutated.",
    )


def test_regime_at_t_independent_of_future_bars_when_split_per_fold():
    """Production path: regime is computed on outer-fold inner_df and then
    sliced per inner fold. The slicing itself must not reintroduce leakage.
    """
    df = _make_synthetic_df(500)
    df["indicator"] = np.linspace(0.0, 1.0, 500)
    cfg = RegimeFilterConfig.from_dict({
        "conditions": [{
            "column": "indicator",
            "operator": ">",
            "value": 0.4,
            "directions": 6,
            "else_directions": 0,
        }],
    })

    inner_df = df.iloc[:400].copy()  # outer fold's train + val
    inner_train = inner_df.iloc[:300]
    inner_val = inner_df.iloc[300:]

    attached_inner = _attach_regime_to_fold(inner_df, cfg)
    inner_train_regime = attached_inner["_regime"].loc[inner_train.index].values
    inner_val_regime = attached_inner["_regime"].loc[inner_val.index].values

    # Sanity: regime is either 6 (allowed) or 0 (blocked).
    assert set(np.unique(inner_train_regime)).issubset({0, 6})
    assert set(np.unique(inner_val_regime)).issubset({0, 6})

    # Mutating future bars (inner_val region) must not change inner_train regime.
    mutated_inner = inner_df.copy()
    mutated_inner.iloc[300:, mutated_inner.columns.get_loc("indicator")] = 999.0
    attached_mut = _attach_regime_to_fold(mutated_inner, cfg)
    np.testing.assert_array_equal(
        inner_train_regime,
        attached_mut["_regime"].loc[inner_train.index].values,
    )
