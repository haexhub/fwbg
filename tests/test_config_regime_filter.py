"""A static regime_filter must reach the grid the optimizer reads."""

from fwbg.core.config import StrategyConfig

ADX = {"column": "adx_14", "operator": ">=", "value": 25, "directions": 4}


def test_static_regime_filter_applies_in_every_combination():
    config = StrategyConfig.from_dict({
        "regime_filter": {"conditions": [ADX]},
        "optimization": {"regime_filter_grid": {"condition_grids": [
            {"column": "macro_vix", "operator": "<", "values": [None, 30]},
        ]}},
    })

    combos = config.optimization.regime_filter_grid.get_combinations()

    fixed = {"column": "adx_14", "operator": ">=", "value": 25,
             "directions": 4, "else_directions": 0}
    assert len(combos) == 2
    assert all(fixed in combo["conditions"] for combo in combos)


def test_static_regime_filter_without_grid():
    config = StrategyConfig.from_dict({"regime_filter": {"conditions": [ADX]}})

    assert config.optimization.regime_filter_grid.get_combinations() == [
        {"conditions": [{"column": "adx_14", "operator": ">=", "value": 25,
                         "directions": 4, "else_directions": 0}]},
    ]
