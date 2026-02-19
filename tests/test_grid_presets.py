"""
Tests for grid preset loading system.

Verifies:
1. _load_json_preset loads JSON files and raises on missing
2. String/dict assignments resolve to correct GridConfig
3. Preset cache prevents cross-contamination between overrides
4. Regime filter resolution (string ref, inline dict, shared, per-asset)
5. Backward compatibility with legacy inline grid format
6. from_json_file resolves presets_dir relative to strategy file
"""
import json
import os
import pytest

from fwbg.core.config import (
    GridConfig,
    StrategyConfig,
    _load_json_preset,
    _parse_grids,
    _resolve_regime_filter,
)


# -- Fixtures --

SAMPLE_GRID = {
    "tp": [1.5, 2.0],
    "sl": [3.0, 4.0],
    "ct": [0.5, 0.55],
    "timeout_bars": [None, 24],
}

SAMPLE_REGIME_FILTER = {
    "condition_grids": [
        {
            "column": "trend_adx_14",
            "operator": ">=",
            "values": [None, 25],
            "directions": 6,
            "else_directions": 0,
        }
    ]
}


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ============================================================
# TestGridPresetLoading
# ============================================================


class TestGridPresetLoading:
    """Tests for _load_json_preset and basic preset assignment resolution."""

    def test_load_preset_file(self, tmp_path):
        """Basic JSON load returns correct dict."""
        presets_dir = tmp_path / "grids"
        presets_dir.mkdir()
        preset_data = {"tp": [1.0, 2.0], "sl": [1.5], "ct": [0.55]}
        _write_json(str(presets_dir / "forex_wide.json"), preset_data)

        result = _load_json_preset("forex_wide", str(presets_dir))
        assert result == preset_data

    def test_preset_not_found_raises(self, tmp_path):
        """FileNotFoundError with clear message when preset missing."""
        presets_dir = tmp_path / "grids"
        presets_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            _load_json_preset("nonexistent", str(presets_dir))

    def test_string_assignment_loads_preset(self, tmp_path):
        """String assignment loads and returns GridConfig from preset file."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "forex_wide.json"), SAMPLE_GRID)

        grids_data = {
            "assignments": {
                "FOREX": "forex_wide",
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        assert "FOREX" in result
        assert isinstance(result["FOREX"], GridConfig)
        assert result["FOREX"].tp == [1.5, 2.0]
        assert result["FOREX"].sl == [3.0, 4.0]

    def test_dict_with_preset_key_overrides(self, tmp_path):
        """Dict with "preset" key loads preset and applies overrides."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)

        grids_data = {
            "assignments": {
                "FOREX": {"preset": "base", "tp": [3.0, 4.0]},
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        assert result["FOREX"].tp == [3.0, 4.0]  # overridden
        assert result["FOREX"].sl == [3.0, 4.0]  # from preset

    def test_override_does_not_mutate_cache(self, tmp_path):
        """Same preset used twice with different overrides — no bleed."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)

        grids_data = {
            "assignments": {
                "FOREX": {"preset": "base", "tp": [10.0]},
                "INDEX": {"preset": "base", "tp": [20.0]},
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        assert result["FOREX"].tp == [10.0]
        assert result["INDEX"].tp == [20.0]
        # Original preset values unchanged in each
        assert result["FOREX"].sl == [3.0, 4.0]
        assert result["INDEX"].sl == [3.0, 4.0]


# ============================================================
# TestRegimeFilterPresets
# ============================================================


class TestRegimeFilterPresets:
    """Tests for regime filter resolution in preset system."""

    def test_regime_filter_string_loads_file(self, tmp_path):
        """String regime_filter_grid loads from regime_filters/ directory."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        regime_dir = tmp_path / "regime_filters"
        regime_dir.mkdir()

        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)
        _write_json(str(regime_dir / "adx_filter.json"), SAMPLE_REGIME_FILTER)

        grids_data = {
            "assignments": {
                "FOREX": {
                    "preset": "base",
                    "regime_filter_grid": "adx_filter",
                },
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        rfg = result["FOREX"].regime_filter_grid
        assert rfg.total_combinations() == 2
        assert len(rfg.condition_grids) == 1
        assert rfg.condition_grids[0]["column"] == "trend_adx_14"

    def test_regime_filter_inline_dict(self, tmp_path):
        """Inline dict regime_filter_grid used as-is."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)

        grids_data = {
            "assignments": {
                "FOREX": {
                    "preset": "base",
                    "regime_filter_grid": SAMPLE_REGIME_FILTER,
                },
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        rfg = result["FOREX"].regime_filter_grid
        assert rfg.total_combinations() == 2

    def test_regime_filter_from_shared(self, tmp_path):
        """Shared strategy-level regime injected when assignment has none."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)

        grids_data = {
            "regime_filter_grid": SAMPLE_REGIME_FILTER,
            "assignments": {
                "FOREX": "base",
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        rfg = result["FOREX"].regime_filter_grid
        assert rfg.total_combinations() == 2

    def test_regime_filter_per_asset_override_string(self, tmp_path):
        """Per-asset string ref takes precedence over shared."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        regime_dir = tmp_path / "regime_filters"
        regime_dir.mkdir()

        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)

        shared_regime = {
            "condition_grids": [
                {"column": "macro_vix", "operator": "<=", "values": [None, 30],
                 "directions": 6, "else_directions": 0},
            ]
        }
        asset_regime = {
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 20, 25],
                 "directions": 6, "else_directions": 0},
            ]
        }
        _write_json(str(regime_dir / "adx_strict.json"), asset_regime)

        grids_data = {
            "regime_filter_grid": shared_regime,
            "assignments": {
                "FOREX": {
                    "preset": "base",
                    "regime_filter_grid": "adx_strict",
                },
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        rfg = result["FOREX"].regime_filter_grid
        # Asset-level has 3 values → 3 combos (not 2 from shared)
        assert rfg.total_combinations() == 3

    def test_regime_filter_per_asset_override_dict(self, tmp_path):
        """Per-asset inline dict takes precedence over shared."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)

        shared_regime = {
            "condition_grids": [
                {"column": "macro_vix", "operator": "<=", "values": [None, 30],
                 "directions": 6, "else_directions": 0},
            ]
        }
        asset_regime = {
            "condition_grids": [
                {"column": "trend_adx_14", "operator": ">=", "values": [None, 20, 25, 30],
                 "directions": 6, "else_directions": 0},
            ]
        }

        grids_data = {
            "regime_filter_grid": shared_regime,
            "assignments": {
                "FOREX": {
                    "preset": "base",
                    "regime_filter_grid": asset_regime,
                },
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        rfg = result["FOREX"].regime_filter_grid
        # Asset-level has 4 values → 4 combos (not 2 from shared)
        assert rfg.total_combinations() == 4

    def test_no_regime_filter(self, tmp_path):
        """Neither assignment nor shared → no regime filter (default empty)."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "base.json"), SAMPLE_GRID)

        grids_data = {
            "assignments": {
                "FOREX": "base",
            },
        }
        result = _parse_grids(grids_data, str(tmp_path))
        rfg = result["FOREX"].regime_filter_grid
        assert rfg.total_combinations() == 1  # default: no conditions


# ============================================================
# TestBackwardCompatibility
# ============================================================


class TestBackwardCompatibility:
    """Tests ensuring legacy inline grid format works identically."""

    def test_legacy_inline_grids_unchanged(self):
        """Old dict format (tp/sl/ct inline) works identically."""
        grids_data = {
            "FOREX": {
                "tp": [1.5, 2.0],
                "sl": [1.0],
                "ct": [0.55],
            },
            "INDEX": {
                "tp": [2.0, 3.0],
                "sl": [2.0],
                "ct": [0.60],
            },
        }
        result = _parse_grids(grids_data)
        assert "FOREX" in result
        assert "INDEX" in result
        assert isinstance(result["FOREX"], GridConfig)
        assert result["FOREX"].tp == [1.5, 2.0]
        assert result["INDEX"].tp == [2.0, 3.0]

    def test_empty_grids(self):
        """Empty grids dict returns empty dict."""
        assert _parse_grids({}) == {}

    def test_get_grid_resolution_with_presets(self, tmp_path):
        """symbol → asset_class → FOREX fallback still works with preset-loaded grids."""
        grids_dir = tmp_path / "grids"
        grids_dir.mkdir()
        _write_json(str(grids_dir / "forex_grid.json"), {
            "tp": [1.5], "sl": [3.0], "ct": [0.5],
        })
        _write_json(str(grids_dir / "index_grid.json"), {
            "tp": [2.5], "sl": [4.0], "ct": [0.6],
        })
        _write_json(str(grids_dir / "eurusd_special.json"), {
            "tp": [5.0], "sl": [1.0], "ct": [0.7],
        })

        strategy_data = {
            "name": "Test",
            "grids": {
                "assignments": {
                    "FOREX": "forex_grid",
                    "INDEX": "index_grid",
                    "EURUSD": "eurusd_special",
                },
            },
            "_strategy_dir": str(tmp_path),
        }
        config = StrategyConfig.from_dict(strategy_data)

        # Symbol-level match
        eurusd_grid = config.get_grid("EURUSD", "FOREX")
        assert eurusd_grid.tp == [5.0]

        # Asset class match
        dax_grid = config.get_grid("DAX", "INDEX")
        assert dax_grid.tp == [2.5]

        # FOREX fallback
        unknown_grid = config.get_grid("UNKNOWN", "UNKNOWN_CLASS")
        assert unknown_grid.tp == [1.5]


# ============================================================
# TestFromJsonFile
# ============================================================


class TestFromJsonFile:
    """Tests for from_json_file with preset resolution."""

    def test_from_json_file_resolves_presets_dir(self, tmp_path):
        """presets_dir resolved relative to strategy file directory."""
        # Create strategy dir structure
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()
        grids_dir = strategy_dir / "grids"
        grids_dir.mkdir()

        _write_json(str(grids_dir / "my_grid.json"), {
            "tp": [1.0, 1.5],
            "sl": [2.0],
            "ct": [0.5],
        })

        strategy = {
            "name": "PresetTest",
            "grids": {
                "assignments": {
                    "FOREX": "my_grid",
                },
            },
        }
        strategy_path = str(strategy_dir / "test_strategy.json")
        _write_json(strategy_path, strategy)

        config = StrategyConfig.from_json_file(strategy_path)
        assert config.name == "PresetTest"
        grid = config.get_grid("EURUSD", "FOREX")
        assert grid.tp == [1.0, 1.5]
        assert grid.sl == [2.0]
