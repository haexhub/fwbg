#!/usr/bin/env python3
"""
Migration script: Convert strategy configs from old format (per-asset grids)
to new format (exit_params as inline dict with arrays + optimization section).

Old format:
  - exit_params: preset string (e.g. "atr_intraday_v1") resolving to scalar values
  - grids: { presets_dir, regime_filters_dir?, regime_filter_grid?, assignments: { ASSET: ... } }

New format:
  - exit_params: inline dict with ALL values as arrays (resolved preset + merged grid tp/sl/timeout)
  - optimization: { ct, regime_filter_grid?, exit_modifier_params_grid?, model_hyperparameters_grid? }
  - grids: REMOVED
"""

import json
import sys
from pathlib import Path
from copy import deepcopy

STRATEGIES_DIR = Path(__file__).parent.parent / "strategies"
CONFIGS_DIR = STRATEGIES_DIR / "configs"
EXIT_PARAMS_DIR = STRATEGIES_DIR / "exit_params"
GRIDS_DIR = STRATEGIES_DIR / "grids"
REGIME_FILTERS_DIR = STRATEGIES_DIR / "regime_filters"

# Keys that are exit-param related and should be in exit_params (as arrays)
EXIT_PARAM_KEYS = {"tp", "sl", "timeout_bars"}
# Keys that should go into the optimization section
OPTIMIZATION_KEYS = {"ct", "regime_filter_grid", "exit_modifier_params_grid", "model_hyperparameters_grid"}
# Keys in grid assignments that are purely metadata / not migrated
GRID_META_KEYS = {"preset", "presets_dir", "regime_filters_dir", "regime_filter_grid",
                  "model_hyperparameters", "required_features"}


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_preset(name: str, presets_dir: Path) -> dict:
    """Load a preset file by name, stripping _meta."""
    # Try exact match first
    path = presets_dir / f"{name}.json"
    if not path.exists():
        # Try glob fallback for versioned names
        matches = list(presets_dir.glob(f"{name}*.json"))
        if matches:
            path = matches[0]
        else:
            print(f"  WARNING: preset '{name}' not found in {presets_dir}")
            return {}

    data = load_json(path)
    data.pop("_meta", None)
    return data


def to_array(value):
    """Convert a scalar value to a single-element array."""
    if isinstance(value, list):
        return value
    return [value]


def sorted_unique(values):
    """Return sorted unique values from a list, handling None specially."""
    nones = [v for v in values if v is None]
    non_nones = sorted(set(v for v in values if v is not None))
    if nones:
        return [None] + non_nones
    return non_nones


def union_lists(lists):
    """Union multiple lists, deduplicating and sorting. Handles None."""
    all_values = []
    for lst in lists:
        if isinstance(lst, list):
            all_values.extend(lst)
        else:
            all_values.append(lst)
    return sorted_unique(all_values)


def union_dict_lists(lists_of_dicts):
    """Union multiple lists of dicts, deduplicating by serialized form."""
    seen = set()
    result = []
    for lst in lists_of_dicts:
        if not isinstance(lst, list):
            continue
        for d in lst:
            key = json.dumps(d, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(d)
    return result


def resolve_grid_assignment(assignment, grids_dir: Path):
    """
    Resolve a single grid assignment entry.
    Can be:
      - string (preset name)
      - dict with "preset" key + overrides
      - inline dict (no preset)
    Returns the fully resolved dict (preset merged with overrides).
    """
    if isinstance(assignment, str):
        return resolve_preset(assignment, grids_dir)

    if isinstance(assignment, dict):
        result = {}
        preset_name = assignment.get("preset")
        if preset_name:
            result = resolve_preset(preset_name, grids_dir)
        # Apply overrides from the assignment (override preset values)
        for key, value in assignment.items():
            if key in ("preset",):
                continue
            result[key] = value
        return result

    return {}


def migrate_strategy(config_path: Path, dry_run: bool = False) -> dict:
    """
    Migrate a single strategy config file.
    Returns a dict with migration info.
    """
    config = load_json(config_path)
    original = deepcopy(config)
    changes = []

    # Skip configs that don't have grids or have empty grids
    grids = config.get("grids")
    if not grids or not isinstance(grids, dict) or not grids.get("assignments"):
        if grids is not None and isinstance(grids, dict) and not grids.get("assignments"):
            # Empty grids object — just remove it
            if "grids" in config:
                del config["grids"]
                changes.append("removed empty grids field")
        if not changes:
            return {"file": config_path.name, "status": "skipped", "reason": "no grids or no assignments"}
        if not dry_run:
            save_json(config_path, config)
        return {"file": config_path.name, "status": "migrated", "changes": changes}

    grids_dir = GRIDS_DIR
    regime_filters_dir = REGIME_FILTERS_DIR

    # Resolve all grid assignments
    assignments = grids.get("assignments", {})
    resolved_assignments = {}
    for asset, assignment in assignments.items():
        resolved_assignments[asset] = resolve_grid_assignment(assignment, grids_dir)

    # --- Step 1: Build exit_params as inline dict ---
    exit_params = config.get("exit_params")
    exit_params_dict = {}

    if isinstance(exit_params, str):
        # Resolve the preset
        exit_params_dict = resolve_preset(exit_params, EXIT_PARAMS_DIR)
        changes.append(f"resolved exit_params preset '{exit_params}'")
    elif isinstance(exit_params, dict):
        exit_params_dict = deepcopy(exit_params)
        changes.append("exit_params already inline dict")
    else:
        exit_params_dict = {}
        changes.append("no exit_params found, creating empty")

    # Convert all scalar exit_params values to arrays
    for key in list(exit_params_dict.keys()):
        exit_params_dict[key] = to_array(exit_params_dict[key])

    # Merge grid tp/sl/timeout_bars from all asset classes (union)
    tp_values = []
    sl_values = []
    timeout_values = []

    for asset, resolved in resolved_assignments.items():
        if "tp" in resolved:
            tp_values.append(resolved["tp"] if isinstance(resolved["tp"], list) else [resolved["tp"]])
        if "sl" in resolved:
            sl_values.append(resolved["sl"] if isinstance(resolved["sl"], list) else [resolved["sl"]])
        if "timeout_bars" in resolved:
            timeout_values.append(resolved["timeout_bars"] if isinstance(resolved["timeout_bars"], list) else [resolved["timeout_bars"]])

    if tp_values:
        merged_tp = union_lists([v for sublist in tp_values for v in [sublist]])
        # In exit_params, tp from grids maps to tp_mult
        exit_params_dict["tp_mult"] = merged_tp
        changes.append(f"merged tp from {len(tp_values)} asset classes -> tp_mult: {merged_tp}")

    if sl_values:
        merged_sl = union_lists([v for sublist in sl_values for v in [sublist]])
        # In exit_params, sl from grids maps to sl_mult
        exit_params_dict["sl_mult"] = merged_sl
        changes.append(f"merged sl from {len(sl_values)} asset classes -> sl_mult: {merged_sl}")

    if timeout_values:
        merged_timeout = union_lists([v for sublist in timeout_values for v in [sublist]])
        exit_params_dict["timeout_bars"] = merged_timeout
        changes.append(f"merged timeout_bars from {len(timeout_values)} asset classes -> {merged_timeout}")

    config["exit_params"] = exit_params_dict

    # --- Step 2: Build optimization section ---
    optimization = {}

    # ct: union from all asset classes
    ct_values = []
    for asset, resolved in resolved_assignments.items():
        if "ct" in resolved:
            ct_values.append(resolved["ct"] if isinstance(resolved["ct"], list) else [resolved["ct"]])
    if ct_values:
        optimization["ct"] = union_lists([v for sublist in ct_values for v in [sublist]])
        changes.append(f"optimization.ct: {optimization['ct']}")

    # regime_filter_grid: from strategy-level grids, resolved
    regime_filter_name = grids.get("regime_filter_grid")
    if regime_filter_name:
        regime_data = resolve_preset(regime_filter_name, regime_filters_dir)
        if regime_data:
            optimization["regime_filter_grid"] = regime_data
            changes.append(f"optimization.regime_filter_grid from '{regime_filter_name}'")

    # Also check for condition_grids inside grid presets (e.g. mr_vwap_index_v1)
    if "regime_filter_grid" not in optimization:
        for asset, resolved in resolved_assignments.items():
            if "condition_grids" in resolved:
                optimization["regime_filter_grid"] = {"condition_grids": resolved["condition_grids"]}
                changes.append(f"optimization.regime_filter_grid from grid preset (asset: {asset})")
                break

    # exit_modifier_params_grid: union from all asset classes
    exit_mod_grids = []
    for asset, resolved in resolved_assignments.items():
        if "exit_modifier_params_grid" in resolved:
            exit_mod_grids.append(resolved["exit_modifier_params_grid"])
    if exit_mod_grids:
        optimization["exit_modifier_params_grid"] = union_dict_lists(exit_mod_grids)
        changes.append(f"optimization.exit_modifier_params_grid: {len(optimization['exit_modifier_params_grid'])} entries")

    # model_hyperparameters_grid: union from all asset classes
    model_hp_grids = []
    for asset, resolved in resolved_assignments.items():
        if "model_hyperparameters_grid" in resolved:
            model_hp_grids.append(resolved["model_hyperparameters_grid"])
    if model_hp_grids:
        optimization["model_hyperparameters_grid"] = union_dict_lists(model_hp_grids)
        changes.append(f"optimization.model_hyperparameters_grid: {len(optimization['model_hyperparameters_grid'])} entries")

    if optimization:
        config["optimization"] = optimization

    # --- Step 3: Remove grids ---
    del config["grids"]
    changes.append("removed grids field")

    # --- Step 4: Write back ---
    if not dry_run:
        save_json(config_path, config)

    return {"file": config_path.name, "status": "migrated", "changes": changes}


def migrate_exit_params_presets():
    """Convert scalar values in exit_params preset files to arrays."""
    results = []
    for path in sorted(EXIT_PARAMS_DIR.glob("*.json")):
        data = load_json(path)
        changed = False
        changes = []

        for key, value in list(data.items()):
            if key == "_meta":
                continue
            if not isinstance(value, list):
                data[key] = [value]
                changes.append(f"{key}: {value} -> [{value}]")
                changed = True

        if changed:
            save_json(path, data)
            results.append({"file": path.name, "status": "converted", "changes": changes})
        else:
            results.append({"file": path.name, "status": "already arrays"})

    return results


def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 70)
    print("Strategy Config Migration: grids -> exit_params arrays + optimization")
    print("=" * 70)
    if dry_run:
        print("DRY RUN MODE - no files will be modified\n")

    # Step 1: Migrate strategy configs
    print("\n--- Migrating strategy configs ---\n")
    config_files = sorted(CONFIGS_DIR.glob("*.json"))
    migrated = 0
    skipped = 0

    for config_path in config_files:
        result = migrate_strategy(config_path, dry_run=dry_run)
        status = result["status"]

        if status == "migrated":
            migrated += 1
            print(f"  MIGRATED: {result['file']}")
            for change in result.get("changes", []):
                print(f"    - {change}")
        elif status == "skipped":
            skipped += 1
            print(f"  SKIPPED:  {result['file']} ({result.get('reason', '')})")

    print(f"\n  Total: {len(config_files)} configs, {migrated} migrated, {skipped} skipped")

    # Step 2: Migrate exit_params presets
    print("\n--- Migrating exit_params presets ---\n")
    if not dry_run:
        preset_results = migrate_exit_params_presets()
        for result in preset_results:
            if result["status"] == "converted":
                print(f"  CONVERTED: {result['file']}")
                for change in result.get("changes", []):
                    print(f"    - {change}")
            else:
                print(f"  OK:        {result['file']} ({result['status']})")
    else:
        print("  (skipped in dry-run mode)")

    print("\n" + "=" * 70)
    print("Migration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
