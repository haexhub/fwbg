#!/usr/bin/env python3
"""Migrate strategy JSON files from single exit_strategy to exit_strategies array.

Converts:
  exit_strategy: "orb_based"
  exit_params: { tp_mult: [1.5, 2.0], sl_mult: [1.0], tp_mode: "atr" }
  exit_modifier: "trailing_stop"
  exit_modifier_params: { breakeven_trigger: 0.5 }
  optimization: { ct: [0.5, 0.55], min_rrr: 1.0 }

To:
  exit_strategies: [
    { name: "orb_based", params: { tp_mult: 1.5, sl_mult: 1.0, tp_mode: "atr" },
      ct: [0.5, 0.55], min_rrr: 1.0,
      exit_modifier: "trailing_stop", exit_modifier_params: { breakeven_trigger: 0.5 } },
    { name: "orb_based", params: { tp_mult: 2.0, sl_mult: 1.0, tp_mode: "atr" },
      ct: [0.5, 0.55], min_rrr: 1.0,
      exit_modifier: "trailing_stop", exit_modifier_params: { breakeven_trigger: 0.5 } },
  ]

Usage:
  python scripts/migrate_exit_strategies.py strategies/
  python scripts/migrate_exit_strategies.py strategies/my_strategy.json --dry-run
"""
import argparse
import itertools
import json
import os
import sys


def _is_grid_array(val):
    """Check if value looks like a grid array (list of scalars)."""
    return isinstance(val, list) and all(isinstance(v, (int, float, str, type(None))) for v in val)


def migrate_strategy(data: dict) -> dict:
    """Migrate a strategy dict from old format to new exit_strategies format."""
    exit_strategy = data.get("exit_strategy")
    if exit_strategy is None and "exit_strategies" in data:
        return data  # Already migrated

    if exit_strategy is None:
        return data  # No exit strategy configured

    exit_params = data.get("exit_params", {})
    exit_modifier = data.get("exit_modifier")
    exit_modifier_params = data.get("exit_modifier_params", {})
    opt = data.get("optimization", {})

    ct = opt.get("ct", [0.5])
    if isinstance(ct, (int, float)):
        ct = [ct]
    long_ct = opt.get("long_ct")
    if isinstance(long_ct, (int, float)):
        long_ct = [long_ct]
    short_ct = opt.get("short_ct")
    if isinstance(short_ct, (int, float)):
        short_ct = [short_ct]
    min_rrr = opt.get("min_rrr", 0)

    # Separate grid arrays from fixed params
    grid_keys = {}
    fixed_params = {}
    for k, v in exit_params.items():
        if _is_grid_array(v) and len(v) > 0:
            grid_keys[k] = v
        else:
            fixed_params[k] = v

    # Build cartesian product of grid arrays
    if grid_keys:
        keys = list(grid_keys.keys())
        values = [grid_keys[k] for k in keys]
        combos = list(itertools.product(*values))
    else:
        keys = []
        combos = [()]  # One instance with no grid variation

    exit_strategies = []
    for combo in combos:
        params = dict(fixed_params)
        for k, v in zip(keys, combo):
            params[k] = v

        instance = {"name": exit_strategy, "params": params, "ct": ct}
        if long_ct:
            instance["long_ct"] = long_ct
        if short_ct:
            instance["short_ct"] = short_ct
        if min_rrr:
            instance["min_rrr"] = min_rrr
        if exit_modifier:
            instance["exit_modifier"] = exit_modifier
            instance["exit_modifier_params"] = exit_modifier_params
        exit_strategies.append(instance)

    # Build new data
    new_data = {}
    for k, v in data.items():
        if k in ("exit_strategy", "exit_params", "exit_modifier", "exit_modifier_params"):
            continue
        if k == "optimization":
            # Remove migrated fields from optimization
            new_opt = {ok: ov for ok, ov in v.items()
                       if ok not in ("ct", "long_ct", "short_ct", "min_rrr",
                                     "exit_modifier_params_grid")}
            if new_opt:
                new_data["optimization"] = new_opt
            continue
        new_data[k] = v

    new_data["exit_strategies"] = exit_strategies
    return new_data


def main():
    parser = argparse.ArgumentParser(description="Migrate strategy JSONs to exit_strategies format")
    parser.add_argument("path", help="Strategy file or directory")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = [os.path.join(args.path, f)
                 for f in os.listdir(args.path) if f.endswith(".json")]
    else:
        print(f"Error: {args.path} not found", file=sys.stderr)
        sys.exit(1)

    for filepath in sorted(files):
        with open(filepath) as f:
            data = json.load(f)

        if "exit_strategy" not in data:
            print(f"  SKIP {filepath} (no exit_strategy field)")
            continue

        migrated = migrate_strategy(data)
        n_instances = len(migrated.get("exit_strategies", []))

        if args.dry_run:
            print(f"  WOULD MIGRATE {filepath} → {n_instances} exit strategy instances")
            print(json.dumps(migrated.get("exit_strategies", []), indent=2))
        else:
            with open(filepath, "w") as f:
                json.dump(migrated, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  MIGRATED {filepath} → {n_instances} exit strategy instances")


if __name__ == "__main__":
    main()
