#!/usr/bin/env python3
"""
Script zum manuellen Testen der Bias-Checks auf bestehenden Results.

Usage:
    python scripts/check_bias_on_results.py [results_dir]

Example:
    python scripts/check_bias_on_results.py test_results/20260207_065056_596686/grid_details
"""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fwbg.optimization.bias_checks import check_asset_bias, check_systematic_bias


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Find latest results
        results_base = Path("test_results")
        if not results_base.exists():
            print("No test_results directory found")
            return

        result_dirs = sorted(results_base.glob("*/grid_details"))
        if not result_dirs:
            print("No grid_details directories found")
            return

        results_dir = result_dirs[-1]

    print(f"Checking results in: {results_dir}")
    print("=" * 80)

    # Support both new (subdirectory) and old (flat JSON) layout
    sym_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if sym_dirs:
        print(f"Found {len(sym_dirs)} asset results\n")
        all_results = []
        for sym_dir in sorted(sym_dirs):
            merged = {}
            for fname in ("config.json", "fold_results.json"):
                fpath = sym_dir / fname
                if fpath.exists():
                    with open(fpath) as f:
                        merged.update(json.load(f))
            if merged:
                all_results.append(merged)
    else:
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            print("No JSON files found")
            return
        print(f"Found {len(json_files)} asset results\n")
        all_results = []
        for json_file in sorted(json_files):
            with open(json_file) as f:
                all_results.append(json.load(f))

    # Run bias check per asset (will print output)
    for result in all_results:
        check_asset_bias(result, verbose=True)

    print()

    # System-wide check
    systematic = check_systematic_bias(all_results, verbose=True)

    # Summary
    print("\nSUMMARY:")
    print(f"  Total Assets: {len(all_results)}")
    print(f"  OK Assets: {systematic.get('ok_assets', 0)}")
    print(f"  Warned Assets: {systematic.get('warned_assets', 0)}")
    print(f"  Biased Assets: {systematic.get('biased_assets', 0)}")
    print(f"  Bias Percentage: {systematic.get('bias_percentage', 0)*100:.1f}%")
    print(f"  Status: {systematic.get('status', 'unknown').upper()}")


if __name__ == "__main__":
    main()
