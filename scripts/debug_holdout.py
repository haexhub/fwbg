#!/usr/bin/env python3
"""Debug script to analyze why holdout evaluation produces negative Kelly."""

import sys
sys.path.insert(0, '/home/haex/Projekte/fwbg')

import json
from pathlib import Path

# Load the grid details to see what happened
grid_file = Path("/home/haex/Projekte/fwbg/test_results/20260129_101655_45251f/grid_details/DAX.json")

with open(grid_file) as f:
    data = json.load(f)

print(f"Symbol: {data['symbol']}")
print(f"Status: {data['status']}")
print(f"Total combinations: {data['total_combinations']}")
print()

# Analyze the best candidates by inner_val_pnl
results = data['grid_results']
sorted_results = sorted(results, key=lambda x: x.get('inner_val_pnl', 0), reverse=True)

print("Top 10 candidates by inner_val_pnl:")
print("-" * 80)
for i, r in enumerate(sorted_results[:10]):
    tp = r['tp_mult']
    sl = r['sl_mult']
    ct = r['conf_thresh']
    rrr = r['rrr']
    pnl = r['inner_val_pnl']
    stability = r['fold_stability']

    # Calculate required win rate for positive Kelly
    required_wr = 1 / (rrr + 1) if rrr > 0 else 1.0

    print(f"{i+1}. TP={tp}, SL={sl}, CT={ct:.2f}")
    print(f"   RRR={rrr:.2f}, Inner PnL={pnl:.1f}, Stability={stability:.1f}")
    print(f"   Required WR for positive Kelly: {required_wr*100:.1f}%")
    print()

# The problem: inner_val_pnl is positive but holdout fails
# Let's calculate what the win rates must have been

print("\n" + "="*80)
print("ANALYSIS: Why no_kelly?")
print("="*80)

# Best candidate: TP=20, SL=100, RRR=0.2, inner_val_pnl=122.2
# For RRR=0.2, required WR = 1/(0.2+1) = 1/1.2 = 83.3%
# This is very high!

best = sorted_results[0]
tp, sl, rrr = best['tp_mult'], best['sl_mult'], best['rrr']
required = 1 / (rrr + 1)
print(f"\nBest candidate: TP={tp}, SL={sl}, RRR={rrr:.2f}")
print(f"Required win rate for positive Kelly: {required*100:.1f}%")
print(f"\nWith RRR={rrr:.2f}, even a 80% win rate gives negative Kelly!")
print(f"Kelly = (0.80 * {rrr:.2f} - 0.20) / {rrr:.2f} = {(0.80 * rrr - 0.20) / rrr:.4f}")

print("\n" + "-"*80)
print("The issue: Very asymmetric TP/SL (small TP, large SL) requires very high win rates")
print("-"*80)

# Show for different RRRs what win rate is needed
print("\nWin rate requirements for positive Kelly:")
for rrr_val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.67, 0.8, 1.0, 1.5, 2.0]:
    req = 1 / (rrr_val + 1)
    print(f"  RRR={rrr_val:.2f} -> WR > {req*100:.1f}%")
