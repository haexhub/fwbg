#!/usr/bin/env python3
"""Debug script to run a single holdout evaluation and see detailed results."""

import sys
sys.path.insert(0, '/home/haex/Projekte/fwbg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from pathlib import Path

from optimizer.data_loader import load_data_aligned
from optimizer.nested_cv import (
    compute_targets, train_model, _get_probs,
    simulate_trades_sequential, evaluate_on_holdout
)
from optimizer.simulation_context import SimulationContext
from optimizer.strategy_config import load_strategy

# Load strategy
strategy = load_strategy("scalping")

# Load DAX data
symbol = "DAX"
csv_path = Path("data") / f"{symbol}_H1.csv"
df = load_data_aligned(str(csv_path))
df = df.reset_index()
df = df.rename(columns={"T": "timestamp", "O": "open", "H": "high", "L": "low", "C": "close"})

print(f"Loaded {len(df)} bars for {symbol}")
print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Create context
ctx = SimulationContext.create(strategy, symbol)
print(f"\nContext: spread={ctx.spread}, min_trades={ctx.min_trades}")
print(f"max_trade_bars={ctx.max_trade_bars}")

# Split into inner (first 80%) and holdout (last 20%)
split_idx = int(len(df) * 0.8)
inner_df = df.iloc[:split_idx].reset_index(drop=True)
holdout_df = df.iloc[split_idx:].reset_index(drop=True)

print(f"\nInner set: {len(inner_df)} bars")
print(f"Holdout set: {len(holdout_df)} bars")

# Test with best candidate params
test_params = [
    {"tp": 20, "sl": 70, "ct": 0.55, "rrr": 0.29},  # Best by inner_val_pnl
    {"tp": 20, "sl": 50, "ct": 0.55, "rrr": 0.40},  # Second best
    {"tp": 50, "sl": 50, "ct": 0.55, "rrr": 1.0},   # Balanced RRR
]

for params in test_params:
    tp, sl, ct = params["tp"], params["sl"], params["ct"]
    rrr = params["rrr"]

    print(f"\n{'='*60}")
    print(f"Testing TP={tp}, SL={sl}, CT={ct}, RRR={rrr:.2f}")
    print(f"Required WR for positive Kelly: {1/(rrr+1)*100:.1f}%")
    print(f"{'='*60}")

    # Compute targets on inner
    targets_long, targets_short, has_long, has_short = compute_targets(inner_df, tp, sl, ctx)
    print(f"Inner targets - Long: {sum(targets_long)}/{len(targets_long)}, Short: {sum(targets_short)}/{len(targets_short)}")

    # Get feature columns
    feature_cols = [c for c in inner_df.columns if c.startswith(('trend_', 'vol_', 'ichi_', 'mom_'))][:10]

    # Train models
    mod_long = train_model(inner_df, targets_long, feature_cols, ctx.min_trades) if has_long else None
    mod_short = train_model(inner_df, targets_short, feature_cols, ctx.min_trades) if has_short else None

    print(f"Models trained - Long: {mod_long is not None}, Short: {mod_short is not None}")

    if not mod_long and not mod_short:
        print("No models trained, skipping...")
        continue

    # Get probabilities on holdout
    probs_long, long_win_idx = _get_probs(mod_long, holdout_df, feature_cols)
    probs_short, short_win_idx = _get_probs(mod_short, holdout_df, feature_cols)

    # Simulate trades
    result = simulate_trades_sequential(
        holdout_df, probs_long, probs_short, long_win_idx, short_win_idx,
        ct, tp, sl, ctx, return_detailed=True
    )

    trades = result["trades"]
    trades_detailed = result["trades_detailed"]

    n_trades = len(trades)
    wins = trades.count(1.0)
    losses = trades.count(-1.0)
    win_rate = wins / n_trades if n_trades > 0 else 0

    print(f"\nHoldout Results:")
    print(f"  Total trades: {n_trades}")
    print(f"  Wins: {wins}, Losses: {losses}")
    print(f"  Win Rate: {win_rate*100:.1f}%")

    # Calculate Kelly
    if rrr > 0:
        kelly = (win_rate * rrr - (1 - win_rate)) / rrr
        print(f"  Kelly: {kelly:.4f}")
        if kelly > 0:
            print(f"  -> POSITIVE KELLY! Would pass filter.")
        else:
            print(f"  -> NEGATIVE KELLY! Would be rejected.")

    # Show some trade details
    if trades_detailed:
        long_trades = [t for t in trades_detailed if t.get("direction") == "LONG"]
        short_trades = [t for t in trades_detailed if t.get("direction") == "SHORT"]
        print(f"\n  Long trades: {len(long_trades)}, Short trades: {len(short_trades)}")

        if long_trades:
            long_wins = sum(1 for t in long_trades if t.get("result") == 1.0)
            print(f"  Long WR: {long_wins}/{len(long_trades)} = {long_wins/len(long_trades)*100:.1f}%")
        if short_trades:
            short_wins = sum(1 for t in short_trades if t.get("result") == 1.0)
            print(f"  Short WR: {short_wins}/{len(short_trades)} = {short_wins/len(short_trades)*100:.1f}%")
