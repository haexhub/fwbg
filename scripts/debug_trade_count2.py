#!/usr/bin/env python3
"""Compare trade counts with different TP/SL ratios."""

import sys
sys.path.insert(0, '/home/haex/Projekte/fwbg')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from pathlib import Path

from optimizer.data_loader import load_data_aligned
from optimizer.simulation import simulate_pro_trade
from optimizer.asset_config import get_asset

# Load DAX data
symbol = "DAX"
csv_path = Path("data/forexsb") / f"{symbol}_HOUR.csv"
df = load_data_aligned(str(csv_path))
df = df.reset_index()
df = df.rename(columns={"T": "timestamp", "O": "open", "H": "high", "L": "low", "C": "close"})

asset = get_asset(symbol)
spread = asset.spread

print(f"Loaded {len(df)} bars for {symbol}")
print(f"Spread: {spread}")

# Extract arrays for simulation
closes = df["close"].values
highs = df["high"].values
lows = df["low"].values
opens = df["open"].values
timestamps = df["timestamp"].values

# Calculate ATR
atr_period = 14
atr = np.zeros(len(df))
for i in range(atr_period, len(df)):
    tr_sum = 0
    for j in range(i - atr_period + 1, i + 1):
        tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1]))
        tr_sum += tr
    atr[i] = tr_sum / atr_period

# Test different TP/SL combinations
test_configs = [
    (20, 50, 0.4),   # Asymmetric - needs 71.4%
    (20, 20, 1.0),   # Balanced - needs 50%
    (30, 30, 1.0),   # Balanced - needs 50%
    (50, 50, 1.0),   # Balanced - needs 50%
    (50, 30, 1.67),  # Slight edge - needs 37.5%
    (70, 50, 1.4),   # Good RRR - needs 41.7%
    (100, 50, 2.0),  # Strong RRR - needs 33.3%
]

print("\nComparing different TP/SL configurations (max_bars=None):")
print("="*80)
print(f"{'TP':>4} | {'SL':>4} | {'RRR':>5} | {'Req WR':>7} | {'Trades':>7} | {'Wins':>6} | {'WR':>6} | {'Kelly':>8}")
print("-"*80)

max_bars = None

for tp_mult, sl_mult, rrr in test_configs:
    trades_total = 0
    wins_total = 0

    # Simulate LONG trades
    i = atr_period
    while i < len(df) - 1:
        trade = simulate_pro_trade(
            closes, highs, lows, atr, i, "LONG", tp_mult, sl_mult, spread,
            timestamps=timestamps, symbol=symbol, opens=opens,
            max_bars=max_bars
        )
        if trade is not None:
            trades_total += 1
            if trade.get("result", 0) > 0:
                wins_total += 1
            exit_idx = trade.get("exit_idx", i + 1)
            i = exit_idx + 1
        else:
            i += 1

    # Simulate SHORT trades
    i = atr_period
    while i < len(df) - 1:
        trade = simulate_pro_trade(
            closes, highs, lows, atr, i, "SHORT", tp_mult, sl_mult, spread,
            timestamps=timestamps, symbol=symbol, opens=opens,
            max_bars=max_bars
        )
        if trade is not None:
            trades_total += 1
            if trade.get("result", 0) > 0:
                wins_total += 1
            exit_idx = trade.get("exit_idx", i + 1)
            i = exit_idx + 1
        else:
            i += 1

    wr = wins_total / trades_total if trades_total > 0 else 0
    req_wr = 1 / (rrr + 1)
    kelly = (wr * rrr - (1 - wr)) / rrr if rrr > 0 else 0

    kelly_str = f"{kelly:.4f}"
    if kelly > 0:
        kelly_str = f"+{kelly:.4f}"

    print(f"{tp_mult:4} | {sl_mult:4} | {rrr:5.2f} | {req_wr*100:6.1f}% | {trades_total:7} | {wins_total:6} | {wr*100:5.1f}% | {kelly_str}")

print("\n" + "="*80)
print("Analysis: Balanced TP/SL (RRR ~1.0) should produce positive Kelly with ~50% WR")
print("Asymmetric Scalping TP/SL requires very high win rates (>70%)")
