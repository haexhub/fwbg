#!/usr/bin/env python3
"""Compare trade counts with and without max_trade_bars limit."""

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

# Test parameters
tp_mult = 20
sl_mult = 50

# Compare WITH and WITHOUT max_trade_bars
for max_bars_label, max_bars in [("48 (old)", 48), ("None (new)", None)]:
    trades_long = 0
    trades_short = 0
    wins_long = 0
    wins_short = 0
    none_count = 0
    total_bars_in_trade = 0

    # Simulate LONG trades
    i = atr_period
    while i < len(df) - 1:
        trade = simulate_pro_trade(
            closes, highs, lows, atr, i, "LONG", tp_mult, sl_mult, spread,
            timestamps=timestamps, symbol=symbol, opens=opens,
            max_bars=max_bars
        )
        if trade is not None:
            trades_long += 1
            if trade.get("result", 0) > 0:
                wins_long += 1
            # Skip to exit bar
            exit_idx = trade.get("exit_idx", i + 1)
            i = exit_idx + 1
        else:
            none_count += 1
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
            trades_short += 1
            if trade.get("result", 0) > 0:
                wins_short += 1
            exit_idx = trade.get("exit_idx", i + 1)
            i = exit_idx + 1
        else:
            none_count += 1
            i += 1

    total_trades = trades_long + trades_short
    total_wins = wins_long + wins_short
    wr = total_wins / total_trades if total_trades > 0 else 0

    print(f"\n{'='*50}")
    print(f"max_trade_bars = {max_bars_label}")
    print(f"{'='*50}")
    print(f"Total signals tested: {2 * (len(df) - atr_period - 1)}")
    print(f"Trades executed: {total_trades} (Long: {trades_long}, Short: {trades_short})")
    print(f"None (ignored): {none_count}")
    print(f"Wins: {total_wins} (Long: {wins_long}, Short: {wins_short})")
    print(f"Win Rate: {wr*100:.1f}%")

    # Kelly calculation
    rrr = tp_mult / sl_mult
    kelly = (wr * rrr - (1 - wr)) / rrr if rrr > 0 else 0
    print(f"RRR: {rrr:.2f}")
    print(f"Kelly: {kelly:.4f} ({'POSITIVE' if kelly > 0 else 'NEGATIVE'})")
