# Plugin Spec — signal_exit

**Kind**: exit_strategy  •  **Version**: 1.0.0

## Capability

Closes trades at the close of the first bar after entry where an indicator's exit column for the trade's side is 1, without take-profit or stop-loss.

## Summary

Rule-based exits for strategies whose exit is a condition rather than a price level (session close, trailing level checked at fixed times, opposite signal). The simulator reads the exit columns named in the exit params; with `entry_delay=0` a rule exit frees the same bar for the next entry, so an exit and an opposite entry on one bar flip the position.

## Inputs

- df: OHLC (O, H, L, C) plus the exit columns
- ctx: SimulationContext providing spread

## Parameters

- `exit_signal_long_column` (str): column whose value 1 closes an open long
- `exit_signal_short_column` (str): column whose value 1 closes an open short
- `entry_delay` (int, default=0): 0 = enter at the signal bar's close

## Outputs

- targets_long / targets_short: 1 where entering at the bar's close (paying 1.5 spreads) and exiting at the next flagged close pays
- durations (with return_durations): bars to the next flag, 0 without one
- resolve_distances: TP/SL distances of ten times the price, which no move reaches
