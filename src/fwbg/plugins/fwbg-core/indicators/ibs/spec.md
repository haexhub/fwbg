# Plugin Spec — ibs

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Internal bar strength: where a bar closed within its range, with flags for a down close and for a close above the previous bar's high.

## Summary

IBS = (C - L) / (H - L). On session bars a low IBS marks a close near the session low, which in equity indices tends to revert over the next sessions (Pagonidis 2013). `ibs_above_prev_high` is a rebound exit for the `signal_exit` exit strategy.

## Inputs

- df: OHLC (O, H, L, C)

## Parameters

None.

## Outputs

- `ibs`: close location in [0, 1]; NaN for a bar without range
- `ibs_down`: 1 when C < previous C
- `ibs_above_prev_high`: 1 when C > previous H

All columns are unshifted (decision at the bar's own close, `entry_delay=0`); the first row is NaN.
