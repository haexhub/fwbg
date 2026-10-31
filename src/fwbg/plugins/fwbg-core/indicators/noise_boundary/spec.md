# Plugin Spec — noise_boundary

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Intraday momentum bands around the cash-session open whose width is the average absolute move from the open to the same time of day over previous sessions, with entry and exit flags for a breakout-and-trail rule.

## Summary

Converts the UTC bar index to a session time zone, anchors each session at max/min of its open and the previous session close, and widens the anchor by the time-of-day noise (mean of the last `lookback_sessions` observations of that slot). At half-hour checks it flags closes beyond the band (and beyond the session VWAP), and flags exits when the close falls back inside the trailing level, at the session's last bar and on every bar outside the session. Based on Zarattini, Aziz and Barbon (2024).

## Inputs

- df: OHLC (O, H, L, C) with optional V; naive index read as UTC, bars labelled by start time

## Parameters

- `session_tz` (str, default="America/New_York"), `session_open` ("09:30"), `session_close` ("16:00")
- `first_check` ("10:00"), `last_check` ("15:30"), `check_every_minutes` (30)
- `lookback_sessions` (int, default=14)
- `use_vwap` (bool, default=True): require the close beyond the VWAP for entries; trail at max/min(band, VWAP)

## Outputs

- `nb_upper`, `nb_lower`, `nb_vwap`: levels on session bars (NaN outside)
- `nb_long`, `nb_short`: entry flags at check bars
- `nb_exit_long`, `nb_exit_short`: exit flags for the `signal_exit` exit strategy

All columns are unshifted: each uses data up to the bar's close and is meant for a decision at that close (`entry_delay=0`).
