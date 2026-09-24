# Plugin Spec — jev_signal

**Kind**: data_loader • **Version**: 1.0.0

## Capability

Maps `{provider}_is_long_win` / `{provider}_is_short_win` base columns
(e.g. `jev_official_is_long_win`), populated by a registered CSV DataSource
reading `scripts/fetch_jev_predictions.py` output, to `_composed_signal_long` /
`_composed_signal_short`, applying the mandatory 1-bar shift.

## Parameters

- `provider` (string, default="jev_official"): which cached provider's
  columns to read (`jev_official` or `jev_semif`).

## Edge Cases

- Missing base columns → composed signal columns are all 0 (no lookahead
  risk, just no signal).
