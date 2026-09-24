# Plugin Spec — jev_signal

**Kind**: data_loading  •  **Version**: 1.0.0

## Capability

Maps `{provider}_is_long_win` / `{provider}_is_short_win` base probability
columns (e.g. `jev_official_is_long_win`) to fwbg's `_composed_signal_long` /
`_composed_signal_short` columns, applying the mandatory 1-bar shift, so the
existing `models/signal` plugin can consume them unmodified.

## Summary

Pure-computation data_loading plugin (no I/O): selects the two
provider-specific probability columns named by the `provider` param,
substitutes an all-zero series for either column that is absent, shifts both
by 1 bar via `shift_features` to prevent lookahead, and concatenates the
result onto ctx.df as `_composed_signal_long` / `_composed_signal_short`.

## Inputs

- ctx.df columns `{provider}_is_long_win` / `{provider}_is_short_win`, where
  `{provider}` is the resolved `provider` param value (e.g.
  `jev_official_is_long_win`) — populated upstream by a registered CSV
  DataSource reading `scripts/fetch_jev_predictions.py`'s cache output.

## Parameters

- `provider` (choice, default="jev_official", choices=["jev_official",
  "jev_semif"]): which cached Jev provider's probability columns to read.

## Outputs

- `_composed_signal_long`
- `_composed_signal_short`

## Acceptance Criteria

- AC-001: Resolves `provider = params.get("provider", "jev_official")` and
  reads `{provider}_is_long_win` as the pre-shift source for
  `_composed_signal_long`, and `{provider}_is_short_win` as the pre-shift
  source for `_composed_signal_short`.
- AC-002: When either source column is missing from ctx.df, substitutes an
  all-0.0 series (same index as ctx.df) for that column instead of raising.
- AC-003: Both output columns are shifted by exactly 1 bar via
  `shift_features` (row 0 is NaN for both), preventing lookahead.
- AC-004: Returns ctx with ctx.df carrying the two new columns; all
  pre-existing df columns are preserved unchanged.

## Edge Cases

- Both base columns missing → both composed signal columns are NaN at row 0
  and 0 thereafter (no lookahead risk, just no signal).
- Only one of the two base columns present → the missing side is all-zero
  while the present side is mapped and shifted normally.

## Assumptions

- ctx exposes a mutable `.df` attribute (a pandas DataFrame) sharing its
  index with the source probability columns.
- The `provider` param value is the full cache-provider name (e.g.
  `jev_official`), matching the prefix already used in the base column
  names — not a short alias like `official`.
