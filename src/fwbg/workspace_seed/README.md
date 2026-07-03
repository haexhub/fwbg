# Workspace seed data

Default presets that `seed_workspace_presets()` copies into the workspace
(`{workspace}/strategies/<section>/`) at API startup — **only when the target
file does not exist yet**. User edits and deletions of *content* are never
overwritten; a deleted file reappears on restart (delete the seed here to
retire it for good).

These mirror `tests/_fixtures/workspace/strategies/` (the CI-validated,
schema-current preset set). When changing a preset, update both places.

Provenance note: these presets are hand-made starting points from early
exploration — convenient building blocks for the dashboard editor, not
scientifically validated configurations.
