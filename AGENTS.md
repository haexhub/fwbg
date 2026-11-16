## Research register

Every backtest run (`python -m fwbg.cli --strategy-file …`) is gated by the
private research register in `$FWBG_WORKSPACE/research`.

- Before proposing a test, look at what was tried:
  `python -m fwbg.cli research list` and `python -m fwbg.cli research show S-NNNN`.
- A new idea gets a card first: `python -m fwbg.cli research new "<title>" --tags …`,
  with its hypothesis and mechanism written before the first run.
- Every run passes `--research-id S-NNNN --note "<what changed and why>"`;
  re-running an identical configuration needs `--reason`. Dry-run with `--check`.
- `python -m fwbg.cli analyze` (MFE/MAE exit analysis) takes the same
  `--research-id`/`--note`/`--reason`, reads only data before the card's cutoff
  and is recorded too; so does `POST /api/exploration/exit-optimization`
  (`research_id`, `note`, `reason` in the body).
- A holdout look (`--holdout`) evaluates the frozen setting once on data after
  the card's cutoff; use it when iterating is done. A further look needs `--reason`.
- Scripts that evaluate strategies outside `python -m fwbg.cli` record every
  evaluated variant (failed and discarded ones too) with
  `fwbg.research.study.register_trial(...)`, stating the config they evaluated
  including its `timeframe` and `end_date`.
- Never set `FWBG_RESEARCH_GATE`; it exists for the test suite only.
- A strategy is closed only by a deliberate `research verdict`, never after
  one failed configuration.
- Never copy research results, hypotheses or card contents into this
  repository; it is public.

## Study scripts

Scripts that test rules outside `python -m fwbg.cli` (monthly regimes, macro
signals, overlays) get their timing from helpers, not from hand-written shifts.
These rules came from errors that invalidated earlier studies.

- Monthly rule: take the daily position from
  `fwbg.research.timing.monthly_position(condition, daily_index)`. A month-end
  decision takes effect at the close of the first session after that month end
  and earns from the session after that. Never write
  `month_end_series.shift(1).reindex(daily_index, method="ffill")`: on the
  monthly index it delays every decision by a whole extra month.
- Windows are calendar time: use `fwbg.research.timing.change_over_weeks`, not
  `Series.diff(n)` on a series that merges weekly and daily data, which counts
  rows instead of weeks.
- Check units before combining series (FRED mixes millions and billions, for
  example WALCL/WTREGEN versus RRPONTSYD) and apply publication lags as a lower
  bound to every component.
- Test the timing on a toy series: a decision on a month end must not change any
  position up to the executing session (see `tests/research/test_timing.py`).
- The duplicate gate ignores the card: the same variant name, window and symbol
  on two cards counts as identical. Pass `reason=` instead of renaming.
- Study scripts and their lessons live in the private research repository
  (`fwbg-research`), not here; read its `AGENTS.md` and `LESSONS.md` before
  starting a study.
