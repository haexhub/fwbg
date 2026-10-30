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
  and is recorded too.
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
