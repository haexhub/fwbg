## Research register

Every backtest run (`python -m fwbg.cli --strategy-file …`) is gated by the
private research register in `$FWBG_WORKSPACE/research`.

- Before proposing a test, look at what was tried:
  `python -m fwbg.cli research list` and `python -m fwbg.cli research show S-NNNN`.
- A new idea gets a card first: `python -m fwbg.cli research new "<title>" --tags …`,
  with its hypothesis and mechanism written before the first run.
- Every run passes `--research-id S-NNNN --note "<what changed and why>"`;
  re-running an identical configuration needs `--reason`. Dry-run with `--check`.
- Never set `FWBG_RESEARCH_GATE`; it exists for the test suite only.
- A strategy is closed only by a deliberate `research verdict`, never after
  one failed configuration.
- Never copy research results, hypotheses or card contents into this
  repository; it is public.
