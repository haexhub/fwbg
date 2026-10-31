## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- One-time setup after cloning: `graphify hook install` to wire the post-commit/post-checkout rebuild. Without it, the graph only refreshes when you call `graphify update .` manually.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->

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
