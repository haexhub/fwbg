# fwbg-mcp

MCP server for the FWBG trading strategy optimizer. Enables AI agents (Claude, Cursor, etc.) to autonomously configure, run, and evaluate strategies.

## Installation

```bash
cd packages/fwbg-mcp
pip install -e .
```

Requires the FWBG API to be running (`fwbg api --host 0.0.0.0 --port 8420`).

## Claude Desktop / Claude Code

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fwbg": {
      "command": "/home/haex/Projekte/fwbg/.venv/bin/fwbg-mcp",
      "env": {
        "FWBG_API_URL": "http://localhost:8420"
      }
    }
  }
}
```

Or if running from the venv directly:

```json
{
  "mcpServers": {
    "fwbg": {
      "command": "uv",
      "args": ["run", "--project", "/home/haex/Projekte/fwbg/packages/fwbg-mcp", "fwbg-mcp"],
      "env": {
        "FWBG_API_URL": "http://localhost:8420"
      }
    }
  }
}
```

## Available Tools

### Strategy Management
| Tool | Description |
|------|-------------|
| `list_strategies()` | List all strategy configs |
| `get_strategy(name)` | Load a strategy config (resolves presets) |
| `save_strategy(name, config)` | Create/update a strategy config |

### Run Management
| Tool | Description |
|------|-------------|
| `start_run(strategy_name, assets?, description?)` | Start a run, returns `run_id` |
| `wait_for_run(run_id, timeout_minutes=120)` | **Block until complete**, return full results |
| `get_run_status(run_id)` | Non-blocking status check |
| `get_run_results(run_id)` | Full results for completed run |
| `get_run_logs(run_id, level?, limit=200)` | Logs for diagnosis |
| `cancel_run(run_id)` | Cancel active run |
| `list_recent_runs(limit=20, strategy?)` | Recent run history |

### Discovery
| Tool | Description |
|------|-------------|
| `list_indicators()` | All indicator plugins with signal_columns |
| `get_indicator_schema(name)` | Full param schema for an indicator |
| `list_exit_strategies()` | All exit strategy plugins with param_schema |
| `list_presets(preset_type)` | Available preset names (pipelines, models, etc.) |

## Autonomous Agent Loop

```
1. list_indicators()                    → discover available building blocks
2. get_strategy("template") or build   → get/create config
3. save_strategy("my_strat", config)   → persist
4. run_id = start_run("my_strat")      → kick off
5. results = wait_for_run(run_id)      → block, get full results
6. analyse PF, WR, fold_stability      → evaluate
7. adjust config → goto 3             → iterate
```

## Key Configuration Concepts

- **signal_rules**: Pre-filter bars before ML model. Without this, model sees all bars → 0 OOS trades.
- **CT (confidence threshold)**: Must be ≤ inner-CV optimal. Start at `[0.35, 0.4, 0.45]`.
- **regime_filter_grid**: Market-regime conditions (bitmask: 4=Long, 2=Short, 6=Both).
- **indicator_grid**: Vary indicator params across grid combinations.
- **Combination limit**: Keep `exit × CT × regime × indicator_grid ≤ 50` to avoid overfit.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FWBG_API_URL` | `http://localhost:8420` | FWBG API base URL |
