"""The CLI command of ``POST /api/runs/start`` and its research-gate preflight."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException

from fwbg.research.runs import CHECK_PREFIX, parse_check_output

if TYPE_CHECKING:
    from fwbg.api.runs import RunStartRequest

# A --check run imports fwbg and selects the data files; it never loads market data.
# On timeout the child is SIGKILLed: if it held the register lock, its git
# index.lock may remain (the register lock itself is released by the kernel).
PREFLIGHT_TIMEOUT = 120.0
_TAIL_LIMIT = 500
# Seam for the tests: they replace this, not the global subprocess.run.
_run = subprocess.run


def cli_command(body: RunStartRequest, strategy_file: Path, job_id: str) -> list[str]:
    """The fwbg CLI argv for a run (separate items, no shell).

    The research values use ``--flag=value`` so text starting with ``-`` parses.
    """
    cmd = [sys.executable, "-m", "fwbg.cli", "--strategy-file", str(strategy_file)]

    if body.assets:
        cmd.extend(["--assets", ",".join(body.assets)])
    if body.asset_classes:
        cmd.extend(["--asset-classes", ",".join(body.asset_classes)])
    if body.description:
        cmd.extend(["-d", body.description])
    if body.start_date:
        cmd.extend(["--start-date", body.start_date])
    if body.end_date:
        cmd.extend(["--end-date", body.end_date])
    if body.cost_multiplier is not None:
        cmd.extend(["--cost-multiplier", str(body.cost_multiplier)])
    cmd.extend(["--run-id", job_id])
    if body.research_id:
        cmd.append(f"--research-id={body.research_id}")
    if body.note:
        cmd.append(f"--note={body.note}")
    if body.reason:
        cmd.append(f"--reason={body.reason}")
    if body.holdout:
        cmd.append("--holdout")
    return cmd


def preflight(cmd: list[str], env: dict) -> None:
    """Run the CLI's research gate (``--check-json``) before the real run starts.

    The CLI stays authoritative (it gates again when the run starts); this gives
    callers such as fwbg-agents the gate's refusal as an HTTP error instead of a
    run that dies. ``"denials" in detail`` marks a gate refusal.

    Raises:
        HTTPException: 409/422 ``{"message", "denials"}`` when the gate refuses;
            422 ``{"message", "output"}`` if the CLI failed before its gate (e.g.
            an unloadable strategy) or printed no valid result; 504 on timeout.
    """
    try:
        result = _run(
            [*cmd, "--check-json"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=PREFLIGHT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "research preflight timed out") from exc
    payload = parse_check_output(result.stdout, result.returncode)
    if payload is None or not _is_denial_list(payload.get("denials", [])):
        raise HTTPException(
            422, {"message": "research preflight failed", "output": _output_tail(result)}
        )
    if not payload["allowed"]:
        denials = payload.get("denials", [])
        status = 409 if any(d.get("status") == 409 for d in denials) else 422
        raise HTTPException(
            status, {"message": "research gate refused the run", "denials": denials}
        )


def _is_denial_list(denials: object) -> bool:
    return isinstance(denials, list) and all(isinstance(d, dict) for d in denials)


def _output_tail(result: subprocess.CompletedProcess) -> str:
    """Stdout (the CLI prints most pre-gate failures there) then stderr, tail-limited."""
    stdout = "\n".join(
        line for line in result.stdout.splitlines() if not line.startswith(CHECK_PREFIX)
    )
    parts = (stdout.strip(), result.stderr.strip())
    return "\n".join(part for part in parts if part)[-_TAIL_LIMIT:]
