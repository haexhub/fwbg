"""Research-gate preflight for ``POST /api/runs/start``."""

import subprocess

from fastapi import HTTPException

from fwbg.research.runs import CHECK_PREFIX, parse_check_output

# A --check run imports fwbg and selects the data files; it never loads market data.
PREFLIGHT_TIMEOUT = 120.0
_TAIL_LIMIT = 500


def preflight(cmd: list[str], env: dict) -> None:
    """Run the CLI's research gate (``--check-json``) before the real run starts.

    The CLI stays authoritative (it gates again when the run starts); this gives
    callers such as fwbg-agents the gate's refusal as an HTTP error instead of a
    run that dies.

    Raises:
        HTTPException: 409/422 with every denial, 422 if the CLI failed before
            its gate (e.g. an unloadable strategy), 504 on timeout.
    """
    try:
        result = subprocess.run(
            [*cmd, "--check-json"],
            capture_output=True,
            text=True,
            env=env,
            timeout=PREFLIGHT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "research preflight timed out") from exc
    payload = parse_check_output(result.stdout, result.returncode)
    if payload is None:
        raise HTTPException(422, f"research preflight failed: {_output_tail(result)}")
    if not payload.get("allowed"):
        denials = payload.get("denials", [])
        status = 409 if any(d.get("status") == 409 for d in denials) else 422
        raise HTTPException(
            status, {"message": "research gate refused the run", "denials": denials}
        )


def _output_tail(result: subprocess.CompletedProcess) -> str:
    """Stdout (the CLI prints most pre-gate failures there) then stderr, tail-limited."""
    stdout = "\n".join(
        line for line in result.stdout.splitlines() if not line.startswith(CHECK_PREFIX)
    )
    parts = (stdout.strip(), result.stderr.strip())
    return "\n".join(part for part in parts if part)[-_TAIL_LIMIT:]
