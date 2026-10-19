"""Capture uncommitted source changes so a run stays reproducible."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from fwbg.core.git_env import isolated_git_env

log = logging.getLogger(__name__)

PATCH_FILENAME = "code.patch"
SOURCE_DIRS = ("src", "packages")


class CodePatchError(Exception):
    """The working tree's changes could not be captured."""


def capture_code_patch(cwd: Path) -> str:
    """Patch from HEAD to the working tree of the repository containing *cwd*.

    ``git diff HEAD`` misses new modules that were never added, so untracked
    files under ``src/`` and ``packages/`` are appended as new-file diffs.
    Returns an empty string for a clean tree.
    """
    root = Path(_git(cwd, "rev-parse", "--show-toplevel").strip())
    parts = [_git(root, "diff", "--binary", "HEAD")]
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *SOURCE_DIRS)
    for name in untracked.splitlines():
        # --no-index exits 1 when the files differ, which they always do here.
        parts.append(
            _git(root, "diff", "--binary", "--no-index", "--", os.devnull, name, ok_codes=(0, 1))
        )
    return "".join(parts)


def write_code_patch(run_dir: str | os.PathLike[str]) -> str | None:
    """Write ``code.patch`` for the repository in the current directory into *run_dir*.

    Returns the file name, or ``None`` when the tree is clean or no patch could
    be captured (no git, not a repository) — a missing patch must never fail a run.
    """
    try:
        patch = capture_code_patch(Path.cwd())
    except CodePatchError as exc:
        log.warning("could not capture %s: %s", PATCH_FILENAME, exc)
        return None
    if not patch:
        return None
    (Path(run_dir) / PATCH_FILENAME).write_text(patch, encoding="utf-8", errors="surrogateescape")
    return PATCH_FILENAME


def _git(cwd: Path, *args: str, ok_codes: tuple[int, ...] = (0,)) -> str:
    # A hook's GIT_DIR or GIT_INDEX_FILE must not redirect or distort the capture.
    env = isolated_git_env()
    try:
        # surrogateescape round-trips non-UTF-8 bytes into code.patch unchanged.
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CodePatchError(f"git {args[0]} failed: {exc}") from exc
    if result.returncode not in ok_codes:
        raise CodePatchError(f"git {args[0]} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout
