"""Capture uncommitted source changes so a run stays reproducible."""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
from pathlib import Path

from fwbg.core.git_env import isolated_git_env

log = logging.getLogger(__name__)

PATCH_FILENAME = "code.patch"
SOURCE_DIRS = ("src", "packages")
# Plumbing-stable output: user config (external diff, textconv, colour, prefixes) must not
# turn the patch into something `git apply` rejects or applies wrongly.
_DIFF_FLAGS = (
    "--binary",
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


class CodePatchError(Exception):
    """The working tree's changes could not be captured."""


def capture_code_patch(cwd: Path) -> bytes:
    """Patch from HEAD to the working tree of the repository containing *cwd*.

    ``git diff HEAD`` misses new modules that were never added, so untracked
    files under ``src/`` and ``packages/`` are appended as new-file diffs.
    Returns empty bytes for a clean tree.

    Known limits: skip-worktree and assume-unchanged files, and gitignored files
    under ``src/`` and ``packages/``, are not captured. Apply the patch with
    ``git apply`` (not ``patch -p1``).
    """
    root = Path(os.fsdecode(_git(cwd, "rev-parse", "--show-toplevel")).strip())
    parts = [_git(root, "diff", *_DIFF_FLAGS, "HEAD")]
    untracked = _git(root, "ls-files", "-z", "--others", "--exclude-standard", "--", *SOURCE_DIRS)
    for raw_name in filter(None, untracked.split(b"\0")):
        name = os.fsdecode(raw_name)
        # --no-index exits 1 both when the files differ and on errors; only an
        # error leaves the output empty (even an empty new file gets a header).
        diff = _git(
            root, "diff", *_DIFF_FLAGS, "--no-index", "--", os.devnull, name, ok_codes=(0, 1)
        )
        if not diff:
            raise CodePatchError(f"could not diff untracked file {name!r}")
        parts.append(diff)
    return b"".join(parts)


def write_code_patch(
    run_dir: str | os.PathLike[str], source: Path
) -> tuple[str | None, str | None]:
    """Write ``code.patch`` for the repository containing *source* into *run_dir*.

    Returns ``(file name, None)`` when a patch was written, ``(None, None)``
    for a clean tree and ``(None, error)`` when no patch could be captured or
    written. A missing patch never fails a run, but it must stay visible.
    """
    try:
        patch = capture_code_patch(source)
    except CodePatchError as exc:
        log.warning("could not capture %s: %s", PATCH_FILENAME, exc)
        return None, str(exc)
    if not patch:
        return None, None
    try:
        (Path(run_dir) / PATCH_FILENAME).write_bytes(patch)
    except OSError as exc:
        log.warning("could not write %s: %s", PATCH_FILENAME, exc)
        # config.json will record no patch, so a truncated file must not stay behind.
        with contextlib.suppress(OSError):
            (Path(run_dir) / PATCH_FILENAME).unlink(missing_ok=True)
        return None, f"could not write {PATCH_FILENAME}: {exc}"
    return PATCH_FILENAME, None


def _git(cwd: Path, *args: str, ok_codes: tuple[int, ...] = (0,)) -> bytes:
    # A hook's GIT_DIR or GIT_INDEX_FILE must not redirect or distort the capture.
    env = isolated_git_env()
    try:
        result = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CodePatchError(f"git {args[0]} failed: {exc}") from exc
    if result.returncode not in ok_codes:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise CodePatchError(f"git {args[0]} exited {result.returncode}: {stderr}")
    return result.stdout
