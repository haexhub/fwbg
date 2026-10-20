"""Layout and git repository of the research register."""

from __future__ import annotations

import fcntl
import logging
import os
import subprocess
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fwbg.core.git_env import isolated_git_env
from fwbg.research.fingerprint import canonical_json, digest
from fwbg.research.ledger import LEDGER_FILENAME

log = logging.getLogger(__name__)

CRITERIA_TEMPLATE = Path("templates") / "criteria-default.yaml"
_DIRECTORIES = ("strategies", "configs", "templates", "reports")
# What a register write changes; every register commit stages all of it, so any
# commit also picks up the changes of an earlier one that failed.
REGISTER_PATHS = ("strategies", "configs", LEDGER_FILENAME)
_GITIGNORE = ".gitignore"
REGISTER_LOCK = ".register.lock"
# *.tmp: store_config writes through temporary files that must never be committed.
_GITIGNORE_TEXT = f".staging/\n{REGISTER_LOCK}\n*.tmp\n"
_GIT_TIMEOUT = 30.0
_GIT_STOP_GRACE = 2.0

DEFAULT_CRITERIA_YAML = """\
# Starting values for the success criteria of new research cards.
# Hand-curated; never derive these values from past results.
# Each card sets its own values at registration — max_drawdown in particular
# depends on the strategy and its position sizing.
min_sharpe: 1.0
min_trades: 100
max_drawdown: 0.40
min_fold_stability: 0.6
max_p_value: 0.05
"""


class ResearchRepoError(Exception):
    """A research-register git or storage operation failed."""


def init_research_repo(research_dir: Path) -> bool:
    """Create the register layout and its git repository.

    Idempotent and recoverable: existing files are never overwritten, and a
    failed first commit is retried on the next call. Returns ``True`` if this
    call created the initial commit. Filesystem errors (``OSError``) pass
    through.

    Raises:
        ResearchRepoError: git failed, or the ledger is missing from an
            initialised register (it is never recreated, so the trial history
            cannot be reset by a re-run).
    """
    research_dir.mkdir(parents=True, exist_ok=True)
    # Empty directories are not tracked, so a clone of the register gets them here.
    for name in _DIRECTORIES:
        (research_dir / name).mkdir(exist_ok=True)
    _write_if_missing(research_dir / CRITERIA_TEMPLATE, DEFAULT_CRITERIA_YAML)
    if not (research_dir / ".git").exists():
        _git(research_dir, "init", "--quiet", "--initial-branch=main")
    ledger = research_dir / LEDGER_FILENAME
    if _has_commit(research_dir):
        if not ledger.exists():
            raise ResearchRepoError(
                f"{LEDGER_FILENAME} is missing from an initialised research register at "
                f"{research_dir}; restore it from git (git checkout -- {LEDGER_FILENAME})"
            )
        return False
    _write_if_missing(research_dir / _GITIGNORE, _GITIGNORE_TEXT)
    ledger.touch()
    # --force: a user's global excludes must not keep register files out; named paths only.
    _git(
        research_dir,
        "add",
        "--force",
        "--",
        CRITERIA_TEMPLATE.as_posix(),
        LEDGER_FILENAME,
        _GITIGNORE,
    )
    _git(research_dir, "commit", "--quiet", "--message", "chore: initialise research register")
    return True


@contextmanager
def register_lock(research_dir: Path) -> Iterator[None]:
    """Serialise register writes (ledger appends plus their commit) across processes.

    Separate from the ``flock`` that ``append_trial`` takes on the ledger itself:
    locking ``trials.jsonl`` again from the same process would deadlock. Not
    re-entrant either: a nested ``register_lock`` in the same process blocks forever.

    On exit, files a root process created are handed to the register's owner
    (:func:`restore_owner`).
    """
    with (research_dir / REGISTER_LOCK).open("a", encoding="utf-8") as handle:
        # ponytail: waits without bound or feedback while another writer holds the lock;
        # poll with LOCK_NB and report the wait if writers ever run long.
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            restore_owner(research_dir)


def restore_owner(research_dir: Path) -> None:
    """Give what a root process created in the register to the register directory's owner.

    The API container runs as root on the register bind-mounted from the host;
    root-owned files (``.git/objects`` above all) would lock the host user out of
    their own register. Never raises: a failure is logged and the next write retries.
    """
    if os.geteuid() != 0:
        return
    try:
        uid, gid = _owner(research_dir)
        if uid == 0:
            return
        for root, dirs, files in os.walk(research_dir):
            for name in (*dirs, *files):
                path = os.path.join(root, name)
                status = os.lstat(path)
                if (status.st_uid, status.st_gid) != (uid, gid):
                    os.chown(path, uid, gid, follow_symlinks=False)
    except OSError as exc:
        log.warning("could not hand register files back to their owner: %s", exc)


def commit_register(research_dir: Path, paths: Sequence[str], message: str) -> bool:
    """Commit the changes under *paths* (relative to the register); ``False`` if none.

    Only the changed files under *paths* are committed: other staged work stays
    staged, a tracked path deleted entirely is committed as deleted, and named
    paths that neither exist nor are tracked are skipped. The register's own
    ``.gitignore`` applies, the user's global excludes do not.

    Raises:
        ValueError: *paths* is empty (git would take that as the whole index).
        ResearchRepoError: git failed.
    """
    if not paths:
        raise ValueError("commit_register needs at least one path")
    stageable = [
        path for path in paths if (research_dir / path).exists() or _tracked(research_dir, path)
    ]
    if stageable:
        # Global excludes (e.g. *.json) must not hide register files.
        _git(research_dir, "add", "--all", "--", *stageable, config=["core.excludesFile=/dev/null"])
    # --no-renames: a rename must list both names, or its deletion stays uncommitted.
    diff = _git(research_dir, "diff", "--cached", "--no-renames", "--name-only", "-z", "--", *paths)
    changed = [name for name in diff.stdout.split("\0") if name]
    if not changed:
        return False
    # Only the changed names: a named directory without tracked files fails as a pathspec.
    _git(research_dir, "commit", "--quiet", "--message", message, "--", *changed)
    return True


def store_config(research_dir: Path, fingerprint: str, config: Mapping[str, Any]) -> Path:
    """Store the fingerprinted *config* as ``configs/<hex>.json`` (content-addressed).

    Call it under :func:`register_lock`. The file is replaced atomically, so a crash
    never leaves a partial config. Filesystem errors (``OSError``, e.g. a missing
    ``configs/``) pass through.

    Raises:
        ValueError: *fingerprint* is malformed or is not the digest of *config*.
        ResearchRepoError: a different file already exists under that name.
    """
    if digest(config) != fingerprint:
        raise ValueError(f"{fingerprint!r} is not the fingerprint of the given config")
    path = research_dir / "configs" / f"{fingerprint.removeprefix('sha256:')}.json"
    data = (canonical_json(config) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise ResearchRepoError(f"{path.name} exists with different content")
        return path
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def _owner(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_uid, status.st_gid


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def _tracked(research_dir: Path, path: str) -> bool:
    """Whether git tracks *path* or files under it (it may be deleted from disk)."""
    return bool(_git(research_dir, "ls-files", "-z", "--", path).stdout)


def _has_commit(research_dir: Path) -> bool:
    """Exit code 1 means no commit yet; any other failure (a broken repository) raises."""
    result = _git(research_dir, "rev-parse", "--verify", "--quiet", "HEAD", ok_codes=(0, 1))
    return result.returncode == 0


def _git(
    cwd: Path, *args: str, ok_codes: tuple[int, ...] = (0,), config: Sequence[str] = ()
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` with ``-c`` *config* overrides; errors name the subcommand."""
    env = isolated_git_env()
    # Never search above the register: a damaged .git must not fall back to an enclosing repo.
    env["GIT_CEILING_DIRECTORIES"] = str(cwd.resolve().parent)
    # The API container (root) works on a register its host user owns; git refuses that
    # without an explicit safe.directory, which only protected config (here: -c) may set.
    settings = (f"safe.directory={cwd.resolve()}", *config)
    overrides = [option for setting in settings for option in ("-c", setting)]
    try:
        with subprocess.Popen(
            ["git", *overrides, *args],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=_GIT_TIMEOUT)
            except subprocess.TimeoutExpired:
                _stop(process)
                raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResearchRepoError(f"git {args[0]} failed: {exc}") from exc
    if process.returncode not in ok_codes:
        detail = stderr.strip() or stdout.strip()
        raise ResearchRepoError(f"git {args[0]} failed: {detail}")
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _stop(process: subprocess.Popen[str]) -> None:
    """SIGTERM first, so git removes its own ``index.lock``; SIGKILL only as a fallback.

    Waits for git alone, not for its output: a hook that git leaves running
    still holds the pipes open.
    """
    process.terminate()
    try:
        process.wait(timeout=_GIT_STOP_GRACE)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
