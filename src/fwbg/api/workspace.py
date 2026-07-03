"""
Workspace management for FWBG.

All runtime artifacts (configs, data, results, accounts, logs) are stored
in a single workspace directory, separate from the application code.

Default: ~/fwbg/
Override: set FWBG_WORKSPACE environment variable.

Individual subdirectories can be further overridden:
  FWBG_STRATEGIES_DIR   → overrides {workspace}/strategies/configs
  FWBG_TEST_RESULTS_DIR → overrides {workspace}/test_results
  FWBG_DATA_DIR         → overrides {workspace}/data
  FWBG_ACCOUNTS_DIR     → overrides {workspace}/accounts
  FWBG_LOGS_DIR         → overrides {workspace}/logs
"""
import logging
import os
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# Packaged default presets, copied into the workspace at startup (create-only).
_SEED_ROOT = Path(__file__).resolve().parent.parent / "workspace_seed" / "strategies"


def get_workspace() -> Path:
    """Return the root workspace directory."""
    return Path(os.environ.get("FWBG_WORKSPACE", Path.home() / "fwbg"))


def _workspace_subdir(env_var: str, *subpath: str) -> Path:
    """Resolve a workspace subdirectory, with env var override support."""
    explicit = os.environ.get(env_var)
    path = Path(explicit) if explicit else get_workspace().joinpath(*subpath)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_strategies_dir() -> Path:
    """Return the strategy configs directory."""
    return _workspace_subdir("FWBG_STRATEGIES_DIR", "strategies", "configs")


def get_test_results_dir() -> Path:
    """Return the test results directory."""
    return _workspace_subdir("FWBG_TEST_RESULTS_DIR", "test_results")


def get_data_dir() -> Path:
    """Return the market data root directory."""
    return _workspace_subdir("FWBG_DATA_DIR", "data")


def get_accounts_dir() -> Path:
    """Return the accounts directory."""
    return _workspace_subdir("FWBG_ACCOUNTS_DIR", "accounts")


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return _workspace_subdir("FWBG_LOGS_DIR", "logs")


def init_workspace() -> Path:
    """
    Ensure the workspace exists and all standard subdirectories are created.
    Returns the workspace root path.
    """
    ws = get_workspace()
    for fn in (get_strategies_dir, get_test_results_dir, get_data_dir,
               get_accounts_dir, get_logs_dir):
        fn()
    return ws


def seed_workspace_presets() -> int:
    """Copy packaged default presets into the workspace, create-only.

    A fresh workspace (new machine, empty volume) has no preset files at all,
    which breaks every strategy that references one by name. This copies each
    file from workspace_seed/strategies/<section>/ to
    {workspace}/strategies/<section>/ unless the target already exists —
    user-edited presets are never overwritten. Returns the number of files
    seeded.
    """
    if not _SEED_ROOT.is_dir():
        log.warning("preset seed dir missing at %s; skipping", _SEED_ROOT)
        return 0

    base = get_strategies_dir().parent
    seeded = 0
    for src in sorted(_SEED_ROOT.glob("*/*.json")):
        target_dir = base / src.parent.name
        target = target_dir / src.name
        if target.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        seeded += 1
    if seeded:
        log.info("seeded %d default preset(s) into %s", seeded, base)
    return seeded
