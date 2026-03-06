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
import os
from pathlib import Path


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
