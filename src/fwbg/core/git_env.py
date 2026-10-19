"""Environment for git subprocesses that must act only on the repository they name."""

from __future__ import annotations

import os

# Git hooks export GIT_DIR and GIT_INDEX_FILE; inheriting them would make git act on
# another repository, e.g. write private data into the public fwbg repository
# (list: `git rev-parse --local-env-vars`).
GIT_LOCAL_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
)


def isolated_git_env() -> dict[str, str]:
    """Environment for git subprocesses: no inherited repository-local variables,
    no optional index locks (a run must not block the operator's own git commands)."""
    env = {name: value for name, value in os.environ.items() if name not in GIT_LOCAL_ENV_VARS}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env
