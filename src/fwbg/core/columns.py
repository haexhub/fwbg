"""Resolve the frame column an exit reads by role (stop distance, range, level).

A configured column is a plain name or an fnmatch glob pattern; a pattern names
the column across indicator-grid variants whose column prefixes differ.
"""

from collections.abc import Callable, Iterable
from fnmatch import fnmatchcase

EXIT_COLUMN_PARAMS = ("sl_dist_column", "range_column", "sl_level_column")
"""exit_params keys that name a frame column an exit strategy reads."""

_GLOB_CHARS = frozenset("*?[")


class ColumnRoleError(ValueError):
    """The frame column for an exit role cannot be determined."""


class AmbiguousColumnError(ColumnRoleError):
    """Several columns qualify for a role and the config does not name one."""


def is_column_pattern(name: str) -> bool:
    return any(ch in _GLOB_CHARS for ch in name)


def named_columns(columns: Iterable[str], names: Iterable[str]) -> list[str]:
    """The *columns* that one of *names* (plain names or glob patterns) selects."""
    names = list(names)
    return [c for c in columns if any(c == n or fnmatchcase(c, n) for n in names)]


def _choose(candidates: list[str], param: str, note: str) -> str:
    if len(candidates) > 1:
        example = next((c for c in candidates if "_orb_s" in c), None)
        hint = (
            f" For indicator-grid variants a glob pattern such as "
            f"'*{example[example.index('_orb_s'):]}' names one column in each."
            if example
            else ""
        )
        raise AmbiguousColumnError(
            f"{len(candidates)} columns qualify for exit_params.{param}: "
            f"{', '.join(candidates)}. Set exit_params.{param} to the one to use.{hint}{note}"
        )
    return candidates[0]


def resolve_role_column(
    columns: Iterable[str],
    explicit: str | None,
    matches: Callable[[str], bool],
    param: str,
    note: str = "",
) -> str | None:
    """The column *explicit* names, else the single column that *matches*.

    *explicit* is a plain name (used when present; when absent, auto-detection
    applies as before) or a glob pattern that must match exactly one column.
    Several candidates raise: taking the first one made the result depend on the
    frame's column order. *param* is the exit_params key that settles the choice.
    """
    columns = list(columns)
    if explicit and explicit in columns:
        return explicit
    if explicit and is_column_pattern(explicit):
        hits = [c for c in columns if fnmatchcase(c, explicit)]
        if not hits:
            raise ColumnRoleError(
                f"exit_params.{param} pattern '{explicit}' matches no column in the frame."
            )
        return _choose(hits, param, note)
    candidates = [c for c in columns if matches(c)]
    return _choose(candidates, param, note) if candidates else None
