"""Resolve the frame column an exit reads by role (stop distance, range, level)."""

from collections.abc import Callable, Iterable

EXIT_COLUMN_PARAMS = ("sl_dist_column", "range_column", "sl_level_column")
"""exit_params keys that name a frame column an exit strategy reads."""


class AmbiguousColumnError(ValueError):
    """Several columns qualify for a role and the config does not name one."""


def resolve_role_column(
    columns: Iterable[str],
    explicit: str | None,
    matches: Callable[[str], bool],
    param: str,
    note: str = "",
) -> str | None:
    """The column named by *explicit* when present, else the single column that *matches*.

    Several matches raise: taking the first one made the result depend on the
    frame's column order. *param* is the exit_params key that settles the choice.
    """
    columns = list(columns)
    if explicit and explicit in columns:
        return explicit
    candidates = [c for c in columns if matches(c)]
    if len(candidates) > 1:
        raise AmbiguousColumnError(
            f"{len(candidates)} columns qualify for exit_params.{param}: "
            f"{', '.join(candidates)}. Set exit_params.{param} to the one to use.{note}"
        )
    return candidates[0] if candidates else None
