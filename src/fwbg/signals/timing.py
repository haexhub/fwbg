"""Fill timing of entry rules that read unshifted signal columns.

An unshifted signal column (``get_signal_columns``) at row ``i`` uses bar ``i``'s
own close. With ``entry_delay=1`` (the default) the order fills at the next bar's
open, after that close is known. With ``entry_delay=0`` it fills at bar ``i``'s
close itself, so the decision needs the close at fill time: only a market-on-close
style order can act on it. The backtest allows that; this module says so.
"""

import logging
from collections.abc import Iterable
from typing import Any

from fwbg.pipeline.features import configured_signal_columns
from fwbg.utils.logging import log as run_log

logger = logging.getLogger(__name__)

# The simulation fills at the next open unless exit_params sets entry_delay.
DEFAULT_ENTRY_DELAY = 1

_COLUMN_KEYS = ("column", "column_a", "column_b")


def entry_delay_zero(exit_strategies: Iterable) -> bool:
    """Whether any configured exit strategy fills entries at the signal bar's close."""
    return any(
        (getattr(es, "params", None) or {}).get("entry_delay", DEFAULT_ENTRY_DELAY) == 0
        for es in exit_strategies or []
    )


def rule_columns(rules: Any) -> list[str]:
    """Column names a rule tree (``signal_rules``) refers to, nested groups included."""
    names: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key in _COLUMN_KEYS:
                if isinstance(node.get(key), str):
                    names.append(node[key])
            for child in node.get("conditions") or []:
                walk(child)
            for direction in (
                "long",
                "short",
            ):  # signal_rules: {"long": {...}, "short": {...}}
                walk(node.get(direction))
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(rules)
    return list(dict.fromkeys(names))


def _names(column: str, signal_columns: Iterable[str]) -> list[str]:
    """The signal columns a rule's (possibly shortened) column name selects."""
    suffix = f"_{column}"
    return [s for s in signal_columns if s == column or s.endswith(suffix)]


def close_decision_columns(
    indicators: Iterable | None,
    signal_rules: dict | None,
    required_features: Iterable[str] | None = (),
) -> list[str]:
    """Unshifted signal columns an entry reads: named by a rule or by ``required_features``.

    Columns known before the bar opens (calendar columns) are not listed: they need
    no close at fill time.
    """
    declared = configured_signal_columns(indicators)
    if not declared:
        return []
    wanted = rule_columns(signal_rules or {})
    wanted += list(required_features or [])
    hit: dict[str, None] = {}
    for name in wanted:
        for col in _names(name, declared):
            hit.setdefault(col, None)
    return list(hit)


def entry_delay_warnings(ctx: Any) -> list[str]:
    """Warnings for an entry that fills at the signal bar's close on a close-derived signal.

    *ctx* is the run's ``SimulationContext``. A strategy with ``entry_delay=0`` and
    an entry rule (``signal_rules``, or a signal model's ``required_features``) on an
    unshifted signal column gets one message naming the columns. It is a warning, not
    an error: with a market-on-close order the strategy is executable.
    """
    if not entry_delay_zero(ctx.exit_strategies):
        return []
    required = ctx.required_features if ctx.model_type == "signal" else ()
    columns = close_decision_columns(ctx.indicator_plugins, ctx.signal_rules, required)
    if not columns:
        return []
    shown = ", ".join(columns[:5]) + (", ..." if len(columns) > 5 else "")
    return [
        f"entry_delay=0 with an entry rule on unshifted signal column(s) {shown}: "
        "the decision uses the signal bar's close and the order fills at that same "
        "close, so it is only executable as a market-on-close style order "
        "(decision and fill at the close). With a market order after the close, "
        "use entry_delay=1."
    ]


def warn_close_decisions(ctx: Any, sym: str = "") -> list[str]:
    """Log ``entry_delay_warnings`` for the run and return the messages."""
    messages = entry_delay_warnings(ctx)
    for message in messages:
        logger.warning(message)
        run_log(1, f"WARNING: {message}", sym)
    return messages


__all__ = [
    "close_decision_columns",
    "entry_delay_warnings",
    "entry_delay_zero",
    "rule_columns",
    "warn_close_decisions",
]
