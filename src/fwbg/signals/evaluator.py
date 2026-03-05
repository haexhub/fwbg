"""Recursive evaluator for nested signal composition rules.

A rule tree looks like::

    {
        "operator": "AND",
        "conditions": [
            {"type": "signal_active", "column": "breakout_up"},
            {"type": "value_check", "column": "adx_14", "op": ">=", "value": 25},
            {
                "type": "group",
                "operator": "OR",
                "conditions": [
                    {"type": "crossing", "column_a": "ema_9", "column_b": "ema_21", "direction": "above"},
                    {"type": "col_compare", "column_a": "close", "column_b": "ema_21", "op": ">"},
                ],
            },
        ],
    }

``evaluate_rules(rules, df)`` returns a boolean ``pd.Series`` aligned to *df*
that is ``True`` wherever the composed signal fires.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

_VALID_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def resolve_column(short_name: str, columns: Sequence[str]) -> str:
    """Resolve *short_name* to a full column name present in *columns*.

    Resolution order:
    1. Exact match.
    2. Unique suffix match where the full column ends with ``_<short_name>``
       or ``_<short_name>`` after the last ``_`` prefix.

    Raises ``KeyError`` when the name cannot be resolved.
    """
    cols = list(columns)

    # 1) exact
    if short_name in cols:
        return short_name

    # 2) suffix match — column ends with "_<short_name>"
    suffix = f"_{short_name}"
    matches = [c for c in cols if c.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(
            f"Ambiguous column '{short_name}' matches multiple columns: {matches}"
        )

    raise KeyError(f"Column '{short_name}' not found in {cols}")


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _apply_op(left: pd.Series, op: str, right: pd.Series | float | int) -> pd.Series:
    """Apply a comparison *op* element-wise, filling NaN results with False."""
    if op not in _VALID_OPS:
        raise ValueError(f"Unsupported operator '{op}'. Valid: {sorted(_VALID_OPS)}")

    if op == "==":
        result = left == right
    elif op == "!=":
        result = left != right
    elif op == "<":
        result = left < right
    elif op == "<=":
        result = left <= right
    elif op == ">":
        result = left > right
    else:  # >=
        result = left >= right

    return result.fillna(False).astype(bool)


# ---------------------------------------------------------------------------
# Condition dispatchers
# ---------------------------------------------------------------------------

def _eval_signal_active(cond: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    col = resolve_column(cond["column"], df.columns)
    return df[col].fillna(0) == 1


def _eval_value_check(cond: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    col = resolve_column(cond["column"], df.columns)
    return _apply_op(df[col], cond["op"], cond["value"])


def _eval_col_compare(cond: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    col_a = resolve_column(cond["column_a"], df.columns)
    col_b = resolve_column(cond["column_b"], df.columns)
    return _apply_op(df[col_a], cond["op"], df[col_b])


def _eval_hour_filter(cond: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """True when the bar's hour is in the allowed set.

    Config examples:
        {"type": "hour_filter", "hours": [14, 15, 16, 17]}
        {"type": "hour_filter", "hours": [14, 15, 16, 17], "exclude": true}
    """
    hours = set(cond["hours"])
    exclude = cond.get("exclude", False)
    mask = df.index.hour.isin(hours)
    return (~mask if exclude else mask).astype(bool)


def _eval_day_filter(cond: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """True when the bar's day of week is in the allowed set.

    Config: {"type": "day_filter", "days": [0, 1, 2, 3, 4]}
    (0=Monday, 6=Sunday)
    """
    days = set(cond["days"])
    exclude = cond.get("exclude", False)
    mask = df.index.dayofweek.isin(days)
    return (~mask if exclude else mask).astype(bool)


def _eval_crossing(cond: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    col_a = resolve_column(cond["column_a"], df.columns)
    col_b = resolve_column(cond["column_b"], df.columns)
    a = df[col_a]
    b = df[col_b]
    direction = cond["direction"]

    if direction == "above":
        # a crosses above b: currently a > b AND previously a <= b
        result = (a > b) & (a.shift(1) <= b.shift(1))
    elif direction == "below":
        # a crosses below b: currently a < b AND previously a >= b
        result = (a < b) & (a.shift(1) >= b.shift(1))
    else:
        raise ValueError(f"Unknown crossing direction '{direction}'. Use 'above' or 'below'.")

    return result.fillna(False).astype(bool)


_CONDITION_DISPATCH: dict[str, Any] = {
    "signal_active": _eval_signal_active,
    "value_check": _eval_value_check,
    "col_compare": _eval_col_compare,
    "crossing": _eval_crossing,
    "hour_filter": _eval_hour_filter,
    "day_filter": _eval_day_filter,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_condition(cond: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """Evaluate a single condition dict against *df*.

    If ``cond["type"]`` is ``"group"``, this recurses via ``evaluate_rules``.
    """
    cond_type = cond["type"]

    if cond_type == "group":
        return evaluate_rules(cond, df)

    handler = _CONDITION_DISPATCH.get(cond_type)
    if handler is None:
        raise ValueError(
            f"Unknown condition type '{cond_type}'. "
            f"Valid: {sorted(_CONDITION_DISPATCH)} + ['group']"
        )
    return handler(cond, df)


def evaluate_rules(rules: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """Recursively evaluate a rule tree and return a boolean Series.

    Parameters
    ----------
    rules : dict
        ``{"operator": "AND"|"OR", "conditions": [...]}``.
    df : pd.DataFrame
        The indicator DataFrame.

    Returns
    -------
    pd.Series[bool]
        True where the composed signal fires.
    """
    operator = rules.get("operator", "AND").upper()
    conditions: list[dict[str, Any]] = rules.get("conditions", [])

    if operator not in ("AND", "OR"):
        raise ValueError(f"Unknown operator '{operator}'. Use 'AND' or 'OR'.")

    # Empty conditions => all True (neutral element for AND)
    if not conditions:
        return pd.Series(True, index=df.index, dtype=bool)

    results = [evaluate_condition(c, df) for c in conditions]

    if operator == "AND":
        combined = results[0]
        for r in results[1:]:
            combined = combined & r
        return combined

    # OR
    combined = results[0]
    for r in results[1:]:
        combined = combined | r
    return combined
