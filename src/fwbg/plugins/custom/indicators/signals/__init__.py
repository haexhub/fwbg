"""Computed Signal Plugin — evaluates rule-based signal definitions.

Each signal definition is a JSON file in the definitions/ directory.
Rules combine conditions on existing indicator columns to produce
new discrete signal columns (0 or 1, NaN where an input is NaN).

Signal definitions are auto-discovered by the PluginRegistry and
registered as separate plugins under the "custom" namespace.
"""

import operator
from typing import Any, Dict, Iterable, List

import pandas as pd

from fwbg_sdk.indicators import BaseIndicator, shift_features
from fwbg_sdk.registry import register_indicator


_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}


# Column names that always belong to the market data and may never be used as
# the output of a signal rule.
RESERVED_COLUMNS = frozenset(
    {
        "O",
        "H",
        "L",
        "C",
        "V",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
)


def validate_output_name(name: Any, existing: Iterable[str] = ()) -> str:
    """Return ``name`` if it is a legal, non-colliding output column name.

    Raises ValueError for non-identifiers, OHLCV names and names that already
    exist as a column of the frame.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"Invalid signal output name {name!r}: not an identifier")
    if name in RESERVED_COLUMNS:
        raise ValueError(f"Signal output name {name!r} collides with an OHLCV column")
    if name in set(existing):
        raise ValueError(f"Signal output name {name!r} already exists as a column")
    return name


def _eval_condition(series: pd.Series, op: str, value: Any) -> pd.Series:
    """Evaluate a single condition against a pandas Series.

    Returns a nullable boolean Series. Rows where the compared input is NaN
    (or, for crossings, the previous input is NaN) are <NA>: an unknown input
    never yields True, also not for ``!=``.
    """
    if op == "crosses_above":
        prev = series.shift(1)
        result = (prev <= value) & (series > value)
        unknown = prev.isna() | series.isna()
    elif op == "crosses_below":
        prev = series.shift(1)
        result = (prev >= value) & (series < value)
        unknown = prev.isna() | series.isna()
    else:
        fn = _OPS.get(op)
        if fn is None:
            raise ValueError(f"Unsupported operator: {op}")
        result = fn(series, value)
        unknown = series.isna()
    return result.astype("boolean").mask(unknown)


def _evaluate_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    """Evaluate a single rule (AND/OR of conditions) against a DataFrame.

    Conditions combine with three-valued (Kleene) logic: the result is 1.0 or
    0.0 when it is decided, and NaN when a NaN input leaves it undecided
    (for example AND of a true condition and an unknown one).
    """
    conditions = rule.get("conditions", [])
    logic = rule.get("logic", "AND").upper()

    if not conditions:
        return pd.Series(0.0, index=df.index)

    if logic == "OR":
        result = pd.Series(False, index=df.index, dtype="boolean")
        for cond in conditions:
            col = cond["column"]
            if col not in df.columns:
                continue
            result = result | _eval_condition(df[col], cond["op"], cond["value"])
    else:
        # AND (default)
        result = pd.Series(True, index=df.index, dtype="boolean")
        for cond in conditions:
            col = cond["column"]
            if col not in df.columns:
                result = pd.Series(False, index=df.index, dtype="boolean")
                break
            result = result & _eval_condition(df[col], cond["op"], cond["value"])

    return result.astype("float64")


@register_indicator("computed_signal")
class ComputedSignalPlugin(BaseIndicator):
    """Base class for rule-based computed signals.

    Subclasses (created dynamically per JSON definition) override
    ``_rules`` with their specific rule list.  The base registration
    under ``computed_signal`` is kept so the class can be imported
    by the dynamic class factory in the registry.
    """

    name = "computed_signal"
    version = "1.1.0"
    group = "custom"

    # Overridden per dynamically-created subclass
    _rules: List[dict] = []

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        features: Dict[str, pd.Series] = {}
        for rule in self._rules:
            output = validate_output_name(
                rule.get("output"), existing=[*df.columns, *features]
            )
            features[output] = _evaluate_rule(df, rule)

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [r["output"] for r in self._rules]

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_param_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {}
