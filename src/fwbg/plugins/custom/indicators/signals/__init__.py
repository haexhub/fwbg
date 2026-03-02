"""Computed Signal Plugin — evaluates rule-based signal definitions.

Each signal definition is a JSON file in the definitions/ directory.
Rules combine conditions on existing indicator columns to produce
new discrete signal columns (0 or 1).

Signal definitions are auto-discovered by the PluginRegistry and
registered as separate plugins under the "custom" namespace.
"""
import operator
from typing import Any, Dict, List

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


def _eval_condition(series: pd.Series, op: str, value: Any) -> pd.Series:
    """Evaluate a single condition against a pandas Series."""
    if op == "crosses_above":
        prev = series.shift(1)
        return (prev <= value) & (series > value)
    if op == "crosses_below":
        prev = series.shift(1)
        return (prev >= value) & (series < value)

    fn = _OPS.get(op)
    if fn is None:
        raise ValueError(f"Unsupported operator: {op}")
    return fn(series, value)


def _evaluate_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    """Evaluate a single rule (AND/OR of conditions) against a DataFrame."""
    conditions = rule.get("conditions", [])
    logic = rule.get("logic", "AND").upper()

    if not conditions:
        return pd.Series(0.0, index=df.index)

    if logic == "OR":
        result = pd.Series(False, index=df.index)
        for cond in conditions:
            col = cond["column"]
            if col not in df.columns:
                continue
            result = result | _eval_condition(df[col], cond["op"], cond["value"])
    else:
        # AND (default)
        result = pd.Series(True, index=df.index)
        for cond in conditions:
            col = cond["column"]
            if col not in df.columns:
                result = pd.Series(False, index=df.index)
                break
            result = result & _eval_condition(df[col], cond["op"], cond["value"])

    return result.astype(float)


@register_indicator("computed_signal")
class ComputedSignalPlugin(BaseIndicator):
    """Base class for rule-based computed signals.

    Subclasses (created dynamically per JSON definition) override
    ``_rules`` with their specific rule list.  The base registration
    under ``computed_signal`` is kept so the class can be imported
    by the dynamic class factory in the registry.
    """

    name = "computed_signal"
    version = "1.0.0"
    group = "custom"

    # Overridden per dynamically-created subclass
    _rules: List[dict] = []

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        features: Dict[str, pd.Series] = {}
        for rule in self._rules:
            features[rule["output"]] = _evaluate_rule(df, rule)

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [r["output"] for r in self._rules]

    def get_signal_columns(self) -> List[str]:
        return [r["output"] for r in self._rules]

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_param_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {}
