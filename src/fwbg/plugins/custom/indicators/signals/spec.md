# Plugin Spec — signals

**Kind**: indicator  •  **Version**: 1.0.0

## Capability

Evaluates JSON-defined rule sets combining AND/OR conditions (comparison and crosses_above/below) on existing indicator columns to produce discrete 0/1 signal feature columns.

## Summary

Base indicator that turns rule-based signal definitions (loaded from JSON files under definitions/) into new binary feature columns. Each rule combines conditions on already-computed indicator columns using AND (default) or OR logic and emits one 0.0/1.0 float column named by the rule's `output`. Subclasses are generated dynamically per JSON definition and override `_rules`; the base class itself carries an empty rule list.

## Inputs

- df: pandas DataFrame containing the indicator columns referenced by each rule's conditions (rule-dependent; determined by the loaded JSON definitions)

## Parameters

- _none_

## Outputs

- One float column per rule, named by rule['output'], holding 0.0 or 1.0 per bar (shifted by one bar via shift_features to prevent lookahead)

## Acceptance Criteria

- AC-001: compute() iterates self._rules and, for each rule, writes a float column named rule['output'] into the returned DataFrame containing 0.0 or 1.0 per bar.
- AC-002: Rule evaluation defaults to AND logic; setting rule['logic'] to 'OR' (case-insensitive) combines its conditions with logical OR instead.
- AC-003: Supports the comparison operators '==', '!=', '>', '<', '>=', '<=' via the _OPS map, applied elementwise between the referenced column and the condition's `value`.
- AC-004: Supports 'crosses_above' (previous bar <= value AND current bar > value) and 'crosses_below' (previous bar >= value AND current bar < value) using series.shift(1).
- AC-005: Under AND logic, if any referenced condition column is missing from df, that rule's output becomes all 0.0 (the loop breaks with an all-False result).
- AC-006: Under OR logic, conditions whose column is missing from df are silently skipped; remaining conditions still contribute to the OR.
- AC-007: A rule with an empty 'conditions' list produces an all-0.0 output column.
- AC-008: Generated feature columns are passed through shift_features before being concatenated onto df, so signal values at bar i are derived from data available at bar i-1 (no lookahead).
- AC-009: If self._rules is empty (e.g., the base class), compute() returns the input DataFrame unchanged.
- AC-010: get_feature_columns() and get_signal_columns() both return the list of rule['output'] names in rule order.
- AC-011: get_default_params() and get_param_schema() return empty dicts — the plugin exposes no user-tunable parameters.

## Edge Cases

- Rule with an empty 'conditions' list → returns a 0.0 series for that rule's output.
- AND rule referencing a column that is not present in df → that rule's output is entirely 0.0.
- OR rule referencing a column that is not present in df → that condition is skipped without error; the rule can still fire on other conditions.
- Unsupported comparison operator (not in _OPS and not 'crosses_above'/'crosses_below') → _eval_condition raises ValueError('Unsupported operator: ...').
- First bar of a 'crosses_above'/'crosses_below' condition: prev = series.shift(1) is NaN, so the crossing comparison evaluates to False for that bar.
- self._rules is empty (base class or a JSON definition with no rules) → compute() returns df unchanged, and get_feature_columns()/get_signal_columns() return [].

## Assumptions

- 'signals' refers to this base computed-signal plugin as a whole; individual JSON-defined subclasses are cataloged separately by the registry and are out of scope for this spec.
- 'value' in a condition is always a scalar comparable to the referenced Series (no column-vs-column comparisons are supported by the current code).

## Needs Clarification

- [NEEDS CLARIFICATION: The class is registered under 'computed_signal' but is described as a base for dynamically-generated per-JSON-file subclasses; confirm whether it is intended to be instantiated directly (with an empty rule set) or only via the registry's class factory.]
- [NEEDS CLARIFICATION: Whether rule['output'] names are expected to already carry a slug/group prefix (per the naming convention that feature columns be prefixed with the plugin slug) or whether the dynamic subclass factory injects that prefix.]
