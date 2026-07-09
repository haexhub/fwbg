# Plugin Spec — signal

**Kind**: model  •  **Version**: 3.0.0

## Capability

Uses pre-composed signal columns (_composed_signal_long/_composed_signal_short) as deterministic entry probabilities instead of training a classifier.

## Summary

A non-learning model that surfaces pipeline-composed signal columns as class-1 probabilities, letting rule-based (e.g. signal_rules) strategies gate entries deterministically via the classification threshold.

## Inputs

- features DataFrame containing the composed signal column referenced by direction (default `_composed_signal_long` for long, `_composed_signal_short` for short)
- TrainingContext providing `direction` ('long' or 'short') to select which signal column to read

## Parameters

- `signal_column_long` (string, default='_composed_signal_long'): Name of the feature column read as the long-direction entry signal (values in [0, 1]).
- `signal_column_short` (string, default='_composed_signal_short'): Name of the feature column read as the short-direction entry signal (values in [0, 1]).

## Outputs

- Class-probability array of shape (n_rows, 2): column 0 = 1 - signal, column 1 = clipped signal value (0..1)
- trained_classes = np.array([0, 1])

## Acceptance Criteria

- AC-001: train() sets internal signal column to `signal_column_long` hyperparameter (or `_composed_signal_long` default) when training_context.direction == 'long', otherwise to `signal_column_short` (or `_composed_signal_short` default).
- AC-002: train() marks the model as fitted without consuming the features or targets — no learning occurs.
- AC-003: predict_probability returns an (n, 2) float64 array where column 1 equals the configured signal column value clipped to [0, 1] and column 0 equals 1 minus that value.
- AC-004: NaN entries in the signal column are treated as 0 before clipping.
- AC-005: If the configured signal column is absent from the feature frame, all rows return probability 0 for class 1 (and 1 for class 0).
- AC-006: trained_classes exposes the fixed binary class array [0, 1].
- AC-007: get_reduced_hyperparameters passes through only `signal_column_long` and `signal_column_short` keys, dropping any others.

## Edge Cases

- Configured signal column is missing from the features DataFrame — output probabilities default to [1.0, 0.0] for every row.
- Signal column contains NaN values — filled with 0 before use.
- Signal column contains values outside [0, 1] — clipped into the [0, 1] range before assignment.
- Empty features DataFrame — returns a (0, 2) probability array without error.
- training_context.direction is neither 'long' nor 'short' — falls through to the short branch and reads the short signal column.

## Assumptions

- _none_
