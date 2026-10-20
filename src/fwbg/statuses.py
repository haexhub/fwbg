"""Shared result-status definitions used by the optimizer and research tools."""

# Per-symbol statuses that mean a backtest actually executed (as opposed to
# bailing out before evaluation: no_data, insufficient_data, no_signal_source,
# no_successful_folds, missing_model_dependencies, error, ...).
EXECUTED_STATUSES = frozenset({"ok", "no_edge", "not_significant", "no_unified_trades"})
