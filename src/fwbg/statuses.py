"""Shared result-status definitions used by the optimizer and research tools."""

# Per-symbol statuses that mean a backtest actually executed (as opposed to
# bailing out before evaluation: NOT_EXECUTED_STATUSES).
EXECUTED_STATUSES = frozenset({"ok", "no_edge", "not_significant", "no_unified_trades"})

# Per-symbol statuses that end before a backtest was evaluated.
NOT_EXECUTED_STATUSES = frozenset(
    {
        "data_invalid",
        "error",
        "insufficient_data",
        "insufficient_data_for_folds",
        "macro_asset",
        "missing_model_dependencies",
        "no_data",
        "no_signal_source",
        "no_successful_folds",
    }
)
