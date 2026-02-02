"""
Results Module - Ergebnis-Speicherung und Visualisierung.
"""
from .storage import (
    generate_run_id,
    create_run_directory,
    save_run_results,
    load_run,
)
from .plotting import (
    create_incremental_plot,
    create_elite_plot,
)

__all__ = [
    "generate_run_id",
    "create_run_directory",
    "save_run_results",
    "load_run",
    "create_incremental_plot",
    "create_elite_plot",
]
