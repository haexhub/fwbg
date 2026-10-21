"""How a run chose its settings, recorded per symbol as ``selection_protocol``.

The research register reads the recorded value from the run files (JSON); it never
imports this module.
"""

# Protocol 2: each fold's setting (signal and ML models) and the indicator variant are
# scored on inner windows of the fold's training data, never on the fold's test set.
# Known remaining test-set influences: ML folds are dropped when they have too few test
# trades (process_fold.py), and the unified merge applies settings from later folds to
# earlier test folds (D6, protocol 3). Runs without the marker predate protocol 2.
SELECTION_PROTOCOL = 2
