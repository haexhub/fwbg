"""How a run chose its settings, recorded per symbol as ``selection_protocol``.

The research register reads the recorded value from the run files (JSON); it never
imports this module.
"""

# Protocol 2: fold settings and the indicator variant are scored on inner training windows.
# Protocol 3: in addition, each test fold is replayed with the indicator variant and the
# setting merged from the folds up to it (never a later fold's), and ML folds are kept
# whatever their test trade count. Remaining hindsight: no_edge and the sizing fallback
# with few causal trades still use all folds (register design doc).
# Runs without the marker predate protocol 2.
SELECTION_PROTOCOL = 3
