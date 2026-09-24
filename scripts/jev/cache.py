from pathlib import Path

import pandas as pd


class PredictionCache:
    """Append-only CSV cache, keyed by timestamp. Safe to resume after a
    partial/interrupted batch run — already-answered timestamps are skipped.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._df = pd.read_csv(self.path, index_col=0)
        else:
            self._df = pd.DataFrame()

    def pending_timestamps(self, all_timestamps: list) -> list:
        done = set(self._df.index)
        return [t for t in all_timestamps if t not in done]

    def append(self, timestamp, values: dict) -> None:
        self._df.loc[timestamp, list(values.keys())] = list(values.values())
        self._df.to_csv(self.path)

    def as_dataframe(self) -> pd.DataFrame:
        return self._df.copy()
