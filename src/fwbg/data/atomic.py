"""Crash-safe file writes: readers see the old file or the new one, never a partial one."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def write_text_atomic(dest: Path | str, text: str) -> None:
    """Write ``text`` to ``dest`` through a temp file in the same directory."""
    dest = Path(dest)
    fd, temp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp, dest)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def write_csv_atomic(df: pd.DataFrame, dest: Path | str, **to_csv_kwargs: Any) -> None:
    """``df.to_csv(dest, ...)`` via a temp file and ``os.replace``.

    An interrupted write leaves any existing ``dest`` untouched. The temp file ends in
    ``.tmp`` so ``*.csv`` globs never pick up a half-written file.
    """
    dest = Path(dest)
    fd, temp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, **to_csv_kwargs)
        os.replace(temp, dest)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
