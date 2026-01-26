"""
Logging-Utilities für den Optimizer.
"""
import os
import sys

# Logging-Level: 0=aus, 1=basic, 2=detail, 3=debug
LOG_LEVEL = int(os.environ.get("OPTIMIZER_LOG", "1"))


def log(level, msg, sym=""):
    """
    Logging-Funktion mit Level-Kontrolle.

    Args:
        level: Logging-Level (0=aus, 1=basic, 2=detail, 3=debug)
        msg: Nachricht
        sym: Symbol-Prefix (optional)
    """
    if level <= LOG_LEVEL:
        prefix = f"[{sym}] " if sym else ""
        print(f"{prefix}{msg}", file=sys.stderr, flush=True)
