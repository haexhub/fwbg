"""
Logging-Utilities für den Optimizer.
"""
import os
import sys

# Logging-Level: 0=aus, 1=basic, 2=detail, 3=debug
LOG_LEVEL = int(os.environ.get("OPTIMIZER_LOG", "1"))

# Flag: Unterdrücke Detail-Logs wenn Progress-UI aktiv ist
_progress_ui_active = False


def set_progress_ui_active(active: bool):
    """Setzt ob die Progress-UI aktiv ist (unterdrückt dann Detail-Logs)."""
    global _progress_ui_active
    _progress_ui_active = active


def log(level, msg, sym=""):
    """
    Logging-Funktion mit Level-Kontrolle.

    Args:
        level: Logging-Level (0=aus, 1=basic, 2=detail, 3=debug)
        msg: Nachricht
        sym: Symbol-Prefix (optional)
    """
    # Unterdrücke Level 2+ Logs wenn Progress-UI aktiv
    if _progress_ui_active and level >= 2:
        return

    if level <= LOG_LEVEL:
        prefix = f"[{sym}] " if sym else ""
        print(f"{prefix}{msg}", file=sys.stderr, flush=True)
