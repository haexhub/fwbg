"""
Logging-Utilities für den Optimizer.
"""
import os
import sys

# Logging-Level: 0=aus, 1=basic, 2=detail, 3=debug
LOG_LEVEL = int(os.environ.get("OPTIMIZER_LOG", "1"))

# Globaler Log-Buffer (pro Worker-Prozess) – None = inaktiv
_log_buffer: list[str] | None = None


def start_log_capture() -> None:
    """Startet das Mitschneiden aller Log-Nachrichten in einen Buffer."""
    global _log_buffer
    _log_buffer = []


def stop_log_capture() -> list[str]:
    """Beendet den Mitschnitt und gibt die gesammelten Nachrichten zurück."""
    global _log_buffer
    captured = _log_buffer or []
    _log_buffer = None
    return captured


def set_progress_ui_active(active: bool):
    """Setzt ob die Progress-UI aktiv ist (unterdrückt dann Detail-Logs)."""
    # Umgebungsvariable setzen, damit auch Worker-Prozesse es sehen
    if active:
        os.environ["_OPTIMIZER_PROGRESS_UI"] = "1"
    else:
        os.environ.pop("_OPTIMIZER_PROGRESS_UI", None)


_LEVEL_MAP = {0: "error", 1: "info", 2: "debug", 3: "debug"}


def log(level, msg, sym=""):
    """
    Logging-Funktion mit Level-Kontrolle.

    Args:
        level: Logging-Level (0=aus, 1=basic, 2=detail, 3=debug)
        msg: Nachricht
        sym: Symbol-Prefix (optional)
    """
    prefix = f"[{sym}] " if sym else ""
    line = f"{prefix}{msg}"

    # Immer in den Buffer schreiben, unabhängig von Level/UI
    if _log_buffer is not None:
        _log_buffer.append(line)

    # Route to structured JSONL logger (logs.jsonl) via progress queue
    from .progress import report_log
    report_log(
        symbol=sym or "",
        stage="processing",
        level=_LEVEL_MAP.get(level, "info"),
        message=msg,
    )

    # Unterdrücke nur Level-1 Logs wenn Progress-UI aktiv (Level 2+ kommen durch)
    # Das verhindert Spam bei normalen Status-Meldungen, zeigt aber wichtige Details
    if os.environ.get("_OPTIMIZER_PROGRESS_UI") and level == 1:
        return

    if level <= LOG_LEVEL:
        print(line, file=sys.stderr, flush=True)
