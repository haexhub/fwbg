"""
Progress-Tracking für den Optimizer.
Ermöglicht Echtzeit-Fortschrittsanzeige im Haupt-Prozess.

HINWEIS: multiprocessing.Manager() wurde entfernt wegen Deadlock mit ProcessPoolExecutor.
         Der SimpleProgressTracker läuft nur im Hauptprozess und wird über Callbacks aktualisiert.
"""
import sys
import time
import threading
from typing import Optional, List

from .logging_utils import set_progress_ui_active


def report_progress(*args, **kwargs):
    """Legacy-Funktion - tut nichts mehr (war für multiprocessing.Manager)."""
    pass


def report_done(symbol: str, status: str = "ok"):
    """Legacy-Funktion - tut nichts mehr (war für multiprocessing.Manager)."""
    pass


class SimpleProgressTracker:
    """
    Einfacher Progress-Tracker für den Hauptprozess.

    Zeigt eine schöne Fortschrittsanzeige an, ohne shared state zwischen Prozessen.
    Wird nur über update_completed() aus dem Hauptprozess aktualisiert.
    """

    def __init__(self, total_assets: int, asset_names: Optional[List[str]] = None):
        self.total_assets = total_assets
        self.completed_assets = 0
        self.asset_names = asset_names or []
        self.completed_symbols: List[str] = []
        self.start_time = None
        self._stop_event = threading.Event()
        self._display_thread = None
        self._last_line_count = 0
        self._lock = threading.Lock()

    def start(self):
        """Startet das Progress-Display."""
        self.start_time = time.time()
        self._stop_event.clear()

        # Detail-Logs unterdrücken während Progress-UI aktiv
        set_progress_ui_active(True)

        # Display-Thread starten
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def update_completed(self, completed: int, symbol: Optional[str] = None):
        """Aktualisiert die Anzahl der fertigen Assets."""
        with self._lock:
            self.completed_assets = completed
            if symbol and symbol not in self.completed_symbols:
                self.completed_symbols.append(symbol)

    def stop(self):
        """Stoppt das Progress-Display."""
        self._stop_event.set()
        if self._display_thread:
            self._display_thread.join(timeout=1)
        # Letzte Zeilen löschen
        self._clear_lines()
        # Detail-Logs wieder erlauben
        set_progress_ui_active(False)

    def _clear_lines(self):
        """Löscht die zuletzt geschriebenen Zeilen."""
        if self._last_line_count > 0:
            sys.stdout.write(f"\033[{self._last_line_count}F")
            for _ in range(self._last_line_count):
                sys.stdout.write("\033[K\n")
            sys.stdout.write(f"\033[{self._last_line_count}F")
            sys.stdout.flush()
            self._last_line_count = 0

    def _display_loop(self):
        """Haupt-Display-Loop (läuft im Thread)."""
        while not self._stop_event.is_set():
            self._render()
            time.sleep(0.5)

    def _render(self):
        """Rendert den aktuellen Status."""
        self._clear_lines()

        with self._lock:
            completed = self.completed_assets
            completed_symbols = self.completed_symbols[:]

        lines = []
        box_width = 74

        elapsed = time.time() - self.start_time if self.start_time else 0
        pct = (completed / self.total_assets * 100) if self.total_assets > 0 else 0
        pending = self.total_assets - completed

        # ETA berechnen
        eta_str = self._calculate_eta(elapsed, completed)
        elapsed_str = self._format_time(elapsed)

        # === HEADER ===
        lines.append(f"╔{'═' * box_width}╗")

        # Status-Zeile
        status_line = f"Done: {completed}  |  Pending: {pending}  |  Total: {self.total_assets}"
        lines.append(f"║ {status_line:^{box_width-2}} ║")

        # === RECENT COMPLETIONS ===
        lines.append(f"╠{'═' * box_width}╣")

        if completed_symbols:
            # Zeige die letzten 3 fertigen Assets
            recent = completed_symbols[-3:]
            recent_str = "  ".join(f"✓ {s}" for s in recent)
            if len(recent_str) > box_width - 4:
                recent_str = recent_str[:box_width-7] + "..."
            lines.append(f"║ {recent_str:<{box_width-2}} ║")
        else:
            lines.append(f"║ {'Processing...':<{box_width-2}} ║")

        # === PROGRESS BAR ===
        lines.append(f"╠{'═' * box_width}╣")

        bar_width_inner = 50
        filled = int(bar_width_inner * completed / self.total_assets) if self.total_assets > 0 else 0
        filled = min(filled, bar_width_inner)
        bar = "█" * filled + "░" * (bar_width_inner - filled)

        progress_line = f" [{bar}] {pct:.1f}%"
        lines.append(f"║{progress_line:<{box_width}}║")

        # Zeit-Info
        time_line = f" Elapsed: {elapsed_str}  |  ETA: {eta_str}"
        lines.append(f"║{time_line:<{box_width}}║")

        lines.append(f"╚{'═' * box_width}╝")

        # Ausgeben
        output = "\n".join(lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

        self._last_line_count = len(lines)

    def _calculate_eta(self, elapsed: float, completed: int) -> str:
        """Berechnet ETA basierend auf bisherigem Fortschritt."""
        if elapsed < 10 or completed < 1:
            return "--:--"

        time_per_asset = elapsed / completed
        remaining = self.total_assets - completed

        if remaining <= 0:
            return "00:00"

        remaining_time = remaining * time_per_asset
        return self._format_time(remaining_time)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Formatiert Sekunden als MM:SS oder HH:MM:SS."""
        if seconds < 0:
            return "--:--"
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
