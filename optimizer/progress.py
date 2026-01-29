"""
Progress-Tracking für den Optimizer.
Ermöglicht Echtzeit-Fortschrittsanzeige im Haupt-Prozess.

Verwendet multiprocessing.Queue statt Manager für deadlock-freie Kommunikation.
"""
import os
import sys
import time
import threading
from multiprocessing import Queue
from queue import Empty
from typing import Optional, List, Dict

from .logging_utils import set_progress_ui_active


# Globale Queue für Worker-Updates (wird in main.py initialisiert)
_progress_queue: Optional[Queue] = None


def init_progress_queue() -> Queue:
    """Erstellt eine neue Progress-Queue."""
    global _progress_queue
    _progress_queue = Queue()
    return _progress_queue


def set_progress_queue(queue: Queue):
    """Setzt die Progress-Queue (für Worker-Prozesse via Initializer)."""
    global _progress_queue
    _progress_queue = queue


def get_progress_queue() -> Optional[Queue]:
    """Gibt die Progress-Queue zurück."""
    return _progress_queue


def report_progress(
    symbol: str,
    fold: int = 0,
    total_folds: int = 0,
    stage: str = "",
    grid_pos: int = 0,
    grid_total: int = 0
):
    """
    Meldet Fortschritt aus einem Worker-Prozess.

    Args:
        symbol: Asset-Symbol (z.B. "EURUSD")
        fold: Aktueller Fold (1-basiert)
        total_folds: Gesamtzahl der Folds
        stage: Optionale Stufe (z.B. "train", "validate", "oos")
        grid_pos: Aktuelle Grid-Position (1-basiert)
        grid_total: Gesamtzahl der Grid-Kombinationen
    """
    if _progress_queue is not None:
        try:
            _progress_queue.put_nowait({
                "type": "progress",
                "pid": os.getpid(),
                "symbol": symbol,
                "fold": fold,
                "total_folds": total_folds,
                "stage": stage,
                "grid_pos": grid_pos,
                "grid_total": grid_total,
                "time": time.time(),
            })
        except Exception:
            pass  # Queue voll oder geschlossen - ignorieren


def report_done(symbol: str, status: str = "ok"):
    """Meldet, dass ein Worker fertig ist."""
    if _progress_queue is not None:
        try:
            _progress_queue.put_nowait({
                "type": "done",
                "pid": os.getpid(),
                "symbol": symbol,
                "status": status,
                "time": time.time(),
            })
        except Exception:
            pass


class ProgressTracker:
    """
    Progress-Tracker mit Queue-basierter Worker-Kommunikation.

    Zeigt Echtzeit-Fortschritt für alle aktiven Worker an.
    Unterstützt sowohl TTY (interaktiv) als auch Nicht-TTY (Log-Datei) Ausgabe.
    """

    # Konfiguration für die Anzeige
    MAX_DISPLAY_LINES = 15  # Maximale Zeilen für Worker-Status
    UPDATE_INTERVAL_TTY = 0.5  # Update-Intervall für TTY
    UPDATE_INTERVAL_NON_TTY = 30.0  # Update-Intervall für Nicht-TTY (Log)

    def __init__(self, total_assets: int, asset_names: Optional[List[str]] = None, queue: Optional[Queue] = None):
        self.total_assets = total_assets
        self.completed_assets = 0
        self.asset_names = asset_names or []
        self.completed_symbols: List[str] = []
        self.queue = queue
        self.worker_status: Dict[int, dict] = {}  # pid -> status
        self.start_time = None
        self._stop_event = threading.Event()
        self._display_thread = None
        self._queue_thread = None
        self._lock = threading.Lock()
        self._is_tty = sys.stdout.isatty()
        self._last_render_time = 0
        self._last_display_lines = 0  # Anzahl der zuletzt angezeigten Zeilen

    def start(self):
        """Startet das Progress-Display."""
        self.start_time = time.time()
        self._stop_event.clear()

        # Detail-Logs unterdrücken während Progress-UI aktiv
        set_progress_ui_active(True)

        # Queue-Reader Thread starten (falls Queue vorhanden)
        if self.queue is not None:
            self._queue_thread = threading.Thread(target=self._queue_reader, daemon=True)
            self._queue_thread.start()

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
        if self._queue_thread:
            self._queue_thread.join(timeout=1)

        # Finale Ausgabe
        elapsed = time.time() - self.start_time if self.start_time else 0

        if self._is_tty:
            # Lösche das Multi-Line-Display
            self._clear_display()

        print(f"\nVerarbeitung abgeschlossen: {self.completed_assets}/{self.total_assets} in {self._format_time(elapsed)}")
        sys.stdout.flush()

        # Detail-Logs wieder erlauben
        set_progress_ui_active(False)

    def _queue_reader(self):
        """Liest Updates aus der Queue."""
        while not self._stop_event.is_set():
            try:
                msg = self.queue.get(timeout=0.1)
                with self._lock:
                    if msg["type"] == "progress":
                        self.worker_status[msg["pid"]] = msg
                    elif msg["type"] == "done":
                        self.worker_status.pop(msg["pid"], None)
            except Empty:
                continue
            except Exception:
                break

    def _clear_display(self):
        """Löscht das gesamte Display (Multi-Line)."""
        if self._is_tty and self._last_display_lines > 0:
            # Cursor hoch und Zeilen löschen
            sys.stdout.write(f"\033[{self._last_display_lines}A")  # Cursor hoch
            for _ in range(self._last_display_lines):
                sys.stdout.write("\033[2K\033[1B")  # Zeile löschen, Cursor runter
            sys.stdout.write(f"\033[{self._last_display_lines}A")  # Cursor zurück
            sys.stdout.flush()

    def _display_loop(self):
        """Haupt-Display-Loop (läuft im Thread)."""
        update_interval = self.UPDATE_INTERVAL_TTY if self._is_tty else self.UPDATE_INTERVAL_NON_TTY

        while not self._stop_event.is_set():
            now = time.time()
            if now - self._last_render_time >= update_interval:
                self._render()
                self._last_render_time = now
            time.sleep(0.1)  # Kurzes Sleep für Queue-Updates

    def _render(self):
        """Rendert den aktuellen Status."""
        with self._lock:
            completed = self.completed_assets
            completed_symbols = self.completed_symbols[:]
            workers = dict(self.worker_status)

        elapsed = time.time() - self.start_time if self.start_time else 0

        # Aktive Worker filtern (nur die letzten 30 Sekunden)
        now = time.time()
        active_workers = {
            pid: info for pid, info in workers.items()
            if now - info.get("time", 0) < 30
        }

        # Gesamt-Fortschritt berechnen (inkl. Grid-Fortschritt der aktiven Worker)
        total_progress = self._calculate_total_progress(completed, active_workers)
        pct = (total_progress / self.total_assets * 100) if self.total_assets > 0 else 0

        # ETA berechnen (inkl. Grid-Fortschritt)
        eta_str = self._calculate_eta(elapsed, total_progress)
        elapsed_str = self._format_time(elapsed)

        if self._is_tty:
            self._render_tty(completed, total_progress, pct, elapsed_str, eta_str, active_workers, completed_symbols)
        else:
            self._render_non_tty(completed, pct, elapsed_str, eta_str, active_workers)

    def _render_tty(self, completed: int, total_progress: float, pct: float,
                    elapsed_str: str, eta_str: str, active_workers: Dict, completed_symbols: List[str]):
        """Rendert für TTY mit Multi-Line Display und Cursor-Steuerung."""
        lines = []

        # Progress bar
        bar_width = 40
        filled = int(bar_width * total_progress / self.total_assets) if self.total_assets > 0 else 0
        filled = min(filled, bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Header-Zeile
        lines.append(f"╔══════════════════════════════════════════════════════════════════╗")
        lines.append(f"║ [{bar}] {pct:5.1f}%")
        lines.append(f"║ Assets: {completed}/{self.total_assets} | Zeit: {elapsed_str} | ETA: {eta_str}")
        lines.append(f"╠══════════════════════════════════════════════════════════════════╣")

        # Worker-Status (max. MAX_DISPLAY_LINES Zeilen)
        if active_workers:
            worker_list = sorted(active_workers.items(), key=lambda x: x[1].get("symbol", ""))
            for i, (pid, info) in enumerate(worker_list[:self.MAX_DISPLAY_LINES - 6]):
                sym = info.get("symbol", "?")
                grid_pos = info.get("grid_pos", 0)
                grid_total = info.get("grid_total", 0)

                if grid_total > 0:
                    worker_pct = int(grid_pos / grid_total * 100)
                    worker_bar_width = 20
                    worker_filled = int(worker_bar_width * grid_pos / grid_total)
                    worker_bar = "▓" * worker_filled + "░" * (worker_bar_width - worker_filled)
                    lines.append(f"║  {sym:<10} [{worker_bar}] {worker_pct:3d}%")
                else:
                    lines.append(f"║  {sym:<10} [starting...]")

            if len(worker_list) > self.MAX_DISPLAY_LINES - 6:
                lines.append(f"║  ... und {len(worker_list) - (self.MAX_DISPLAY_LINES - 6)} weitere Worker")
        else:
            lines.append(f"║  Starte Worker...")

        # Kürzlich fertige Assets
        if completed_symbols:
            recent = completed_symbols[-3:]
            lines.append(f"╠══════════════════════════════════════════════════════════════════╣")
            lines.append(f"║ Fertig: " + ", ".join(f"✓{s}" for s in recent))

        lines.append(f"╚══════════════════════════════════════════════════════════════════╝")

        # Altes Display löschen
        self._clear_display()

        # Neues Display ausgeben
        output = "\n".join(lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

        self._last_display_lines = len(lines)

    def _render_non_tty(self, completed: int, pct: float, elapsed_str: str,
                        eta_str: str, active_workers: Dict):
        """Rendert für Nicht-TTY (Log-Datei) - kompakte einzeilige Ausgabe."""
        # Worker-Status zusammenfassen
        if active_workers:
            worker_info = []
            for pid, info in sorted(active_workers.items(), key=lambda x: x[1].get("symbol", "")):
                sym = info.get("symbol", "?")[:8]
                grid_pos = info.get("grid_pos", 0)
                grid_total = info.get("grid_total", 0)
                if grid_total > 0:
                    grid_pct = int(grid_pos / grid_total * 100)
                    worker_info.append(f"{sym}:{grid_pct}%")
            workers_str = " ".join(worker_info[:6])  # Max 6 Worker anzeigen
            if len(active_workers) > 6:
                workers_str += f" +{len(active_workers)-6}"
        else:
            workers_str = "starting..."

        # Kompakte Ausgabe für Logs
        print(f"[PROGRESS] {pct:5.1f}% | {completed}/{self.total_assets} | {elapsed_str} | ETA: {eta_str} | {workers_str}")
        sys.stdout.flush()

    def _calculate_total_progress(self, completed: int, active_workers: Dict) -> float:
        """Berechnet Gesamt-Fortschritt inkl. Grid-Progress der aktiven Worker."""
        total_progress = float(completed)

        for info in active_workers.values():
            grid_pos = info.get("grid_pos", 0)
            grid_total = info.get("grid_total", 1)
            if grid_total > 0:
                total_progress += grid_pos / grid_total

        return total_progress

    def _calculate_eta(self, elapsed: float, total_progress: float) -> str:
        """Berechnet ETA basierend auf Gesamt-Fortschritt."""
        if elapsed < 10 or total_progress < 0.1:
            return "--:--"

        # total_progress ist der Fortschritt in "Asset-Einheiten" (0 bis total_assets)
        # Berechne verbleibende Zeit: elapsed / progress * remaining
        remaining_progress = self.total_assets - total_progress

        if remaining_progress <= 0:
            return "00:00"

        # Zeit pro Fortschritts-Einheit * verbleibender Fortschritt
        remaining_time = (elapsed / total_progress) * remaining_progress
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


# Alias für Rückwärtskompatibilität
SimpleProgressTracker = ProgressTracker
