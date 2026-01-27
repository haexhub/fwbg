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
    """

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
        self._clear_line()
        print(f"Verarbeitung abgeschlossen: {self.completed_assets}/{self.total_assets} in {self._format_time(elapsed)}")
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

    def _clear_line(self):
        """Löscht die aktuelle Zeile."""
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _display_loop(self):
        """Haupt-Display-Loop (läuft im Thread)."""
        while not self._stop_event.is_set():
            self._render()
            time.sleep(0.5)

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

        # Progress bar
        bar_width = 25
        filled = int(bar_width * total_progress / self.total_assets) if self.total_assets > 0 else 0
        filled = min(filled, bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Worker-Status zusammenfassen
        if active_workers:
            worker_info = []
            for pid, info in sorted(active_workers.items(), key=lambda x: x[1].get("symbol", "")):
                sym = info.get("symbol", "?")[:6]
                grid_pos = info.get("grid_pos", 0)
                grid_total = info.get("grid_total", 0)
                if grid_total > 0:
                    grid_pct = int(grid_pos / grid_total * 100)
                    worker_info.append(f"{sym}:{grid_pct}%")
                else:
                    worker_info.append(f"{sym}")
            workers_str = " ".join(worker_info[:4])  # Max 4 Worker anzeigen
            if len(active_workers) > 4:
                workers_str += f" +{len(active_workers)-4}"
        elif completed_symbols:
            recent = completed_symbols[-2:]
            workers_str = " ".join(f"✓{s}" for s in recent)
        else:
            workers_str = "starting..."

        # Einzeilige Ausgabe
        line = f"[{bar}] {pct:5.1f}% | {completed}/{self.total_assets} | {elapsed_str} | ETA: {eta_str} | {workers_str}"

        # Zeile auf max 100 Zeichen begrenzen
        if len(line) > 100:
            line = line[:97] + "..."

        self._clear_line()
        sys.stdout.write(line)
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

        time_per_unit = elapsed / total_progress
        remaining = self.total_assets - total_progress

        if remaining <= 0:
            return "00:00"

        remaining_time = remaining * time_per_unit
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
