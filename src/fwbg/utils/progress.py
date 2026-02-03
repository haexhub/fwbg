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

from .logging import set_progress_ui_active


# Globale Queue für Worker-Updates (wird in main.py initialisiert)
_progress_queue: Optional[Queue] = None

# Thread-lokale Daten für parallele Feature-Group-Verarbeitung
_thread_local = threading.local()


def set_parallel_mode(enabled: bool):
    """
    Aktiviert/Deaktiviert den Parallel-Modus für Feature-Group-Threads.

    Im Parallel-Modus werden Progress-Updates unterdrückt, um Chaos
    in der Progress-Bar zu vermeiden wenn mehrere Threads gleichzeitig laufen.
    """
    _thread_local.parallel_mode = enabled


def is_parallel_mode() -> bool:
    """Prüft ob der aktuelle Thread im Parallel-Modus läuft."""
    return getattr(_thread_local, 'parallel_mode', False)


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
    # Im Parallel-Modus Progress-Updates unterdrücken
    # (parallele Feature-Group-Threads würden sich sonst gegenseitig überschreiben)
    if is_parallel_mode():
        return

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


def report_phase(symbol: str, phase: str):
    """
    Meldet die aktuelle Verarbeitungsphase für ein Symbol.

    Diese Phase wird im Progress-UI angezeigt, solange keine neue Phase gemeldet wird.
    Beispiele: "Lade Daten...", "Berechne Indikatoren...", "Grid-Search...", etc.

    Args:
        symbol: Asset-Symbol (z.B. "EURUSD")
        phase: Beschreibung der aktuellen Phase
    """
    if _progress_queue is not None:
        try:
            _progress_queue.put_nowait({
                "type": "phase",
                "pid": os.getpid(),
                "symbol": symbol,
                "phase": phase,
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
    MAX_DISPLAY_LINES = 20  # Maximale Zeilen für Worker-Status
    UPDATE_INTERVAL_TTY = 0.3  # Update-Intervall für TTY (schnell für flüssige Anzeige)
    UPDATE_INTERVAL_NON_TTY = 10.0  # Update-Intervall für Nicht-TTY (seltener)

    def __init__(self, total_assets: int, asset_names: Optional[List[str]] = None, queue: Optional[Queue] = None):
        self.total_assets = total_assets
        self.completed_assets = 0
        self.asset_names = asset_names or []
        self.completed_symbols: List[str] = []
        self.queue = queue
        self.worker_status: Dict[int, dict] = {}  # pid -> status
        self.worker_phases: Dict[str, str] = {}  # symbol -> aktuelle Phase
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
                        # Index by symbol, not PID (multiple threads share same PID)
                        symbol = msg.get("symbol", msg["pid"])
                        self.worker_status[symbol] = msg
                    elif msg["type"] == "phase":
                        # Phase-Update: Symbol -> Phase-Text speichern
                        self.worker_phases[msg["symbol"]] = msg["phase"]
                    elif msg["type"] == "done":
                        # Remove by symbol
                        symbol = msg.get("symbol")
                        if symbol:
                            self.worker_status.pop(symbol, None)
                            self.worker_phases.pop(symbol, None)
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
            time.sleep(0.05)  # Kurzes Sleep für Queue-Updates

    def _render(self):
        """Rendert den aktuellen Status."""
        with self._lock:
            completed = self.completed_assets
            completed_symbols = self.completed_symbols[:]
            workers = dict(self.worker_status)
            phases = dict(self.worker_phases)

        elapsed = time.time() - self.start_time if self.start_time else 0

        # Aktive Worker filtern (nur die letzten 5 Minuten)
        now = time.time()
        active_workers = {
            pid: info for pid, info in workers.items()
            if now - info.get("time", 0) < 300
        }

        # Gesamt-Fortschritt berechnen (inkl. Grid-Fortschritt der aktiven Worker)
        total_progress = self._calculate_total_progress(completed, active_workers)
        pct = (total_progress / self.total_assets * 100) if self.total_assets > 0 else 0

        # ETA berechnen (inkl. Grid-Fortschritt)
        eta_str = self._calculate_eta(elapsed, total_progress)
        elapsed_str = self._format_time(elapsed)

        if self._is_tty:
            # TTY: Fixes Fenster mit Cursor-Steuerung
            self._render_fixed_window(completed, total_progress, pct, elapsed_str, eta_str, active_workers, completed_symbols, phases)
        else:
            # Non-TTY: Kompakte einzeilige Ausgabe
            self._render_compact(completed, pct, elapsed_str, eta_str, phases)

    def _render_fixed_window(self, completed: int, total_progress: float, pct: float,
                              elapsed_str: str, eta_str: str, active_workers: Dict,
                              completed_symbols: List[str], phases: Dict[str, str]):
        """Rendert ein fixes Fenster mit allen Assets und deren Fortschritt."""
        lines = []

        # Feste Fensterbreite für konsistentes Layout
        WIDTH = 78

        # Progress bar für Gesamtfortschritt
        bar_width = 50
        filled = int(bar_width * total_progress / self.total_assets) if self.total_assets > 0 else 0
        filled = min(filled, bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Header
        lines.append("╔" + "═" * (WIDTH - 2) + "╗")
        lines.append(f"║  FWBG Optimizer - {completed}/{self.total_assets} Assets | {elapsed_str} | ETA: {eta_str}".ljust(WIDTH - 1) + "║")
        lines.append(f"║  [{bar}] {pct:5.1f}%".ljust(WIDTH - 1) + "║")
        lines.append("╠" + "═" * (WIDTH - 2) + "╣")

        # Alle Assets aus asset_names anzeigen (nicht nur aktive)
        displayed = 0
        max_assets = self.MAX_DISPLAY_LINES - 7  # Platz für Header/Footer

        for sym in self.asset_names[:max_assets]:
            displayed += 1

            # Status ermitteln
            if sym in completed_symbols:
                # Fertig
                lines.append(f"║  ✓ {sym:<10} [████████████████] 100%  Fertig".ljust(WIDTH - 1) + "║")
            else:
                # Noch aktiv oder wartend
                # Worker-Info direkt per Symbol abrufen (nicht mehr per PID)
                worker_info = active_workers.get(sym)

                grid_pos = worker_info.get("grid_pos", 0) if worker_info else 0
                grid_total = worker_info.get("grid_total", 0) if worker_info else 0
                phase = phases.get(sym, "")

                if grid_total > 0:
                    # Hat Grid-Fortschritt - Progress-Bar und Text aus gleicher Quelle!
                    worker_pct = int(grid_pos / grid_total * 100)
                    worker_bar_width = 16
                    worker_filled = int(worker_bar_width * grid_pos / grid_total)
                    worker_bar = "▓" * worker_filled + "░" * (worker_bar_width - worker_filled)

                    # Generiere konsistenten Phase-Text aus grid_pos/grid_total
                    grid_phase = f"Grid: {grid_pos}/{grid_total} ({worker_pct}%)"

                    lines.append(f"║  → {sym:<10} [{worker_bar}] {worker_pct:3d}%  {grid_phase}".ljust(WIDTH - 1) + "║")
                elif phase:
                    # Nur Phase, noch kein Grid
                    max_phase = WIDTH - 20
                    if len(phase) > max_phase:
                        phase = phase[:max_phase-2] + ".."
                    lines.append(f"║  → {sym:<10} {phase}".ljust(WIDTH - 1) + "║")
                else:
                    # Wartend
                    lines.append(f"║    {sym:<10} [················]       Wartend".ljust(WIDTH - 1) + "║")

        # Falls mehr Assets als angezeigt
        if len(self.asset_names) > max_assets:
            remaining = len(self.asset_names) - max_assets
            lines.append(f"║  ... und {remaining} weitere Assets".ljust(WIDTH - 1) + "║")

        lines.append("╚" + "═" * (WIDTH - 2) + "╝")

        # Altes Display löschen und neues ausgeben
        if self._is_tty:
            self._clear_display()

        output = "\n".join(lines)
        sys.stdout.write(output + "\n")
        sys.stdout.flush()

        self._last_display_lines = len(lines)

    def _render_compact(self, completed: int, pct: float, elapsed_str: str,
                        eta_str: str, phases: Dict[str, str]):
        """Rendert kompakte einzeilige Ausgabe für Non-TTY (Logs/Pipes)."""
        # Sammle Asset-Status
        asset_info = []
        for sym in self.asset_names[:6]:  # Max 6 anzeigen
            if sym in self.completed_symbols:
                asset_info.append(f"{sym}:✓")
            elif sym in phases:
                # Phase kürzen
                phase = phases[sym][:15]
                asset_info.append(f"{sym}:{phase}")
            else:
                asset_info.append(f"{sym}:...")

        status = " | ".join(asset_info)
        if len(self.asset_names) > 6:
            status += f" +{len(self.asset_names)-6}"

        print(f"[{pct:5.1f}%] {completed}/{self.total_assets} | {elapsed_str} | ETA: {eta_str} | {status}")
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
        # Mindestens 30 Sekunden warten und 1% Fortschritt für zuverlässige ETA
        min_progress = 0.01 * self.total_assets  # 1% des Gesamtfortschritts
        if elapsed < 30 or total_progress < min_progress:
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
