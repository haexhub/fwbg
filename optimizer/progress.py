"""
Progress-Tracking für den Optimizer.
Ermöglicht Echtzeit-Fortschrittsanzeige über mehrere Worker-Prozesse.
"""
import os
import sys
import time
import threading
from multiprocessing import Manager
from typing import Optional, Dict, Any, List

# Globaler Progress-Manager (wird in main.py initialisiert)
_progress_manager: Optional["ProgressTracker"] = None
_progress_dict: Optional[Dict] = None


def init_progress_tracking():
    """Initialisiert das Progress-Tracking (muss im Main-Prozess aufgerufen werden)."""
    global _progress_manager, _progress_dict
    manager = Manager()
    _progress_dict = manager.dict()
    _progress_manager = ProgressTracker(_progress_dict)
    return _progress_manager, _progress_dict


def set_progress_dict(shared_dict: Dict):
    """Setzt das Progress-Dict (für Worker-Prozesse via Initializer)."""
    global _progress_dict
    _progress_dict = shared_dict


def get_progress_dict():
    """Gibt das shared dict zurück (für Worker-Prozesse)."""
    return _progress_dict


def report_progress(
    symbol: str,
    fold: int,
    total_folds: int,
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
    pid = os.getpid()
    if _progress_dict is not None:
        try:
            _progress_dict[pid] = {
                "symbol": symbol,
                "fold": fold,
                "total_folds": total_folds,
                "stage": stage,
                "grid_pos": grid_pos,
                "grid_total": grid_total,
                "updated": time.time(),
            }
        except Exception:
            pass  # Ignoriere Fehler bei Manager-Disconnect


def report_done(symbol: str, status: str = "ok"):
    """Meldet, dass ein Worker fertig ist."""
    pid = os.getpid()
    if _progress_dict is not None:
        try:
            if pid in _progress_dict:
                del _progress_dict[pid]
        except (KeyError, Exception):
            pass


class ProgressTracker:
    """
    Zeigt Echtzeit-Fortschritt für alle aktiven Worker an.
    """

    def __init__(self, shared_dict: Dict):
        self.shared_dict = shared_dict
        self.total_assets = 0
        self.completed_assets = 0
        self.completed_symbols: List[str] = []
        self.pending_symbols: List[str] = []
        self.start_time = None
        self._stop_event = threading.Event()
        self._display_thread = None
        self._last_line_count = 0
        self._lock = threading.Lock()

    def start(self, total_assets: int, asset_names: List[str] = None):
        """Startet das Progress-Display."""
        self.total_assets = total_assets
        self.completed_assets = 0
        self.completed_symbols = []
        self.pending_symbols = asset_names[:] if asset_names else []
        self.start_time = time.time()
        self._stop_event.clear()

        # Display-Thread starten
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def update_completed(self, completed: int, symbol: str = None):
        """Aktualisiert die Anzahl der fertigen Assets."""
        with self._lock:
            self.completed_assets = completed
            if symbol:
                if symbol not in self.completed_symbols:
                    self.completed_symbols.append(symbol)
                if symbol in self.pending_symbols:
                    self.pending_symbols.remove(symbol)

    def stop(self):
        """Stoppt das Progress-Display."""
        self._stop_event.set()
        if self._display_thread:
            self._display_thread.join(timeout=1)
        # Letzte Zeilen löschen
        self._clear_lines()

    def _clear_lines(self):
        """Löscht die zuletzt geschriebenen Zeilen."""
        if self._last_line_count > 0:
            # Cursor hoch bewegen und jede Zeile löschen
            # \033[F = Cursor eine Zeile hoch + an Zeilenanfang
            # \033[K = Rest der Zeile löschen
            sys.stdout.write(f"\033[{self._last_line_count}F")  # N Zeilen hoch
            for _ in range(self._last_line_count):
                sys.stdout.write("\033[K\n")  # Zeile löschen, nächste Zeile
            sys.stdout.write(f"\033[{self._last_line_count}F")  # Wieder hoch
            sys.stdout.flush()
            self._last_line_count = 0

    def _display_loop(self):
        """Haupt-Display-Loop (läuft im Thread)."""
        while not self._stop_event.is_set():
            self._render()
            time.sleep(0.5)  # Update alle 500ms

    def _render(self):
        """Rendert den aktuellen Status."""
        # Zeilen löschen
        self._clear_lines()

        lines = []
        box_width = 74

        # Aktive Worker ermitteln (früh, für ETA-Berechnung)
        try:
            workers = dict(self.shared_dict)
        except Exception:
            workers = {}
        now = time.time()

        active_workers = {
            pid: info for pid, info in workers.items()
            if isinstance(info, dict) and now - info.get("updated", 0) < 30
        }
        active_count = len(active_workers)
        pending_count = self.total_assets - self.completed_assets - active_count

        # Gesamtfortschritt berechnen (inkl. Grid-Progress der aktiven Worker)
        elapsed = time.time() - self.start_time if self.start_time else 0
        total_progress = self._calculate_total_progress(active_workers)
        pct = (total_progress / self.total_assets * 100) if self.total_assets > 0 else 0

        # ETA berechnen basierend auf Gesamt-Fortschritt
        eta_str = self._calculate_eta(elapsed, total_progress)

        elapsed_str = self._format_time(elapsed)

        # === HEADER mit Status-Zusammenfassung ===
        lines.append(f"╔{'═' * box_width}╗")

        # Status-Zeile: Done | Active | Pending
        status_line = f"Done: {self.completed_assets}  |  Active: {active_count}  |  Pending: {max(0, pending_count)}"
        lines.append(f"║ {status_line:^{box_width-2}} ║")

        # === ACTIVE WORKERS ===
        if active_workers:
            lines.append(f"╠{'═' * box_width}╣")

            # Sortiere nach Symbol
            sorted_workers = sorted(active_workers.items(), key=lambda x: x[1].get("symbol", ""))

            for pid, info in sorted_workers[:6]:  # Max 6 Worker anzeigen
                sym = info.get("symbol", "???")[:8].ljust(8)
                fold = info.get("fold", 0)
                total = info.get("total_folds", 0)
                stage = info.get("stage", "")[:8].ljust(8)
                grid_pos = info.get("grid_pos", 0)
                grid_total = info.get("grid_total", 0)

                # Mini-Progressbar für Fold
                if total > 0:
                    fold_bar = "●" * fold + "○" * (total - fold)
                    fold_str = f"F{fold}/{total}[{fold_bar}]"
                else:
                    fold_str = "Loading...".ljust(18)

                # Grid-Info
                if grid_total > 0:
                    grid_pct = int(grid_pos / grid_total * 100)
                    grid_str = f"G:{grid_pos}/{grid_total}({grid_pct}%)"
                else:
                    grid_str = ""

                worker_line = f" {sym} {fold_str:<18} {grid_str:<14} {stage}"
                lines.append(f"║{worker_line:<{box_width}}║")

            if len(active_workers) > 6:
                more_line = f" ... +{len(active_workers) - 6} weitere"
                lines.append(f"║{more_line:<{box_width}}║")
        else:
            # Keine aktiven Worker - zeige "Waiting..."
            lines.append(f"╠{'═' * box_width}╣")
            lines.append(f"║{' Waiting for workers...':<{box_width}}║")

        # === PROGRESS BAR (unten) ===
        lines.append(f"╠{'═' * box_width}╣")

        # Progress-Bar (nutzt total_progress inkl. Grid-Fortschritt)
        bar_width_inner = 40
        filled = int(bar_width_inner * total_progress / self.total_assets) if self.total_assets > 0 else 0
        filled = min(filled, bar_width_inner)  # Nicht über 100%
        bar = "█" * filled + "░" * (bar_width_inner - filled)

        progress_line = f" [{bar}] {total_progress:.2f}/{self.total_assets} ({pct:.1f}%)"
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

    def _calculate_total_progress(self, active_workers: Dict) -> float:
        """
        Berechnet Gesamt-Fortschritt als Dezimalzahl.

        Berücksichtigt:
        - Fertige Assets (zählen zu 100%)
        - Grid-Fortschritt der aktiven Worker (anteilig)
        """
        total_progress = float(self.completed_assets)

        for info in active_workers.values():
            grid_pos = info.get("grid_pos", 0)
            grid_total = info.get("grid_total", 1)
            if grid_total > 0:
                total_progress += grid_pos / grid_total

        return total_progress

    def _calculate_eta(self, elapsed: float, total_progress: float) -> str:
        """Berechnet ETA basierend auf Gesamt-Fortschritt."""
        if elapsed < 5:  # Mindestens 5 Sekunden für sinnvolle Schätzung
            return "--:--"

        if self.total_assets == 0 or total_progress < 0.01:
            return "--:--"

        # Zeit pro "Asset-Einheit"
        time_per_unit = elapsed / total_progress

        # Verbleibende "Asset-Einheiten"
        remaining_units = self.total_assets - total_progress

        if remaining_units <= 0:
            return "00:00"

        remaining_time = remaining_units * time_per_unit
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
