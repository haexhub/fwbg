"""
Progress-Tracking für den Optimizer.
Ermöglicht Echtzeit-Fortschrittsanzeige im Haupt-Prozess.

Verwendet multiprocessing.Queue statt Manager für deadlock-freie Kommunikation.
"""
import os
import sys
import time
import threading
from datetime import datetime, timezone
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from typing import Any, Optional, List, Dict

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
    # Im Parallel-Modus Progress-Updates unterdrücken, AUSSER bei grid_search
    # (grid_search Updates sind aggregiert und sollten immer durchkommen)
    if is_parallel_mode() and stage != "grid_search":
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


def report_meta(symbol: str, **kwargs):
    """
    Meldet statische Metadaten für ein Symbol (einmalig pro Run).

    Wird im TUI neben dem Grid-Fortschritt angezeigt, damit
    ersichtlich ist, warum manche Assets länger brauchen.

    Args:
        symbol: Asset-Symbol (z.B. "EURUSD")
        **kwargs: Beliebige Metadaten, z.B. indicator_count=11, feature_count=352
    """
    if _progress_queue is not None:
        try:
            msg = {
                "type": "meta",
                "pid": os.getpid(),
                "symbol": symbol,
                "time": time.time(),
            }
            msg.update(kwargs)
            _progress_queue.put_nowait(msg)
        except Exception:
            pass


def report_result(symbol: str, status: str, summary: str):
    """
    Meldet ein fertiges Ergebnis für sofortige Anzeige im Progress-UI.

    Args:
        symbol: Asset-Symbol (z.B. "EURUSD")
        status: Status ("ok", "no_kelly", "not_significant", etc.)
        summary: Kurze Zusammenfassung (eine Zeile)
    """
    if _progress_queue is not None:
        try:
            _progress_queue.put_nowait({
                "type": "result",
                "pid": os.getpid(),
                "symbol": symbol,
                "status": status,
                "summary": summary,
                "time": time.time(),
            })
        except Exception:
            pass


def report_log(
    symbol: str,
    stage: str,
    level: str,
    message: str,
    **data: Any,
) -> None:
    """Send structured log entry through progress queue."""
    if _progress_queue is not None:
        try:
            _progress_queue.put_nowait({
                "type": "log",
                "level": level,
                "symbol": symbol,
                "stage": stage,
                "message": message,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
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

    def __init__(self, total_assets: int, asset_names: Optional[List[str]] = None,
                 queue: Optional[Queue] = None,
                 run_directory: Optional[Path] = None, run_id: str = "",
                 strategy_name: str = ""):
        self.total_assets = total_assets
        self.completed_assets = 0
        self.asset_names = asset_names or []
        self.completed_symbols: List[str] = []
        self.queue = queue
        self.worker_status: Dict[int, dict] = {}  # pid -> status
        self.worker_phases: Dict[str, str] = {}  # symbol -> aktuelle Phase
        self.worker_phase_times: Dict[str, float] = {}  # symbol -> Zeitstempel der letzten Phase
        self.worker_results: Dict[str, str] = {}  # symbol -> Ergebnis-Zusammenfassung
        self.worker_meta: Dict[str, dict] = {}  # symbol -> statische Metadaten (indicator_count, etc.)
        self.start_time = None
        self._stop_event = threading.Event()
        self._display_thread = None
        self._queue_thread = None
        self._lock = threading.Lock()
        self._is_tty = sys.stdout.isatty()
        self._last_render_time = 0
        self._last_display_lines = 0  # Anzahl der zuletzt angezeigten Zeilen

        # Persistent progress + structured logging
        self._run_progress_writer = None
        self._run_logger = None
        if run_directory is not None:
            from .run_progress import RunProgressWriter
            from .run_logger import RunLogger
            self._run_progress_writer = RunProgressWriter(
                run_directory, run_id, self.asset_names,
                strategy_name=strategy_name,
            )
            self._run_logger = RunLogger(run_directory)

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

        # WICHTIG: Queue schließen um Worker-Prozesse freizugeben
        # Ohne close() und join_thread() können Worker beim put() blockieren
        if self.queue is not None:
            try:
                # close() verhindert weitere put() Calls
                self.queue.close()
                # join_thread() wartet bis alle gepufferten Daten geschrieben wurden
                self.queue.join_thread()
            except Exception:
                pass  # Queue könnte bereits geschlossen sein

        # Finale Ausgabe
        elapsed = time.time() - self.start_time if self.start_time else 0

        if self._is_tty:
            # Lösche das Multi-Line-Display
            self._clear_display()

        print(f"\nVerarbeitung abgeschlossen: {self.completed_assets}/{self.total_assets} in {self._format_time(elapsed)}")
        sys.stdout.flush()

        # Complete persistent progress file
        if self._run_progress_writer:
            self._run_progress_writer.complete_run()
        if self._run_logger:
            self._run_logger.close()

        # Detail-Logs wieder erlauben
        set_progress_ui_active(False)

    def _queue_reader(self):
        """Liest Updates aus der Queue."""
        while not self._stop_event.is_set():
            try:
                msg = self.queue.get(timeout=0.1)
                msg_type = msg.get("type")

                if msg_type == "log":
                    # Structured log entry — write to JSONL file
                    if self._run_logger:
                        self._run_logger.write_raw(msg)
                elif msg_type == "result":
                    # Ergebnis-Zusammenfassung speichern für "Fertig"-Zeile
                    sym = msg.get("symbol")
                    with self._lock:
                        if sym:
                            status = msg.get("status", "?")
                            summary = msg.get("summary", "")
                            icon = "✓" if status == "ok" else "✗"
                            self.worker_results[sym] = f"{icon} {summary}"
                    # Sofortige Ergebnis-Anzeige (außerhalb des Locks)
                    self._show_result(msg)
                    # Persist to progress file
                    if self._run_progress_writer and sym:
                        self._run_progress_writer.complete_asset(
                            sym, msg.get("summary", ""))
                else:
                    with self._lock:
                        if msg_type == "progress":
                            # Index by symbol, not PID (multiple threads share same PID)
                            symbol = msg.get("symbol", msg["pid"])
                            self.worker_status[symbol] = msg
                            # Update file writer with grid_search stage progress
                            if self._run_progress_writer and isinstance(symbol, str):
                                grid_pos = msg.get("grid_pos", 0)
                                grid_total = msg.get("grid_total", 0)
                                fold = msg.get("fold", 0)
                                total_folds = msg.get("total_folds", 0)
                                if grid_total > 0:
                                    frac = grid_pos / grid_total
                                    desc = f"Fold {fold}/{total_folds} ({grid_pos}/{grid_total})" if fold else f"{grid_pos}/{grid_total}"
                                    self._run_progress_writer.update_asset_stage(
                                        symbol, "grid_search", "running",
                                        description=desc, progress_fraction=frac,
                                        details={"grid_pos": grid_pos, "grid_total": grid_total,
                                                 "fold": fold, "total_folds": total_folds},
                                    )
                        elif msg_type == "phase":
                            # Phase-Update: Symbol -> Phase-Text speichern
                            sym = msg["symbol"]
                            self.worker_phases[sym] = msg["phase"]
                            self.worker_phase_times[sym] = msg.get("time", time.time())
                            # Persist phase to file writer
                            if self._run_progress_writer:
                                phase = msg["phase"]
                                # Map known phase text to stage names
                                stage_name = self._phase_to_stage(phase)
                                self._run_progress_writer.begin_asset(sym)
                                self._run_progress_writer.update_asset_stage(
                                    sym, stage_name, "running", description=phase,
                                )
                        elif msg_type == "meta":
                            # Statische Metadaten (indicator_count, etc.)
                            symbol = msg.get("symbol")
                            if symbol:
                                self.worker_meta[symbol] = {
                                    k: v for k, v in msg.items()
                                    if k not in ("type", "pid", "symbol", "time")
                                }
                        elif msg_type == "done":
                            # Remove by symbol
                            symbol = msg.get("symbol")
                            if symbol:
                                self.worker_status.pop(symbol, None)
                                self.worker_phases.pop(symbol, None)
                                self.worker_phase_times.pop(symbol, None)
                                self.worker_meta.pop(symbol, None)
                                # Sofort als fertig markieren für Display
                                # (Result kommt später per IPC/Pickle)
                                if symbol not in self.completed_symbols:
                                    self.completed_symbols.append(symbol)
            except Empty:
                continue
            except Exception:
                break

    @staticmethod
    def _phase_to_stage(phase: str) -> str:
        """Map phase text to a known stage name."""
        phase_lower = phase.lower()
        if "daten" in phase_lower or "data" in phase_lower or "lade" in phase_lower:
            return "data_loading"
        if "indikator" in phase_lower or "indicator" in phase_lower or "feature" in phase_lower:
            return "indicators"
        if "grid" in phase_lower:
            return "grid_search"
        if "model" in phase_lower or "train" in phase_lower:
            return "model_training"
        if "eval" in phase_lower or "holdout" in phase_lower or "walk" in phase_lower:
            return "evaluation"
        return phase_lower[:30]

    def _show_result(self, msg: dict):
        """Zeigt ein Ergebnis sofort an (unter der Progress-UI)."""
        symbol = msg.get("symbol", "?")
        status = msg.get("status", "?")
        summary = msg.get("summary", "")

        # Status-Symbol
        if status == "ok":
            icon = "✓"
        else:
            icon = "✗"

        # Ausgabe formatieren
        result_line = f"{icon} {symbol}: {summary}"

        if self._is_tty:
            # TTY: Display temporär löschen, Ergebnis ausgeben, Display neu zeichnen
            self._clear_display()
            print(result_line)
            sys.stdout.flush()
            # Display wird beim nächsten Render automatisch neu gezeichnet
            self._last_display_lines = 0
        else:
            # Non-TTY: Einfach ausgeben
            print(result_line)
            sys.stdout.flush()

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
            # completed_symbols wird sofort bei "done" aktualisiert (via Queue),
            # completed_assets erst wenn das Result per IPC/Pickle ankommt.
            # Verwende den höheren Wert für korrekte Anzeige.
            completed = max(self.completed_assets, len(self.completed_symbols))
            completed_symbols = self.completed_symbols[:]
            workers = dict(self.worker_status)
            phases = dict(self.worker_phases)
            phase_times = dict(self.worker_phase_times)
            results = dict(self.worker_results)
            meta = dict(self.worker_meta)

        elapsed = time.time() - self.start_time if self.start_time else 0

        # Aktive Worker filtern (nur die letzte Stunde - Combos mit Preprocessing
        # können bei großen Grids >5 Minuten pro Combo dauern)
        now = time.time()
        active_workers = {
            pid: info for pid, info in workers.items()
            if now - info.get("time", 0) < 3600
        }

        # Gesamt-Fortschritt berechnen (inkl. Grid-Fortschritt der aktiven Worker)
        total_progress = self._calculate_total_progress(completed, active_workers)
        pct = (total_progress / self.total_assets * 100) if self.total_assets > 0 else 0

        # ETA berechnen (inkl. Grid-Fortschritt)
        eta_str = self._calculate_eta(elapsed, total_progress)
        elapsed_str = self._format_time(elapsed)

        if self._is_tty:
            # TTY: Fixes Fenster mit Cursor-Steuerung
            self._render_fixed_window(completed, total_progress, pct, elapsed_str, eta_str, active_workers, completed_symbols, phases, phase_times, results, meta)
        else:
            # Non-TTY: Kompakte einzeilige Ausgabe
            self._render_compact(completed, pct, elapsed_str, eta_str, phases)

    def _render_fixed_window(self, completed: int, total_progress: float, pct: float,
                              elapsed_str: str, eta_str: str, active_workers: Dict,
                              completed_symbols: List[str], phases: Dict[str, str],
                              phase_times: Dict[str, float] = None,
                              results: Dict[str, str] = None,
                              meta: Dict[str, dict] = None):
        """Rendert ein fixes Fenster mit allen Assets und deren Fortschritt."""
        lines = []

        # Feste Fensterbreite für konsistentes Layout
        WIDTH = 78

        # Progress bar für Gesamtfortschritt
        bar_width = 50
        filled = int(bar_width * total_progress / self.total_assets) if self.total_assets > 0 else 0
        filled = min(filled, bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Header mit mehr Details
        lines.append("╔" + "═" * (WIDTH - 2) + "╗")
        lines.append(f"║  FWBG Optimizer │ {completed}/{self.total_assets} Assets │ {elapsed_str} │ ETA: {eta_str}".ljust(WIDTH - 1) + "║")
        lines.append(f"║  [{bar}] {pct:5.1f}%".ljust(WIDTH - 1) + "║")
        lines.append("╠" + "═" * (WIDTH - 2) + "╣")

        # Alle Assets aus asset_names anzeigen (nicht nur aktive)
        displayed = 0
        max_assets = self.MAX_DISPLAY_LINES - 7  # Platz für Header/Footer

        for sym in self.asset_names[:max_assets]:
            displayed += 1

            # Status ermitteln
            if sym in completed_symbols:
                # Fertig - mit Ergebnis-Zusammenfassung wenn vorhanden
                result_summary = results.get(sym) if results else None
                if result_summary:
                    # Kürzen falls nötig (Platz: WIDTH - "║  ✓ SYM      " - "║")
                    max_summary = WIDTH - 16
                    if len(result_summary) > max_summary:
                        result_summary = result_summary[:max_summary-2] + ".."
                    lines.append(f"║  {sym:<8} {result_summary}".ljust(WIDTH - 1) + "║")
                else:
                    lines.append(f"║  ✓ {sym:<8} Fertig".ljust(WIDTH - 1) + "║")
            else:
                # Noch aktiv oder wartend
                worker_info = active_workers.get(sym)

                grid_pos = worker_info.get("grid_pos", 0) if worker_info else 0
                grid_total = worker_info.get("grid_total", 0) if worker_info else 0
                fold = worker_info.get("fold", 0) if worker_info else 0
                total_folds = worker_info.get("total_folds", 0) if worker_info else 0
                progress_time = worker_info.get("time", 0) if worker_info else 0
                phase = phases.get(sym, "")
                phase_time = phase_times.get(sym, 0) if phase_times else 0

                # Show phase text if it's newer than last progress update
                # (e.g. between folds during indicator computation)
                show_grid = grid_total > 0 and progress_time >= phase_time

                if show_grid:
                    bar_width = 12

                    # Feature/Regime-Info aus Meta-Daten
                    sym_meta = meta.get(sym, {}) if meta else {}
                    feat_count = sym_meta.get("feature_count", 0)
                    ind_count = sym_meta.get("indicator_count", 0)
                    regime_combos = sym_meta.get("regime_combos", 0)
                    meta_parts = []
                    if feat_count:
                        meta_parts.append(f"{feat_count}F")
                    elif ind_count:
                        meta_parts.append(f"{ind_count}P")
                    if regime_combos > 1:
                        meta_parts.append(f"{regime_combos}R")
                    ind_suffix = f" {'/'.join(meta_parts)}" if meta_parts else ""

                    if fold > 0 and total_folds > 0:
                        # Gesamt-Asset-Fortschritt über alle Folds (monoton steigend)
                        asset_progress = ((fold - 1) + (grid_pos / grid_total)) / total_folds
                        asset_pct = asset_progress * 100
                        filled = int(bar_width * asset_progress)
                        bar = "▓" * filled + "░" * (bar_width - filled)

                        # Format: SYMBOL [████░░░░] 18.8% F2/8 (12/16) 352 Feat
                        line = f"║  → {sym:<8} [{bar}] {asset_pct:5.1f}% F{fold}/{total_folds} ({grid_pos}/{grid_total}){ind_suffix}"
                    else:
                        fold_pct = grid_pos / grid_total * 100
                        filled = int(bar_width * grid_pos / grid_total)
                        bar = "▓" * filled + "░" * (bar_width - filled)
                        line = f"║  → {sym:<8} [{bar}] {fold_pct:5.1f}% ({grid_pos}/{grid_total}){ind_suffix}"

                    lines.append(line.ljust(WIDTH - 1) + "║")
                elif phase:
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
        # Kopiere worker_status für Thread-Sicherheit
        with self._lock:
            workers = dict(self.worker_status)

        # Sammle Asset-Status
        asset_info = []
        for sym in self.asset_names[:6]:  # Max 6 anzeigen
            if sym in self.completed_symbols:
                asset_info.append(f"{sym}:✓")
            elif sym in workers and workers[sym].get("grid_total", 0) > 0:
                # Grid-Progress anzeigen
                info = workers[sym]
                grid_pos = info.get("grid_pos", 0)
                grid_total = info.get("grid_total", 1)
                fold = info.get("fold", 0)
                total_folds = info.get("total_folds", 0)
                if fold > 0 and total_folds > 0:
                    asset_pct = ((fold - 1) + (grid_pos / grid_total)) / total_folds * 100
                    asset_info.append(f"{sym}:F{fold}/{total_folds} {asset_pct:.0f}%")
                else:
                    pct_done = grid_pos / grid_total * 100
                    asset_info.append(f"{sym}:{pct_done:.0f}%")
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
        """Berechnet Gesamt-Fortschritt inkl. Fold- und Grid-Progress der aktiven Worker."""
        total_progress = float(completed)

        for info in active_workers.values():
            fold = info.get("fold", 0)
            total_folds = info.get("total_folds", 1)
            grid_pos = info.get("grid_pos", 0)
            grid_total = info.get("grid_total", 1)

            if total_folds > 0 and grid_total > 0 and fold > 0:
                # Fortschritt = (abgeschlossene Folds + aktueller Fold-Fortschritt) / total_folds
                # fold ist 1-basiert, also fold-1 für abgeschlossene Folds
                fold_progress = max(0, (fold - 1)) / total_folds  # Abgeschlossene Folds (nie negativ)
                current_fold_progress = (grid_pos / grid_total) / total_folds  # Aktueller Fold
                total_progress += fold_progress + current_fold_progress
            elif grid_total > 0:
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
