"""
Adaptive Resource Manager für den Optimizer.
Steuert die Anzahl paralleler Prozesse basierend auf RAM-Verfügbarkeit.
"""
import os
import sys
import time
import signal
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Any, Optional, Dict

import psutil

# Globale Referenz auf aktiven Executor für Cleanup
_active_executor = None
_active_futures = []

# Shared Progress-Queue für Worker-Initialisierung
_shared_progress_queue = None

# Original signal handlers to restore after cleanup
_original_sigint = None
_original_sigterm = None


def _init_worker(progress_queue):
    """Initializer für Worker-Prozesse - setzt die Progress-Queue und ignoriert SIGINT."""
    global _shared_progress_queue
    _shared_progress_queue = progress_queue

    # Worker sollen SIGINT ignorieren - nur der Hauptprozess soll es behandeln
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Importiere und setze in progress.py
    try:
        from fwbg.utils.progress import set_progress_queue
        set_progress_queue(progress_queue)
    except ImportError:
        pass


def _cleanup_on_interrupt():
    """Beendet alle aktiven Worker-Prozesse bei Interrupt."""
    global _active_executor, _active_futures

    if _active_futures:
        print("\n[ResourceManager] Beende laufende Worker...", file=sys.stderr)
        for future in _active_futures:
            try:
                future.cancel()
            except Exception:
                pass
        _active_futures.clear()

    if _active_executor:
        try:
            _active_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _active_executor = None

    # Alle Kindprozesse des aktuellen Prozesses beenden
    _kill_child_processes()


def _kill_child_processes():
    """Beendet alle Kindprozesse des aktuellen Prozesses."""
    current_pid = os.getpid()
    try:
        current_process = psutil.Process(current_pid)
        children = current_process.children(recursive=True)

        if not children:
            return

        # Erst SIGTERM
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Warte max 2 Sekunden auf sauberes Beenden
        gone, alive = psutil.wait_procs(children, timeout=2)

        # SIGKILL für hartnäckige Prozesse
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass


def _signal_handler(signum, frame):
    """Handler für SIGINT/SIGTERM."""
    print("\n[ResourceManager] Abbruch-Signal empfangen, räume auf...", file=sys.stderr)
    _cleanup_on_interrupt()
    # Original handler wiederherstellen und Signal erneut senden
    if signum == signal.SIGINT and _original_sigint:
        signal.signal(signal.SIGINT, _original_sigint)
    elif signum == signal.SIGTERM and _original_sigterm:
        signal.signal(signal.SIGTERM, _original_sigterm)
    sys.exit(1)


class AdaptivePoolManager:
    """
    Verwaltet einen Pool von Worker-Prozessen mit dynamischer Skalierung
    basierend auf verfügbarem RAM.

    Regeln:
    - Maximal 80% der CPU-Kerne nutzen
    - Mindestens 20% RAM müssen frei bleiben
    - Neue Prozesse nur starten, wenn genug RAM verfügbar ist
    """

    def __init__(
        self,
        max_cpu_percent: float = 0.80,
        min_free_ram_percent: float = 0.25,
        ram_per_worker_gb: float = 4.0,
        check_interval: float = 2.0,
        verbose: bool = True,
        progress_dict: Optional[Dict] = None,
        progress_queue = None
    ):
        """
        Args:
            max_cpu_percent: Maximaler Anteil der CPU-Kerne (0.0-1.0)
            min_free_ram_percent: Minimaler freier RAM-Anteil (0.0-1.0)
            ram_per_worker_gb: Geschätzter Peak-RAM pro Worker in GB
            check_interval: Sekunden zwischen RAM-Checks
            verbose: Detaillierte Ausgaben
            progress_dict: DEPRECATED - nicht mehr verwendet
            progress_queue: multiprocessing.Queue für Progress-Updates
        """
        # Normalisiere Prozent-Werte: 80 -> 0.80, 0.80 -> 0.80
        self.max_cpu_percent = max_cpu_percent / 100 if max_cpu_percent > 1 else max_cpu_percent
        self.min_free_ram_percent = min_free_ram_percent / 100 if min_free_ram_percent > 1 else min_free_ram_percent
        self.ram_per_worker_gb = ram_per_worker_gb
        self.check_interval = check_interval
        self.verbose = verbose
        self.progress_queue = progress_queue

        # Systeminfo
        self.total_cores = mp.cpu_count()
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)

        # Berechne max Workers basierend auf CPU UND RAM
        cpu_limit = max(1, int(self.total_cores * self.max_cpu_percent))

        # RAM-Limit: (Gesamt-RAM - Reserve) / RAM pro Worker
        reserved_ram = self.total_ram_gb * self.min_free_ram_percent
        available_for_workers = self.total_ram_gb - reserved_ram
        ram_limit = max(1, int(available_for_workers / self.ram_per_worker_gb))

        self.max_workers = min(cpu_limit, ram_limit)

        # Stats
        self.peak_workers = 0
        self.ram_throttle_count = 0

    def get_free_ram_percent(self) -> float:
        """Gibt den Anteil des freien RAMs zurück (0.0-1.0)."""
        mem = psutil.virtual_memory()
        return mem.available / mem.total

    def get_free_ram_gb(self) -> float:
        """Gibt den freien RAM in GB zurück."""
        return psutil.virtual_memory().available / (1024**3)

    def get_cpu_percent(self, samples: int = 3) -> float:
        """
        Gibt die aktuelle CPU-Auslastung zurück (0.0-100.0).

        Nimmt mehrere Samples und gibt den Durchschnitt zurück,
        um kurzfristige Schwankungen auszugleichen.
        """
        if samples <= 1:
            return psutil.cpu_percent(interval=0.2)

        readings = []
        for _ in range(samples):
            readings.append(psutil.cpu_percent(interval=0.15))
        return sum(readings) / len(readings)

    def can_spawn_worker(self, current_workers: int) -> bool:
        """
        Prüft, ob ein neuer Worker gestartet werden kann.

        Berücksichtigt:
        - Hartes Worker-Limit (basierend auf CPU und RAM)
        - Aktuell freier RAM
        - Aktuelle CPU-Auslastung
        - Geschätzter RAM-Bedarf für laufende + neuen Worker

        Returns:
            True wenn genug Ressourcen für einen weiteren Worker
        """
        # Hartes Limit prüfen
        if current_workers >= self.max_workers:
            return False

        # CPU-Check: Nicht starten wenn CPU bereits über max_cpu_percent
        # Mehrere Samples nehmen um Momentan-Schwankungen auszugleichen
        current_cpu = self.get_cpu_percent(samples=3)
        cpu_threshold = self.max_cpu_percent * 100

        if current_cpu > cpu_threshold:
            self.ram_throttle_count += 1  # Reuse counter for any throttle
            return False

        # Berechne benötigten RAM für alle Worker (inkl. neuem)
        needed_workers = current_workers + 1
        needed_ram_gb = needed_workers * self.ram_per_worker_gb

        # Verfügbarer RAM für Worker (nach Reserve)
        reserved_ram = self.total_ram_gb * self.min_free_ram_percent
        available_ram = self.total_ram_gb - reserved_ram

        if needed_ram_gb > available_ram:
            self.ram_throttle_count += 1
            return False

        # Zusätzlich: Aktuellen freien RAM prüfen (falls System schon belastet)
        current_free = self.get_free_ram_gb()
        if current_free < (reserved_ram + self.ram_per_worker_gb):
            self.ram_throttle_count += 1
            return False

        return True

    def log(self, msg: str, force: bool = False):
        """Optionale Ausgabe auf stderr."""
        if not self.verbose:
            return
        # Unterdrücke Logs wenn Progress-UI aktiv (außer force=True)
        if os.environ.get("_OPTIMIZER_PROGRESS_UI") and not force:
            return
        print(f"[ResourceManager] {msg}", file=sys.stderr, flush=True)

    def get_status(self) -> dict:
        """Gibt aktuellen Ressourcen-Status zurück."""
        mem = psutil.virtual_memory()
        return {
            "total_cores": self.total_cores,
            "max_workers": self.max_workers,
            "total_ram_gb": round(self.total_ram_gb, 1),
            "free_ram_gb": round(mem.available / (1024**3), 1),
            "free_ram_percent": round(mem.available / mem.total * 100, 1),
            "ram_throttle_count": self.ram_throttle_count,
            "peak_workers": self.peak_workers,
        }

    def map_adaptive(
        self,
        func: Callable,
        items: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        result_callback: Optional[Callable[[Any], None]] = None
    ) -> List[Any]:
        """
        Verarbeitet Items mit adaptiver Worker-Anzahl.

        Startet mit wenigen Workern und skaliert hoch, solange RAM verfügbar ist.
        Reduziert Worker-Anzahl wenn RAM knapp wird.

        Args:
            func: Funktion die auf jedes Item angewendet wird
            items: Liste der zu verarbeitenden Items
            progress_callback: Optional callback(completed, total)
            result_callback: Optional callback(result) - wird für jedes fertige Ergebnis aufgerufen

        Returns:
            Liste der Ergebnisse (None-Werte werden gefiltert)
        """
        if not items:
            return []

        total = len(items)
        results = []
        completed = 0

        # CPU-Warm-up: Erster Aufruf von cpu_percent() gibt immer 0 zurück
        # Wir machen einen kurzen Warm-up um realistische Werte zu bekommen
        psutil.cpu_percent(interval=None)  # Initialisierung
        time.sleep(0.2)
        initial_cpu = psutil.cpu_percent(interval=0.1)

        # Status ausgeben
        status = self.get_status()
        self.log(f"System: {status['total_cores']} Cores, {status['total_ram_gb']:.1f} GB RAM")
        self.log(f"Limits: max {self.max_workers} Workers (CPU<{self.max_cpu_percent*100:.0f}%, RAM={self.ram_per_worker_gb}GB/Worker)")
        self.log(f"Reserve: {self.min_free_ram_percent*100:.0f}% RAM = {self.total_ram_gb * self.min_free_ram_percent:.1f} GB")
        self.log(f"Aktuell: {status['free_ram_gb']:.1f} GB RAM frei, CPU bei {initial_cpu:.0f}%")

        global _active_executor, _active_futures, _original_sigint, _original_sigterm

        # Signal-Handler für Ctrl+C registrieren (nur während Pool aktiv)
        _original_sigint = signal.signal(signal.SIGINT, _signal_handler)
        _original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

        # Erstelle Executor mit Initializer für Progress-Tracking
        executor_kwargs = {"max_workers": self.max_workers}
        if self.progress_queue is not None:
            executor_kwargs["initializer"] = _init_worker
            executor_kwargs["initargs"] = (self.progress_queue,)

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            _active_executor = executor

            # Futures verwalten
            futures = {}
            items_iter = iter(enumerate(items))
            active_count = 0

            # KONSERVATIV STARTEN: Nur 1 Worker initial!
            # Die echte CPU-Last entsteht erst bei Grid-Search, nicht beim Start.
            # Weitere Worker werden erst gestartet wenn die Last tatsächlich messbar ist.
            try:
                idx, item = next(items_iter)
                future = executor.submit(func, item)
                futures[future] = idx
                _active_futures.append(future)
                active_count += 1
            except StopIteration:
                pass

            self.peak_workers = active_count
            self.log(f"Initial gestartet: {active_count} Worker (skaliert dynamisch bis max {self.max_workers})", force=True)

            # Tracking für periodisches Scaling
            last_scale_check = time.time()
            # WICHTIG: Langsames Scaling um System nicht zu überlasten
            # CPU-Last entsteht erst bei Grid-Search, nicht bei Indikator-Berechnung
            # Daher: Lange Wartezeiten zwischen Worker-Starts
            initial_wait = 90.0  # 90 Sekunden warten bevor erstes Scaling
            scale_check_interval = 60.0  # Danach alle 60 Sekunden ein weiterer Worker
            first_scale_done = False
            items_remaining = True

            while futures or items_remaining:
                # Fertige Tasks einsammeln
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                # Ergebnisse sammeln
                for future in done_futures:
                    idx = futures.pop(future)
                    active_count -= 1
                    completed += 1

                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            if result_callback:
                                try:
                                    result_callback(result)
                                except Exception as cb_err:
                                    self.log(f"Result-Callback Fehler: {cb_err}")
                    except Exception as e:
                        self.log(f"Worker-Fehler: {e}")

                    if progress_callback:
                        progress_callback(completed, total)

                # Periodisch prüfen ob wir skalieren können (auch wenn kein Task fertig)
                now = time.time()
                current_interval = initial_wait if not first_scale_done else scale_check_interval

                if now - last_scale_check >= current_interval:
                    last_scale_check = now
                    first_scale_done = True

                    # KONSERVATIV: Max 1 Worker pro Scale-Check
                    # Das gibt dem System Zeit, die CPU/RAM-Last zu messen
                    # bevor weitere Worker gestartet werden
                    spawned = 0
                    max_spawn_per_check = 1

                    while items_remaining and spawned < max_spawn_per_check:
                        if not self.can_spawn_worker(active_count):
                            break

                        try:
                            idx, item = next(items_iter)
                            future = executor.submit(func, item)
                            futures[future] = idx
                            _active_futures.append(future)
                            active_count += 1
                            spawned += 1

                            if active_count > self.peak_workers:
                                self.peak_workers = active_count
                        except StopIteration:
                            items_remaining = False
                            break

                    if spawned > 0:
                        self.log(f"Skaliert: +{spawned} Worker (jetzt {active_count} aktiv)", force=True)

                # Ressourcen-Status loggen bei Throttling
                if self.ram_throttle_count > 0 and self.ram_throttle_count % 20 == 0:
                    free_gb = self.get_free_ram_gb()
                    cpu_pct = self.get_cpu_percent()
                    self.log(f"Throttling: {free_gb:.1f} GB RAM frei, CPU {cpu_pct:.0f}%, {active_count}/{self.max_workers} Worker")

                # Kurz warten bevor nächster Check
                if futures:
                    time.sleep(0.2)

            # Cleanup nach erfolgreicher Beendigung
            _active_futures.clear()

        _active_executor = None

        # Signal-Handler wiederherstellen
        signal.signal(signal.SIGINT, _original_sigint if _original_sigint else signal.SIG_DFL)
        signal.signal(signal.SIGTERM, _original_sigterm if _original_sigterm else signal.SIG_DFL)

        # Finale Stats
        self.log(f"Fertig: {completed} Items, Peak {self.peak_workers} Workers")
        if self.ram_throttle_count > 0:
            self.log(f"RAM-Throttling wurde {self.ram_throttle_count}x aktiviert")

        return results


def get_resource_info() -> dict:
    """Gibt aktuelle Systemressourcen zurück."""
    mem = psutil.virtual_memory()
    cpu_count = mp.cpu_count()

    return {
        "cpu_cores": cpu_count,
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "ram_available_gb": round(mem.available / (1024**3), 1),
        "ram_used_percent": round(mem.percent, 1),
        "ram_free_percent": round(100 - mem.percent, 1),
    }


def calculate_safe_workers(
    max_cpu_percent: float = 0.80,
    min_free_ram_percent: float = 0.25,
    estimated_ram_per_worker_gb: float = 2.5
) -> int:
    """
    Berechnet eine sichere Anzahl paralleler Worker.

    Args:
        max_cpu_percent: Max. Anteil der CPU-Kerne
        min_free_ram_percent: Min. freier RAM-Anteil
        estimated_ram_per_worker_gb: Geschätzter RAM pro Worker

    Returns:
        Anzahl sicherer Worker
    """
    info = get_resource_info()

    # CPU-Limit
    cpu_limit = max(1, int(info["cpu_cores"] * max_cpu_percent))

    # RAM-Limit
    available_for_workers = info["ram_available_gb"] - (info["ram_total_gb"] * min_free_ram_percent)
    ram_limit = max(1, int(available_for_workers / estimated_ram_per_worker_gb))

    return min(cpu_limit, ram_limit)
