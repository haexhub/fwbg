"""
Adaptive Resource Manager für den Optimizer.
Steuert die Anzahl paralleler Prozesse basierend auf RAM-Verfügbarkeit.
"""
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Any, Optional

import psutil


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
        min_free_ram_percent: float = 0.20,
        check_interval: float = 2.0,
        verbose: bool = True
    ):
        """
        Args:
            max_cpu_percent: Maximaler Anteil der CPU-Kerne (0.0-1.0)
            min_free_ram_percent: Minimaler freier RAM-Anteil (0.0-1.0)
            check_interval: Sekunden zwischen RAM-Checks
            verbose: Detaillierte Ausgaben
        """
        self.max_cpu_percent = max_cpu_percent
        self.min_free_ram_percent = min_free_ram_percent
        self.check_interval = check_interval
        self.verbose = verbose

        # Systeminfo
        self.total_cores = mp.cpu_count()
        self.max_workers = max(1, int(self.total_cores * max_cpu_percent))
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)

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

    def can_spawn_worker(self, current_workers: int) -> bool:
        """
        Prüft, ob ein neuer Worker gestartet werden kann.

        Returns:
            True wenn: CPU-Limit nicht erreicht UND genug RAM frei
        """
        # CPU-Limit prüfen
        if current_workers >= self.max_workers:
            return False

        # RAM-Limit prüfen
        free_ram = self.get_free_ram_percent()
        if free_ram < self.min_free_ram_percent:
            self.ram_throttle_count += 1
            return False

        return True

    def log(self, msg: str):
        """Optionale Ausgabe."""
        if self.verbose:
            print(f"[ResourceManager] {msg}")

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
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        Verarbeitet Items mit adaptiver Worker-Anzahl.

        Startet mit wenigen Workern und skaliert hoch, solange RAM verfügbar ist.
        Reduziert Worker-Anzahl wenn RAM knapp wird.

        Args:
            func: Funktion die auf jedes Item angewendet wird
            items: Liste der zu verarbeitenden Items
            progress_callback: Optional callback(completed, total)

        Returns:
            Liste der Ergebnisse (None-Werte werden gefiltert)
        """
        if not items:
            return []

        total = len(items)
        results = []
        completed = 0

        # Status ausgeben
        status = self.get_status()
        self.log(f"System: {status['total_cores']} Cores, {status['total_ram_gb']:.1f} GB RAM")
        self.log(f"Limits: max {self.max_workers} Workers, min {self.min_free_ram_percent*100:.0f}% free RAM")
        self.log(f"Aktuell frei: {status['free_ram_gb']:.1f} GB ({status['free_ram_percent']:.0f}%)")

        # Start mit konservativer Worker-Anzahl
        initial_workers = max(1, min(self.max_workers // 2, 4))

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Futures verwalten
            futures = {}
            items_iter = iter(enumerate(items))
            active_count = 0

            # Initial einige Tasks starten
            for _ in range(initial_workers):
                try:
                    idx, item = next(items_iter)
                    future = executor.submit(func, item)
                    futures[future] = idx
                    active_count += 1
                except StopIteration:
                    break

            self.peak_workers = active_count

            while futures:
                # Auf nächstes Ergebnis warten (mit Timeout für RAM-Checks)
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # Kurz warten und erneut prüfen
                    time.sleep(0.1)
                    continue

                # Ergebnisse sammeln
                for future in done_futures:
                    idx = futures.pop(future)
                    active_count -= 1
                    completed += 1

                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        self.log(f"Worker-Fehler: {e}")

                    if progress_callback:
                        progress_callback(completed, total)

                # Neue Tasks starten wenn möglich
                while True:
                    if not self.can_spawn_worker(active_count):
                        break

                    try:
                        idx, item = next(items_iter)
                        future = executor.submit(func, item)
                        futures[future] = idx
                        active_count += 1

                        if active_count > self.peak_workers:
                            self.peak_workers = active_count
                    except StopIteration:
                        break

                # RAM-Status loggen bei Throttling
                if self.ram_throttle_count > 0 and self.ram_throttle_count % 10 == 0:
                    free_gb = self.get_free_ram_gb()
                    self.log(f"RAM-Throttling aktiv: {free_gb:.1f} GB frei, {active_count} aktive Worker")

        # Finale Stats
        final_status = self.get_status()
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
    min_free_ram_percent: float = 0.20,
    estimated_ram_per_worker_gb: float = 3.0
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
