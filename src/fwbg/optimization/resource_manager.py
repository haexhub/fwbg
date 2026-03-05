"""Simple process pool with signal handling for the optimizer."""
import os
import sys
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Any, Optional

import psutil


# Global references for cleanup
_active_executor = None
_active_futures = []
_original_sigint = None
_original_sigterm = None


def _init_worker(progress_queue):
    """Initializer for worker processes — sets progress queue, ignores SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from fwbg.utils.progress import set_progress_queue
        set_progress_queue(progress_queue)
    except ImportError:
        pass


def _cleanup_on_interrupt():
    """Terminate all active worker processes on interrupt."""
    global _active_executor, _active_futures

    if _active_futures:
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

    _kill_child_processes()


def _kill_child_processes():
    """Terminate all child processes."""
    try:
        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
        if not children:
            return
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gone, alive = psutil.wait_procs(children, timeout=2)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except Exception:
        pass


def _signal_handler(signum, frame):
    """Handler for SIGINT/SIGTERM."""
    print("\n[Pool] Interrupt received, cleaning up...", file=sys.stderr)
    _cleanup_on_interrupt()
    # Use os._exit to bypass ProcessPoolExecutor.__exit__ which calls
    # shutdown(wait=True) and blocks until workers finish.
    os._exit(1)


class SimplePoolManager:
    """Process pool with a fixed number of workers and signal handling.

    KISS: max_concurrent_assets is the only control knob.
    """

    def __init__(self, max_concurrent_assets: int = 1, progress_queue=None):
        self.max_workers = max(1, max_concurrent_assets)
        self.progress_queue = progress_queue
        self.peak_workers = 0

    def map_adaptive(
        self,
        func: Callable,
        items: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        result_callback: Optional[Callable[[Any], None]] = None,
    ) -> List[Any]:
        """Process items with a fixed-size pool."""
        if not items:
            return []

        total = len(items)
        results = []
        completed = 0

        global _active_executor, _active_futures, _original_sigint, _original_sigterm

        _original_sigint = signal.signal(signal.SIGINT, _signal_handler)
        _original_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

        executor_kwargs = {"max_workers": self.max_workers}
        if self.progress_queue is not None:
            executor_kwargs["initializer"] = _init_worker
            executor_kwargs["initargs"] = (self.progress_queue,)

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            _active_executor = executor

            futures = {}
            for idx, item in enumerate(items):
                future = executor.submit(func, item)
                futures[future] = idx
                _active_futures.append(future)

            self.peak_workers = min(self.max_workers, total)

            while futures:
                done_futures = [f for f in futures if f.done()]

                for future in done_futures:
                    idx = futures.pop(future)
                    completed += 1

                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            if result_callback:
                                try:
                                    result_callback(result)
                                except Exception:
                                    pass
                    except Exception as e:
                        print(f"[Pool] Worker error #{idx}: {e}", file=sys.stderr)

                    if progress_callback:
                        progress_callback(completed, total)

                if futures:
                    time.sleep(0.5)

            _active_futures.clear()

            # Drain progress queue before executor __exit__ joins workers.
            # Workers can't exit if their feeder threads are blocked on a full pipe.
            if self.progress_queue is not None:
                from queue import Empty
                try:
                    while True:
                        self.progress_queue.get_nowait()
                except (Empty, OSError):
                    pass

        _active_executor = None

        signal.signal(signal.SIGINT, _original_sigint if _original_sigint else signal.SIG_DFL)
        signal.signal(signal.SIGTERM, _original_sigterm if _original_sigterm else signal.SIG_DFL)

        return results

    def get_status(self) -> dict:
        return {
            "max_workers": self.max_workers,
            "peak_workers": self.peak_workers,
        }


def get_resource_info() -> dict:
    """Return current system resource info."""
    import multiprocessing as mp
    mem = psutil.virtual_memory()
    return {
        "cpu_cores": mp.cpu_count(),
        "ram_total_gb": round(mem.total / (1024**3), 1),
        "ram_available_gb": round(mem.available / (1024**3), 1),
        "ram_used_percent": round(mem.percent, 1),
        "ram_free_percent": round(100 - mem.percent, 1),
    }
