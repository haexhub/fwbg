#!/usr/bin/env python3
"""
Diagnose-Skript um zu sehen wo der Optimizer hängt.
"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def timed(name):
    """Context manager für Timing."""
    class Timer:
        def __init__(self, name):
            self.name = name
        def __enter__(self):
            self.start = time.time()
            print(f"[START] {self.name}...", flush=True)
            return self
        def __exit__(self, *args):
            print(f"[DONE]  {self.name} ({time.time() - self.start:.2f}s)", flush=True)
    return Timer(name)

print("="*60)
print("DIAGNOSE: Optimizer Startup")
print("="*60)

with timed("Import optimizer.config"):
    from optimizer.config import DATA_PATH, TIMEFRAME, MACRO_INDICATORS

with timed("Import optimizer.data_loader"):
    from optimizer.data_loader import load_data_aligned, load_macro_csv

with timed("Import optimizer.process"):
    from optimizer.process import process_symbol

with timed("Import glob"):
    import glob

# Prüfe Dateien
print(f"\nDATA_PATH: {DATA_PATH}")
print(f"TIMEFRAME: {TIMEFRAME}")

files = sorted(glob.glob(f"{DATA_PATH}/*_{TIMEFRAME}.csv"))
print(f"Gefundene Dateien: {len(files)}")
if files:
    print(f"Erste 5: {[os.path.basename(f) for f in files[:5]]}")

# Prüfe Makro-Dateien
print(f"\nMakro-Dateien ({len(MACRO_INDICATORS)}):")
for filename in list(MACRO_INDICATORS.keys())[:5]:
    path = f"{DATA_PATH}/{filename}.csv"
    exists = os.path.exists(path)
    if exists:
        df = load_macro_csv(path)
        status = f"OK ({len(df)} rows)" if df is not None else "FEHLER beim Laden"
    else:
        status = "NICHT GEFUNDEN"
    print(f"  {filename}: {status}")

# Teste einzelnes Asset laden
if files:
    test_file = files[0]
    print(f"\n{'='*60}")
    print(f"Test: Lade {os.path.basename(test_file)}")
    print("="*60)

    with timed("load_data_aligned"):
        df = load_data_aligned(test_file)

    if df is not None:
        print(f"  Zeilen: {len(df)}")
        print(f"  Spalten: {list(df.columns)}")
        print(f"  Index: {df.index[0]} bis {df.index[-1]}")
    else:
        print("  FEHLER: df ist None")

    # Teste process_symbol mit Timeout
    print(f"\n{'='*60}")
    print(f"Test: process_symbol (max 30s)")
    print("="*60)

    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("process_symbol dauert > 30s")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

    try:
        with timed("process_symbol"):
            result = process_symbol(test_file)
        signal.alarm(0)

        if result:
            print(f"  Symbol: {result['symbol']}")
            print(f"  Trades: {len(result['tr_trace'])}")
            print(f"  Sharpe: {result['sharpe']:.2f}")
        else:
            print("  Result: None (Asset nicht profitabel)")
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        print("  -> Verarbeitung dauert zu lange, prüfe Grid-Größe")
    except Exception as e:
        print(f"  FEHLER: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSE ABGESCHLOSSEN")
print("="*60)
