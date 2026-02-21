"""
Konvertiert Dukascopy-Rohdaten (Unix-ms-Timestamps, dukascopy-Dateinamen)
in das Standard-Format des FWBG-Optimizers.

Input:  data/dukascopy/raw/{prefix}_m15.csv
        Spalten: timestamp (Unix ms), open, high, low, close
        Kein Volume.

Output: data/dukascopy/datasource/{SYMBOL}_MINUTE_15.csv
        Spalten: T (ISO datetime), O, H, L, C, V=0
        Nanosekunden-Timestamps → kein Timezone-Header

Verwendung:
    python scripts/convert_dukascopy.py
    python scripts/convert_dukascopy.py --raw-dir data/dukascopy/raw --out-dir data/dukascopy/datasource
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


CONFIG_PATH = Path("data/dukascopy/config.json")


def load_symbol_map(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("symbol_map", {})


def convert_file(src: Path, dst: Path, symbol: str) -> int:
    """Liest eine Dukascopy-CSV und schreibt Standard-Format."""
    df = pd.read_csv(src)

    if "timestamp" not in df.columns:
        raise ValueError(f"{src}: Keine 'timestamp'-Spalte gefunden")

    df["T"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["T"] = df["T"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cols_in = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if len(cols_in) < 4:
        raise ValueError(f"{src}: Erwartete Spalten fehlen (gefunden: {list(df.columns)})")

    out = df[["T"] + cols_in].copy()
    out.columns = ["T", "O", "H", "L", "C"]
    out["V"] = 0

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return len(out)


def main():
    parser = argparse.ArgumentParser(description="Konvertiert Dukascopy-Rohdaten ins FWBG-Format")
    parser.add_argument("--raw-dir", default="data/dukascopy/raw",
                        help="Verzeichnis mit Dukascopy-Rohdaten")
    parser.add_argument("--out-dir", default="data/dukascopy/datasource",
                        help="Ausgabe-Verzeichnis für konvertierte Dateien")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="Pfad zur dukascopy config.json")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    config_path = Path(args.config)

    if not raw_dir.exists():
        print(f"Raw-Verzeichnis nicht gefunden: {raw_dir}", file=sys.stderr)
        sys.exit(1)

    symbol_map = load_symbol_map(config_path)
    if not symbol_map:
        print(f"Kein symbol_map in {config_path}", file=sys.stderr)
        sys.exit(1)

    raw_files = sorted(raw_dir.glob("*_m15.csv"))
    if not raw_files:
        print(f"Keine *_m15.csv Dateien in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    converted = 0
    skipped = 0
    for raw_file in raw_files:
        # Extrahiere Präfix: DE40_DAX aus DE40_DAX_m15.csv
        stem = raw_file.stem          # DE40_DAX_m15
        prefix = stem.rsplit("_m15", 1)[0]  # DE40_DAX

        symbol = symbol_map.get(prefix)
        if not symbol:
            print(f"  SKIP {raw_file.name} — kein Symbol-Mapping für '{prefix}'")
            skipped += 1
            continue

        dst = out_dir / f"{symbol}_MINUTE_15.csv"
        try:
            n = convert_file(raw_file, dst, symbol)
            print(f"  OK   {raw_file.name} → {dst.name} ({n} Bars)")
            converted += 1
        except Exception as e:
            print(f"  FAIL {raw_file.name}: {e}", file=sys.stderr)

    print(f"\nFertig: {converted} konvertiert, {skipped} übersprungen.")
    if converted == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
