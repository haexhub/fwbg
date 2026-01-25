#!/usr/bin/env python3
"""
ForexSB Historical Data Crawler

Lädt automatisch historische Forex-Daten von https://forexsb.com/historical-forex-data
für alle Symbole und gewünschten Timeframes herunter.

Nutzung:
    python scripts/forexsb_crawler.py [--symbols EURUSD,GBPUSD] [--timeframes M15,M30,H1]

Standardmäßig werden alle verfügbaren Symbole und die Timeframes M15, M30, H1 geladen.
"""
import argparse
import os
import re
import time
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Zielverzeichnis für Downloads
DATA_DIR = Path(__file__).parent.parent / "data" / "forexsb"

# Alle verfügbaren Symbole auf ForexSB
# Format: (dropdown_value, unser_dateiname)
ALL_SYMBOLS = [
    # FX Major
    ("EURUSD", "EURUSD"),
    ("GBPUSD", "GBPUSD"),
    ("USDCAD", "USDCAD"),
    ("USDCHF", "USDCHF"),
    ("USDJPY", "USDJPY"),
    # FX Minor
    ("AUDCAD", "AUDCAD"),
    ("AUDCHF", "AUDCHF"),
    ("AUDJPY", "AUDJPY"),
    ("AUDNZD", "AUDNZD"),
    ("AUDUSD", "AUDUSD"),
    ("CADCHF", "CADCHF"),
    ("CADJPY", "CADJPY"),
    ("CHFJPY", "CHFJPY"),
    ("EURAUD", "EURAUD"),
    ("EURCAD", "EURCAD"),
    ("EURCHF", "EURCHF"),
    ("EURGBP", "EURGBP"),
    ("EURJPY", "EURJPY"),
    ("EURNZD", "EURNZD"),
    ("GBPAUD", "GBPAUD"),
    ("GBPCAD", "GBPCAD"),
    ("GBPCHF", "GBPCHF"),
    ("GBPJPY", "GBPJPY"),
    ("GBPNZD", "GBPNZD"),
    ("NZDCAD", "NZDCAD"),
    ("NZDCHF", "NZDCHF"),
    ("NZDJPY", "NZDJPY"),
    ("NZDUSD", "NZDUSD"),
    # Commodity
    ("BRENTCMDUSD", "BRENT"),
    ("XAGUSD", "XAGUSD"),
    ("XAUUSD", "XAUUSD"),
    # Indices
    ("DEUIDXEUR", "DAX"),
    ("GBRIDXGBP", "FTSE100"),
    ("USA30IDXUSD", "DOW30"),
    ("USA500IDXUSD", "SPX500"),
    ("USATECHIDXUSD", "NAS100"),
    # Crypto USD
    ("BTCUSD", "BTCUSD"),
    ("ETHUSD", "ETHUSD"),
]

# Nur die Dateinamen für einfache Nutzung
SYMBOL_NAMES = [s[1] for s in ALL_SYMBOLS]

# Unsere gewünschten Timeframes für die Optimierung
DEFAULT_TIMEFRAMES = ["M15", "M30", "H1"]


def get_target_filename(symbol: str, timeframe: str) -> str:
    """Generiert den Ziel-Dateinamen im Format das unser Optimizer erwartet."""
    if timeframe == "M15":
        return f"{symbol}_MINUTE_15.csv"
    elif timeframe == "M30":
        return f"{symbol}_MINUTE_30.csv"
    elif timeframe == "H1":
        return f"{symbol}_HOUR.csv"
    elif timeframe == "H4":
        return f"{symbol}_HOUR_4.csv"
    elif timeframe == "D1":
        return f"{symbol}_DAY.csv"
    else:
        return f"{symbol}_{timeframe}.csv"


def download_symbol_data(page, dropdown_value: str, file_symbol: str, timeframes: list[str], download_dir: Path) -> dict:
    """Lädt Daten für ein Symbol herunter."""
    results = {"symbol": file_symbol, "downloaded": [], "failed": []}

    try:
        # Warte auf den richtigen iframe (data-app-frame)
        iframe = page.frame_locator("#data-app-frame")

        # Warte bis das Select-Element geladen ist
        dropdown = iframe.locator("#select-symbol")
        dropdown.wait_for(state="visible", timeout=10000)

        # Wähle Symbol direkt über Value
        print(f"  Wähle Symbol: {dropdown_value}")
        dropdown.select_option(value=dropdown_value)
        time.sleep(1)

        # Format auf MetaTrader (CSV) setzen (value=0)
        print(f"  Setze Format auf MetaTrader (CSV)...")
        format_dropdown = iframe.locator("#select-format")
        format_dropdown.wait_for(state="visible", timeout=5000)
        format_dropdown.select_option(value="0")  # MetaTrader (CSV)
        time.sleep(1)

        # Load Data Button klicken
        print(f"  Klicke 'Load data'...")
        load_btn = iframe.locator("button:has-text('Load data'), .btn:has-text('Load data')")
        load_btn.click()

        # Warte bis Tabelle mit Downloads erscheint
        print(f"  Warte auf Daten...")
        time.sleep(8)  # Server braucht Zeit

        # Suche Download-Links für gewünschte Timeframes
        for tf in timeframes:
            try:
                print(f"  Download {tf}...")

                # Finde Download-Link für diesen Timeframe
                # Die Tabelle hat Spalten: Symbol, Period, Bars, From, To, Download
                download_link = None

                # Finde die Zeile mit dem Period (in tbody#table-acquisition)
                rows = iframe.locator("#table-acquisition tr").all()
                for row in rows:
                    cells = row.locator("td").all()
                    if len(cells) >= 6:
                        period_text = (cells[1].text_content() or "").strip()
                        if period_text == tf:
                            # Download-Link ist in der letzten Spalte
                            link = cells[5].locator("a").first
                            if link.count() > 0:
                                download_link = link
                                break

                if not download_link:
                    print(f"    {tf}: Nicht in Tabelle gefunden")
                    results["failed"].append(tf)
                    continue

                # Klicke Download-Link und fange Download ab
                with page.expect_download(timeout=60000) as download_info:
                    download_link.click()

                download = download_info.value

                # Speichere mit korrektem Namen (nutze file_symbol, nicht dropdown_value)
                target_name = get_target_filename(file_symbol, tf)
                target_path = download_dir / target_name

                download.save_as(target_path)
                print(f"    {tf}: OK -> {target_name}")
                results["downloaded"].append(tf)

                time.sleep(1)

            except PlaywrightTimeout:
                print(f"    {tf}: Timeout beim Download")
                results["failed"].append(tf)
            except Exception as e:
                print(f"    {tf}: Fehler - {e}")
                results["failed"].append(tf)

    except Exception as e:
        print(f"  FEHLER bei {symbol}: {e}")
        results["failed"] = timeframes

    return results


def crawl_forexsb(symbols: list[str], timeframes: list[str], headless: bool = True):
    """Hauptfunktion: Crawlt ForexSB für alle angegebenen Symbole."""

    # Erstelle Download-Verzeichnis
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"ForexSB Crawler")
    print(f"===============")
    print(f"Symbole: {len(symbols)}")
    print(f"Timeframes: {timeframes}")
    print(f"Zielverzeichnis: {DATA_DIR}")
    print()

    all_results = []

    with sync_playwright() as p:
        # Browser starten
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        # Zur Seite navigieren
        print("Lade ForexSB...")
        page.goto("https://forexsb.com/historical-forex-data", wait_until="networkidle")
        time.sleep(5)  # Warte auf iframe

        for i, (dropdown_value, file_symbol) in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] {file_symbol} ({dropdown_value})")

            # Lade Seite neu für jedes Symbol (sauberer Zustand)
            if i > 0:
                page.reload(wait_until="networkidle")
                time.sleep(5)

            result = download_symbol_data(page, dropdown_value, file_symbol, timeframes, DATA_DIR)
            all_results.append(result)

            # Kurze Pause zwischen Symbolen
            time.sleep(2)

        browser.close()

    # Zusammenfassung
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)

    total_downloaded = 0
    total_failed = 0

    for r in all_results:
        downloaded = len(r["downloaded"])
        failed = len(r["failed"])
        total_downloaded += downloaded
        total_failed += failed

        status = "OK" if failed == 0 else "TEILWEISE" if downloaded > 0 else "FEHLER"
        print(f"{r['symbol']}: {status} ({downloaded} geladen, {failed} fehlt)")

    print()
    print(f"Gesamt: {total_downloaded} Dateien heruntergeladen, {total_failed} fehlgeschlagen")
    print(f"Dateien in: {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="ForexSB Historical Data Crawler")
    parser.add_argument(
        "--symbols", "-s",
        type=str,
        default=None,
        help="Komma-getrennte Liste von Symbolen (z.B. EURUSD,GBPUSD). Standard: alle"
    )
    parser.add_argument(
        "--timeframes", "-t",
        type=str,
        default=",".join(DEFAULT_TIMEFRAMES),
        help=f"Komma-getrennte Liste von Timeframes. Standard: {','.join(DEFAULT_TIMEFRAMES)}"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Browser im Headless-Modus starten"
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Browser sichtbar starten (Standard)"
    )

    args = parser.parse_args()

    # Parse Symbole
    if args.symbols:
        # Benutzer hat Symbole angegeben - finde passende Tuples
        requested = [s.strip().upper() for s in args.symbols.split(",")]
        symbols = [(dv, fs) for dv, fs in ALL_SYMBOLS if fs in requested or dv in requested]
        if not symbols:
            print(f"FEHLER: Keine der angegebenen Symbole gefunden: {requested}")
            print(f"Verfügbar: {SYMBOL_NAMES}")
            return
    else:
        symbols = ALL_SYMBOLS

    # Parse Timeframes
    timeframes = [t.strip().upper() for t in args.timeframes.split(",")]

    # Headless-Modus (Standard: sichtbar)
    headless = args.headless and not args.visible

    crawl_forexsb(symbols, timeframes, headless)


if __name__ == "__main__":
    main()
