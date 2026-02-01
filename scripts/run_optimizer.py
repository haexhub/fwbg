#!/usr/bin/env python3
"""
Walk-Forward Optimizer - Einstiegspunkt

Verwendung:
    python run_optimizer.py                              # Standard-Run
    python run_optimizer.py -d "Test mit neuen Makros"   # Mit Beschreibung
    python run_optimizer.py --list                       # Alle Runs anzeigen
    python run_optimizer.py --compare RUN1 RUN2          # Runs vergleichen

    TIMEFRAME=MINUTE_15 python run_optimizer.py          # Andere Timeframes
    ACCOUNT_NAME=live python run_optimizer.py            # Anderer Account
"""
from optimizer import main

if __name__ == "__main__":
    main()
