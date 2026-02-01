# FWBG Adapter Package Template

Dieses Verzeichnis zeigt, wie ein FWBG Adapter als eigenständiges PyPI Package erstellt wird.

## Struktur

```
adapter_package/
├── pyproject.toml              # Package-Definition mit Entry Points
├── README.md                   # Diese Datei
└── src/
    └── fwbg_adapter_binance/
        └── __init__.py         # Adapter-Implementation
```

## Lokal installieren & testen

```bash
cd examples/adapter_package
pip install -e .
```

Nach der Installation wird der Adapter automatisch erkannt:

```python
from fwbg.core import discover_plugins, list_execution_adapters

discover_plugins()
print(list_execution_adapters())  # ['ig', 'binance']
```

## Eigenen Adapter erstellen

1. **Verzeichnis kopieren:**
   ```bash
   cp -r examples/adapter_package my-adapter
   cd my-adapter
   ```

2. **Umbenennen:**
   - Ordner: `src/fwbg_adapter_binance` → `src/fwbg_adapter_BROKER`
   - In `pyproject.toml`: Name und Entry Point anpassen
   - In `__init__.py`: Klasse anpassen

3. **Implementieren:**
   - `connect()` / `disconnect()`
   - `submit_order()`
   - `cancel_order()`
   - `get_positions()`
   - `get_account_info()`

4. **Testen:**
   ```bash
   pip install -e .
   python -c "from fwbg.core import discover_plugins; discover_plugins()"
   ```

5. **Veröffentlichen:**
   ```bash
   pip install build twine
   python -m build
   twine upload dist/*
   ```

## Entry Points

Der Schlüssel ist der Entry Point in `pyproject.toml`:

```toml
[project.entry-points."fwbg.execution_adapters"]
binance = "fwbg_adapter_binance:BinanceAdapter"
```

- `fwbg.execution_adapters` = Entry Point Group (für Broker)
- `fwbg.data_adapters` = Entry Point Group (für Datenquellen)
- `binance` = Name unter dem der Adapter registriert wird
- `fwbg_adapter_binance:BinanceAdapter` = Module:Klasse

## Siehe auch

- [docs/adapters.md](../../docs/adapters.md) - Vollständige Dokumentation
- [examples/custom_adapter.py](../custom_adapter.py) - Einfaches Beispiel ohne Package
