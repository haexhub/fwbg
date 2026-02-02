# Contributing to FWBG

Vielen Dank für dein Interesse an FWBG!

## Development Setup

```bash
# Repository klonen
git clone https://github.com/haexhub/fwbg.git
cd fwbg

# Virtual Environment erstellen
python -m venv .venv
source .venv/bin/activate

# Mit Development-Dependencies installieren
pip install -e ".[dev,ig]"
```

## Code Style

- **Formatter**: Black (line-length: 100)
- **Import Sorting**: isort (black profile)
- **Type Hints**: mypy

```bash
# Formatierung
black src/ tests/ bots/
isort src/ tests/ bots/

# Type Checking
mypy src/
```

## Tests

```bash
# Alle Tests ausführen
pytest

# Mit Coverage
pytest --cov=src/fwbg
```

## Pull Requests

1. Fork das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/mein-feature`)
3. Committe deine Änderungen
4. Push zum Branch (`git push origin feature/mein-feature`)
5. Öffne einen Pull Request

## Projektstruktur

```
src/fwbg/           # Core Library
├── builtins/       # Built-in Plugins (Indicators, Exit Strategies, etc.)
├── optimizer/      # Walk-Forward Optimization Engine
├── adapters/       # Data & Execution Adapters
└── core/           # Plugin Registry, Config, Base Classes

bots/               # Trading Bots
tests/              # Test Suite
```

## Plugin-Entwicklung

Siehe [docs/ADAPTERS.md](docs/ADAPTERS.md) für Informationen zur Entwicklung eigener Adapter und Plugins.
