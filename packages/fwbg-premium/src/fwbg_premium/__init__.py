"""FWBG Premium Plugins - advanced indicators, exit strategies & more."""
from pathlib import Path

__version__ = "1.0.0"


def get_plugins_dir() -> Path:
    """Return path to premium plugins directory for auto-discovery."""
    return Path(__file__).parent / "plugins"
