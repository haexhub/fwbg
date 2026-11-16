"""Explicit, reproducible input contract for portfolio research."""

import hashlib
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import pandas as pd

from .engine import PortfolioSettings, validate_daily_index
from .overlay import OverlaySettings, exposure_mask, input_files, overlay_condition


def read_daily_csv(path: Path) -> pd.DataFrame:
    """Read numeric CSV columns indexed by sorted, unique, timezone-free dates."""
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError(f"{path}: missing date column")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("date"), format="%Y-%m-%d"))
    frame.index.name = "date"
    validate_daily_index(frame.index, str(path))
    return frame.apply(pd.to_numeric, errors="raise")


def file_fingerprint(path: Path) -> dict:
    """Return the resolved file path and SHA-256 digest of its contents."""
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_inputs(config_path: Path) -> tuple[dict, PortfolioSettings, pd.DataFrame, dict]:
    """Validate a config and load settings, prices and provenance for a backtest.

    Resolve input paths relative to the config file and require explicit
    total-return prices and a base currency matching any supplied metadata.
    """
    config = json.loads(config_path.read_text())
    allowed = {f.name for f in fields(PortfolioSettings)} | {
        "schema_version",
        "name",
        "prices",
        "base_currency",
        "price_basis",
        "metadata",
        "split_date",
        "overlay",
    }
    if extra := set(config) - allowed:
        raise ValueError(f"unknown portfolio config fields: {sorted(extra)}")
    if config.get("schema_version") != 1 or config.get("price_basis") != "total_return":
        raise ValueError("schema_version=1 and price_basis=total_return are required")
    if not isinstance(config.get("base_currency"), str) or len(config["base_currency"]) != 3:
        raise ValueError("base_currency must be an explicit three-letter currency code")
    for key in ("name", "prices", "weights", "equity_benchmark"):
        if not config.get(key):
            raise ValueError(f"missing required config field: {key}")
    setting_names = {f.name for f in fields(PortfolioSettings)} - {"overlay"}
    settings = PortfolioSettings(**{k: v for k, v in config.items() if k in setting_names})
    settings.validate()
    price_path = config_path.parent / config["prices"]
    prices = read_daily_csv(price_path)
    provenance = {
        "config": file_fingerprint(config_path),
        "prices": file_fingerprint(price_path),
        "implementation": {
            name: file_fingerprint(Path(__file__).with_name(name))
            for name in ("engine.py", "rules.py", "data.py", "cli.py", "robustness.py", "overlay.py")
        },
    }
    if "overlay" in config:
        overlay_config = config["overlay"]
        if not isinstance(overlay_config, dict):
            raise ValueError("overlay must be an object")
        if set(overlay_config) - {"name", "weeks", "lag_days", "threshold", "data_directory"}:
            raise ValueError("unknown overlay config fields")
        for key in ("name", "data_directory"):
            if not overlay_config.get(key):
                raise ValueError(f"missing required overlay field: {key}")
        data_directory = (config_path.parent / overlay_config["data_directory"]).resolve()
        overlay_settings = OverlaySettings(
            **{k: v for k, v in overlay_config.items() if k != "data_directory"}
        )
        condition = overlay_condition(overlay_settings, data_directory)
        settings = replace(settings, overlay=exposure_mask(condition, prices.index))
        provenance["overlay"] = {
            "settings": asdict(overlay_settings),
            "first_condition": condition.index[0].date().isoformat(),
            "invested_share": float(settings.overlay.mean()),
            "inputs": {
                path.name: file_fingerprint(path)
                for path in input_files(overlay_settings, data_directory)
            },
        }
    if config.get("metadata"):
        meta_path = config_path.parent / config["metadata"]
        provenance["metadata"] = json.loads(meta_path.read_text())
        provenance["metadata_file"] = file_fingerprint(meta_path)
        if provenance["metadata"].get("base_currency") != config["base_currency"]:
            raise ValueError("metadata and config base_currency do not match")
    return config, settings, prices, provenance
