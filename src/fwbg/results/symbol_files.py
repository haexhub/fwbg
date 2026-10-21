"""Per-symbol run files beyond the optimizer's classic set."""

import json
import os

from fwbg.data.config import convert_numpy

_CONFIG_BLOCKS = ("overfitting", "holdout")


def config_extras(result: dict) -> dict:
    """Blocks a symbol's config.json carries for the research register."""
    return {key: result[key] for key in _CONFIG_BLOCKS if result.get(key)}


def write_development(sym_dir: str, result: dict, encoder: type[json.JSONEncoder]) -> None:
    """development.json: a holdout look's development outcome, kept apart from its metrics."""
    if not result.get("development"):
        return
    with open(os.path.join(sym_dir, "development.json"), "w") as f:
        json.dump(convert_numpy(result["development"]), f, indent=2, cls=encoder)
