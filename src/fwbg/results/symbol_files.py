"""Per-symbol run files beyond the optimizer's classic set."""

_CONFIG_BLOCKS = ("overfitting",)


def config_extras(result: dict) -> dict:
    """Blocks a symbol's config.json carries for the research register."""
    return {key: result[key] for key in _CONFIG_BLOCKS if result.get(key)}
