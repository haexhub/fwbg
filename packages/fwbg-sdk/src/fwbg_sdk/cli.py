"""CLI for fwbg-sdk: scaffold new plugin packages."""
import json
import textwrap
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Plugin type -> (base class, registration decorator, category dir name)
# ---------------------------------------------------------------------------
PLUGIN_TYPES = {
    "indicator": {
        "base_class": "BaseIndicator",
        "register_decorator": "register_indicator",
        "category_dir": "indicators",
        "manifest_phase": "indicators",
        "extra_imports": "shift_features, safe_divide",
    },
    "preprocessor": {
        "base_class": "BasePreprocessor",
        "register_decorator": "register_preprocessor",
        "category_dir": "preprocessors",
        "manifest_phase": "preprocessors",
        "extra_imports": None,
    },
    "feature_selector": {
        "base_class": "BaseFeatureSelector",
        "register_decorator": "register_feature_selector",
        "category_dir": "feature_selectors",
        "manifest_phase": "feature_selectors",
        "extra_imports": None,
    },
    "exit_strategy": {
        "base_class": "BaseExitStrategy",
        "register_decorator": "register_exit_strategy",
        "category_dir": "exit_strategies",
        "manifest_phase": "exit_strategies",
        "extra_imports": None,
    },
    "risk_manager": {
        "base_class": "BaseRiskManager",
        "register_decorator": "register_risk_manager",
        "category_dir": "risk_management",
        "manifest_phase": "risk_management",
        "extra_imports": None,
    },
    "data_loader": {
        "base_class": "BaseDataLoader",
        "register_decorator": "register_data_loader",
        "category_dir": "data_loading",
        "manifest_phase": "data_loading",
        "extra_imports": None,
    },
}


# ---------------------------------------------------------------------------
# Code templates per plugin type
# ---------------------------------------------------------------------------


def _indicator_template(class_name: str, plugin_name: str) -> str:
    return textwrap.dedent(f'''\
        """Indicator plugin: {plugin_name}."""
        from typing import List
        import pandas as pd
        from fwbg_sdk import (
            BaseIndicator, shift_features, safe_divide,
            register_indicator,
        )


        @register_indicator("{plugin_name}")
        class {class_name}(BaseIndicator):
            """Custom indicator plugin."""

            name = "{plugin_name}"
            version = "1.0.0"
            group = "custom"

            def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
                features = {{}}
                # TODO: compute your features here
                # Example: features["feat_example"] = safe_divide(df["C"], df["O"])
                features_df = shift_features(features, df.index)
                return pd.concat([df, features_df], axis=1)

            def get_feature_columns(self) -> List[str]:
                return []
    ''')


def _preprocessor_template(class_name: str, plugin_name: str) -> str:
    return textwrap.dedent(f'''\
        """Preprocessor plugin: {plugin_name}."""
        import pandas as pd
        from fwbg_sdk import BasePreprocessor, register_preprocessor


        @register_preprocessor("{plugin_name}")
        class {class_name}(BasePreprocessor):
            """Custom preprocessor plugin."""

            name = "{plugin_name}"
            version = "1.0.0"

            def fit(self, df: pd.DataFrame, **params) -> "BasePreprocessor":
                # TODO: learn parameters from training data
                self.fitted_ = True
                return self

            def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
                super().transform(df, **params)
                # TODO: apply transformation
                return df
    ''')


def _feature_selector_template(class_name: str, plugin_name: str) -> str:
    return textwrap.dedent(f'''\
        """Feature selector plugin: {plugin_name}."""
        from typing import List, Tuple
        import numpy as np
        import pandas as pd
        from fwbg_sdk import BaseFeatureSelector, register_feature_selector


        @register_feature_selector("{plugin_name}")
        class {class_name}(BaseFeatureSelector):
            """Custom feature selector plugin."""

            name = "{plugin_name}"
            version = "1.0.0"

            def select_features(
                self,
                X: pd.DataFrame,
                y: np.ndarray,
                max_features: int = None,
                **params,
            ) -> Tuple[List[str], dict]:
                # TODO: implement feature selection logic
                selected = list(X.columns[:max_features]) if max_features else list(X.columns)
                return selected, {{"method": "{plugin_name}"}}
    ''')


def _exit_strategy_template(class_name: str, plugin_name: str) -> str:
    return textwrap.dedent(f'''\
        """Exit strategy plugin: {plugin_name}."""
        from typing import Tuple, Iterator
        import numpy as np
        import pandas as pd
        from fwbg_sdk import BaseExitStrategy, AssetInfo, register_exit_strategy


        @register_exit_strategy("{plugin_name}")
        class {class_name}(BaseExitStrategy):
            """Custom exit strategy plugin."""

            name = "{plugin_name}"
            version = "1.0.0"

            def compute_targets(
                self,
                df: pd.DataFrame,
                ctx: "AssetInfo",
                **params,
            ) -> Tuple[np.ndarray, np.ndarray]:
                # TODO: compute win/loss targets for long and short
                n = len(df)
                targets_long = np.zeros(n)
                targets_short = np.zeros(n)
                return targets_long, targets_short

            def iterate_grid(
                self,
                grid_config: dict,
                ctx: "AssetInfo",
            ) -> Iterator[dict]:
                # TODO: yield parameter combinations
                yield {{}}

            def get_cache_key(self, params: dict) -> str:
                return "{plugin_name}_" + "_".join(f"{{k}}={{v}}" for k, v in sorted(params.items()))
    ''')


def _risk_manager_template(class_name: str, plugin_name: str) -> str:
    return textwrap.dedent(f'''\
        """Risk manager plugin: {plugin_name}."""
        from typing import Dict, Any, List
        from fwbg_sdk import BaseRiskManager, register_risk_manager


        @register_risk_manager("{plugin_name}")
        class {class_name}(BaseRiskManager):
            """Custom risk manager plugin."""

            name = "{plugin_name}"
            version = "1.0.0"

            def compute_risk_params(
                self,
                trades: List[float],
                win_rate: float,
                rrr: float,
                **params,
            ) -> Dict[str, Any]:
                # TODO: implement risk computation
                return {{
                    "risk_per_trade": 0.01,
                    "trade_returns": trades,
                    "circuit_breaker": {{"pause_after_losses": 3, "pause_bars": 10, "enabled": False}},
                    "risk_adjustment": {{"original_risk": 0.01, "scale_factor": 1.0, "target_dd": 0.10}},
                }}
    ''')


def _data_loader_template(class_name: str, plugin_name: str) -> str:
    return textwrap.dedent(f'''\
        """Data loader plugin: {plugin_name}."""
        from fwbg_sdk import BaseDataLoader, register_data_loader


        @register_data_loader("{plugin_name}")
        class {class_name}(BaseDataLoader):
            """Custom data loader plugin."""

            name = "{plugin_name}"
            version = "1.0.0"

            def execute(self, ctx, **params):
                # TODO: compute derived features from raw data in ctx.df
                return ctx
    ''')


_TEMPLATE_MAP = {
    "indicator": _indicator_template,
    "preprocessor": _preprocessor_template,
    "feature_selector": _feature_selector_template,
    "exit_strategy": _exit_strategy_template,
    "risk_manager": _risk_manager_template,
    "data_loader": _data_loader_template,
}


def _tests_template(plugin_name: str, plugin_type: str) -> str:
    info = PLUGIN_TYPES[plugin_type]
    return textwrap.dedent(f'''\
        """Tests for {plugin_name} plugin."""
        import pytest


        def test_{plugin_name}_loads():
            """Verify the plugin module can be imported."""
            from . import __init__  # noqa: F401
    ''')


def _to_class_name(snake_name: str) -> str:
    """Convert snake_case to PascalCase and append plugin type suffix."""
    return "".join(word.capitalize() for word in snake_name.split("_"))


# ---------------------------------------------------------------------------
# Init command
# ---------------------------------------------------------------------------

def _create_plugin_dir(
    plugins_dir: Path,
    package_name: str,
    plugin_type: str,
    plugin_name: str,
) -> None:
    """Create a single plugin directory with __init__.py, manifest.json, tests.py."""
    info = PLUGIN_TYPES[plugin_type]
    category_dir = info["category_dir"]
    plugin_dir = plugins_dir / package_name / category_dir / plugin_name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # __init__.py
    class_name = _to_class_name(plugin_name)
    template_fn = _TEMPLATE_MAP[plugin_type]
    (plugin_dir / "__init__.py").write_text(template_fn(class_name, plugin_name))

    # manifest.json
    manifest = {
        "name": plugin_name,
        "version": "1.0.0",
        "description": f"{plugin_type.replace('_', ' ').title()} plugin: {plugin_name}",
        "phase": info["manifest_phase"],
    }
    (plugin_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # tests.py
    (plugin_dir / "tests.py").write_text(_tests_template(plugin_name, plugin_type))


def _update_package_manifest(
    plugins_dir: Path,
    package_name: str,
    plugin_type: str,
    plugin_name: str,
) -> None:
    """Update (or create) the package-level manifest.json."""
    info = PLUGIN_TYPES[plugin_type]
    category = info["category_dir"]
    manifest_path = plugins_dir / package_name / "manifest.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "name": package_name,
            "version": "1.0.0",
            "description": f"FWBG plugin package: {package_name}",
            "author": "",
            "license": "MIT",
            "plugins": {},
        }

    plugins_section = manifest.setdefault("plugins", {})
    plugin_list = plugins_section.setdefault(category, [])
    if plugin_name not in plugin_list:
        plugin_list.append(plugin_name)

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def _parse_plugin_spec(spec: str) -> tuple:
    """Parse 'type:name' into (type, name). Raises click.BadParameter on invalid input."""
    if ":" not in spec:
        raise click.BadParameter(
            f"Invalid plugin spec '{spec}'. Expected format: type:name "
            f"(e.g. indicator:my_rsi). Valid types: {', '.join(PLUGIN_TYPES)}",
            param_hint="'--plugin'",
        )
    ptype, pname = spec.split(":", 1)
    if ptype not in PLUGIN_TYPES:
        raise click.BadParameter(
            f"Unknown plugin type '{ptype}'. Valid types: {', '.join(PLUGIN_TYPES)}",
            param_hint="'--plugin'",
        )
    return ptype, pname


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def main():
    """FWBG SDK - scaffold and manage plugin packages."""
    pass


@main.command()
@click.argument("name")
@click.option(
    "--plugin", "-p",
    multiple=True,
    help="Plugin to create: type:name (e.g. indicator:my_rsi). Can be repeated.",
)
@click.option(
    "--output-dir", "-o",
    default=".",
    type=click.Path(),
    help="Parent directory for the new package (default: current dir).",
)
def init(name: str, plugin: tuple, output_dir: str):
    """Create a new FWBG plugin package.

    NAME is the package name (e.g. my-indicators).
    """
    output = Path(output_dir)
    pkg_dir = output / name
    module_name = name.replace("-", "_")
    src_dir = pkg_dir / "src" / module_name
    plugins_dir = src_dir / "plugins"

    # Create directory structure
    src_dir.mkdir(parents=True, exist_ok=True)
    plugins_dir.mkdir(parents=True, exist_ok=True)

    # pyproject.toml
    pyproject = textwrap.dedent(f"""\
        [build-system]
        requires = ["setuptools>=61.0", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{name}"
        version = "0.1.0"
        description = "FWBG plugin package: {name}"
        requires-python = ">=3.11"
        license = {{text = "MIT"}}
        dependencies = [
            "fwbg-sdk>=1.0.0",
        ]

        [project.entry-points."fwbg.plugin_packages"]
        {name} = "{module_name}:get_plugins_dir"

        [tool.setuptools.packages.find]
        where = ["src"]
    """)
    (pkg_dir / "pyproject.toml").write_text(pyproject)

    # __init__.py with get_plugins_dir
    init_py = textwrap.dedent(f'''\
        """FWBG plugin package: {name}."""
        from pathlib import Path


        def get_plugins_dir() -> Path:
            """Return the path to the plugins directory for fwbg discovery."""
            return Path(__file__).parent / "plugins"
    ''')
    (src_dir / "__init__.py").write_text(init_py)

    # Create each plugin
    parsed_plugins = []
    for spec in plugin:
        ptype, pname = _parse_plugin_spec(spec)
        parsed_plugins.append((ptype, pname))
        _create_plugin_dir(plugins_dir, name, ptype, pname)
        _update_package_manifest(plugins_dir, name, ptype, pname)

    click.echo(f"Created plugin package '{name}' at {pkg_dir}")
    for ptype, pname in parsed_plugins:
        click.echo(f"  - {ptype}: {pname}")
    click.echo(f"\nNext steps:")
    click.echo(f"  cd {pkg_dir}")
    click.echo(f"  pip install -e .")


@main.command()
@click.argument("plugin_type", type=click.Choice(list(PLUGIN_TYPES.keys())))
@click.argument("plugin_name")
@click.option(
    "--package-dir", "-d",
    default=".",
    type=click.Path(exists=True),
    help="Root of the plugin package (default: current dir).",
)
def add(plugin_type: str, plugin_name: str, package_dir: str):
    """Add a new plugin to an existing package.

    PLUGIN_TYPE is one of: indicator, preprocessor, feature_selector,
    exit_strategy, risk_manager, data_loader.

    PLUGIN_NAME is the snake_case name for the plugin.
    """
    pkg_root = Path(package_dir)

    # Find the src/<module>/plugins dir
    src_dir = pkg_root / "src"
    if not src_dir.exists():
        raise click.ClickException(f"No 'src/' directory found in {pkg_root}")

    # Find the module directory (first subdir of src/)
    module_dirs = [d for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not module_dirs:
        raise click.ClickException(f"No module directory found in {src_dir}")
    module_dir = module_dirs[0]

    plugins_dir = module_dir / "plugins"
    if not plugins_dir.exists():
        raise click.ClickException(f"No 'plugins/' directory found in {module_dir}")

    # Find the package name (first subdir of plugins/)
    pkg_names = [d.name for d in plugins_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if not pkg_names:
        raise click.ClickException(f"No plugin package found in {plugins_dir}")
    package_name = pkg_names[0]

    _create_plugin_dir(plugins_dir, package_name, plugin_type, plugin_name)
    _update_package_manifest(plugins_dir, package_name, plugin_type, plugin_name)

    click.echo(f"Added {plugin_type} '{plugin_name}' to {package_name}")
