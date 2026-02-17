"""Tests for fwbg-sdk CLI (init, add commands)."""
import json
from pathlib import Path
from click.testing import CliRunner
from fwbg_sdk.cli import main


def test_init_creates_package(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "init", "my-indicators",
        "--plugin", "indicator:my_rsi",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output

    pkg_dir = tmp_path / "my-indicators"
    assert (pkg_dir / "pyproject.toml").exists()
    assert (pkg_dir / "src" / "my_indicators" / "__init__.py").exists()

    # Check plugin files
    plugin_dir = pkg_dir / "src" / "my_indicators" / "plugins" / "my-indicators" / "indicators" / "my_rsi"
    assert (plugin_dir / "__init__.py").exists()
    assert (plugin_dir / "manifest.json").exists()
    assert (plugin_dir / "tests.py").exists()

    # Check package manifest
    manifest = json.loads((pkg_dir / "src" / "my_indicators" / "plugins" / "my-indicators" / "manifest.json").read_text())
    assert manifest["name"] == "my-indicators"
    assert "my_rsi" in manifest["plugins"]["indicators"]


def test_init_pyproject_has_entry_point(tmp_path):
    runner = CliRunner()
    runner.invoke(main, [
        "init", "my-indicators",
        "--plugin", "indicator:my_rsi",
        "--output-dir", str(tmp_path),
    ])
    content = (tmp_path / "my-indicators" / "pyproject.toml").read_text()
    assert "fwbg.plugin_packages" in content
    assert "fwbg-sdk" in content


def test_add_plugin_to_existing(tmp_path):
    runner = CliRunner()
    # First create package
    runner.invoke(main, [
        "init", "my-indicators",
        "--plugin", "indicator:my_rsi",
        "--output-dir", str(tmp_path),
    ])
    # Then add another plugin
    result = runner.invoke(main, [
        "add", "indicator", "my_macd",
        "--package-dir", str(tmp_path / "my-indicators"),
    ])
    assert result.exit_code == 0, result.output

    plugin_dir = tmp_path / "my-indicators" / "src" / "my_indicators" / "plugins" / "my-indicators" / "indicators" / "my_macd"
    assert (plugin_dir / "__init__.py").exists()


def test_init_multiple_plugins(tmp_path):
    """Test creating a package with multiple plugins of different types."""
    runner = CliRunner()
    result = runner.invoke(main, [
        "init", "my-toolkit",
        "--plugin", "indicator:my_rsi",
        "--plugin", "exit_strategy:my_atr",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output

    pkg_dir = tmp_path / "my-toolkit"
    plugins_base = pkg_dir / "src" / "my_toolkit" / "plugins" / "my-toolkit"

    # Both plugin dirs exist
    assert (plugins_base / "indicators" / "my_rsi" / "__init__.py").exists()
    assert (plugins_base / "exit_strategies" / "my_atr" / "__init__.py").exists()

    # Package manifest lists both
    manifest = json.loads((plugins_base / "manifest.json").read_text())
    assert "my_rsi" in manifest["plugins"]["indicators"]
    assert "my_atr" in manifest["plugins"]["exit_strategies"]


def test_add_updates_manifest(tmp_path):
    """Test that adding a plugin updates the package manifest."""
    runner = CliRunner()
    runner.invoke(main, [
        "init", "my-pkg",
        "--plugin", "indicator:ind_a",
        "--output-dir", str(tmp_path),
    ])
    runner.invoke(main, [
        "add", "indicator", "ind_b",
        "--package-dir", str(tmp_path / "my-pkg"),
    ])
    manifest_path = tmp_path / "my-pkg" / "src" / "my_pkg" / "plugins" / "my-pkg" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "ind_a" in manifest["plugins"]["indicators"]
    assert "ind_b" in manifest["plugins"]["indicators"]


def test_init_generates_valid_indicator_code(tmp_path):
    """Test that generated indicator code is valid Python."""
    runner = CliRunner()
    runner.invoke(main, [
        "init", "test-pkg",
        "--plugin", "indicator:my_rsi",
        "--output-dir", str(tmp_path),
    ])
    plugin_init = tmp_path / "test-pkg" / "src" / "test_pkg" / "plugins" / "test-pkg" / "indicators" / "my_rsi" / "__init__.py"
    code = plugin_init.read_text()
    assert "BaseIndicator" in code
    assert "shift_features" in code
    assert "register_indicator" in code
    # Verify it's valid Python
    compile(code, str(plugin_init), "exec")


def test_init_generates_valid_exit_strategy_code(tmp_path):
    """Test that generated exit strategy code is valid Python."""
    runner = CliRunner()
    runner.invoke(main, [
        "init", "test-pkg",
        "--plugin", "exit_strategy:my_exit",
        "--output-dir", str(tmp_path),
    ])
    plugin_init = tmp_path / "test-pkg" / "src" / "test_pkg" / "plugins" / "test-pkg" / "exit_strategies" / "my_exit" / "__init__.py"
    code = plugin_init.read_text()
    assert "BaseExitStrategy" in code
    assert "AssetInfo" in code
    assert "register_exit_strategy" in code
    compile(code, str(plugin_init), "exec")


def test_plugin_manifest_has_correct_phase(tmp_path):
    """Test that individual plugin manifests have the right phase."""
    runner = CliRunner()
    runner.invoke(main, [
        "init", "test-pkg",
        "--plugin", "indicator:my_ind",
        "--plugin", "preprocessor:my_prep",
        "--output-dir", str(tmp_path),
    ])
    base = tmp_path / "test-pkg" / "src" / "test_pkg" / "plugins" / "test-pkg"

    ind_manifest = json.loads((base / "indicators" / "my_ind" / "manifest.json").read_text())
    assert ind_manifest["phase"] == "indicators"

    prep_manifest = json.loads((base / "preprocessors" / "my_prep" / "manifest.json").read_text())
    assert prep_manifest["phase"] == "preprocessors"
