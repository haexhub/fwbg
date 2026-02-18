"""Plugin endpoints."""
import subprocess
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from fwbg.api.deps import get_plugin_registry
from fwbg_sdk import PluginPhase

router = APIRouter(prefix="/plugins", tags=["plugins"])


def _plugin_to_dict(fqn: str) -> dict:
    """Convert a plugin to its API representation."""
    registry = get_plugin_registry()
    plugin_cls = registry.get(fqn)
    namespace, plugin_name = fqn.split(":", 1)
    manifest = registry.get_plugin_manifest(fqn)

    # Some plugins override get_default_params/get_param_schema as instance methods
    try:
        defaults = plugin_cls.get_default_params()
    except TypeError:
        defaults = plugin_cls().get_default_params()

    try:
        param_schema = plugin_cls.get_param_schema()
    except TypeError:
        param_schema = plugin_cls().get_param_schema()

    # Get feature columns for indicator plugins
    feature_columns: list[str] = []
    signal_columns: list[str] = []
    plot_columns: list[str] = []
    if hasattr(plugin_cls, "get_feature_columns"):
        try:
            feature_columns = plugin_cls.get_feature_columns()
        except TypeError:
            try:
                feature_columns = plugin_cls().get_feature_columns()
            except Exception:
                pass
    if hasattr(plugin_cls, "get_signal_columns"):
        try:
            signal_columns = plugin_cls.get_signal_columns()
        except TypeError:
            try:
                signal_columns = plugin_cls().get_signal_columns()
            except Exception:
                pass
    if hasattr(plugin_cls, "get_plot_columns"):
        try:
            plot_columns = plugin_cls.get_plot_columns()
        except TypeError:
            try:
                plot_columns = plugin_cls().get_plot_columns()
            except Exception:
                plot_columns = [c for c in feature_columns if c not in signal_columns]

    return {
        "fqn": fqn,
        "name": plugin_cls.name,
        "namespace": namespace,
        "phase": plugin_cls.phase.value if isinstance(plugin_cls.phase, PluginPhase) else str(plugin_cls.phase),
        "version": plugin_cls.version,
        "description": manifest.get("description", ""),
        "stateful": plugin_cls.stateful,
        "cacheable": plugin_cls.cacheable,
        "param_schema": param_schema,
        "defaults": defaults,
        "feature_columns": feature_columns,
        "signal_columns": signal_columns,
        "plot_columns": plot_columns,
    }


@router.get("")
def list_plugins(phase: Optional[str] = Query(None, description="Filter by phase")) -> list[dict]:
    """List all registered plugins with their schemas."""
    registry = get_plugin_registry()

    phase_filter = None
    if phase:
        try:
            phase_filter = PluginPhase(phase)
        except ValueError:
            raise HTTPException(400, f"Invalid phase: {phase}")

    fqns = registry.list_plugins(phase=phase_filter)
    return [_plugin_to_dict(fqn) for fqn in sorted(fqns)]


@router.get("/{fqn:path}")
def get_plugin(fqn: str) -> dict:
    """Get plugin details by fully qualified name."""
    registry = get_plugin_registry()
    try:
        return _plugin_to_dict(fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {fqn}")


@router.post("/{fqn:path}/tests/run")
def run_plugin_tests(fqn: str) -> dict:
    """Run tests for a specific plugin."""
    registry = get_plugin_registry()

    try:
        plugin_cls = registry.get(fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {fqn}")

    namespace, plugin_name = fqn.split(":", 1)
    phase = plugin_cls.phase

    # Map phase to directory name
    phase_dir_map = {
        PluginPhase.INDICATORS: "indicators",
        PluginPhase.PREPROCESSING: "preprocessing",
        PluginPhase.FEATURE_SELECTION: "feature_selection",
        PluginPhase.EXIT_STRATEGIES: "exit_strategies",
        PluginPhase.RISK_MANAGEMENT: "risk_management",
        PluginPhase.DATA_LOADING: "data_loading",
    }

    phase_dir = phase_dir_map.get(phase, str(phase))

    # Look for test files in the plugin's directory
    from fwbg.pipeline.registry import get_core_plugins_dir

    plugin_test_dir = get_core_plugins_dir() / namespace / phase_dir / plugin_name
    test_file = plugin_test_dir / "tests.py"

    # Also check installed packages
    if not test_file.exists():
        # Try entry-point discovered packages
        from importlib.metadata import entry_points
        try:
            eps = entry_points(group="fwbg.plugin_packages")
            for ep in eps:
                get_dir = ep.load()
                pkg_dir = get_dir()
                candidate = pkg_dir / namespace / phase_dir / plugin_name / "tests.py"
                if candidate.exists():
                    test_file = candidate
                    break
        except Exception:
            pass

    if not test_file.exists():
        return {
            "fqn": fqn,
            "has_tests": False,
            "status": "no_tests",
            "message": f"No test file found for {fqn}",
        }

    # Run pytest on the test file
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    return {
        "fqn": fqn,
        "has_tests": True,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
