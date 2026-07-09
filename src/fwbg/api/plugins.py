"""Plugin endpoints."""
import mimetypes
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from fwbg.api.deps import get_plugin_registry
from fwbg_sdk import PluginPhase
from fwbg_sdk.registry import ENTRY_MODIFIER_REGISTRY, EXIT_MODIFIER_REGISTRY

router = APIRouter(prefix="/plugins", tags=["plugins"])
exit_modifiers_router = APIRouter(prefix="/exit-modifiers", tags=["exit-modifiers"])
entry_modifiers_router = APIRouter(prefix="/entry-modifiers", tags=["entry-modifiers"])


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

    column_group_labels: dict[str, str] = {}
    if hasattr(plugin_cls, "get_column_group_labels"):
        try:
            column_group_labels = plugin_cls.get_column_group_labels()
        except TypeError:
            try:
                column_group_labels = plugin_cls().get_column_group_labels()
            except Exception:
                pass

    return {
        "fqn": fqn,
        "name": plugin_cls.name,
        "namespace": namespace,
        "phase": plugin_cls.phase.value if isinstance(plugin_cls.phase, PluginPhase) else str(plugin_cls.phase),
        "version": plugin_cls.version,
        "description": manifest.get("description", ""),
        "group": getattr(plugin_cls, "group", "custom"),
        "stateful": plugin_cls.stateful,
        "cacheable": plugin_cls.cacheable,
        "has_docs": plugin_cls.get_docs_dir() is not None,
        "param_schema": param_schema,
        "defaults": defaults,
        "feature_columns": feature_columns,
        "signal_columns": signal_columns,
        "plot_columns": plot_columns,
        "column_group_labels": column_group_labels,
    }


def _get_validated_docs_dir(fqn: str) -> Path:
    """Get a plugin's docs dir, raising 403 if validation fails."""
    registry = get_plugin_registry()
    try:
        plugin_cls = registry.get(fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {fqn}")

    docs_dir = plugin_cls.get_docs_dir()
    if docs_dir is None:
        raise HTTPException(404, f"No documentation for plugin: {fqn}")

    result = plugin_cls.validate_docs()
    if not result.valid:
        reasons = [f"{v.file}:{v.line} {v.reason}" for v in result.violations]
        raise HTTPException(
            403,
            f"Documentation blocked due to validation errors: {'; '.join(reasons)}",
        )
    return docs_dir


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


# --- Docs endpoints (must be before the catch-all GET /{fqn:path}) ---


@router.get("/{fqn:path}/docs")
def get_plugin_docs(fqn: str) -> dict:
    """Get plugin documentation overview."""
    registry = get_plugin_registry()
    try:
        plugin_cls = registry.get(fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {fqn}")

    docs_dir = plugin_cls.get_docs_dir()
    if docs_dir is None:
        return {"fqn": fqn, "has_docs": False, "readme": None, "files": [], "validation": {"valid": True, "violations": []}}

    result = plugin_cls.validate_docs()
    validation = {
        "valid": result.valid,
        "violations": [
            {"file": v.file, "line": v.line, "link": v.link, "reason": v.reason}
            for v in result.violations
        ],
    }

    # Block content if validation fails
    if not result.valid:
        return {"fqn": fqn, "has_docs": True, "readme": None, "files": [], "validation": validation}

    readme = plugin_cls.get_docs_readme()
    return {"fqn": fqn, "has_docs": True, "readme": readme, "files": result.files, "validation": validation}


@router.get("/{fqn:path}/docs/{doc_path:path}")
def get_plugin_doc_file(fqn: str, doc_path: str) -> Response:
    """Serve a specific documentation file with path validation."""
    docs_dir = _get_validated_docs_dir(fqn)

    # Validate the requested path
    if ".." in Path(doc_path).parts or doc_path.startswith("/"):
        raise HTTPException(403, "Path traversal not allowed")

    file_path = (docs_dir / doc_path).resolve()

    # Must be within docs_dir
    try:
        file_path.relative_to(docs_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Path traversal not allowed")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, f"File not found: {doc_path}")

    content = file_path.read_bytes()
    media_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    return Response(content=content, media_type=media_type)


@router.get("/{fqn:path}/source")
def get_plugin_source(fqn: str) -> dict:
    """Return the plugin's Python source.

    The backend is the single source of truth for plugins, so fwbg-agents fetches
    example source over HTTP (for the PluginPlanner) rather than reading the repo
    off disk. Resolved via ``inspect`` so it works for core, premium and
    entry-point-installed plugins alike.
    """
    import inspect

    registry = get_plugin_registry()
    try:
        plugin_cls = registry.get(fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {fqn}")

    try:
        src_file = inspect.getsourcefile(plugin_cls) or inspect.getfile(plugin_cls)
    except TypeError:
        src_file = None
    if not src_file or not Path(src_file).is_file():
        raise HTTPException(404, f"No source available for plugin: {fqn}")

    source = Path(src_file).read_text(encoding="utf-8")
    return {"fqn": fqn, "filename": Path(src_file).name, "source": source}


@router.get("/{fqn:path}/spec")
def get_plugin_spec(fqn: str) -> dict:
    """Return the plugin's speckit spec.md (co-located with its source).

    The backend is the single source of truth for plugins, so fwbg-agents reads
    each plugin's structured spec over HTTP (for duplicate detection) rather
    than off disk. 404 when the plugin has no spec yet.
    """
    import inspect

    registry = get_plugin_registry()
    try:
        plugin_cls = registry.get(fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {fqn}")

    try:
        src_file = inspect.getsourcefile(plugin_cls) or inspect.getfile(plugin_cls)
    except TypeError:
        src_file = None
    if not src_file:
        raise HTTPException(404, f"No source location for plugin: {fqn}")

    spec_path = Path(src_file).parent / "spec.md"
    if not spec_path.is_file():
        raise HTTPException(404, f"No spec available for plugin: {fqn}")

    return {"fqn": fqn, "spec": spec_path.read_text(encoding="utf-8")}


# --- Tests endpoint ---


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
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "fqn": fqn,
            "has_tests": True,
            "status": "timeout",
            "returncode": -1,
            "stdout": (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": f"Test run exceeded {exc.timeout:.0f}s timeout and was killed.",
        }

    return {
        "fqn": fqn,
        "has_tests": True,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# --- Catch-all plugin detail endpoint (must be LAST) ---


@router.get("/{fqn:path}")
def get_plugin(fqn: str) -> dict:
    """Get plugin details by fully qualified name."""
    _registry = get_plugin_registry()
    try:
        return _plugin_to_dict(fqn)
    except Exception:
        raise HTTPException(404, f"Plugin not found: {fqn}")


# --- Exit Modifiers ---


@exit_modifiers_router.get("")
def list_exit_modifiers_endpoint() -> list[dict]:
    """List all registered exit modifiers with their param schemas."""
    registry = get_plugin_registry()

    result = []
    for name, cls in EXIT_MODIFIER_REGISTRY.items():
        # Find manifest from plugin registry (stored during package discovery)
        manifest = {}
        for fqn, m in registry._plugin_manifests.items():
            if m.get("name") == name and m.get("phase") == "exit_modifiers":
                manifest = m
                break

        try:
            defaults = cls.get_default_params()
        except TypeError:
            defaults = cls().get_default_params()

        try:
            param_schema = cls.get_param_schema()
        except TypeError:
            param_schema = cls().get_param_schema()

        result.append({
            "name": name,
            "description": manifest.get("description", ""),
            "version": manifest.get("version", "0.1.0"),
            "param_schema": param_schema,
            "defaults": defaults,
        })

    return result


# --- Entry Modifiers ---


@entry_modifiers_router.get("")
def list_entry_modifiers_endpoint() -> list[dict]:
    """List all registered entry modifiers with their param schemas."""
    registry = get_plugin_registry()

    result = []
    for name, cls in ENTRY_MODIFIER_REGISTRY.items():
        # Find manifest from plugin registry (stored during package discovery)
        manifest = {}
        for fqn, m in registry._plugin_manifests.items():
            if m.get("name") == name and m.get("phase") == "entry_modifiers":
                manifest = m
                break

        try:
            defaults = cls.get_default_params()
        except TypeError:
            defaults = cls().get_default_params()

        try:
            param_schema = cls.get_param_schema()
        except TypeError:
            param_schema = cls().get_param_schema()

        result.append({
            "name": name,
            "description": manifest.get("description", ""),
            "version": manifest.get("version", "0.1.0"),
            "param_schema": param_schema,
            "defaults": defaults,
        })

    return result
