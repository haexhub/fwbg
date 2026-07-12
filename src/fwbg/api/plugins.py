"""Plugin endpoints."""
import importlib.util
import inspect
import json
import mimetypes
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from fwbg.api.deps import get_plugin_registry
from fwbg_sdk import BasePlugin, PluginPhase
from fwbg_sdk.registry import ENTRY_MODIFIER_REGISTRY, EXIT_MODIFIER_REGISTRY

router = APIRouter(prefix="/plugins", tags=["plugins"])
exit_modifiers_router = APIRouter(prefix="/exit-modifiers", tags=["exit-modifiers"])
entry_modifiers_router = APIRouter(prefix="/entry-modifiers", tags=["entry-modifiers"])

_AGENT_AUTHORED_NAMESPACE = "agent-authored"

# Singular PluginKind (fwbg-agents) → plural category dir (fwbg registry)
_KIND_TO_CATEGORY: dict[str, str] = {
    "indicator": "indicators",
    "model": "models",
    "exit_strategy": "exit_strategies",
    "risk_management": "risk_management",
    "entry_modifier": "entry_modifiers",
    "preprocessing": "preprocessing",
    "feature_selection": "feature_selection",
    "data_loading": "data_loading",
}

_BUNDLE_MANIFEST = {
    "name": _AGENT_AUTHORED_NAMESPACE,
    "version": "1.0.0",
    "description": "Agent-authored plugins",
    "author": "fwbg-agents",
    "license": "MIT",
}


class RegisterPluginPayload(BaseModel):
    slug: str
    python_code: str
    kind: str
    description: str = ""
    spec_md: str = ""
    tests_code: str = ""
    version: str = "1.0.0"
    overwrite: bool = False


def _import_plugin_module(init_py: Path, slug: str) -> tuple[Optional[type], str]:
    """Load a plugin module and return (plugin_cls, error). error is '' on success."""
    module_name = f"_fwbg_register_{slug}"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, init_py)
    if spec is None or spec.loader is None:
        return None, "Could not create module spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        return None, f"Import error: {exc}"
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, BasePlugin)
            and attr is not BasePlugin
            and not inspect.isabstract(attr)
            and getattr(attr, "name", None) == slug
        ):
            return attr, ""
    sys.modules.pop(module_name, None)
    return None, f"No non-abstract BasePlugin subclass with name='{slug}' found"


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


@router.post("")
def register_plugin(payload: RegisterPluginPayload) -> dict:
    """Register an agent-authored plugin into the fwbg registry.

    Writes verified plugin code to ``~/.fwbg/plugins/agent-authored/<category>/<slug>/``,
    validates it (module import + tests), and refreshes the registry so the plugin
    appears immediately in ``GET /api/plugins``.
    Returns ``{"fqn": "agent-authored:<slug>", ...}`` or a 4xx error on failure.
    """
    from fwbg.pipeline.registry import get_user_plugins_dir, reset_registry

    category = _KIND_TO_CATEGORY.get(payload.kind)
    if category is None:
        raise HTTPException(
            422,
            f"Unknown plugin kind: {payload.kind!r}. Valid: {sorted(_KIND_TO_CATEGORY)}",
        )

    fqn = f"{_AGENT_AUTHORED_NAMESPACE}:{payload.slug}"
    target_dir = get_user_plugins_dir() / _AGENT_AUTHORED_NAMESPACE / category / payload.slug

    if target_dir.exists() and not payload.overwrite:
        raise HTTPException(409, f"Plugin '{fqn}' already registered. Set overwrite=true to replace.")

    # Validate in a temp directory so nothing is written on failure.
    with tempfile.TemporaryDirectory(prefix="fwbg_register_") as tmp:
        tmp_path = Path(tmp)
        init_py = tmp_path / "__init__.py"
        init_py.write_text(payload.python_code, encoding="utf-8")

        plugin_cls, err = _import_plugin_module(init_py, payload.slug)
        if err:
            raise HTTPException(422, f"Plugin validation failed: {err}")

        if payload.tests_code:
            tests_py = tmp_path / "tests.py"
            tests_py.write_text(payload.tests_code, encoding="utf-8")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(tests_py), "-v", "--tb=short", "--no-header", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(422, "Plugin tests timed out (120 s)")
            if result.returncode != 0:
                raise HTTPException(
                    422,
                    f"Plugin tests failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )

    # All checks passed — write to the persistent user-plugins directory.
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "__init__.py").write_text(payload.python_code, encoding="utf-8")
    (target_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": payload.slug,
                "version": payload.version,
                "description": payload.description,
                "phase": category,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if payload.spec_md:
        (target_dir / "spec.md").write_text(payload.spec_md, encoding="utf-8")
    if payload.tests_code:
        (target_dir / "tests.py").write_text(payload.tests_code, encoding="utf-8")

    # Ensure the bundle-level manifest.json (required by discover_package).
    bundle_manifest = target_dir.parent.parent / "manifest.json"
    if not bundle_manifest.exists():
        bundle_manifest.parent.mkdir(parents=True, exist_ok=True)
        bundle_manifest.write_text(json.dumps(_BUNDLE_MANIFEST, indent=2), encoding="utf-8")

    # Refresh the global registry so the new plugin is visible immediately.
    reset_registry()
    get_plugin_registry.cache_clear()

    return {"fqn": fqn, "slug": payload.slug, "category": category}


@router.get("")
def list_plugins(
    phase: Optional[str] = Query(None, description="Filter by phase"),
    namespace: Optional[str] = Query(
        None, description="Filter by namespace (e.g. agent-authored, fwbg-core)"
    ),
) -> list[dict]:
    """List all registered plugins with their schemas.

    An unknown ``namespace`` yields an empty list (namespaces are free-form,
    not an enum), so the caller can filter without pre-validating.
    """
    registry = get_plugin_registry()

    phase_filter = None
    if phase:
        try:
            phase_filter = PluginPhase(phase)
        except ValueError:
            raise HTTPException(400, f"Invalid phase: {phase}")

    fqns = registry.list_plugins(phase=phase_filter, namespace=namespace)
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


def _resolve_plugin_src_file(fqn: str) -> Path:
    """Resolve the source file path for a plugin, or raise 404."""
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
        raise HTTPException(404, f"No source location for plugin: {fqn}")

    return Path(src_file)


@router.get("/{fqn:path}/source")
def get_plugin_source(fqn: str) -> dict:
    """Return the plugin's Python source.

    The backend is the single source of truth for plugins, so fwbg-agents fetches
    example source over HTTP (for the PluginPlanner) rather than reading the repo
    off disk. Resolved via ``inspect`` so it works for core, premium and
    entry-point-installed plugins alike.
    """
    src_file = _resolve_plugin_src_file(fqn)
    source = src_file.read_text(encoding="utf-8")
    return {"fqn": fqn, "filename": src_file.name, "source": source}


@router.get("/{fqn:path}/spec")
def get_plugin_spec(fqn: str) -> dict:
    """Return the plugin's speckit spec.md (co-located with its source).

    The backend is the single source of truth for plugins, so fwbg-agents reads
    each plugin's structured spec over HTTP (for duplicate detection) rather
    than off disk. 404 when the plugin has no spec yet.
    """
    src_file = _resolve_plugin_src_file(fqn)
    spec_path = src_file.parent / "spec.md"
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

    # Fallback: locate tests.py relative to the plugin's source file (covers
    # user / agent-authored plugins not under the core or entry-point dirs).
    if not test_file.exists():
        try:
            src = _resolve_plugin_src_file(fqn)
            candidate = src.parent / "tests.py"
            if candidate.exists():
                test_file = candidate
        except HTTPException:
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
