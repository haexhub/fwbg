# Plugin Pipeline System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a modular plugin system with fixed pipeline phases (DataLoading, Preprocessing, Indicators, FeatureSelection, Labeling, Model, Validation) where each phase supports configurable plugins via strategy JSON.

**Architecture:** Replace the current ad-hoc pipeline in `process.py` with a formal `PipelineRunner` that executes phases in order. Each phase has a base class; plugins register themselves via decorators. Plugins are discovered from core (`src/fwbg/plugins/`) and user (`~/.fwbg/plugins/`) directories. Strategy config uses new format: `[{"name": "...", "params": {...}}]`. Plugin packages include `manifest.json` for dependencies.

**Tech Stack:** Python 3.10+, pandas, ABC for interfaces, importlib for plugin discovery, JSON Schema for validation

---

## Phase 1: Core Plugin Infrastructure

### Task 1: Create PipelineContext dataclass

**Files:**
- Create: `src/fwbg/pipeline/context.py`
- Test: `tests/pipeline/test_context.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_context.py
import pytest
import pandas as pd
from fwbg.pipeline.context import PipelineContext


def test_pipeline_context_creation():
    """PipelineContext should store DataFrame and metadata."""
    df = pd.DataFrame({"O": [1, 2], "H": [2, 3], "L": [0.5, 1.5], "C": [1.5, 2.5]})
    ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="FOREX")

    assert ctx.df is not None
    assert len(ctx.df) == 2
    assert ctx.symbol == "EURUSD"
    assert ctx.asset_class == "FOREX"
    assert ctx.metadata == {}


def test_pipeline_context_metadata():
    """PipelineContext should allow storing arbitrary metadata."""
    df = pd.DataFrame({"C": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="BTCUSD", asset_class="CRYPTO")

    ctx.metadata["fitted_d"] = 0.4
    ctx.metadata["selected_features"] = ["rsi_14", "ema_20"]

    assert ctx.metadata["fitted_d"] == 0.4
    assert len(ctx.metadata["selected_features"]) == 2


def test_pipeline_context_immutable_df_reference():
    """Updating df should create new reference, not mutate."""
    df1 = pd.DataFrame({"C": [1, 2]})
    ctx = PipelineContext(df=df1, symbol="TEST", asset_class="FOREX")

    df2 = pd.DataFrame({"C": [1, 2, 3, 4]})
    ctx.df = df2

    assert len(ctx.df) == 4
    assert len(df1) == 2  # Original unchanged
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_context.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'fwbg.pipeline'"

**Step 3: Create directory structure and minimal implementation**

```bash
mkdir -p src/fwbg/pipeline
touch src/fwbg/pipeline/__init__.py
```

```python
# src/fwbg/pipeline/context.py
"""Pipeline context for passing data between phases."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import pandas as pd


@dataclass
class PipelineContext:
    """
    Context object passed through all pipeline phases.

    Carries the DataFrame and metadata that plugins can read/write.
    Each plugin receives this context, processes it, and returns
    an updated context (or the same one with modified df/metadata).

    Attributes:
        df: The main DataFrame being processed
        symbol: Asset symbol (e.g., "EURUSD")
        asset_class: Asset class (e.g., "FOREX", "CRYPTO")
        metadata: Arbitrary key-value store for inter-plugin communication
        fold_info: Optional fold information for walk-forward validation
    """
    df: pd.DataFrame
    symbol: str
    asset_class: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    fold_info: Optional[Dict[str, Any]] = None

    def clone(self) -> "PipelineContext":
        """Create a shallow copy with a new DataFrame copy."""
        return PipelineContext(
            df=self.df.copy(),
            symbol=self.symbol,
            asset_class=self.asset_class,
            metadata=self.metadata.copy(),
            fold_info=self.fold_info.copy() if self.fold_info else None,
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_context.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/fwbg/pipeline/ tests/pipeline/
git commit -m "feat(pipeline): add PipelineContext dataclass"
```

---

### Task 2: Create BasePlugin abstract class

**Files:**
- Create: `src/fwbg/pipeline/base.py`
- Test: `tests/pipeline/test_base.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_base.py
import pytest
import pandas as pd
from abc import ABC
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext


def test_plugin_phase_enum():
    """PluginPhase should define all pipeline phases."""
    assert PluginPhase.DATA_LOADING.value == "data_loading"
    assert PluginPhase.PREPROCESSING.value == "preprocessing"
    assert PluginPhase.INDICATORS.value == "indicators"
    assert PluginPhase.FEATURE_SELECTION.value == "feature_selection"
    assert PluginPhase.LABELING.value == "labeling"
    assert PluginPhase.MODEL.value == "model"
    assert PluginPhase.VALIDATION.value == "validation"


def test_base_plugin_is_abstract():
    """BasePlugin should be abstract and not instantiable."""
    with pytest.raises(TypeError):
        BasePlugin()


def test_base_plugin_required_attributes():
    """Concrete plugin must define required class attributes."""
    class IncompletePlugin(BasePlugin):
        pass

    with pytest.raises(TypeError):
        IncompletePlugin()


def test_concrete_plugin_implementation():
    """Concrete plugin should be instantiable with all required attributes."""
    class TestPlugin(BasePlugin):
        name = "test_plugin"
        version = "1.0.0"
        phase = PluginPhase.PREPROCESSING

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            ctx.df["test_col"] = 1
            return ctx

        def validate(self) -> bool:
            return True

    plugin = TestPlugin()
    assert plugin.name == "test_plugin"
    assert plugin.version == "1.0.0"
    assert plugin.phase == PluginPhase.PREPROCESSING
    assert plugin.stateful == False  # Default
    assert plugin.cacheable == True  # Default


def test_plugin_execute():
    """Plugin execute should process context and return updated context."""
    class AddColumnPlugin(BasePlugin):
        name = "add_column"
        version = "1.0.0"
        phase = PluginPhase.INDICATORS

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            col_name = params.get("column_name", "new_col")
            ctx.df[col_name] = params.get("value", 0)
            return ctx

        def validate(self) -> bool:
            return True

    df = pd.DataFrame({"C": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="TEST", asset_class="FOREX")

    plugin = AddColumnPlugin()
    result = plugin.execute(ctx, column_name="my_col", value=42)

    assert "my_col" in result.df.columns
    assert result.df["my_col"].iloc[0] == 42


def test_plugin_stateful_flag():
    """Stateful plugins should declare stateful=True."""
    class StatefulPlugin(BasePlugin):
        name = "stateful_test"
        version = "1.0.0"
        phase = PluginPhase.PREPROCESSING
        stateful = True

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            return ctx

        def fit(self, ctx: PipelineContext, **params) -> None:
            self.fitted_value_ = params.get("value", 1.0)

        def validate(self) -> bool:
            return True

    plugin = StatefulPlugin()
    assert plugin.stateful == True

    df = pd.DataFrame({"C": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="TEST", asset_class="FOREX")
    plugin.fit(ctx, value=0.5)
    assert plugin.fitted_value_ == 0.5


def test_plugin_default_params():
    """Plugin should provide default parameters."""
    class ParamPlugin(BasePlugin):
        name = "param_test"
        version = "1.0.0"
        phase = PluginPhase.INDICATORS

        @classmethod
        def get_default_params(cls) -> dict:
            return {"window": 14, "method": "sma"}

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            return ctx

        def validate(self) -> bool:
            return True

    defaults = ParamPlugin.get_default_params()
    assert defaults["window"] == 14
    assert defaults["method"] == "sma"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_base.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'fwbg.pipeline.base'"

**Step 3: Write minimal implementation**

```python
# src/fwbg/pipeline/base.py
"""Base plugin class and phase definitions."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fwbg.pipeline.context import PipelineContext


class PluginPhase(Enum):
    """Pipeline phases in execution order."""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    INDICATORS = "indicators"
    FEATURE_SELECTION = "feature_selection"
    LABELING = "labeling"
    MODEL = "model"
    VALIDATION = "validation"


class BasePlugin(ABC):
    """
    Abstract base class for all pipeline plugins.

    Every plugin must:
    1. Define name, version, and phase class attributes
    2. Implement execute() method
    3. Implement validate() method for pre-run checks

    Optional:
    - Set stateful=True if plugin needs fit/transform pattern
    - Set cacheable=False if results shouldn't be cached
    - Override get_default_params() to provide defaults
    - Override get_feature_columns() for indicator plugins
    """

    # Required class attributes (must be overridden)
    name: str
    version: str
    phase: PluginPhase

    # Optional class attributes with defaults
    stateful: bool = False
    cacheable: bool = True

    # Instance state for stateful plugins
    _fitted: bool = False

    def __init_subclass__(cls, **kwargs):
        """Validate that required class attributes are defined."""
        super().__init_subclass__(**kwargs)
        # Only check concrete classes (not intermediate ABCs)
        if not getattr(cls, '__abstractmethods__', None):
            for attr in ('name', 'version', 'phase'):
                if not hasattr(cls, attr) or getattr(cls, attr, None) is None:
                    raise TypeError(
                        f"Plugin class {cls.__name__} must define '{attr}' class attribute"
                    )

    @abstractmethod
    def execute(self, ctx: "PipelineContext", **params) -> "PipelineContext":
        """
        Execute the plugin on the given context.

        Args:
            ctx: Pipeline context with DataFrame and metadata
            **params: Plugin-specific parameters

        Returns:
            Updated PipelineContext (may be same object or new one)
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate that plugin is correctly configured and ready to run.

        Called before pipeline execution starts. Should check:
        - Required dependencies are available
        - Configuration is valid
        - Any external resources are accessible

        Returns:
            True if plugin is ready, raises exception otherwise
        """
        pass

    def fit(self, ctx: "PipelineContext", **params) -> None:
        """
        Fit plugin on training data (for stateful plugins).

        Override this for plugins that learn from training data.
        After fit(), execute() should use the learned parameters.

        Args:
            ctx: Pipeline context with training DataFrame
            **params: Plugin-specific parameters
        """
        self._fitted = True

    def reset(self) -> None:
        """Reset plugin state (for stateful plugins)."""
        self._fitted = False

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Return default parameters for this plugin.

        Override to provide sensible defaults that users can override
        in their strategy configuration.

        Returns:
            Dictionary of parameter names to default values
        """
        return {}

    def get_feature_columns(self) -> List[str]:
        """
        Return list of feature column names created by this plugin.

        Override for indicator plugins that add columns to the DataFrame.
        Used for feature tracking and selection.

        Returns:
            List of column names this plugin creates
        """
        return []

    def report_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        callback: Optional[callable] = None
    ) -> None:
        """
        Report progress during long-running operations.

        Args:
            current: Current step number
            total: Total number of steps
            message: Optional status message
            callback: Optional callback function(current, total, message)
        """
        if callback:
            callback(current, total, message)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_base.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/fwbg/pipeline/base.py tests/pipeline/test_base.py
git commit -m "feat(pipeline): add BasePlugin abstract class with phase enum"
```

---

### Task 3: Create PluginRegistry with directory-based discovery

**Files:**
- Create: `src/fwbg/pipeline/registry.py`
- Test: `tests/pipeline/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_registry.py
import pytest
import tempfile
import os
import json
from pathlib import Path
from fwbg.pipeline.registry import PluginRegistry, PluginNotFoundError, PluginValidationError
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext


class MockPlugin(BasePlugin):
    """Test plugin for registry tests."""
    name = "mock_plugin"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        return ctx

    def validate(self) -> bool:
        return True


def test_registry_register_plugin():
    """Registry should allow manual plugin registration."""
    registry = PluginRegistry()
    registry.register(MockPlugin)

    assert "mock_plugin" in registry.list_plugins(PluginPhase.PREPROCESSING)


def test_registry_get_plugin():
    """Registry should return plugin class by name."""
    registry = PluginRegistry()
    registry.register(MockPlugin)

    plugin_cls = registry.get("mock_plugin")
    assert plugin_cls == MockPlugin


def test_registry_get_unknown_plugin():
    """Registry should raise PluginNotFoundError for unknown plugins."""
    registry = PluginRegistry()

    with pytest.raises(PluginNotFoundError) as exc:
        registry.get("unknown_plugin")

    assert "unknown_plugin" in str(exc.value)


def test_registry_list_by_phase():
    """Registry should list plugins by phase."""
    registry = PluginRegistry()

    class PreprocessPlugin(BasePlugin):
        name = "preprocess1"
        version = "1.0.0"
        phase = PluginPhase.PREPROCESSING
        def execute(self, ctx, **p): return ctx
        def validate(self): return True

    class IndicatorPlugin(BasePlugin):
        name = "indicator1"
        version = "1.0.0"
        phase = PluginPhase.INDICATORS
        def execute(self, ctx, **p): return ctx
        def validate(self): return True

    registry.register(PreprocessPlugin)
    registry.register(IndicatorPlugin)

    preprocess_plugins = registry.list_plugins(PluginPhase.PREPROCESSING)
    indicator_plugins = registry.list_plugins(PluginPhase.INDICATORS)

    assert "preprocess1" in preprocess_plugins
    assert "indicator1" not in preprocess_plugins
    assert "indicator1" in indicator_plugins


def test_registry_discover_from_directory(tmp_path):
    """Registry should discover plugins from directory with manifest."""
    # Create plugin package structure
    pkg_dir = tmp_path / "test_plugins"
    pkg_dir.mkdir()

    # Create manifest.json
    manifest = {
        "package": "test_plugins",
        "version": "1.0.0",
        "author": "test",
        "dependencies": {},
        "plugin_dependencies": {}
    }
    (pkg_dir / "manifest.json").write_text(json.dumps(manifest))

    # Create __init__.py with plugin class
    init_content = '''
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext

class DiscoveredPlugin(BasePlugin):
    name = "discovered_plugin"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        ctx.df["discovered"] = 1
        return ctx

    def validate(self) -> bool:
        return True
'''
    (pkg_dir / "__init__.py").write_text(init_content)

    # Discover from directory
    registry = PluginRegistry()
    registry.discover_from_directory(tmp_path)

    assert "discovered_plugin" in registry.list_plugins(PluginPhase.INDICATORS)


def test_registry_validate_all():
    """Registry should validate all registered plugins."""
    registry = PluginRegistry()

    class ValidPlugin(BasePlugin):
        name = "valid"
        version = "1.0.0"
        phase = PluginPhase.PREPROCESSING
        def execute(self, ctx, **p): return ctx
        def validate(self): return True

    class InvalidPlugin(BasePlugin):
        name = "invalid"
        version = "1.0.0"
        phase = PluginPhase.PREPROCESSING
        def execute(self, ctx, **p): return ctx
        def validate(self): raise RuntimeError("Plugin broken!")

    registry.register(ValidPlugin)
    registry.register(InvalidPlugin)

    results = registry.validate_all()

    assert results["valid"]["valid"] == True
    assert results["invalid"]["valid"] == False
    assert "broken" in results["invalid"]["error"].lower()


def test_registry_get_plugin_info():
    """Registry should return plugin metadata."""
    registry = PluginRegistry()
    registry.register(MockPlugin)

    info = registry.get_info("mock_plugin")

    assert info["name"] == "mock_plugin"
    assert info["version"] == "1.0.0"
    assert info["phase"] == "preprocessing"
    assert info["stateful"] == False
    assert info["cacheable"] == True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_registry.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'fwbg.pipeline.registry'"

**Step 3: Write minimal implementation**

```python
# src/fwbg/pipeline/registry.py
"""Plugin registry with directory-based discovery."""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type

from fwbg.pipeline.base import BasePlugin, PluginPhase


class PluginNotFoundError(Exception):
    """Raised when a requested plugin is not found in registry."""
    pass


class PluginValidationError(Exception):
    """Raised when plugin validation fails."""
    pass


class PluginRegistry:
    """
    Central registry for all pipeline plugins.

    Supports:
    - Manual registration via register()
    - Directory-based discovery via discover_from_directory()
    - Plugin lookup by name
    - Listing plugins by phase
    - Bulk validation of all plugins
    """

    def __init__(self):
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._manifests: Dict[str, dict] = {}  # package_name -> manifest

    def register(self, plugin_cls: Type[BasePlugin]) -> None:
        """
        Register a plugin class.

        Args:
            plugin_cls: Plugin class (subclass of BasePlugin)

        Raises:
            TypeError: If plugin_cls is not a valid plugin
        """
        if not isinstance(plugin_cls, type) or not issubclass(plugin_cls, BasePlugin):
            raise TypeError(f"{plugin_cls} is not a BasePlugin subclass")

        if plugin_cls is BasePlugin:
            raise TypeError("Cannot register abstract BasePlugin")

        self._plugins[plugin_cls.name] = plugin_cls

    def get(self, name: str) -> Type[BasePlugin]:
        """
        Get plugin class by name.

        Args:
            name: Plugin name

        Returns:
            Plugin class

        Raises:
            PluginNotFoundError: If plugin not found
        """
        if name not in self._plugins:
            available = list(self._plugins.keys())
            raise PluginNotFoundError(
                f"Plugin '{name}' not found. Available: {available}"
            )
        return self._plugins[name]

    def list_plugins(self, phase: Optional[PluginPhase] = None) -> List[str]:
        """
        List registered plugin names.

        Args:
            phase: Optional filter by phase

        Returns:
            List of plugin names
        """
        if phase is None:
            return list(self._plugins.keys())

        return [
            name for name, cls in self._plugins.items()
            if cls.phase == phase
        ]

    def get_info(self, name: str) -> dict:
        """
        Get plugin metadata.

        Args:
            name: Plugin name

        Returns:
            Dict with plugin info
        """
        cls = self.get(name)
        return {
            "name": cls.name,
            "version": cls.version,
            "phase": cls.phase.value,
            "stateful": cls.stateful,
            "cacheable": cls.cacheable,
            "default_params": cls.get_default_params(),
        }

    def validate_all(self) -> Dict[str, dict]:
        """
        Validate all registered plugins.

        Returns:
            Dict mapping plugin name to validation result
        """
        results = {}
        for name, cls in self._plugins.items():
            try:
                instance = cls()
                instance.validate()
                results[name] = {"valid": True, "error": None}
            except Exception as e:
                results[name] = {"valid": False, "error": str(e)}
        return results

    def discover_from_directory(self, directory: Path) -> List[str]:
        """
        Discover plugins from a directory.

        Looks for subdirectories with manifest.json files.
        Each subdirectory is treated as a plugin package.

        Args:
            directory: Root directory to scan

        Returns:
            List of discovered plugin names
        """
        discovered = []
        directory = Path(directory)

        if not directory.exists():
            return discovered

        for pkg_dir in directory.iterdir():
            if not pkg_dir.is_dir():
                continue

            manifest_path = pkg_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            # Load manifest
            try:
                manifest = json.loads(manifest_path.read_text())
                self._manifests[manifest.get("package", pkg_dir.name)] = manifest
            except json.JSONDecodeError:
                continue

            # Load plugin module
            init_path = pkg_dir / "__init__.py"
            if not init_path.exists():
                continue

            try:
                # Create module spec and load
                module_name = f"fwbg_plugins_{pkg_dir.name}"
                spec = importlib.util.spec_from_file_location(module_name, init_path)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Find and register all BasePlugin subclasses
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BasePlugin)
                        and attr is not BasePlugin
                        and hasattr(attr, 'name')
                    ):
                        self.register(attr)
                        discovered.append(attr.name)

            except Exception as e:
                # Log but continue with other packages
                print(f"Warning: Failed to load plugin package {pkg_dir.name}: {e}")
                continue

        return discovered

    def get_manifest(self, package_name: str) -> Optional[dict]:
        """Get manifest for a plugin package."""
        return self._manifests.get(package_name)


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry, creating if needed."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_registry.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/fwbg/pipeline/registry.py tests/pipeline/test_registry.py
git commit -m "feat(pipeline): add PluginRegistry with directory discovery"
```

---

### Task 4: Create PluginConfig parser for new strategy format

**Files:**
- Create: `src/fwbg/pipeline/config.py`
- Test: `tests/pipeline/test_config.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_config.py
import pytest
from fwbg.pipeline.config import PluginConfig, PipelineConfig, parse_pipeline_config


def test_plugin_config_from_dict():
    """PluginConfig should parse from dict format."""
    data = {"name": "fractional_diff", "params": {"d": 0.4}}
    config = PluginConfig.from_dict(data)

    assert config.name == "fractional_diff"
    assert config.params["d"] == 0.4
    assert config.stateful is None  # Not overridden
    assert config.cacheable is None


def test_plugin_config_with_overrides():
    """PluginConfig should support stateful/cacheable overrides."""
    data = {
        "name": "my_plugin",
        "params": {"window": 20},
        "stateful": True,
        "cacheable": False
    }
    config = PluginConfig.from_dict(data)

    assert config.name == "my_plugin"
    assert config.stateful == True
    assert config.cacheable == False


def test_plugin_config_empty_params():
    """PluginConfig should handle missing params."""
    data = {"name": "simple_plugin"}
    config = PluginConfig.from_dict(data)

    assert config.name == "simple_plugin"
    assert config.params == {}


def test_pipeline_config_parse():
    """PipelineConfig should parse full pipeline section."""
    data = {
        "pipeline": {
            "data_loading": [
                {"name": "csv_loader", "params": {"aligned": True}}
            ],
            "preprocessing": [
                {"name": "fractional_diff", "params": {"d": 0.4}},
                {"name": "robust_scaler", "params": {}}
            ],
            "indicators": [
                {"name": "trend", "params": {"ema_periods": [8, 21]}},
                {"name": "momentum", "params": {}}
            ],
            "feature_selection": [
                {"name": "boruta", "params": {"max_features": 20}}
            ],
            "labeling": [
                {"name": "fixed_tp_sl", "params": {}}
            ],
            "model": [
                {"name": "xgboost", "params": {"n_estimators": 200}}
            ],
            "validation": [
                {"name": "walk_forward", "params": {"folds": 8}}
            ]
        }
    }

    config = parse_pipeline_config(data)

    assert len(config.data_loading) == 1
    assert config.data_loading[0].name == "csv_loader"

    assert len(config.preprocessing) == 2
    assert config.preprocessing[0].name == "fractional_diff"
    assert config.preprocessing[1].name == "robust_scaler"

    assert len(config.indicators) == 2
    assert len(config.feature_selection) == 1
    assert len(config.labeling) == 1
    assert len(config.model) == 1
    assert len(config.validation) == 1


def test_pipeline_config_empty_phases():
    """PipelineConfig should handle missing phases."""
    data = {
        "pipeline": {
            "indicators": [
                {"name": "trend", "params": {}}
            ]
        }
    }

    config = parse_pipeline_config(data)

    assert len(config.data_loading) == 0
    assert len(config.preprocessing) == 0
    assert len(config.indicators) == 1
    assert len(config.feature_selection) == 0


def test_pipeline_config_validation_error():
    """PipelineConfig should reject invalid plugin entries."""
    data = {
        "pipeline": {
            "indicators": [
                {"invalid": "no_name_field"}  # Missing 'name'
            ]
        }
    }

    with pytest.raises(ValueError) as exc:
        parse_pipeline_config(data)

    assert "name" in str(exc.value).lower()
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'fwbg.pipeline.config'"

**Step 3: Write minimal implementation**

```python
# src/fwbg/pipeline/config.py
"""Pipeline configuration parsing."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PluginConfig:
    """
    Configuration for a single plugin instance.

    Attributes:
        name: Plugin name (must match registered plugin)
        params: Plugin-specific parameters
        stateful: Override plugin's default stateful setting
        cacheable: Override plugin's default cacheable setting
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    stateful: Optional[bool] = None
    cacheable: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PluginConfig":
        """
        Parse PluginConfig from dictionary.

        Args:
            data: Dict with 'name' and optional 'params', 'stateful', 'cacheable'

        Returns:
            PluginConfig instance

        Raises:
            ValueError: If 'name' is missing
        """
        if "name" not in data:
            raise ValueError(f"Plugin config must have 'name' field: {data}")

        return cls(
            name=data["name"],
            params=data.get("params", {}),
            stateful=data.get("stateful"),
            cacheable=data.get("cacheable"),
        )


@dataclass
class PipelineConfig:
    """
    Configuration for the complete pipeline.

    Each phase contains a list of PluginConfig in execution order.
    """
    data_loading: List[PluginConfig] = field(default_factory=list)
    preprocessing: List[PluginConfig] = field(default_factory=list)
    indicators: List[PluginConfig] = field(default_factory=list)
    feature_selection: List[PluginConfig] = field(default_factory=list)
    labeling: List[PluginConfig] = field(default_factory=list)
    model: List[PluginConfig] = field(default_factory=list)
    validation: List[PluginConfig] = field(default_factory=list)

    def get_phase(self, phase_name: str) -> List[PluginConfig]:
        """Get plugin configs for a specific phase."""
        return getattr(self, phase_name, [])

    def all_plugins(self) -> List[PluginConfig]:
        """Get all plugin configs across all phases."""
        return (
            self.data_loading +
            self.preprocessing +
            self.indicators +
            self.feature_selection +
            self.labeling +
            self.model +
            self.validation
        )


def parse_pipeline_config(data: dict) -> PipelineConfig:
    """
    Parse pipeline configuration from strategy dict.

    Args:
        data: Strategy dict with 'pipeline' section

    Returns:
        PipelineConfig instance

    Raises:
        ValueError: If configuration is invalid
    """
    pipeline_data = data.get("pipeline", {})

    def parse_phase(phase_name: str) -> List[PluginConfig]:
        phase_data = pipeline_data.get(phase_name, [])
        configs = []
        for i, item in enumerate(phase_data):
            try:
                configs.append(PluginConfig.from_dict(item))
            except ValueError as e:
                raise ValueError(f"Invalid plugin config in {phase_name}[{i}]: {e}")
        return configs

    return PipelineConfig(
        data_loading=parse_phase("data_loading"),
        preprocessing=parse_phase("preprocessing"),
        indicators=parse_phase("indicators"),
        feature_selection=parse_phase("feature_selection"),
        labeling=parse_phase("labeling"),
        model=parse_phase("model"),
        validation=parse_phase("validation"),
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_config.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/fwbg/pipeline/config.py tests/pipeline/test_config.py
git commit -m "feat(pipeline): add PluginConfig and PipelineConfig parsers"
```

---

### Task 5: Create PipelineRunner

**Files:**
- Create: `src/fwbg/pipeline/runner.py`
- Test: `tests/pipeline/test_runner.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_runner.py
import pytest
import pandas as pd
from fwbg.pipeline.runner import PipelineRunner
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.config import PipelineConfig, PluginConfig
from fwbg.pipeline.registry import PluginRegistry


class AddColumnPlugin(BasePlugin):
    name = "add_column"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        col_name = params.get("column_name", "added")
        ctx.df[col_name] = params.get("value", 1)
        return ctx

    def validate(self) -> bool:
        return True


class MultiplyPlugin(BasePlugin):
    name = "multiply"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        factor = params.get("factor", 2)
        ctx.df["C"] = ctx.df["C"] * factor
        return ctx

    def validate(self) -> bool:
        return True


class StatefulPlugin(BasePlugin):
    name = "stateful_scaler"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING
    stateful = True

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        if hasattr(self, "mean_"):
            ctx.df["C"] = ctx.df["C"] - self.mean_
        return ctx

    def fit(self, ctx: PipelineContext, **params) -> None:
        self.mean_ = ctx.df["C"].mean()
        self._fitted = True

    def validate(self) -> bool:
        return True


@pytest.fixture
def registry():
    reg = PluginRegistry()
    reg.register(AddColumnPlugin)
    reg.register(MultiplyPlugin)
    reg.register(StatefulPlugin)
    return reg


def test_runner_execute_single_plugin(registry):
    """Runner should execute a single plugin."""
    config = PipelineConfig(
        indicators=[PluginConfig(name="add_column", params={"column_name": "test", "value": 42})]
    )
    runner = PipelineRunner(registry, config)

    df = pd.DataFrame({"C": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="TEST", asset_class="FOREX")

    result = runner.run(ctx)

    assert "test" in result.df.columns
    assert result.df["test"].iloc[0] == 42


def test_runner_execute_multiple_plugins(registry):
    """Runner should execute plugins in order."""
    config = PipelineConfig(
        preprocessing=[PluginConfig(name="multiply", params={"factor": 10})],
        indicators=[PluginConfig(name="add_column", params={"column_name": "flag", "value": 1})]
    )
    runner = PipelineRunner(registry, config)

    df = pd.DataFrame({"C": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="TEST", asset_class="FOREX")

    result = runner.run(ctx)

    # Preprocessing ran first (multiply by 10)
    assert result.df["C"].iloc[0] == 10
    # Then indicators (add column)
    assert "flag" in result.df.columns


def test_runner_stateful_plugin_fit_transform(registry):
    """Runner should call fit() for stateful plugins on training data."""
    config = PipelineConfig(
        preprocessing=[PluginConfig(name="stateful_scaler", params={})]
    )
    runner = PipelineRunner(registry, config)

    # Training data
    train_df = pd.DataFrame({"C": [10, 20, 30]})  # mean = 20
    train_ctx = PipelineContext(df=train_df, symbol="TEST", asset_class="FOREX")

    # Fit on training data
    runner.fit(train_ctx)

    # Transform training data
    train_result = runner.run(train_ctx)
    assert train_result.df["C"].iloc[0] == -10  # 10 - 20
    assert train_result.df["C"].iloc[1] == 0    # 20 - 20

    # Transform test data (using fitted mean=20)
    test_df = pd.DataFrame({"C": [25, 35]})
    test_ctx = PipelineContext(df=test_df, symbol="TEST", asset_class="FOREX")
    test_result = runner.run(test_ctx)

    assert test_result.df["C"].iloc[0] == 5   # 25 - 20
    assert test_result.df["C"].iloc[1] == 15  # 35 - 20


def test_runner_validate_before_run(registry):
    """Runner should validate all plugins before running."""
    config = PipelineConfig(
        indicators=[PluginConfig(name="add_column", params={})]
    )
    runner = PipelineRunner(registry, config)

    validation = runner.validate()
    assert validation["add_column"]["valid"] == True


def test_runner_unknown_plugin_error(registry):
    """Runner should fail fast if plugin not found."""
    config = PipelineConfig(
        indicators=[PluginConfig(name="nonexistent", params={})]
    )
    runner = PipelineRunner(registry, config)

    with pytest.raises(Exception) as exc:
        runner.validate()

    assert "nonexistent" in str(exc.value)


def test_runner_reset_stateful_plugins(registry):
    """Runner should reset stateful plugins."""
    config = PipelineConfig(
        preprocessing=[PluginConfig(name="stateful_scaler", params={})]
    )
    runner = PipelineRunner(registry, config)

    # Fit once
    train_df = pd.DataFrame({"C": [10, 20, 30]})
    train_ctx = PipelineContext(df=train_df, symbol="TEST", asset_class="FOREX")
    runner.fit(train_ctx)

    # Reset
    runner.reset()

    # Plugin should no longer be fitted
    # Running without fit should not subtract mean
    result = runner.run(train_ctx)
    assert result.df["C"].iloc[0] == 10  # Not transformed
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_runner.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'fwbg.pipeline.runner'"

**Step 3: Write minimal implementation**

```python
# src/fwbg/pipeline/runner.py
"""Pipeline runner for executing plugins in sequence."""
from typing import Dict, List, Optional, Callable
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.config import PipelineConfig, PluginConfig
from fwbg.pipeline.registry import PluginRegistry, PluginNotFoundError


class PipelineRunner:
    """
    Executes pipeline plugins in phase order.

    Handles:
    - Plugin instantiation from registry
    - Phase ordering (data_loading -> preprocessing -> ... -> validation)
    - Stateful plugin fit/transform pattern
    - Progress reporting
    """

    # Fixed phase execution order
    PHASE_ORDER = [
        ("data_loading", PluginPhase.DATA_LOADING),
        ("preprocessing", PluginPhase.PREPROCESSING),
        ("indicators", PluginPhase.INDICATORS),
        ("feature_selection", PluginPhase.FEATURE_SELECTION),
        ("labeling", PluginPhase.LABELING),
        ("model", PluginPhase.MODEL),
        ("validation", PluginPhase.VALIDATION),
    ]

    def __init__(
        self,
        registry: PluginRegistry,
        config: PipelineConfig,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ):
        """
        Initialize pipeline runner.

        Args:
            registry: Plugin registry with available plugins
            config: Pipeline configuration
            progress_callback: Optional callback(phase_name, current, total)
        """
        self.registry = registry
        self.config = config
        self.progress_callback = progress_callback

        # Instantiated plugins (name -> instance)
        self._instances: Dict[str, BasePlugin] = {}

        # Ordered list of (config, instance) for execution
        self._execution_order: List[tuple] = []

        self._initialized = False

    def _initialize(self) -> None:
        """Initialize plugin instances from config."""
        if self._initialized:
            return

        self._instances.clear()
        self._execution_order.clear()

        for phase_name, phase_enum in self.PHASE_ORDER:
            phase_configs = self.config.get_phase(phase_name)

            for plugin_config in phase_configs:
                # Get plugin class from registry
                plugin_cls = self.registry.get(plugin_config.name)

                # Create instance
                instance = plugin_cls()

                # Apply config overrides
                if plugin_config.stateful is not None:
                    instance.stateful = plugin_config.stateful
                if plugin_config.cacheable is not None:
                    instance.cacheable = plugin_config.cacheable

                self._instances[plugin_config.name] = instance
                self._execution_order.append((plugin_config, instance))

        self._initialized = True

    def validate(self) -> Dict[str, dict]:
        """
        Validate all configured plugins.

        Returns:
            Dict mapping plugin name to validation result

        Raises:
            PluginNotFoundError: If any plugin not found
        """
        self._initialize()

        results = {}
        for plugin_config, instance in self._execution_order:
            try:
                instance.validate()
                results[plugin_config.name] = {"valid": True, "error": None}
            except Exception as e:
                results[plugin_config.name] = {"valid": False, "error": str(e)}

        return results

    def fit(self, ctx: PipelineContext, **global_params) -> None:
        """
        Fit stateful plugins on training data.

        Args:
            ctx: Pipeline context with training data
            **global_params: Additional parameters passed to all plugins
        """
        self._initialize()

        for plugin_config, instance in self._execution_order:
            if instance.stateful:
                # Merge default params, config params, and global params
                params = {
                    **instance.get_default_params(),
                    **plugin_config.params,
                    **global_params,
                }
                instance.fit(ctx, **params)

    def run(
        self,
        ctx: PipelineContext,
        phases: Optional[List[str]] = None,
        **global_params
    ) -> PipelineContext:
        """
        Execute pipeline on context.

        Args:
            ctx: Pipeline context to process
            phases: Optional list of phases to run (default: all)
            **global_params: Additional parameters passed to all plugins

        Returns:
            Updated PipelineContext
        """
        self._initialize()

        # Determine which phases to run
        if phases is None:
            phases_to_run = {p[0] for p in self.PHASE_ORDER}
        else:
            phases_to_run = set(phases)

        total_plugins = len(self._execution_order)

        for i, (plugin_config, instance) in enumerate(self._execution_order):
            # Check if this plugin's phase should run
            phase_name = instance.phase.value
            if phase_name not in phases_to_run:
                continue

            # Report progress
            if self.progress_callback:
                self.progress_callback(phase_name, i + 1, total_plugins)

            # Merge parameters
            params = {
                **instance.get_default_params(),
                **plugin_config.params,
                **global_params,
            }

            # Execute plugin
            ctx = instance.execute(ctx, **params)

        return ctx

    def reset(self) -> None:
        """Reset all stateful plugins."""
        for instance in self._instances.values():
            if instance.stateful:
                instance.reset()

    def get_instance(self, name: str) -> Optional[BasePlugin]:
        """Get instantiated plugin by name."""
        self._initialize()
        return self._instances.get(name)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_runner.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/fwbg/pipeline/runner.py tests/pipeline/test_runner.py
git commit -m "feat(pipeline): add PipelineRunner with phase ordering and stateful support"
```

---

### Task 6: Update pipeline __init__.py exports

**Files:**
- Modify: `src/fwbg/pipeline/__init__.py`
- Test: `tests/pipeline/test_imports.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_imports.py
import pytest


def test_pipeline_public_api():
    """Pipeline module should export main classes."""
    from fwbg.pipeline import (
        PipelineContext,
        BasePlugin,
        PluginPhase,
        PluginRegistry,
        PluginNotFoundError,
        PipelineConfig,
        PluginConfig,
        PipelineRunner,
        get_registry,
    )

    # All imports should work
    assert PipelineContext is not None
    assert BasePlugin is not None
    assert PluginPhase is not None
    assert PluginRegistry is not None
    assert PipelineRunner is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_imports.py -v`
Expected: FAIL with "ImportError: cannot import name 'PipelineContext'"

**Step 3: Write minimal implementation**

```python
# src/fwbg/pipeline/__init__.py
"""
FWBG Pipeline System

A modular plugin system for building trading strategy pipelines.

Phases (in execution order):
1. data_loading - Load raw market data
2. preprocessing - Transform data (e.g., fractional differentiation)
3. indicators - Compute technical indicators
4. feature_selection - Select relevant features
5. labeling - Generate training labels
6. model - Train/predict with ML models
7. validation - Validate strategy performance

Usage:
    from fwbg.pipeline import PipelineRunner, PipelineConfig, get_registry

    # Load strategy config
    config = parse_pipeline_config(strategy_dict)

    # Create runner
    runner = PipelineRunner(get_registry(), config)

    # Validate plugins
    runner.validate()

    # Fit on training data
    runner.fit(train_ctx)

    # Run pipeline
    result = runner.run(test_ctx)
"""

from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.registry import (
    PluginRegistry,
    PluginNotFoundError,
    PluginValidationError,
    get_registry,
    reset_registry,
)
from fwbg.pipeline.config import (
    PluginConfig,
    PipelineConfig,
    parse_pipeline_config,
)
from fwbg.pipeline.runner import PipelineRunner

__all__ = [
    # Context
    "PipelineContext",
    # Base
    "BasePlugin",
    "PluginPhase",
    # Registry
    "PluginRegistry",
    "PluginNotFoundError",
    "PluginValidationError",
    "get_registry",
    "reset_registry",
    # Config
    "PluginConfig",
    "PipelineConfig",
    "parse_pipeline_config",
    # Runner
    "PipelineRunner",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_imports.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/fwbg/pipeline/__init__.py tests/pipeline/test_imports.py
git commit -m "feat(pipeline): export public API from pipeline module"
```

---

## Phase 2: Migrate Existing Plugins

### Task 7: Migrate FractionalDiff preprocessor to new system

**Files:**
- Modify: `src/fwbg/builtins/preprocessing/fractional_diff/__init__.py`
- Test: `tests/pipeline/test_fractional_diff_migration.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_fractional_diff_migration.py
import pytest
import pandas as pd
import numpy as np
from fwbg.pipeline import BasePlugin, PluginPhase, PipelineContext


def test_fractional_diff_is_baseplugin():
    """FractionalDiff should be a BasePlugin subclass."""
    from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor

    assert issubclass(FractionalDiffPreprocessor, BasePlugin)
    assert FractionalDiffPreprocessor.phase == PluginPhase.PREPROCESSING


def test_fractional_diff_attributes():
    """FractionalDiff should have required plugin attributes."""
    from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor

    assert FractionalDiffPreprocessor.name == "fractional_diff"
    assert FractionalDiffPreprocessor.version is not None
    assert FractionalDiffPreprocessor.stateful == True  # Requires fit


def test_fractional_diff_execute():
    """FractionalDiff execute should transform DataFrame."""
    from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor

    # Create test data
    np.random.seed(42)
    df = pd.DataFrame({
        "O": np.cumsum(np.random.randn(500)) + 100,
        "H": np.cumsum(np.random.randn(500)) + 101,
        "L": np.cumsum(np.random.randn(500)) + 99,
        "C": np.cumsum(np.random.randn(500)) + 100,
    })

    ctx = PipelineContext(df=df.copy(), symbol="TEST", asset_class="FOREX")

    plugin = FractionalDiffPreprocessor()
    plugin.fit(ctx, auto_d=False, default_d=0.4, columns=["C"])

    result = plugin.execute(ctx, auto_d=False, default_d=0.4, columns=["C"])

    # C should be transformed (fractionally differenced)
    assert "C" in result.df.columns
    # Values should be different from original
    assert not np.allclose(result.df["C"].dropna().values[:10], df["C"].values[:10])


def test_fractional_diff_validate():
    """FractionalDiff validate should return True when ready."""
    from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor

    plugin = FractionalDiffPreprocessor()
    assert plugin.validate() == True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_fractional_diff_migration.py -v`
Expected: FAIL (FractionalDiffPreprocessor doesn't inherit from BasePlugin yet)

**Step 3: Modify the existing FractionalDiff to inherit from BasePlugin**

First, read the current implementation:

```python
# src/fwbg/builtins/preprocessing/fractional_diff/__init__.py
"""
Fractional Differentiation Preprocessor (migrated to new plugin system).
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from statsmodels.tsa.stattools import adfuller

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.core.registry import register_preprocessor


def _get_weights(d: float, size: int) -> np.ndarray:
    """Compute weights for fractional differentiation."""
    weights = [1.0]
    for k in range(1, size):
        weights.append(-weights[-1] * (d - k + 1) / k)
    return np.array(weights[::-1])


def _frac_diff(series: pd.Series, d: float, threshold: float = 1e-5, max_window: int = 500) -> pd.Series:
    """Apply fractional differentiation to a series."""
    weights = _get_weights(d, min(len(series), max_window))
    weights = weights[np.abs(weights) > threshold]
    width = len(weights)

    if width > len(series):
        return pd.Series(index=series.index, dtype=float)

    result = np.convolve(series.values, weights, mode='valid')

    # Align index
    idx = series.index[width - 1:]
    return pd.Series(result, index=idx)


def _find_optimal_d(series: pd.Series, max_d: float = 1.0, step: float = 0.1, p_value: float = 0.05) -> float:
    """Find minimum d for stationarity via ADF test."""
    for d in np.arange(0.0, max_d + step, step):
        diff_series = _frac_diff(series, d)
        diff_series = diff_series.dropna()

        if len(diff_series) < 20:
            continue

        try:
            adf_result = adfuller(diff_series, maxlag=1, autolag=None)
            if adf_result[1] < p_value:
                return round(d, 2)
        except Exception:
            continue

    return max_d


@register_preprocessor("fractional_diff")
class FractionalDiffPreprocessor(BasePlugin):
    """
    Fractional Differentiation Preprocessor.

    Transforms price series to achieve stationarity while preserving
    memory/signal. Uses fractional differencing with configurable d parameter.

    This is a STATEFUL plugin:
    - fit(): Learns optimal d from training data (if auto_d=True)
    - execute(): Applies transformation using learned/configured d
    """

    name = "fractional_diff"
    version = "2.0.0"
    phase = PluginPhase.PREPROCESSING
    stateful = True
    cacheable = False  # Transform depends on fitted state

    def __init__(self):
        super().__init__()
        self.d_: Optional[float] = None
        self.history_: Optional[pd.DataFrame] = None

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "auto_d": False,
            "default_d": 0.4,
            "columns": ["O", "H", "L", "C"],
            "threshold": 1e-5,
            "max_window": 500,
        }

    def validate(self) -> bool:
        """Validate plugin is ready to run."""
        # Check dependencies
        try:
            import statsmodels
            return True
        except ImportError:
            raise ImportError("statsmodels required for fractional_diff plugin")

    def fit(self, ctx: PipelineContext, **params) -> None:
        """
        Fit preprocessor on training data.

        If auto_d=True, finds optimal d via ADF test.
        Stores history for handling validation/test data.
        """
        merged_params = {**self.get_default_params(), **params}

        auto_d = merged_params["auto_d"]
        default_d = merged_params["default_d"]
        columns = merged_params["columns"]

        # Find or use default d
        if auto_d and "C" in ctx.df.columns:
            self.d_ = _find_optimal_d(ctx.df["C"])
        else:
            self.d_ = default_d

        # Store history for validation/test transformation
        # Need enough history to compute fractional diff without NaNs
        history_size = min(500, len(ctx.df))
        self.history_ = ctx.df[columns].iloc[-history_size:].copy()

        # Store in metadata for downstream access
        ctx.metadata["fractional_diff_d"] = self.d_

        self._fitted = True

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        """
        Apply fractional differentiation.

        For training data: transform directly
        For test data: prepend history to avoid edge effects
        """
        merged_params = {**self.get_default_params(), **params}

        columns = merged_params["columns"]
        threshold = merged_params["threshold"]
        max_window = merged_params["max_window"]

        # Use fitted d or default
        d = self.d_ if self.d_ is not None else merged_params["default_d"]

        df = ctx.df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col]

            # If we have history and this looks like test data, prepend history
            if self.history_ is not None and col in self.history_.columns:
                # Check if this is different data than what we fitted on
                if len(series) < len(self.history_) or not series.index.equals(self.history_.index):
                    # Prepend history
                    combined = pd.concat([self.history_[col], series])
                    transformed = _frac_diff(combined, d, threshold, max_window)
                    # Keep only the original portion
                    df[col] = transformed.reindex(series.index)
                    continue

            # Direct transformation
            df[col] = _frac_diff(series, d, threshold, max_window)

        ctx.df = df
        return ctx

    def reset(self) -> None:
        """Reset fitted state."""
        super().reset()
        self.d_ = None
        self.history_ = None

    # Legacy compatibility methods (for old code that uses fit/transform pattern)
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Legacy transform method for backward compatibility."""
        ctx = PipelineContext(df=df, symbol="", asset_class="")
        result = self.execute(ctx, **params)
        return result.df
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_fractional_diff_migration.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/fwbg/builtins/preprocessing/fractional_diff/__init__.py tests/pipeline/test_fractional_diff_migration.py
git commit -m "refactor(fractional_diff): migrate to BasePlugin interface"
```

---

### Task 8: Create base indicator plugin and migrate TrendIndicators

**Files:**
- Modify: `src/fwbg/builtins/indicators/trend/__init__.py`
- Test: `tests/pipeline/test_trend_migration.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_trend_migration.py
import pytest
import pandas as pd
import numpy as np
from fwbg.pipeline import BasePlugin, PluginPhase, PipelineContext


def test_trend_is_baseplugin():
    """Trend should be a BasePlugin subclass."""
    from fwbg.builtins.indicators.trend import TrendIndicators

    assert issubclass(TrendIndicators, BasePlugin)
    assert TrendIndicators.phase == PluginPhase.INDICATORS


def test_trend_attributes():
    """Trend should have required plugin attributes."""
    from fwbg.builtins.indicators.trend import TrendIndicators

    assert TrendIndicators.name == "trend"
    assert TrendIndicators.version is not None
    assert TrendIndicators.stateful == False  # Indicators are stateless


def test_trend_execute():
    """Trend execute should add indicator columns."""
    from fwbg.builtins.indicators.trend import TrendIndicators

    # Create test OHLCV data
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "O": np.cumsum(np.random.randn(n)) + 100,
        "H": np.cumsum(np.random.randn(n)) + 101,
        "L": np.cumsum(np.random.randn(n)) + 99,
        "C": np.cumsum(np.random.randn(n)) + 100,
        "V": np.random.randint(1000, 10000, n),
    })
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)

    ctx = PipelineContext(df=df.copy(), symbol="TEST", asset_class="FOREX")

    plugin = TrendIndicators()
    result = plugin.execute(ctx, ema_periods=[8, 21])

    # Should have EMA columns
    assert "trend_ema_8" in result.df.columns or any("ema" in c.lower() for c in result.df.columns)


def test_trend_get_feature_columns():
    """Trend should report created feature columns."""
    from fwbg.builtins.indicators.trend import TrendIndicators

    plugin = TrendIndicators()

    # After compute, should report features
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "O": np.cumsum(np.random.randn(n)) + 100,
        "H": np.cumsum(np.random.randn(n)) + 101,
        "L": np.cumsum(np.random.randn(n)) + 99,
        "C": np.cumsum(np.random.randn(n)) + 100,
        "V": np.random.randint(1000, 10000, n),
    })
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)

    ctx = PipelineContext(df=df, symbol="TEST", asset_class="FOREX")
    result = plugin.execute(ctx, ema_periods=[8])

    features = plugin.get_feature_columns()
    assert len(features) > 0


def test_trend_validate():
    """Trend validate should return True when ready."""
    from fwbg.builtins.indicators.trend import TrendIndicators

    plugin = TrendIndicators()
    assert plugin.validate() == True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_trend_migration.py -v`
Expected: FAIL (TrendIndicators doesn't inherit from BasePlugin yet)

**Step 3: Modify TrendIndicators to inherit from BasePlugin**

The implementation will:
1. Keep existing indicator logic
2. Add BasePlugin inheritance
3. Wrap compute() in execute()
4. Track created feature columns

```python
# src/fwbg/builtins/indicators/trend/__init__.py
"""
Trend Indicators Plugin (migrated to new plugin system).
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import ta

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.core.registry import register_indicator
from fwbg.plugins.indicator import safe_divide, shift_features


@register_indicator("trend")
class TrendIndicators(BasePlugin):
    """
    Trend Indicators Plugin.

    Computes trend-following indicators:
    - ADX (Average Directional Index)
    - EMA (Exponential Moving Average)
    - SMA (Simple Moving Average)
    - MACD (Moving Average Convergence Divergence)
    - CCI (Commodity Channel Index)
    - Aroon Indicator
    - Efficiency Ratio

    This is a STATELESS plugin - no fit required.
    """

    name = "trend"
    version = "2.0.0"
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True

    def __init__(self):
        super().__init__()
        self._feature_columns: List[str] = []

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "adx_periods": [7, 14, 21],
            "ema_periods": [8, 21, 50, 100, 200],
            "sma_periods": [20, 50, 200],
            "cci_periods": [14, 20],
            "aroon_period": 25,
            "er_periods": [10, 20, 50],
        }

    def validate(self) -> bool:
        """Validate plugin is ready to run."""
        try:
            import ta
            return True
        except ImportError:
            raise ImportError("ta library required for trend indicators")

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        """Compute trend indicators and add to DataFrame."""
        merged_params = {**self.get_default_params(), **params}

        df = ctx.df
        features: Dict[str, pd.Series] = {}

        high = df["H"]
        low = df["L"]
        close = df["C"]

        # ADX
        for period in merged_params["adx_periods"]:
            try:
                adx = ta.trend.ADXIndicator(high, low, close, window=period)
                features[f"trend_adx_{period}"] = adx.adx()
                features[f"trend_plus_di_{period}"] = adx.adx_pos()
                features[f"trend_minus_di_{period}"] = adx.adx_neg()
                features[f"trend_di_diff_{period}"] = features[f"trend_plus_di_{period}"] - features[f"trend_minus_di_{period}"]
            except Exception:
                pass

        # EMA
        for period in merged_params["ema_periods"]:
            ema = close.ewm(span=period, adjust=False).mean()
            features[f"trend_ema_{period}"] = ema
            features[f"trend_ema_{period}_dist"] = safe_divide(close - ema, ema) * 100

        # EMA Cross features
        ema_periods = sorted(merged_params["ema_periods"])
        for i, fast in enumerate(ema_periods[:-1]):
            slow = ema_periods[i + 1]
            fast_ema = close.ewm(span=fast, adjust=False).mean()
            slow_ema = close.ewm(span=slow, adjust=False).mean()
            features[f"trend_ema_cross_{fast}_{slow}"] = safe_divide(fast_ema - slow_ema, slow_ema) * 100

        # SMA
        for period in merged_params["sma_periods"]:
            sma = close.rolling(window=period).mean()
            features[f"trend_sma_{period}"] = sma
            features[f"trend_sma_{period}_dist"] = safe_divide(close - sma, sma) * 100

        # MACD
        try:
            macd = ta.trend.MACD(close)
            features["trend_macd"] = macd.macd()
            features["trend_macd_signal"] = macd.macd_signal()
            features["trend_macd_diff"] = macd.macd_diff()
        except Exception:
            pass

        # CCI
        for period in merged_params["cci_periods"]:
            try:
                features[f"trend_cci_{period}"] = ta.trend.CCIIndicator(high, low, close, window=period).cci()
            except Exception:
                pass

        # Aroon
        aroon_period = merged_params["aroon_period"]
        try:
            aroon = ta.trend.AroonIndicator(high, low, window=aroon_period)
            features["trend_aroon_up"] = aroon.aroon_up()
            features["trend_aroon_down"] = aroon.aroon_down()
            features["trend_aroon_ind"] = aroon.aroon_indicator()
        except Exception:
            pass

        # Efficiency Ratio
        for period in merged_params["er_periods"]:
            change = (close - close.shift(period)).abs()
            volatility = close.diff().abs().rolling(window=period).sum()
            features[f"trend_er_{period}"] = safe_divide(change, volatility)

        # Convert to DataFrame and shift to prevent lookahead
        features_df = shift_features(features, df.index)

        # Store feature columns
        self._feature_columns = list(features_df.columns)

        # Concatenate with original DataFrame
        ctx.df = pd.concat([df, features_df], axis=1)

        return ctx

    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns created by this plugin."""
        return self._feature_columns

    # Legacy compatibility
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Legacy compute method for backward compatibility."""
        ctx = PipelineContext(df=df, symbol="", asset_class="")
        result = self.execute(ctx, **params)
        return result.df
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_trend_migration.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/fwbg/builtins/indicators/trend/__init__.py tests/pipeline/test_trend_migration.py
git commit -m "refactor(trend): migrate to BasePlugin interface"
```

---

### Task 9: Create manifest.json for core plugins

**Files:**
- Create: `src/fwbg/plugins/fwbg_core/manifest.json`
- Test: `tests/pipeline/test_core_manifest.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_core_manifest.py
import pytest
import json
from pathlib import Path


def test_core_manifest_exists():
    """Core plugins should have a manifest.json."""
    manifest_path = Path("src/fwbg/plugins/fwbg_core/manifest.json")
    assert manifest_path.exists(), f"Missing {manifest_path}"


def test_core_manifest_valid_json():
    """Core manifest should be valid JSON."""
    manifest_path = Path("src/fwbg/plugins/fwbg_core/manifest.json")
    content = manifest_path.read_text()
    manifest = json.loads(content)

    assert "package" in manifest
    assert "version" in manifest


def test_core_manifest_required_fields():
    """Core manifest should have all required fields."""
    manifest_path = Path("src/fwbg/plugins/fwbg_core/manifest.json")
    manifest = json.loads(manifest_path.read_text())

    assert manifest["package"] == "fwbg_core"
    assert "version" in manifest
    assert "author" in manifest
    assert "dependencies" in manifest
    assert isinstance(manifest["dependencies"], dict)


def test_core_manifest_dependencies():
    """Core manifest should declare Python package dependencies."""
    manifest_path = Path("src/fwbg/plugins/fwbg_core/manifest.json")
    manifest = json.loads(manifest_path.read_text())

    deps = manifest["dependencies"]

    # Core plugins need these
    assert "pandas" in deps
    assert "numpy" in deps
    assert "ta" in deps  # For indicators
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/pipeline/test_core_manifest.py -v`
Expected: FAIL with "AssertionError: Missing src/fwbg/plugins/fwbg_core/manifest.json"

**Step 3: Create the manifest and directory structure**

```bash
mkdir -p src/fwbg/plugins/fwbg_core
```

```json
{
  "package": "fwbg_core",
  "version": "2.0.0",
  "author": "fwbg-team",
  "description": "Core plugins for FWBG trading strategy backtester",
  "license": "MIT",
  "dependencies": {
    "pandas": ">=1.5.0",
    "numpy": ">=1.20.0",
    "ta": ">=0.10.0",
    "statsmodels": ">=0.13.0",
    "xgboost": ">=1.7.0",
    "scikit-learn": ">=1.0.0"
  },
  "plugin_dependencies": {}
}
```

Save to: `src/fwbg/plugins/fwbg_core/manifest.json`

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/pipeline/test_core_manifest.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/fwbg/plugins/fwbg_core/
git commit -m "feat(plugins): add manifest.json for core plugins"
```

---

### Task 10: Migrate remaining indicator plugins (template task - repeat for each)

**Note:** This task should be repeated for each indicator plugin:
- momentum
- volatility
- regime
- structure
- risk
- price_action
- time_season
- distribution
- dynamics
- multi_timeframe
- cross_features
- ichimoku
- microstructure
- macro_surprise

The pattern is identical to Task 8 (TrendIndicators). For each:

1. Add `from fwbg.pipeline.base import BasePlugin, PluginPhase`
2. Add `from fwbg.pipeline.context import PipelineContext`
3. Change class to inherit from `BasePlugin`
4. Add class attributes: `name`, `version`, `phase`, `stateful`, `cacheable`
5. Add `execute(self, ctx, **params) -> PipelineContext` wrapping existing `compute()`
6. Add `validate(self) -> bool`
7. Add `get_feature_columns(self) -> List[str]`
8. Keep legacy `compute()` for backward compatibility

**Example for MomentumIndicators:**

```python
@register_indicator("momentum")
class MomentumIndicators(BasePlugin):
    name = "momentum"
    version = "2.0.0"
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True

    # ... rest follows same pattern as TrendIndicators
```

---

## Phase 3: Integrate with Process.py

### Task 11: Create pipeline integration in process.py

**Files:**
- Modify: `src/fwbg/optimization/process.py`
- Test: `tests/optimization/test_process_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/optimization/test_process_pipeline.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def test_process_uses_pipeline_runner():
    """process_symbol should use PipelineRunner for data processing."""
    from fwbg.optimization.process import _create_pipeline_runner
    from fwbg.core.config import StrategyConfig

    # Create minimal strategy config with new pipeline format
    strategy_dict = {
        "name": "Test",
        "pipeline": {
            "preprocessing": [
                {"name": "fractional_diff", "params": {"d": 0.4}}
            ],
            "indicators": [
                {"name": "trend", "params": {"ema_periods": [8, 21]}}
            ]
        }
    }

    strategy = StrategyConfig.from_dict(strategy_dict)
    runner = _create_pipeline_runner(strategy)

    assert runner is not None
    # Should have configured plugins
    assert runner.get_instance("fractional_diff") is not None
    assert runner.get_instance("trend") is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/optimization/test_process_pipeline.py -v`
Expected: FAIL (function doesn't exist)

**Step 3: Add pipeline integration to process.py**

Add a new function at the top of process.py:

```python
# Add to imports at top of src/fwbg/optimization/process.py
from fwbg.pipeline import (
    PipelineRunner,
    PipelineContext,
    parse_pipeline_config,
    get_registry,
)

def _create_pipeline_runner(strategy: StrategyConfig) -> PipelineRunner:
    """
    Create a PipelineRunner from strategy configuration.

    Args:
        strategy: Strategy configuration (new or legacy format)

    Returns:
        Configured PipelineRunner
    """
    registry = get_registry()

    # Check if strategy uses new pipeline format
    if hasattr(strategy, 'pipeline') and strategy.pipeline:
        config = parse_pipeline_config({"pipeline": strategy.pipeline})
    else:
        # Convert legacy format to new format
        config = _convert_legacy_config(strategy)

    return PipelineRunner(registry, config)


def _convert_legacy_config(strategy: StrategyConfig) -> "PipelineConfig":
    """Convert legacy strategy config to new pipeline config."""
    from fwbg.pipeline.config import PipelineConfig, PluginConfig

    preprocessing = []
    for name in (strategy.preprocessing or []):
        params = strategy.preprocessing_params.get(name, {})
        preprocessing.append(PluginConfig(name=name, params=params))

    indicators = []
    for name in (strategy.indicators or []):
        params = strategy.indicator_params.get(name, {})
        indicators.append(PluginConfig(name=name, params=params))

    feature_selection = []
    if strategy.feature_selector:
        params = strategy.feature_params or {}
        feature_selection.append(PluginConfig(name=strategy.feature_selector, params=params))

    return PipelineConfig(
        preprocessing=preprocessing,
        indicators=indicators,
        feature_selection=feature_selection,
    )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/optimization/test_process_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/fwbg/optimization/process.py tests/optimization/test_process_pipeline.py
git commit -m "feat(process): add pipeline runner integration"
```

---

### Task 12: Replace indicator computation with pipeline runner

**Files:**
- Modify: `src/fwbg/optimization/process.py`
- Test: `tests/optimization/test_process_indicators.py`

This task modifies the main processing loop in `process_symbol()` to use the PipelineRunner instead of direct `compute_indicator_pool()` calls.

**Key changes:**
1. Create PipelineRunner at start
2. Validate all plugins before processing
3. For stateless indicators (no preprocessing): run once, slice
4. For stateful preprocessing: fit on train, run per fold

The detailed implementation follows the same TDD pattern as previous tasks.

---

## Phase 4: User Plugin Directory Support

### Task 13: Add user plugin directory discovery

**Files:**
- Modify: `src/fwbg/pipeline/registry.py`
- Test: `tests/pipeline/test_user_plugins.py`

**Step 1: Write the failing test**

```python
# tests/pipeline/test_user_plugins.py
import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch


def test_discover_user_plugins_directory():
    """Registry should discover plugins from ~/.fwbg/plugins/"""
    from fwbg.pipeline.registry import PluginRegistry, get_user_plugins_dir

    # Should return path to user plugins dir
    user_dir = get_user_plugins_dir()
    assert "fwbg" in str(user_dir).lower()
    assert "plugins" in str(user_dir).lower()


def test_registry_auto_discover():
    """Registry auto_discover should load from core and user directories."""
    from fwbg.pipeline.registry import PluginRegistry

    registry = PluginRegistry()

    # Should have method for auto-discovery
    assert hasattr(registry, 'auto_discover')


def test_user_plugin_loaded(tmp_path):
    """User plugins should be loaded and available."""
    from fwbg.pipeline.registry import PluginRegistry
    from fwbg.pipeline.base import BasePlugin, PluginPhase

    # Create a mock user plugin
    user_plugins = tmp_path / "plugins"
    user_plugins.mkdir()

    pkg_dir = user_plugins / "my_custom_plugins"
    pkg_dir.mkdir()

    # Manifest
    manifest = {"package": "my_custom_plugins", "version": "1.0.0", "author": "test", "dependencies": {}}
    (pkg_dir / "manifest.json").write_text(json.dumps(manifest))

    # Plugin code
    plugin_code = '''
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext

class MyCustomIndicator(BasePlugin):
    name = "my_custom_indicator"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS

    def execute(self, ctx, **params):
        ctx.df["custom"] = 42
        return ctx

    def validate(self):
        return True
'''
    (pkg_dir / "__init__.py").write_text(plugin_code)

    # Discover
    registry = PluginRegistry()
    discovered = registry.discover_from_directory(user_plugins)

    assert "my_custom_indicator" in discovered

    # Should be usable
    plugin_cls = registry.get("my_custom_indicator")
    assert plugin_cls.name == "my_custom_indicator"
```

**Step 2-5:** Follow TDD pattern to implement.

Add to `registry.py`:

```python
def get_user_plugins_dir() -> Path:
    """Get user plugins directory (~/.fwbg/plugins/)."""
    return Path.home() / ".fwbg" / "plugins"


def get_core_plugins_dir() -> Path:
    """Get core plugins directory."""
    return Path(__file__).parent.parent / "plugins"


class PluginRegistry:
    # ... existing code ...

    def auto_discover(self) -> List[str]:
        """
        Automatically discover plugins from core and user directories.

        Returns:
            List of discovered plugin names
        """
        discovered = []

        # Core plugins first
        core_dir = get_core_plugins_dir()
        if core_dir.exists():
            discovered.extend(self.discover_from_directory(core_dir))

        # User plugins (can override core)
        user_dir = get_user_plugins_dir()
        if user_dir.exists():
            discovered.extend(self.discover_from_directory(user_dir))

        return discovered
```

---

## Phase 5: Update Strategy Config Schema

### Task 14: Update StrategyConfig to support new pipeline format

**Files:**
- Modify: `src/fwbg/core/config.py`
- Test: `tests/core/test_config_pipeline.py`

Add `pipeline` field to StrategyConfig that accepts the new format while maintaining backward compatibility with legacy format.

---

## Phase 6: Documentation and Examples

### Task 15: Create example custom plugin

**Files:**
- Create: `examples/plugins/example_indicator/__init__.py`
- Create: `examples/plugins/example_indicator/manifest.json`
- Test: `tests/examples/test_example_plugin.py`

Create a fully documented example plugin that third-party developers can use as a template.

---

### Task 16: Update strategy example files

**Files:**
- Modify: `strategies/exploration.json` (convert to new format)
- Create: `strategies/exploration_v2.json` (new format example)

---

## Summary

**Total Tasks:** 16 main tasks (Task 10 repeats for 14 indicators = ~30 total)

**Key Deliverables:**
1. `src/fwbg/pipeline/` - New pipeline module
   - `context.py` - PipelineContext
   - `base.py` - BasePlugin, PluginPhase
   - `registry.py` - PluginRegistry with directory discovery
   - `config.py` - PluginConfig, PipelineConfig parsers
   - `runner.py` - PipelineRunner orchestration

2. Migrated plugins:
   - All 15 indicator plugins
   - FractionalDiff preprocessor
   - Boruta feature selector

3. Core plugin manifest: `src/fwbg/plugins/fwbg_core/manifest.json`

4. Integration: `process.py` updated to use PipelineRunner

5. User plugins: `~/.fwbg/plugins/` directory support

**Testing Strategy:**
- Each task has dedicated tests
- Run full test suite after each task: `PYTHONPATH=src pytest tests/ -v`
- Integration test at end with real strategy file
